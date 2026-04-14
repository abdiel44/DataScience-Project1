"""Neural models for local-first sleep staging experiments."""

from __future__ import annotations

from typing import Any, Dict

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    class _ModuleBase:  # pragma: no cover - import-time fallback only
        pass

    class _NNFallback:  # pragma: no cover - import-time fallback only
        Module = _ModuleBase

    nn = _NNFallback()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover - runtime guard
        raise RuntimeError("torch is required for deep Phase E models.")


class EpochCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, ceil_mode=True),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feats = self.features(x).squeeze(-1)
        return self.proj(feats)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x).transpose(1, 2)
        out = F.glu(self.pointwise_in(out), dim=1)
        out = self.depthwise(out)
        out = self.batch_norm(out)
        out = F.silu(out)
        out = self.pointwise_out(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        *,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        q = self.attn_norm(x)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class TemporalConformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(x + self.pos_embedding[:, : x.size(1), :])
        for block in self.blocks:
            out = block(out)
        return self.final_norm(out)


class SleepCNNClassifier(nn.Module):
    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.epoch_encoder = EpochCNNEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = x[:, x.size(1) // 2, :]
        emb = self.epoch_encoder(center)
        return self.head(emb)


class SleepConformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        embedding_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        conv_kernel_size: int,
    ) -> None:
        super().__init__()
        self.epoch_encoder = EpochCNNEncoder(embedding_dim=embedding_dim)
        self.temporal_encoder = TemporalConformerEncoder(
            sequence_length=sequence_length,
            d_model=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_samples = x.shape
        flat = x.reshape(bsz * seq_len, n_samples)
        emb = self.epoch_encoder(flat).reshape(bsz, seq_len, -1)
        return self.temporal_encoder(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode_sequence(x)
        center = seq[:, seq.size(1) // 2, :]
        return self.head(center)


class ContrastiveConformerModel(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        embedding_dim: int,
        projection_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        conv_kernel_size: int,
    ) -> None:
        super().__init__()
        self.epoch_encoder = EpochCNNEncoder(embedding_dim=embedding_dim)
        self.temporal_encoder = TemporalConformerEncoder(
            sequence_length=sequence_length,
            d_model=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
        )
        self.projection_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_samples = x.shape
        flat = x.reshape(bsz * seq_len, n_samples)
        emb = self.epoch_encoder(flat).reshape(bsz, seq_len, -1)
        seq = self.temporal_encoder(emb)
        return seq[:, seq.size(1) // 2, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection_head(self.encode(x))
        return F.normalize(z, dim=-1)


def build_supervised_model(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    _require_torch()
    model_type = str(model_cfg.get("type", "conformer")).lower()
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    if model_type == "cnn":
        return SleepCNNClassifier(embedding_dim=embedding_dim, num_classes=num_classes)
    if model_type == "conformer":
        return SleepConformerClassifier(
            sequence_length=int(model_cfg.get("sequence_length", 9)),
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_layers=int(model_cfg.get("conformer_blocks", 2)),
            num_heads=int(model_cfg.get("attention_heads", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 31)),
        )
    raise ValueError(f"Unsupported deep model type {model_type!r}. Use 'cnn' or 'conformer'.")


def build_ssl_model(model_cfg: Dict[str, Any], ssl_cfg: Dict[str, Any]) -> ContrastiveConformerModel:
    _require_torch()
    return ContrastiveConformerModel(
        sequence_length=int(model_cfg.get("sequence_length", 9)),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        projection_dim=int(ssl_cfg.get("projection_dim", 64)),
        num_layers=int(model_cfg.get("conformer_blocks", 2)),
        num_heads=int(model_cfg.get("attention_heads", 4)),
        ffn_dim=int(model_cfg.get("ffn_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        conv_kernel_size=int(model_cfg.get("conv_kernel_size", 31)),
    )


def load_pretrained_encoder_weights(model: nn.Module, checkpoint: Dict[str, Any]) -> None:
    epoch_state = checkpoint.get("epoch_encoder_state_dict")
    temporal_state = checkpoint.get("temporal_encoder_state_dict")
    if epoch_state and hasattr(model, "epoch_encoder"):
        model.epoch_encoder.load_state_dict(epoch_state, strict=True)
    if temporal_state and hasattr(model, "temporal_encoder"):
        model.temporal_encoder.load_state_dict(temporal_state, strict=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
