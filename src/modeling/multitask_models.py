"""Multitask deep models for apnea/no-apnea + optional sleep staging."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]

    class _ModuleBase:  # pragma: no cover
        pass

    class _NNFallback:  # pragma: no cover
        Module = _ModuleBase

    nn = _NNFallback()  # type: ignore[assignment]

from modeling.deep_models import EpochCNNEncoder, TemporalConformerEncoder


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - runtime guard
        raise RuntimeError("torch is required for multitask deep models.")


class MultiTaskCNNModel(nn.Module):
    def __init__(self, *, embedding_dim: int, stage_num_classes: int) -> None:
        super().__init__()
        self.epoch_encoder = EpochCNNEncoder(embedding_dim=embedding_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.apnea_head = nn.Linear(64, 1)
        self.stage_head = nn.Linear(64, stage_num_classes)

    def _shared_features(self, x: "torch.Tensor") -> "torch.Tensor":
        center = x[:, x.size(1) // 2, :]
        return self.shared(self.epoch_encoder(center))

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        feats = self._shared_features(x)
        return {
            "apnea_logits": self.apnea_head(feats).squeeze(-1),
            "stage_logits": self.stage_head(feats),
        }


class MultiTaskConformerModel(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        embedding_dim: int,
        stage_num_classes: int,
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
        self.shared = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.apnea_head = nn.Linear(64, 1)
        self.stage_head = nn.Linear(64, stage_num_classes)

    def encode_sequence(self, x: "torch.Tensor") -> "torch.Tensor":
        bsz, seq_len, n_samples = x.shape
        flat = x.reshape(bsz * seq_len, n_samples)
        emb = self.epoch_encoder(flat).reshape(bsz, seq_len, -1)
        seq = self.temporal_encoder(emb)
        return seq[:, seq.size(1) // 2, :]

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        feats = self.shared(self.encode_sequence(x))
        return {
            "apnea_logits": self.apnea_head(feats).squeeze(-1),
            "stage_logits": self.stage_head(feats),
        }


def build_multitask_model(model_cfg: Mapping[str, Any], *, stage_num_classes: int) -> "nn.Module":
    _require_torch()
    model_type = str(model_cfg.get("type", "conformer")).lower()
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    if model_type == "cnn":
        return MultiTaskCNNModel(embedding_dim=embedding_dim, stage_num_classes=stage_num_classes)
    if model_type == "conformer":
        return MultiTaskConformerModel(
            sequence_length=int(model_cfg.get("sequence_length", 9)),
            embedding_dim=embedding_dim,
            stage_num_classes=stage_num_classes,
            num_layers=int(model_cfg.get("conformer_blocks", 2)),
            num_heads=int(model_cfg.get("attention_heads", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            conv_kernel_size=int(model_cfg.get("conv_kernel_size", 31)),
        )
    raise ValueError(f"Unsupported multitask model type {model_type!r}.")


def load_encoder_weights_from_checkpoint(model: "nn.Module", checkpoint: Mapping[str, Any]) -> None:
    """Load shared encoder weights from SSL, supervised, or multitask checkpoints."""
    epoch_state = checkpoint.get("epoch_encoder_state_dict")
    temporal_state = checkpoint.get("temporal_encoder_state_dict")
    if epoch_state:
        model.epoch_encoder.load_state_dict(dict(epoch_state), strict=True)
    if temporal_state and hasattr(model, "temporal_encoder"):
        model.temporal_encoder.load_state_dict(dict(temporal_state), strict=True)
    if epoch_state or temporal_state:
        return

    model_state = checkpoint.get("model_state_dict")
    if not model_state:
        return
    epoch_sub = {
        k.split("epoch_encoder.", 1)[1]: v
        for k, v in dict(model_state).items()
        if str(k).startswith("epoch_encoder.")
    }
    temporal_sub = {
        k.split("temporal_encoder.", 1)[1]: v
        for k, v in dict(model_state).items()
        if str(k).startswith("temporal_encoder.")
    }
    if epoch_sub:
        model.epoch_encoder.load_state_dict(epoch_sub, strict=True)
    if temporal_sub and hasattr(model, "temporal_encoder"):
        model.temporal_encoder.load_state_dict(temporal_sub, strict=True)
