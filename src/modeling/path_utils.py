from __future__ import annotations

from pathlib import Path


def resolve_path_any(path_str: str, config_path: Path, *, expect_dir: bool = False) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        kind = "directory" if expect_dir else "path"
        raise FileNotFoundError(f"{kind.capitalize()} not found (absolute path): {raw}")
    cand_cwd = (Path.cwd() / raw).resolve()
    if cand_cwd.exists():
        return cand_cwd
    cand_cfg = (config_path.resolve().parent / raw).resolve()
    if cand_cfg.exists():
        return cand_cfg
    kind = "directory" if expect_dir else "path"
    raise FileNotFoundError(f"{kind.capitalize()} not found: {path_str!r}\n  Tried: {cand_cwd}\n  Tried: {cand_cfg}")
