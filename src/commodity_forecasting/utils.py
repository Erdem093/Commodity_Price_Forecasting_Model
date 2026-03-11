from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def ensure_run_dirs(base_dir: str | Path, run_id: str) -> dict[str, Path]:
    root = Path(base_dir) / run_id
    subdirs = {
        "root": root,
        "datasets": root / "datasets",
        "models": root / "models",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "shap": root / "shap",
        "reports": root / "reports",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs
