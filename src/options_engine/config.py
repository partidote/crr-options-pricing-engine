"""Configuration loader and project path resolution."""

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML pipeline configuration from explicit path, CWD, or package root."""
    candidates = (
        [Path(path)]
        if path is not None
        else [Path.cwd() / "configs" / "config.yaml", DEFAULT_CONFIG_PATH]
    )
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found in {[str(c) for c in candidates]}")


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the active repository root and ensure it exists."""
    base = Path.cwd() if (Path.cwd() / "configs").exists() else PROJECT_ROOT
    full_path = base / relative_path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path