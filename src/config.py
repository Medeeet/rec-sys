"""Centralized configuration management."""

import yaml
import torch
from pathlib import Path
from copy import deepcopy


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class Config:
    """Loads and merges YAML configs."""

    @staticmethod
    def load(*config_paths: str, overrides: dict = None) -> dict:
        """Merge multiple YAML files, then apply overrides."""
        merged = {}
        for path in config_paths:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, data)
        if overrides:
            merged = _deep_merge(merged, overrides)
        return merged

    @staticmethod
    def get_device() -> torch.device:
        """Return cuda if available, else cpu."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_project_root() -> Path:
        """Return project root (works in Colab and locally)."""
        # Colab: typically /content/disser
        colab_path = Path("/content/disser")
        if colab_path.exists():
            return colab_path
        # Local: find by traversing up from this file
        current = Path(__file__).resolve().parent.parent
        return current

    @staticmethod
    def parse_cli_overrides(args: list[str]) -> dict:
        """Parse key=value pairs into nested dict.

        Example: ['model.embed_dim=128', 'training.lr=0.01']
        """
        overrides = {}
        if not args:
            return overrides
        for arg in args:
            key, value = arg.split("=", 1)
            parts = key.split(".")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            # Try to parse as int, float, bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            d[parts[-1]] = value
        return overrides
