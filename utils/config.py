"""YAML configuration loader with CLI override support."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_cli_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI argument overrides into config dict.

    Supports dot-notation keys like 'curriculum.enabled=false'.
    CLI values take precedence over config file values.

    Args:
        config: Base configuration dictionary.
        overrides: Dictionary of key-value pairs to override.

    Returns:
        Updated configuration dictionary.
    """
    for key, value in overrides.items():
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def get_nested(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from a nested config using dot-notation.

    Args:
        config: Configuration dictionary.
        key: Dot-separated key path (e.g., 'curriculum.enabled').
        default: Default value if key not found.

    Returns:
        The value at the key path, or default.
    """
    keys = key.split(".")
    d = config
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d
