"""
Configuration Parser with Dot-Notation Attribute Access.
Supports YAML, JSON, and Python dictionary configs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml


class ConfigDict(dict):
    """Dictionary subclass that enables attribute-style dot access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = self._wrap(value)

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert ConfigDict back to native python dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, ConfigDict) else v for v in value]
            else:
                result[key] = value
        return result


class Config:
    """Main Configuration container."""

    @classmethod
    def fromfile(cls, filename: Union[str, Path]) -> ConfigDict:
        """Load configuration from a YAML or JSON file."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.suffix in [".yaml", ".yml"]:
                cfg_dict = yaml.safe_load(f) or {}
            elif filepath.suffix == ".json":
                cfg_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {filepath.suffix}. Supported: [.yaml, .yml, .json]")

        return ConfigDict(cfg_dict)

    @classmethod
    def fromdict(cls, cfg_dict: Dict[str, Any]) -> ConfigDict:
        """Wrap a python dictionary into a ConfigDict."""
        return ConfigDict(cfg_dict)

    @staticmethod
    def dump(cfg: Union[ConfigDict, Dict[str, Any]], filepath: Optional[Union[str, Path]] = None) -> str:
        """Dump configuration to a YAML string or save to file."""
        d = cfg.to_dict() if isinstance(cfg, ConfigDict) else cfg
        yaml_str = yaml.dump(d, default_flow_style=False, sort_keys=False)
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str
