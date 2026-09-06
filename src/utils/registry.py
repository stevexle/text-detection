"""
Modular Registry Pattern for Plug-and-Play Component Management.
Inspired by OpenMMLab / MMDetection registry architecture.
"""

from typing import Any, Callable, Dict, Optional, Union
import inspect


class Registry:
    """Registry to map component names to classes and build instances from configs."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, type]:
        return self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict or key.lower() in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    def get(self, key: str) -> Optional[type]:
        """Get module class by name."""
        if key in self._module_dict:
            return self._module_dict[key]
        if key.lower() in self._module_dict:
            return self._module_dict[key.lower()]
        return None

    def _register_module(self, module_class: type, module_name: Optional[str] = None, force: bool = False):
        if not inspect.isclass(module_class) and not inspect.isfunction(module_class):
            raise TypeError(f"module must be a class or function, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__

        names = [module_name, module_name.lower()]
        for name in names:
            if not force and name in self._module_dict and self._module_dict[name] != module_class:
                raise KeyError(f"'{name}' is already registered in '{self.name}' registry.")
            self._module_dict[name] = module_class

    def register_module(
        self,
        name: Optional[Union[str, type]] = None,
        force: bool = False,
        module: Optional[type] = None,
    ) -> Union[type, Callable]:
        """Register module either as a decorator or direct function call."""
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        if inspect.isclass(name) or inspect.isfunction(name):
            self._register_module(module_class=name, force=force)
            return name

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def build(self, cfg: Union[Dict[str, Any], Any], **default_args) -> Any:
        """Build a module from configuration dictionary."""
        if not isinstance(cfg, dict):
            return cfg

        args = cfg.copy()
        if default_args:
            for k, v in default_args.items():
                args.setdefault(k, v)

        if "type" not in args:
            raise KeyError(f"Config must contain the key 'type' to build from {self.name} registry, got: {args}")

        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"'{obj_type}' is not registered in '{self.name}' registry. Available: {list(self._module_dict.keys())}")
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or class, but got {type(obj_type)}")

        try:
            return obj_cls(**args)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate '{obj_cls.__name__}' from '{self.name}' registry: {e}") from e
