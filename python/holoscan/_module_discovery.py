# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy aliases for independently packaged Holoscan modules."""

from __future__ import annotations

import keyword
import sys
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from importlib.metadata import entry_points
from threading import RLock
from types import ModuleType
from typing import Protocol

_ENTRY_POINT_GROUP = "holoscan.modules"
_PUBLIC_PACKAGE = "holoscan"


class _EntryPoint(Protocol):
    """Metadata used from one ``importlib.metadata`` entry point."""

    name: str
    value: str

    def load(self) -> object: ...


@dataclass
class _Provider:
    """Process-local state for one validated provider entry point."""

    entry_point: _EntryPoint
    public_name: str
    canonical_name: str
    lock: RLock = field(default_factory=RLock)
    module: ModuleType | None = None
    failure: str | None = None
    loading: bool = False


_index_lock = RLock()
_providers: dict[str, _Provider] = {}
_rejections: dict[str, str] = {}
_root_module: ModuleType | None = None
_reserved_names: frozenset[str] = frozenset()


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _metadata_entry_points() -> tuple[_EntryPoint, ...]:
    return tuple(entry_points(group=_ENTRY_POINT_GROUP))


def _canonical_module(entry_point: _EntryPoint) -> str:
    value = entry_point.value.strip()
    module_name, separator, _ = value.partition(":")
    if separator:
        raise ValueError("entry point must reference a module package, not an attribute")
    if not module_name or any(
        not part.isidentifier() or keyword.iskeyword(part) for part in module_name.split(".")
    ):
        raise ValueError(f"invalid provider package {module_name!r}")
    if module_name == _PUBLIC_PACKAGE or module_name.startswith(f"{_PUBLIC_PACKAGE}."):
        raise ValueError("provider package cannot target the public Holoscan namespace")
    return module_name


def _validate_public_name(name: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise ValueError(f"invalid public module name {name!r}")
    if name in _reserved_names:
        raise ValueError(f"public module name {name!r} is reserved by Holoscan Core")
    if (
        _root_module is not None
        and hasattr(_root_module, "__path__")
        and PathFinder.find_spec(f"{_PUBLIC_PACKAGE}.{name}", _root_module.__path__) is not None
    ):
        raise ValueError(f"public module name {name!r} would shadow an existing package")


def _build_index(entries: tuple[_EntryPoint, ...]) -> None:
    candidates: dict[str, list[_EntryPoint]] = {}
    for entry_point in entries:
        candidates.setdefault(entry_point.name, []).append(entry_point)

    for name, matches in candidates.items():
        try:
            _validate_public_name(name)
            if len(matches) != 1:
                raise ValueError(
                    f"Conflicting entry: multiple providers claim public module name {name!r}: "
                    + ", ".join(sorted(match.value for match in matches))
                )
            entry_point = matches[0]
            canonical_name = _canonical_module(entry_point)
        except (AttributeError, TypeError, ValueError) as error:
            message = f"Ignoring invalid {_ENTRY_POINT_GROUP} entry {name!r}: {error}"
            _rejections[name] = message
            _warn(message)
            continue
        _providers[name] = _Provider(entry_point, name, canonical_name)


def _alias_loaded_descendants(provider: _Provider) -> None:
    canonical_prefix = f"{provider.canonical_name}."
    public_prefix = f"{_PUBLIC_PACKAGE}.{provider.public_name}."
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, ModuleType) and name.startswith(canonical_prefix):
            sys.modules[f"{public_prefix}{name.removeprefix(canonical_prefix)}"] = module


def _publish_root(provider: _Provider, module: ModuleType) -> None:
    public_name = f"{_PUBLIC_PACKAGE}.{provider.public_name}"
    sys.modules[public_name] = module
    _alias_loaded_descendants(provider)
    if _root_module is not None:
        setattr(_root_module, provider.public_name, module)


def _load_root(provider: _Provider, placeholder: ModuleType | None = None) -> ModuleType:
    with provider.lock:
        if provider.module is not None:
            _publish_root(provider, provider.module)
            return provider.module
        if provider.failure is not None:
            raise ModuleNotFoundError(provider.failure)
        if provider.loading:
            raise ModuleNotFoundError(
                f"circular import while loading Holoscan module {provider.public_name!r}"
            )

        provider.loading = True
        public_name = f"{_PUBLIC_PACKAGE}.{provider.public_name}"
        if placeholder is not None and sys.modules.get(public_name) is placeholder:
            del sys.modules[public_name]
        try:
            loaded = provider.entry_point.load()
            if not isinstance(loaded, ModuleType):
                raise TypeError("entry point did not resolve to a module")
            if getattr(getattr(loaded, "__spec__", None), "_initializing", False):
                raise ImportError(
                    f"Circular Holoscan module alias import: circular import of "
                    f"{provider.canonical_name!r}"
                )
            if loaded.__name__ != provider.canonical_name:
                raise TypeError(
                    f"entry point resolved to {loaded.__name__!r}, expected "
                    f"{provider.canonical_name!r}"
                )
            if not hasattr(loaded, "__path__"):
                raise TypeError("entry point must reference a module package")
        except Exception as error:
            message = (
                f"Unable to load Holoscan module {provider.public_name!r} "
                f"from {provider.entry_point.value!r}: {error}"
            )
            provider.failure = message
            raise ModuleNotFoundError(message, name=provider.canonical_name) from error
        finally:
            provider.loading = False

        provider.module = loaded
        _publish_root(provider, loaded)
        return loaded


def _load_alias(fullname: str, placeholder: ModuleType) -> ModuleType:
    relative_name = fullname.removeprefix(f"{_PUBLIC_PACKAGE}.")
    provider_name, _, descendant = relative_name.partition(".")
    provider = _providers[provider_name]
    root = _load_root(provider, placeholder if not descendant else None)
    if not descendant:
        return root

    canonical_name = f"{provider.canonical_name}.{descendant}"
    if sys.modules.get(fullname) is placeholder:
        del sys.modules[fullname]
    try:
        module = import_module(canonical_name)
    except Exception as error:
        raise ModuleNotFoundError(
            f"Unable to load Holoscan module alias {fullname!r} from {canonical_name!r}: {error}"
        ) from error
    if not isinstance(module, ModuleType):
        raise ModuleNotFoundError(
            f"Holoscan module alias target {canonical_name!r} is not a module"
        )
    sys.modules[fullname] = module
    _alias_loaded_descendants(provider)
    return module


class _AliasLoader(Loader):
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        _load_alias(self.fullname, module)


class _AliasFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del path, target
        prefix = f"{_PUBLIC_PACKAGE}."
        if not fullname.startswith(prefix):
            return None
        provider_name = fullname.removeprefix(prefix).partition(".")[0]
        if provider_name not in _providers:
            return None
        return ModuleSpec(fullname, _AliasLoader(fullname), is_package=True)


_finder = _AliasFinder()


def install(root_module: ModuleType, reserved_names: frozenset[str]) -> None:
    """Install the process-wide finder after Core's namespace is initialized."""

    global _root_module, _reserved_names
    with _index_lock:
        _root_module = root_module
        _reserved_names = reserved_names
        if not _providers and not _rejections:
            try:
                _build_index(_metadata_entry_points())
            except Exception as error:
                _warn(f"Unable to discover {_ENTRY_POINT_GROUP} entries: {error}")
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)


def load_attribute(name: str) -> ModuleType:
    """Load one registered provider for package attribute access."""

    if name not in _providers:
        # Rejected registrations must allow normal submodule import fallback.
        raise AttributeError(f"module {_PUBLIC_PACKAGE!r} has no attribute {name!r}")
    return import_module(f"{_PUBLIC_PACKAGE}.{name}")


def registered_names() -> frozenset[str]:
    """Return validated provider names without importing their packages."""

    return frozenset(_providers)


def refresh_for_tests(entries: tuple[_EntryPoint, ...] | None = None) -> None:
    """Replace the cached metadata index for isolated test processes."""

    with _index_lock:
        for provider in _providers.values():
            public_name = f"{_PUBLIC_PACKAGE}.{provider.public_name}"
            for name in tuple(sys.modules):
                if name == public_name or name.startswith(f"{public_name}."):
                    del sys.modules[name]
            if (
                _root_module is not None
                and vars(_root_module).get(provider.public_name) is provider.module
            ):
                delattr(_root_module, provider.public_name)
        _providers.clear()
        _rejections.clear()
        try:
            _build_index(_metadata_entry_points() if entries is None else entries)
        except Exception as error:
            _warn(f"Unable to discover {_ENTRY_POINT_GROUP} entries: {error}")
