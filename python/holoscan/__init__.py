# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Holoscan SDK Python package scaffold."""

from importlib.metadata import PackageNotFoundError, version


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _package_version() -> str:
    for distribution_name in ("holoscan-core-cu13", "holoscan-core-cu12", "holoscan-core"):
        distribution_version = _distribution_version(distribution_name)
        if distribution_version is not None:
            return distribution_version
    return "0+unknown"


__version__ = _package_version()

__all__ = ["__version__"]

# Discover metadata only; optional native packages are loaded when requested.
# Entry points alias independently named packages into holoscan.<module>.
# Unlike pkgutil.extend_path, this does not discover physical holoscan/<module>
# directories in other installation roots; registered packages need no path extension.
import sys as _sys  # noqa: E402

from . import _module_discovery as _module_discovery  # noqa: E402

_module_discovery.install(
    _sys.modules[__name__],
    frozenset({*globals(), "core", "modules"}),
)


def __getattr__(name: str) -> object:
    """Load an independently packaged Holoscan module on first access."""

    return _module_discovery.load_attribute(name)


def __dir__() -> list[str]:
    """Include discoverable providers without importing their packages."""

    return sorted(set(globals()) | _module_discovery.registered_names())
