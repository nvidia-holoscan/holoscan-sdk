# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan SDK Python authoring API."""

from __future__ import annotations as _annotations

import builtins as _builtins
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_metadata_version

from ._public_api import export_public_names as _export_public_names


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return _distribution_metadata_version(distribution_name)
    except _PackageNotFoundError:
        return None


def _package_version() -> str:
    for distribution_name in ("holoscan-core-cu13", "holoscan-core-cu12", "holoscan-core"):
        distribution_version = _distribution_version(distribution_name)
        if distribution_version is not None:
            return distribution_version
    return "0+unknown"


__version__ = _package_version()

_root_public_names = _export_public_names(__name__, globals())  # noqa: PLE0604
__all__ = [  # noqa: PLE0604
    "__version__",
    *(name for name in _root_public_names if not hasattr(_builtins, name)),
]

# Discover metadata only; optional native packages are loaded when requested.
# Entry points alias independently named packages into holoscan.<module>.
# Unlike pkgutil.extend_path, this does not discover physical holoscan/<module>
# directories in other installation roots; registered packages need no path extension.
import sys as _sys  # noqa: E402

from . import _module_discovery as _module_discovery  # noqa: E402
from . import channel as channel  # noqa: E402
from . import clocks as clocks  # noqa: E402
from . import config as config  # noqa: E402
from . import diagnostics as diagnostics  # noqa: E402
from . import errors as errors  # noqa: E402
from . import execution as execution  # noqa: E402
from . import graphs as graphs  # noqa: E402
from . import lifecycle as lifecycle  # noqa: E402
from . import operators as operators  # noqa: E402
from . import params as params  # noqa: E402
from . import payload as payload  # noqa: E402
from . import registry as registry  # noqa: E402
from . import schema as schema  # noqa: E402
from . import sensor_io as sensor_io  # noqa: E402
from . import tensor as tensor  # noqa: E402
from . import time as time  # noqa: E402
from . import topics as topics  # noqa: E402

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
