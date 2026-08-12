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
    for distribution_name in ("holoscan-cu13", "holoscan-cu12", "holoscan"):
        distribution_version = _distribution_version(distribution_name)
        if distribution_version is not None:
            return distribution_version
    return "0+unknown"


__version__ = _package_version()

__all__ = ["__version__"]
