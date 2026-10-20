<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Development container source archives

The Holoscan development container (`Dockerfile.dev`) keeps source archives for
third-party packages under `/opt/sources/`. The helper scripts in this directory
collect sources for installed APT and Python packages, while committed exclusion
lists let the image build handle packages with a separate source archive or
without an available source distribution.

The container build records the base image package lists before installing SDK
dependencies. After installation, it fetches sources for packages added by the
build and copies the resulting archives into the final image. APT sources are
stored under `/opt/sources/apt/`; Python source distributions are stored under
`/opt/sources/pip/`. Selected upstream archives are stored in package-specific
directories under `/opt/sources/`.

The APT helper accepts multiple exclusion files through `EXCLUDE_LISTS`. The
Python helper accepts repeated `--exclude-list PATH` arguments. Each list uses
one package per line: `name` excludes any version, and `name==version` excludes
only that version. Blank lines and lines beginning with `#` are ignored.
