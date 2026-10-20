# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Version file for the Holoscan-provided magic_enum config (see magic_enum-config.cmake). The bundled
# magic_enum headers are header-only, so any requested version is treated as compatible.
set(PACKAGE_VERSION "0.9.7")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
if("${PACKAGE_FIND_VERSION}" VERSION_EQUAL "${PACKAGE_VERSION}")
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
