# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(imgui)
add_subdirectory(nvpro_core)

# Disable clang-tidy for third-party targets to avoid warnings from external code.
# These targets compile code from _deps/imgui-src/ and thirdparty/nvpro_core/.
set_target_properties(imgui_backend holoviz_imgui nvpro_core
    PROPERTIES CXX_CLANG_TIDY ""
)
