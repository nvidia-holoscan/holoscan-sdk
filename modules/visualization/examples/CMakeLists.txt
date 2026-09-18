# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT COMMAND holoscan_configure_example_install)
    include(HoloscanExampleInstall)
endif()

if(WIN32)
    set(HOLOVIZ_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/holoscan/examples/holoviz")
else()
    set(HOLOVIZ_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/holoscan/examples/holoviz")
endif()

add_subdirectory(core)
add_subdirectory(text)
add_subdirectory(imgui)
add_subdirectory(operators)
