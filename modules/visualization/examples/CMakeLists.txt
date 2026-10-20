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

# Ship the sources beside the binaries the way the SDK-level examples do, so an installed user can
# copy an example out and build it standalone (each operator cpp/ carries a CMakeLists.min.txt;
# python/ ships the script, which runs against the installed Python package).
if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}"
        COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
        FILES_MATCHING
            PATTERN "CMakeLists.txt"
            PATTERN "CMakeLists.min.txt"
            PATTERN "*.cpp"
            PATTERN "*.cu"
            PATTERN "*.hpp"
            PATTERN "*.py"
            PATTERN "*.md"
            PATTERN "*.png"
            PATTERN "*.slang"
            PATTERN "*.h"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "build" EXCLUDE
        PATTERN "build-*" EXCLUDE
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "_deps" EXCLUDE
    )
endif()
