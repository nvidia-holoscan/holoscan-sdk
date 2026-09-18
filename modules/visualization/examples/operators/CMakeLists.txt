# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Examples that use the module's Holoscan operator (HolovizOp) in a real Graph. They link
# holoscan::core and holoscan::visualization::op_holoviz like an application would, never the
# HoloViz backend directly (the samples under ../core and ../text cover that).
add_subdirectory(gray_scale_example)

# Ship the sources beside the binaries the way the SDK-level examples do, so an installed user can
# copy an operator example out and build it standalone (each cpp/ carries a CMakeLists.min.txt).
if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators"
        COMPONENT holoscan-examples
        FILES_MATCHING
            PATTERN "CMakeLists.txt"
            PATTERN "CMakeLists.min.txt"
            PATTERN "*.cpp"
            PATTERN "*.cu"
            PATTERN "*.hpp"
            PATTERN "*.md"
    )
endif()
