# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_vision_pipeline_example LANGUAGES CXX)
    include(CTest)
    find_package(CUDAToolkit REQUIRED)
    find_package(holoscan COMPONENTS vision REQUIRED)
endif()

include(GNUInstallDirs)

if(NOT HOLOSCAN_VISION_EXAMPLES_INSTALL_COMPONENT)
    set(HOLOSCAN_VISION_EXAMPLES_INSTALL_COMPONENT holoscan-vision-examples)
endif()
if(NOT HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR)
    if(WIN32)
        set(HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/holoscan/examples")
    else()
        set(HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/holoscan/examples")
    endif()
endif()

add_executable(vision_pipeline vision_pipeline.cpp)
target_compile_features(vision_pipeline PRIVATE cxx_std_20)
target_link_libraries(vision_pipeline PRIVATE holoscan::vision CUDA::cudart)
if(NOT WIN32)
    set_target_properties(vision_pipeline PROPERTIES INSTALL_RPATH "$ORIGIN/../../../..")
endif()

install(TARGETS vision_pipeline
    RUNTIME DESTINATION "${HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR}/vision_pipeline/cpp"
    COMPONENT ${HOLOSCAN_VISION_EXAMPLES_INSTALL_COMPONENT}
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../"
        DESTINATION "${HOLOSCAN_VISION_EXAMPLES_INSTALL_DIR}/vision_pipeline"
        COMPONENT ${HOLOSCAN_VISION_EXAMPLES_INSTALL_COMPONENT}
        FILES_MATCHING
            PATTERN "CMakeLists.txt"
            PATTERN "*.cmake"
            PATTERN "*.cpp"
            PATTERN "*.cu"
            PATTERN "*.hpp"
            PATTERN "*.json"
            PATTERN "*.yaml"
            PATTERN "*.sh"
            PATTERN "*.md"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "build" EXCLUDE
        PATTERN "build-*" EXCLUDE
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "_deps" EXCLUDE
    )
endif()

if(BUILD_TESTING)
    add_test(NAME VISION_PIPELINE_EXAMPLE COMMAND vision_pipeline)
    set_tests_properties(VISION_PIPELINE_EXAMPLE
        PROPERTIES
            LABELS "vision;examples"
            PASS_REGULAR_EXPRESSION "vision_pipeline: image and detection branches passed"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )
endif()
