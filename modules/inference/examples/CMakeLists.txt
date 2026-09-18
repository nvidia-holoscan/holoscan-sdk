# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_inference_examples LANGUAGES CXX CUDA)

if(holoscan_inference_examples_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(GNUInstallDirs)
if(NOT HOLOSCAN_EXAMPLES_INSTALL_DIR)
    if(WIN32)
        set(HOLOSCAN_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/holoscan/examples")
    else()
        set(HOLOSCAN_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/holoscan/examples")
    endif()
endif()

add_executable(holoinfer_example holoinfer_example.cpp)
target_link_libraries(holoinfer_example
    PRIVATE
        holoscan::core
        holoscan::op_inference
        CUDA::cudart
)
holoscan_configure_example_install(holoinfer_example
    RUNTIME_DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference/cpp"
)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    include(HoloscanInferenceExampleHelper)
    holoscan_inference_add_example_test("01_single_model.yaml")
    holoscan_inference_add_example_test("02_multi_model_independent.yaml")
    holoscan_inference_add_example_test("03_multi_model_chained.yaml")
    holoscan_inference_add_example_test("04_multi_model_concurrent.yaml")
    holoscan_inference_add_example_test("05_multi_model_concurrent_graphs.yaml")
    holoscan_inference_add_example_test("06_multi_model_concurrent_isolated.yaml")
    holoscan_inference_add_example_test("07_dynamic_shape_single.yaml")
    holoscan_inference_add_example_test("08_dynamic_shape_concurrent_isolated.yaml")
endif()

install(TARGETS holoinfer_example
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference/cpp"
    COMPONENT holoscan-inference-examples
)
install(DIRECTORY configs
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
    COMPONENT holoscan-inference-examples
)
install(DIRECTORY models
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
    COMPONENT holoscan-inference-examples
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/"
        DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
        COMPONENT holoscan-inference-examples
        FILES_MATCHING
            PATTERN "CMakeLists.txt"
            PATTERN "CMakeLists.min.txt"
            PATTERN "*.cmake"
            PATTERN "*.cpp"
            PATTERN "*.cu"
            PATTERN "*.hpp"
            PATTERN "*.fbs"
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
