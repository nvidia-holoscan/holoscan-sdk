# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# holoinfer_example lives directly under modules/inference/examples/ (no per-target subdir).

cmake_minimum_required(VERSION 4.0)

project(holoscan_inference_examples LANGUAGES CXX)

if(holoscan_inference_examples_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

if(NOT DEFINED HOLOSCAN_EXAMPLES_SOURCE_DIR)
    set(HOLOSCAN_EXAMPLES_SOURCE_DIR "${CMAKE_SOURCE_DIR}/examples")
endif()
if(NOT DEFINED HOLOSCAN_EXAMPLES_INSTALL_DIR)
    if(WIN32)
        set(HOLOSCAN_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/holoscan/examples")
    else()
        set(HOLOSCAN_EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/holoscan/examples")
    endif()
endif()

if(NOT COMMAND holoscan_configure_example_install)
    include(HoloscanExampleInstall)
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

if(HOLOSCAN_BUILD_INFERENCE AND BUILD_TESTING)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
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

# Ship the source files (holoinfer_example.cpp, CMakeLists.min.txt, cmake/ helpers, README.md)
# under the same component as the binary so a downstream user who installs only
# holoscan-inference-examples has everything needed to rebuild the example externally.
if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
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
            PATTERN "*.py"
            PATTERN "*.sh"
            PATTERN "*.yaml"
            PATTERN "*.md"
            PATTERN "*.onnx"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "build" EXCLUDE
        PATTERN "build-*" EXCLUDE
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "_deps" EXCLUDE
    )
endif()
