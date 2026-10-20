# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(holoscan_green_context LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(green_context green_context.cu)
target_compile_features(green_context PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(green_context
    PRIVATE
        CUDA::cudart
        holoscan::core
)
holoscan_configure_example_install(green_context)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_GREEN_CONTEXT_TEST COMMAND green_context)
    set_tests_properties(EXAMPLE_CPP_GREEN_CONTEXT_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "green_context example complete"
            SKIP_RETURN_CODE 77
            TIMEOUT 15
    )
endif()

install(TARGETS green_context
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/green_context/cpp"
    COMPONENT holoscan-core-examples
)
