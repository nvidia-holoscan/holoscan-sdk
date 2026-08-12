# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_hello_cccl LANGUAGES CXX CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_cccl hello_cccl.cu)
target_compile_features(hello_cccl PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(hello_cccl
    PRIVATE
        CCCL::CCCL
        CUDA::cudart
        holoscan::core
)
holoscan_configure_example_install(hello_cccl)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_CCCL_TEST COMMAND hello_cccl)
    set_tests_properties(EXAMPLE_CPP_HELLO_CCCL_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "CCCL CUDA sample complete"
            SKIP_RETURN_CODE 77
    )
endif()

install(TARGETS hello_cccl
    RUNTIME DESTINATION examples/hello_cccl/cpp
    COMPONENT holoscan-examples
)
