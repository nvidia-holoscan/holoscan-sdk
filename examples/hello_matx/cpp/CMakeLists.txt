# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_hello_matx LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_matx hello_matx.cu)
target_compile_features(hello_matx PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(hello_matx
    PRIVATE
        CCCL::CCCL
        CUDA::cudart
        holoscan::core
        matx::matx
)
holoscan_configure_example_install(hello_matx)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_MATX_TEST COMMAND hello_matx)
    set_tests_properties(EXAMPLE_CPP_HELLO_MATX_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "MatX CUDA sample complete"
            SKIP_RETURN_CODE 77
    )
endif()

install(TARGETS hello_matx
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/hello_matx/cpp"
    COMPONENT holoscan-core-examples
)
