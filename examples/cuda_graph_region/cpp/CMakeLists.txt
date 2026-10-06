# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_cuda_graph_region LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(cuda_graph_region cuda_graph_region.cu)
target_compile_features(cuda_graph_region PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(cuda_graph_region
    PRIVATE
        CUDA::cudart
        holoscan::core
)
holoscan_configure_example_install(cuda_graph_region)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_CUDA_GRAPH_REGION_NORMAL_TEST
        COMMAND cuda_graph_region --normal)
    add_test(NAME EXAMPLE_CPP_CUDA_GRAPH_REGION_CAPTURED_TEST
        COMMAND cuda_graph_region --cuda-graph)
    set_tests_properties(EXAMPLE_CPP_CUDA_GRAPH_REGION_NORMAL_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "cuda_graph_region NORMAL PASS"
            SKIP_RETURN_CODE 77
            LABELS "examples;graph;cuda"
            TIMEOUT 30
    )
    set_tests_properties(EXAMPLE_CPP_CUDA_GRAPH_REGION_CAPTURED_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "cuda_graph_region CUDA_GRAPH PASS"
            SKIP_RETURN_CODE 77
            LABELS "examples;graph;cuda"
            TIMEOUT 30
    )
endif()

install(TARGETS cuda_graph_region
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/cuda_graph_region/cpp"
    COMPONENT holoscan-core-examples
)
