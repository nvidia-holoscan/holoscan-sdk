# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_automatic_partitioning LANGUAGES CXX CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(automatic_partitioning automatic_partitioning.cu)
target_compile_features(automatic_partitioning PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(automatic_partitioning
    PRIVATE
        CUDA::cudart
        holoscan::core
)
holoscan_configure_example_install(automatic_partitioning)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_AUTOMATIC_PARTITIONING_CPU_TEST
        COMMAND automatic_partitioning --mode cpu
    )
    set_tests_properties(EXAMPLE_CPP_AUTOMATIC_PARTITIONING_CPU_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "automatic_partitioning cpu example complete"
            LABELS "examples;graph;partitioning"
            TIMEOUT 30
    )

    add_test(NAME EXAMPLE_CPP_AUTOMATIC_PARTITIONING_HYBRID_TEST
        COMMAND automatic_partitioning --mode hybrid
    )
    set_tests_properties(EXAMPLE_CPP_AUTOMATIC_PARTITIONING_HYBRID_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "automatic_partitioning hybrid example complete"
            SKIP_RETURN_CODE 77
            LABELS "examples;gpu;graph;partitioning"
            TIMEOUT 30
    )
endif()

install(TARGETS automatic_partitioning
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/automatic_partitioning/cpp"
    COMPONENT holoscan-examples
)
