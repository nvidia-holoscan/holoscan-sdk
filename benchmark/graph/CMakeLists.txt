# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(graph_authoring_benchmark graph_authoring_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets graph_authoring_benchmark)
target_link_libraries(graph_authoring_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_GRAPH_AUTHORING
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:graph_authoring_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_GRAPH_AUTHORING
        PROPERTIES
            LABELS benchmark
    )
endif()
