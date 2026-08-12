# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(time_benchmark time_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets time_benchmark)
target_include_directories(time_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(time_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_TIME
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:time_benchmark>"
                --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_TIME
        PROPERTIES
            LABELS benchmark
    )
endif()
