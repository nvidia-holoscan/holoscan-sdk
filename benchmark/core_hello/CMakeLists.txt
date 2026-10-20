# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(core_hello_benchmark core_hello_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets core_hello_benchmark)
target_link_libraries(core_hello_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(
        NAME BENCHMARK_CORE_HELLO
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:core_hello_benchmark>"
                --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_CORE_HELLO
        PROPERTIES
            LABELS benchmark
    )
endif()
