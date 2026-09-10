# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(temporal_contract_benchmark temporal_contract_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets temporal_contract_benchmark)
target_include_directories(temporal_contract_benchmark
    PRIVATE ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(temporal_contract_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_TEMPORAL_CONTRACT
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:temporal_contract_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_TEMPORAL_CONTRACT
        PROPERTIES
            LABELS benchmark
    )
endif()
