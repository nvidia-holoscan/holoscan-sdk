# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(channel_pod_benchmark
    channel_pod_benchmark.cpp
    ../common/channel_benchmark_common.hpp
)
add_dependencies(holoscan-benchmark-targets channel_pod_benchmark)
target_link_libraries(channel_pod_benchmark
    PRIVATE
        benchmark::benchmark_main
        fmt::fmt-header-only
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_CHANNEL_POD
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:channel_pod_benchmark>"
                --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_CHANNEL_POD
        PROPERTIES
            LABELS benchmark
    )
endif()
