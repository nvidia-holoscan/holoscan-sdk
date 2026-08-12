# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# holomq_tensor helpers are HOLOSCAN_CORE_EXPORT'ed from holoscan::core; compiling the source into
# this target again would need non-exported channel_common internals and fails to link.
add_executable(channel_tensor_benchmark
    channel_tensor_benchmark.cpp
    ../common/channel_benchmark_common.hpp
)
add_dependencies(holoscan-benchmark-targets channel_tensor_benchmark)
target_link_libraries(channel_tensor_benchmark
    PRIVATE
        benchmark::benchmark_main
        fmt::fmt-header-only
        holoscan::core
        holomq::holomq
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_CHANNEL_TENSOR
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:channel_tensor_benchmark>"
                --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_CHANNEL_TENSOR
        PROPERTIES
            LABELS benchmark
    )
endif()
