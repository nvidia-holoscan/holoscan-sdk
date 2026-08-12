# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(channel_schema_benchmark
    channel_schema_benchmark.cpp
    ../common/channel_benchmark_common.hpp
)
add_dependencies(channel_schema_benchmark holoscan_flatbuffer_schemas_generated)
add_dependencies(holoscan-benchmark-targets channel_schema_benchmark)

target_include_directories(channel_schema_benchmark
    PRIVATE
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)

target_link_libraries(channel_schema_benchmark
    PRIVATE
        benchmark::benchmark_main
        flatbuffers::flatbuffers
        fmt::fmt-header-only
        holomq::holomq
        holoscan::core
        holoscan::schema
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_CHANNEL_SCHEMA
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:channel_schema_benchmark>"
                --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_CHANNEL_SCHEMA
        PROPERTIES
            LABELS benchmark
    )
endif()
