# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(endpoint_catalog_benchmark endpoint_catalog_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets endpoint_catalog_benchmark)
target_include_directories(endpoint_catalog_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(endpoint_catalog_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_ENDPOINT_CATALOG
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:endpoint_catalog_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_ENDPOINT_CATALOG
        PROPERTIES
            LABELS benchmark
    )
endif()
