# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# holomq_tensor helpers are HOLOSCAN_CORE_EXPORT'ed from holoscan::core; compiling the source into
# this target again would need non-exported channel_common internals and fails to link.
add_executable(channel_tensor_benchmark
    channel_tensor_benchmark.cpp
    ../common/channel_benchmark_common.hpp
)
add_dependencies(holoscan-benchmark-targets channel_tensor_benchmark)
target_include_directories(channel_tensor_benchmark PRIVATE "${holoscan_SOURCE_DIR}/src")
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

add_executable(imported_tensor_retirement_benchmark imported_tensor_retirement_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets imported_tensor_retirement_benchmark)
target_include_directories(imported_tensor_retirement_benchmark PRIVATE "${holoscan_SOURCE_DIR}/src")
target_link_libraries(imported_tensor_retirement_benchmark
    PRIVATE benchmark::benchmark_main holoscan::core holomq::holomq
)
if(BUILD_TESTING)
    add_test(NAME BENCHMARK_IMPORTED_TENSOR_RETIREMENT
        COMMAND ${Python3_EXECUTABLE} "${HOLOSCAN_BENCHMARK_RUNNER}"
            "$<TARGET_FILE:imported_tensor_retirement_benchmark>"
            --benchmark_min_time=0.01s
    )
    set_tests_properties(BENCHMARK_IMPORTED_TENSOR_RETIREMENT PROPERTIES
        LABELS "benchmark;cuda;holomq" TIMEOUT 30
    )
endif()

# Role orchestration remains outside the timed native graph, as with Channel domain comparisons.
add_executable(cuda_tensor_interprocess_benchmark cuda_tensor_interprocess_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets cuda_tensor_interprocess_benchmark)
target_include_directories(cuda_tensor_interprocess_benchmark PRIVATE "${holoscan_SOURCE_DIR}/src")
target_link_libraries(cuda_tensor_interprocess_benchmark
    PRIVATE holoscan::core holomq::holomq CUDA::cudart)
if(BUILD_TESTING)
    add_test(NAME BENCHMARK_CUDA_TENSOR_INTERPROCESS
        COMMAND cuda_tensor_interprocess_benchmark 2 2 0 0 32)
    add_test(NAME BENCHMARK_CUDA_TENSOR_INTERPROCESS_PINNED_CALLER
        COMMAND cuda_tensor_interprocess_benchmark 2 2 1 1 32)
    set_tests_properties(BENCHMARK_CUDA_TENSOR_INTERPROCESS
        BENCHMARK_CUDA_TENSOR_INTERPROCESS_PINNED_CALLER PROPERTIES
        LABELS "benchmark;cuda;holomq;interprocess" TIMEOUT 45 SKIP_RETURN_CODE 77)
endif()
