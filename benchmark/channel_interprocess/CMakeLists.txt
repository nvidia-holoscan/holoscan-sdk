# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(channel_interprocess_benchmark
    channel_interprocess_benchmark.cpp
    ../common/channel_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
)
add_dependencies(holoscan-benchmark-targets channel_interprocess_benchmark)
target_link_libraries(channel_interprocess_benchmark
    PRIVATE
        fmt::fmt-header-only
        holomq::holomq
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_CHANNEL_RUNNER_PYTHON_TESTS
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_run_channel_benchmark.py"
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTERPROCESS_POD_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --samples 100
                --warmup 10
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTERPROCESS_POD_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --expected-subscribers 2
                --samples 100
                --warmup 10
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_POD_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess
                --expected-subscribers 2
                --samples 100
                --warmup 10
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_VALIDATION_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --payload tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --payload-validation all
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTERPROCESS_IMAGE_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --payload image
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --payload-validation all
                --expected-subscribers 2
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_ASYNC_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --payload tensor
                --tensor-storage device
                --tensor-bytes 4096
                --async-cuda-work
                --skip-if-no-cuda
                --samples 300
                --warmup 4
                --timeout-sec 30
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess_sidecar
                --payload tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --payload-validation all
                --expected-subscribers 2
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_SERIAL_TENSOR_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess_serial
                --payload tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --payload-validation all
                --expected-subscribers 1
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 10
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_LATENCY_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess_sidecar
                --payload tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --workload latency
                --publisher-period-us 100
                --publisher-pacing spin
                --subscriber-queue-depth 1
                --payload-validation all
                --expected-subscribers 1
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 30
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_IMAGE_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess_sidecar
                --payload image
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --payload-validation all
                --expected-subscribers 2
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 30
    )
    add_test(
        NAME BENCHMARK_CHANNEL_INTRAPROCESS_IMAGE_DEVICE_FANOUT_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_channel_benchmark.py"
                "$<TARGET_FILE:channel_interprocess_benchmark>"
                --domain intraprocess
                --payload image
                --tensor-storage device
                --tensor-bytes 4096
                --payload-validation all
                --expected-subscribers 2
                --skip-if-no-cuda
                --samples 32
                --warmup 4
                --timeout-sec 30
    )
    set_tests_properties(
        BENCHMARK_CHANNEL_RUNNER_PYTHON_TESTS
        BENCHMARK_CHANNEL_INTERPROCESS_POD_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_POD_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_POD_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_VALIDATION_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_IMAGE_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_ASYNC_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SERIAL_TENSOR_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_LATENCY_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_IMAGE_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_IMAGE_DEVICE_FANOUT_SMOKE
        PROPERTIES
            LABELS benchmark
    )
    set_tests_properties(
        BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_VALIDATION_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_IMAGE_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTERPROCESS_TENSOR_ASYNC_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SERIAL_TENSOR_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_TENSOR_LATENCY_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_SIDECAR_IMAGE_FANOUT_SMOKE
        BENCHMARK_CHANNEL_INTRAPROCESS_IMAGE_DEVICE_FANOUT_SMOKE
        PROPERTIES SKIP_RETURN_CODE 77
    )
endif()
