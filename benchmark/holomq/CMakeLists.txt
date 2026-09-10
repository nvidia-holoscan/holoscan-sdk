# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(benchmark CONFIG QUIET)

set(HOLOMQ_BENCHMARK_LIBRARY holoscan::holomq)

add_executable(holomq_pubsub_benchmarks
    typed_pubsub_benchmark.cpp
)
add_executable(holomq::pubsub_benchmarks ALIAS holomq_pubsub_benchmarks)
set_target_properties(holomq_pubsub_benchmarks PROPERTIES CXX_CLANG_TIDY "")
target_link_libraries(holomq_pubsub_benchmarks
    PRIVATE
        ${HOLOMQ_BENCHMARK_LIBRARY}
        benchmark::benchmark_main
)

if(BUILD_TESTING)
    add_test(
        NAME holomq_pubsub_waitset_benchmark_smoke
        COMMAND
            $<TARGET_FILE:holomq_pubsub_benchmarks>
            --benchmark_filter=^BM_PubSubPOD_OneToOne_Blocking_LatchedWaitSet/real_time$
            --benchmark_min_time=0.0001s
    )
    set_tests_properties(holomq_pubsub_waitset_benchmark_smoke PROPERTIES
        LABELS "holomq;benchmark"
        TIMEOUT 30
    )
    function(add_holomq_concurrent_benchmark_smoke test_name benchmark_filter)
        add_test(
            NAME ${test_name}
            COMMAND
                $<TARGET_FILE:holomq_pubsub_benchmarks>
                "--benchmark_filter=${benchmark_filter}"
                --benchmark_min_time=0.0001s
        )
        set_tests_properties(${test_name} PROPERTIES
            LABELS "holomq;benchmark"
            RESOURCE_LOCK holomq_pubsub_concurrent_benchmark
            TIMEOUT 30
        )
    endfunction()

    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_polling_throughput_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentPolling_Throughput/real_time$"
    )
    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_polling_latency_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentPolling_Latency/manual_time$"
    )
    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_waitset_throughput_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentWaitSet_Throughput/real_time$"
    )
    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_waitset_latency_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentWaitSet_Latency/manual_time$"
    )
    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_waitset_publisher_send_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentWaitSet_PublisherSend/[0-9]+/manual_time$"
    )
    add_holomq_concurrent_benchmark_smoke(
        holomq_pubsub_concurrent_polling_publisher_send_benchmark_smoke
        "^BM_PubSubPOD_ConcurrentPolling_PublisherSend/[0-9]+/manual_time$"
    )
    add_test(
        NAME holomq_pubsub_fanout_backend_breakdown_smoke
        COMMAND
            $<TARGET_FILE:holomq_pubsub_benchmarks>
            "--benchmark_filter=^BM_PubSubFixed_Fanout.*<2>/1/manual_time$"
            --benchmark_min_time=0.0001s
    )
    set_tests_properties(holomq_pubsub_fanout_backend_breakdown_smoke PROPERTIES
        LABELS "holomq;benchmark"
        TIMEOUT 30
    )
endif()

if(TARGET holomq_objects)
    add_executable(holomq_cuda_sync_benchmarks
        cuda_sync_benchmark.cpp
    )
    add_executable(holomq::cuda_sync_benchmarks ALIAS holomq_cuda_sync_benchmarks)
    set_target_properties(holomq_cuda_sync_benchmarks PROPERTIES CXX_CLANG_TIDY "")
    target_include_directories(holomq_cuda_sync_benchmarks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src/holomq
    )
    target_link_libraries(holomq_cuda_sync_benchmarks
        PRIVATE
            # The benchmark exercises source-private stream-generation helpers,
            # whose symbols remain hidden from the installed HoloMQ library.
            holomq_objects
            benchmark::benchmark_main
    )

    if(BUILD_TESTING)
        add_test(
            NAME holomq_host_buffer_control_reuse_benchmark_smoke
            COMMAND
                $<TARGET_FILE:holomq_cuda_sync_benchmarks>
                "--benchmark_filter=^bm_host_buffer_control_lifecycle_(cold|reused)/1$"
                --benchmark_min_time=0.0001s
        )
        set_tests_properties(holomq_host_buffer_control_reuse_benchmark_smoke PROPERTIES
            LABELS "holomq;benchmark"
            TIMEOUT 30
        )
    endif()
endif()

add_executable(holomq_transport_benchmark
    transport_benchmark.cpp
)
add_executable(holomq::transport_benchmark ALIAS holomq_transport_benchmark)
set_target_properties(holomq_transport_benchmark PROPERTIES CXX_CLANG_TIDY "")
target_link_libraries(holomq_transport_benchmark
    PRIVATE
        ${HOLOMQ_BENCHMARK_LIBRARY}
)

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(
        NAME holomq_transport_benchmark_smoke
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_holomq_benchmark.py
            --binary $<TARGET_FILE:holomq_transport_benchmark>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/transport-smoke
            --payload-bytes 64
            --samples 100
            --warmup 10
            --queue-depth 8
            --subscriber-count 2
            --payload-validation all
            --timeout-sec 15
    )
    set_tests_properties(holomq_transport_benchmark_smoke PROPERTIES
        LABELS "holomq;benchmark"
        TIMEOUT 30
    )
    add_test(
        NAME holomq_transport_benchmark_latency_smoke
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_holomq_benchmark.py
            --binary $<TARGET_FILE:holomq_transport_benchmark>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/transport-latency-smoke
            --workload latency
            --payload-bytes 64
            --samples 100
            --warmup 10
            --queue-depth 1
            --publisher-period-us 10
            --publisher-pacing spin
            --payload-validation all
            --timeout-sec 15
    )
    set_tests_properties(holomq_transport_benchmark_latency_smoke PROPERTIES
        LABELS "holomq;benchmark"
        TIMEOUT 30
    )
    add_test(
        NAME holomq_transport_benchmark_pageable_host_buffer_smoke
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_holomq_benchmark.py
            --binary $<TARGET_FILE:holomq_transport_benchmark>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/transport-host-buffer-smoke
            --payload buffer_handle
            --buffer-bytes 4096
            --samples 100
            --warmup 10
            --queue-depth 8
            --pool-slots 10
            --payload-validation all
            --timeout-sec 15
            --skip-if-no-cuda
    )
    set_tests_properties(holomq_transport_benchmark_pageable_host_buffer_smoke PROPERTIES
        LABELS "holomq;benchmark"
        SKIP_RETURN_CODE 77
        TIMEOUT 30
    )
    add_test(
        NAME holomq_transport_benchmark_pinned_host_buffer_smoke
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_holomq_benchmark.py
            --binary $<TARGET_FILE:holomq_transport_benchmark>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/transport-pinned-buffer-smoke
            --payload buffer_handle
            --storage pinned_host
            --buffer-bytes 4096
            --samples 100
            --warmup 10
            --queue-depth 8
            --pool-slots 10
            --payload-validation all
            --timeout-sec 15
            --skip-if-no-cuda
    )
    set_tests_properties(holomq_transport_benchmark_pinned_host_buffer_smoke PROPERTIES
        LABELS "holomq;benchmark"
        SKIP_RETURN_CODE 77
        TIMEOUT 30
    )
    add_test(
        NAME holomq_transport_benchmark_device_buffer_smoke
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_holomq_benchmark.py
            --binary $<TARGET_FILE:holomq_transport_benchmark>
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/transport-device-buffer-smoke
            --payload buffer_handle
            --storage device
            --device-sync ready
            --buffer-bytes 4096
            --samples 100
            --warmup 10
            --queue-depth 8
            --pool-slots 10
            --payload-validation final
            --timeout-sec 15
            --skip-if-no-cuda
    )
    set_tests_properties(holomq_transport_benchmark_device_buffer_smoke PROPERTIES
        LABELS "holomq;benchmark"
        SKIP_RETURN_CODE 77
        TIMEOUT 30
    )
    add_test(
        NAME holomq_transport_benchmark_python_tests
        COMMAND
            ${Python3_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/test_run_holomq_benchmark.py
    )
    set_tests_properties(holomq_transport_benchmark_python_tests PROPERTIES
        LABELS "holomq;benchmark"
        TIMEOUT 30
    )
endif()

add_custom_target(holomq_all_benchmarks
    DEPENDS
        holomq_cuda_sync_benchmarks
        holomq_pubsub_benchmarks
        holomq_transport_benchmark
)
