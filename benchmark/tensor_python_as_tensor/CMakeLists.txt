# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(tensor_python_as_tensor_benchmark
    SOURCES
        tensor_as_tensor_fastpath_benchmark.py
        tensor_stream_lifetime_benchmark.py
        tensor_stream_retirement_soak.py
        test_tensor_stream_lifetime_benchmark.py
)
add_dependencies(tensor_python_as_tensor_benchmark holoscan-python)
add_dependencies(holoscan-python-benchmark-targets tensor_python_as_tensor_benchmark)

if(BUILD_TESTING)
    add_test(NAME BENCHMARK_TENSOR_CUPY_DELIVERY_WAIT_TESTS
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_stream_lifetime_benchmark.py")
    set_tests_properties(BENCHMARK_TENSOR_CUPY_DELIVERY_WAIT_TESTS PROPERTIES
        LABELS "benchmark;python" TIMEOUT 20)
    # A finite smoke must tolerate its entire 50-frame burst if a consumer is descheduled.
    # Measurement runs keep the default depth two; these checks add no implicit retries or loss.
    foreach(policy IN ITEMS shared per_partition)
        add_test(NAME BENCHMARK_TENSOR_CUPY_SOAK_${policy}
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_retirement_soak.py"
                --seconds 2 --sample-seconds 1 --policy ${policy}
                --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_soak_${policy}.json")
        set_tests_properties(BENCHMARK_TENSOR_CUPY_SOAK_${policy} PROPERTIES
            LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_lifetime_benchmark.py"
                --frames 40 --warmup 10 --policy ${policy} --queue-depth 50
                --delivery-timeout-seconds 10 --retirement-timeout-seconds 10
                --diagnostics --stage-timings --release-timings --sample-every 10
                --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_retirement_${policy}.json")
        set_tests_properties(BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}
            PROPERTIES LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
    add_test(NAME BENCHMARK_TENSOR_CUPY_IMPORT_RETIRE
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_lifetime_benchmark.py"
            --mode import_retire --frames 100 --diagnostics --stage-timings --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_import_retire.json")
    set_tests_properties(BENCHMARK_TENSOR_CUPY_IMPORT_RETIRE
        PROPERTIES LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
    foreach(policy IN ITEMS shared per_partition)
        add_test(NAME BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}_PLAIN
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_lifetime_benchmark.py"
                --frames 40 --warmup 10 --policy ${policy} --queue-depth 50
                --delivery-timeout-seconds 10 --retirement-timeout-seconds 10
                --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_retirement_${policy}_plain.json")
        set_tests_properties(BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}_PLAIN
            PROPERTIES LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}_PACED
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_lifetime_benchmark.py"
                --frames 40 --warmup 10 --policy ${policy} --queue-depth 50
                --delivery-timeout-seconds 10 --retirement-timeout-seconds 10 --gpu-work-us 100
                --period-us 10 --pace-window 4 --release-timings
                --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_retirement_${policy}_paced.json")
        set_tests_properties(BENCHMARK_TENSOR_CUPY_RETIREMENT_${policy}_PACED
            PROPERTIES LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
    add_test(NAME BENCHMARK_TENSOR_CUPY_IMPORT_RETIRE_PLAIN
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tensor_stream_lifetime_benchmark.py"
            --mode import_retire --frames 100
            --output "${CMAKE_CURRENT_BINARY_DIR}/cupy_import_retire_plain.json")
    set_tests_properties(BENCHMARK_TENSOR_CUPY_IMPORT_RETIRE_PLAIN
        PROPERTIES LABELS "benchmark;python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_test(
        NAME BENCHMARK_TENSOR_PYTHON_AS_TENSOR
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tensor_as_tensor_fastpath_benchmark.py"
                --iterations 2000
                --warmup 200
                --repeats 5
    )
    set_tests_properties(BENCHMARK_TENSOR_PYTHON_AS_TENSOR
        PROPERTIES
            LABELS benchmark
    )
endif()
