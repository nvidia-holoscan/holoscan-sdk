# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(tensor_python_as_tensor_benchmark
    SOURCES
        tensor_as_tensor_fastpath_benchmark.py
)
add_dependencies(holoscan-python-benchmark-targets tensor_python_as_tensor_benchmark)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_TENSOR_PYTHON_AS_TENSOR
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib"
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
