# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.SABIModule)
holoscan_add_flatbuffer_schema(
    TARGET python_vector_record_generated
    SCHEMA vector_record.fbs
    MAX_SERIALIZED_SIZE 1048576
    PYTHON
    PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
)
Python3_add_library(python_payload_pod_fixtures MODULE USE_SABI 3.12 WITH_SOABI pod_fixtures.cpp)
target_compile_features(python_payload_pod_fixtures PRIVATE cxx_std_20)
target_link_libraries(python_payload_pod_fixtures PRIVATE holoscan::core)
set_target_properties(python_payload_pod_fixtures PROPERTIES
    OUTPUT_NAME _pod_fixtures
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(python_payload_conversion_benchmark SOURCES payload_conversion_benchmark.py)
add_dependencies(python_payload_conversion_benchmark
    holoscan-python python_vector_record_generated python_payload_pod_fixtures)
add_dependencies(holoscan-python-benchmark-targets python_payload_conversion_benchmark)

if(BUILD_TESTING)
    add_test(NAME BENCHMARK_PYTHON_PAYLOAD_CONVERSION
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}:${CMAKE_CURRENT_BINARY_DIR}"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/payload_conversion_benchmark.py"
            --iterations 16 --warmup 4 --repeats 1 --require-byte-vectors)
    set_tests_properties(BENCHMARK_PYTHON_PAYLOAD_CONVERSION PROPERTIES LABELS benchmark TIMEOUT 120)
endif()
