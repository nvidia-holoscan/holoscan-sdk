# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_GRAPH_SURFACE_SCANNER OR
   NOT DEFINED HOLOSCAN_GRAPH_SURFACE_FIXTURE_ROOT)
    message(FATAL_ERROR "Scanner and fixture-root paths are required")
endif()

set(fixture_include "${HOLOSCAN_GRAPH_SURFACE_FIXTURE_ROOT}/include/holoscan")
set(fixture_header "${fixture_include}/core/execution_plan.hpp")
set(fixture_graph_header "${fixture_include}/core/graph.hpp")
file(REMOVE_RECURSE "${HOLOSCAN_GRAPH_SURFACE_FIXTURE_ROOT}")
file(MAKE_DIRECTORY "${fixture_include}/core")

file(WRITE "${fixture_header}"
    "struct AlgorithmKey {};\n"
    "struct ExecutionPlanRecord { AlgorithmKey ready_policy{}; };\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE allowed_result
    OUTPUT_VARIABLE allowed_output
    ERROR_VARIABLE allowed_error
)
if(NOT allowed_result EQUAL 0)
    message(FATAL_ERROR
        "The exact compiler-selected READY evidence field was rejected:\n"
        "${allowed_output}${allowed_error}"
    )
endif()

# REQ: COMPILE-058
file(WRITE "${fixture_graph_header}"
    "#include <string_view>\n"
    "struct PartitionHandle {};\n"
    "struct Graph { PartitionHandle partition(std::string_view); };\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE partition_allowed_result
    OUTPUT_VARIABLE partition_allowed_output
    ERROR_VARIABLE partition_allowed_error
)
if(NOT partition_allowed_result EQUAL 0)
    message(FATAL_ERROR
        "The public Graph partition authoring member was rejected:\n"
        "${partition_allowed_output}${partition_allowed_error}"
    )
endif()

file(WRITE "${fixture_graph_header}"
    "#include <string_view>\n"
    "struct PartitionHandle {};\n"
    "struct Graph {\n"
    "  PartitionHandle partition(std::string_view);\n"
    "  void set_partition_algorithm();\n"
    "};\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE partition_selector_rejected_result
    OUTPUT_VARIABLE partition_selector_rejected_output
    ERROR_VARIABLE partition_selector_rejected_error
)
if(partition_selector_rejected_result EQUAL 0)
    message(FATAL_ERROR
        "A forbidden public Graph partition-selector member escaped the scanner"
    )
endif()
file(REMOVE "${fixture_graph_header}")

file(APPEND "${fixture_header}"
    "struct ForbiddenExtension { AlgorithmKey ready_policy_extra{}; };\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE rejected_result
    OUTPUT_VARIABLE rejected_output
    ERROR_VARIABLE rejected_error
)
if(rejected_result EQUAL 0)
    message(FATAL_ERROR "A similarly named public READY selector escaped the scanner")
endif()

# REQ: CHAN-065
file(WRITE "${fixture_header}"
    "struct AlgorithmKey {};\n"
    "struct ExecutionPlanRecord { AlgorithmKey ready_policy{}; };\n"
    "struct ProcessLocalChannelServicePlanRecord {};\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE private_record_result
    OUTPUT_VARIABLE private_record_output
    ERROR_VARIABLE private_record_error
)
if(private_record_result EQUAL 0)
    message(FATAL_ERROR
        "A private process-local Channel construction record escaped the scanner"
    )
endif()

# Generated CTest metadata may contain private test-only selector configuration. It is not
# installed package or public API evidence and must remain outside the public-surface scan.
set(fixture_build "${HOLOSCAN_GRAPH_SURFACE_FIXTURE_ROOT}/build")
file(MAKE_DIRECTORY "${fixture_build}/benchmark/runtime")
file(WRITE "${fixture_header}"
    "struct AlgorithmKey {};\n"
    "struct ExecutionPlanRecord { AlgorithmKey ready_policy{}; };\n"
)
file(WRITE "${fixture_build}/benchmark/runtime/CTestTestfile.cmake"
    "set_tests_properties(private_test PROPERTIES "
    "ENVIRONMENT HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM=source_chain/v1)\n"
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
        -DHOLOSCAN_PUBLIC_INCLUDE_DIR=${fixture_include}
        -DHOLOSCAN_BUILD_ROOT=${fixture_build}
        -P ${HOLOSCAN_GRAPH_SURFACE_SCANNER}
    RESULT_VARIABLE generated_test_metadata_result
    OUTPUT_VARIABLE generated_test_metadata_output
    ERROR_VARIABLE generated_test_metadata_error
)
if(NOT generated_test_metadata_result EQUAL 0)
    message(FATAL_ERROR
        "Generated build metadata was incorrectly treated as public API evidence:\n"
        "${generated_test_metadata_output}${generated_test_metadata_error}"
    )
endif()
