# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: GRAPH-001

cmake_policy(SET CMP0009 NEW)

set(forbidden_root_type_regex
    "(^|[^A-Za-z0-9_])(holoscan::)?(Application|Pipeline)([^A-Za-z0-9_]|$)"
)
set(forbidden_selector_regex
    "(^|[^A-Za-z0-9_])(PartitionAlgorithm|PartitionPlanner|PartitionAlgorithmSelector|PartitionPlannerRegistry|ReadyPolicy|partition_algorithm|partition_algorithm_selector|partition_of|partition_planner|partition_planner_plugin|ready_policy|register_partition_planner|set_partition_algorithm|set_ready_policy|serial_baseline|source_chain|HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM)([^A-Za-z0-9_]|$)"
)

function(check_graph_public_contract_file contract_file)
    file(READ "${contract_file}" contract_contents)
    string(REGEX MATCH "${forbidden_root_type_regex}" forbidden_root_type "${contract_contents}")
    if(forbidden_root_type)
        message(FATAL_ERROR
            "Forbidden Application/Pipeline export or binding found in ${contract_file}"
        )
    endif()
    string(REGEX MATCH "${forbidden_selector_regex}" forbidden_selector "${contract_contents}")
    if(forbidden_selector)
        message(FATAL_ERROR
            "Forbidden public partition selector/configuration/plugin route found in ${contract_file}"
        )
    endif()
endfunction()

# Production C++ may contain private compiler implementations of partition
# algorithms in later components. Reject alternate roots everywhere, but reject
# partition names in C++ implementation only when they are attached to an
# exported declaration or a language-binding registration. Package/configuration
# files remain subject to the stricter token scan above.
function(check_graph_production_source_file source_file)
    file(READ "${source_file}" source_contents)

    foreach(forbidden_class IN ITEMS Application Pipeline)
        string(REGEX MATCH
            "(class|struct)[ \t\r\n]+(HOLOSCAN_[A-Z_]+_EXPORT[ \t\r\n]+)?${forbidden_class}([ \t\r\n:{;]|$)"
            forbidden_declaration
            "${source_contents}"
        )
        string(REGEX MATCH
            "using[ \t\r\n]+${forbidden_class}[ \t\r\n]*="
            forbidden_using_alias
            "${source_contents}"
        )
        string(REGEX MATCH
            "typedef[ \t\r\n]+[^;]*[ \t\r\n]+${forbidden_class}[ \t\r\n]*;"
            forbidden_typedef_alias
            "${source_contents}"
        )
        if(forbidden_declaration OR forbidden_using_alias OR forbidden_typedef_alias)
            message(FATAL_ERROR
                "Forbidden production ${forbidden_class} declaration found in ${source_file}"
            )
        endif()
    endforeach()

    foreach(forbidden_identifier IN ITEMS
            Application
            Pipeline
            PartitionPlanner
            PartitionAlgorithm
            PartitionAlgorithmSelector
            PartitionPlannerRegistry
            ReadyPolicy
            partition_algorithm
            partition_algorithm_selector
            partition_of
            partition_planner
            partition_planner_plugin
            ready_policy
            register_partition_planner
            set_partition_algorithm
            set_ready_policy
            serial_baseline
            source_chain
            HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM)
        string(REGEX MATCH
            "(HOLOSCAN_[A-Z_]+_EXPORT|pybind11|nanobind|py::|nb::|\\.def|\\.attr)[^;\n\r]*${forbidden_identifier}"
            forbidden_export_or_binding
            "${source_contents}"
        )
        if(forbidden_export_or_binding)
            message(FATAL_ERROR
                "Forbidden exported/bound Graph route '${forbidden_identifier}' found in ${source_file}"
            )
        endif()
    endforeach()
endfunction()

if(NOT DEFINED HOLOSCAN_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR "HOLOSCAN_PUBLIC_INCLUDE_DIR is required")
endif()

foreach(forbidden_header IN ITEMS
        application.h
        application.hpp
        pipeline.h
        pipeline.hpp)
    file(GLOB_RECURSE forbidden_header_matches
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/${forbidden_header}"
        "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/*/${forbidden_header}"
    )
    if(forbidden_header_matches)
        message(FATAL_ERROR
            "Forbidden public Graph-root header found: ${forbidden_header_matches}"
        )
    endif()
endforeach()

set(graph_header "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/core/graph.hpp")
if(EXISTS "${graph_header}")
    file(READ "${graph_header}" graph_contents)
    foreach(forbidden_graph_member IN ITEMS
            compile
            run
            run_async
            start
            stop
            restart
            request_restart
            scheduler
            set_scheduler
            set_partition_algorithm
            ready_policy
            set_ready_policy)
        string(REGEX MATCH
            "(^|[;{}\n\r])[ \t]*(template[^\n\r]*[\n\r]+[ \t]*)?[^;{}\n\r]*[ \t*&]${forbidden_graph_member}[ \t\r\n]*\\("
            forbidden_member_declaration
            "${graph_contents}"
        )
        if(forbidden_member_declaration)
            message(FATAL_ERROR
                "Forbidden Graph lifecycle/scheduling member '${forbidden_graph_member}' found in ${graph_header}"
            )
        endif()
    endforeach()
endif()

if(DEFINED HOLOSCAN_PACKAGE_ROOT AND EXISTS "${HOLOSCAN_PACKAGE_ROOT}")
    file(GLOB_RECURSE package_manifests
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_PACKAGE_ROOT}/lib*/cmake/holoscan/*.cmake"
        "${HOLOSCAN_PACKAGE_ROOT}/lib*/cmake/holoscan/*.cmake.in"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.cpp"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.h"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.hpp"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/python/holoscan/*.py"
        "${HOLOSCAN_PACKAGE_ROOT}/python/holoscan/*.pyi"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.py"
        "${HOLOSCAN_PACKAGE_ROOT}/lib/python*/site-packages/holoscan/*.pyi"
        "${HOLOSCAN_PACKAGE_ROOT}/Lib/site-packages/holoscan/*.py"
        "${HOLOSCAN_PACKAGE_ROOT}/Lib/site-packages/holoscan/*.pyi"
        "${HOLOSCAN_PACKAGE_ROOT}/share/holoscan/*.cmake"
        "${HOLOSCAN_PACKAGE_ROOT}/share/holoscan/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/share/holoscan/*.yaml"
        "${HOLOSCAN_PACKAGE_ROOT}/share/holoscan/*.yml"
        "${HOLOSCAN_PACKAGE_ROOT}/lib*/holoscan/*.cmake"
        "${HOLOSCAN_PACKAGE_ROOT}/lib*/holoscan/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/plugins/*.cmake"
        "${HOLOSCAN_PACKAGE_ROOT}/plugins/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/registry/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/registries/*.json"
        "${HOLOSCAN_PACKAGE_ROOT}/examples/CMakeLists.txt"
        "${HOLOSCAN_PACKAGE_ROOT}/examples/*.cmake"
        "${HOLOSCAN_PACKAGE_ROOT}/python/CMakeLists.txt"
        "${HOLOSCAN_PACKAGE_ROOT}/python/*.cmake"
    )
    foreach(manifest IN LISTS package_manifests)
        check_graph_public_contract_file("${manifest}")
    endforeach()
endif()

if(DEFINED HOLOSCAN_BUILD_ROOT AND EXISTS "${HOLOSCAN_BUILD_ROOT}")
    file(GLOB_RECURSE build_contract_files
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_BUILD_ROOT}/*.cmake"
        "${HOLOSCAN_BUILD_ROOT}/*.cmake.in"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.cpp"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.h"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.hpp"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.py"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.pyi"
        "${HOLOSCAN_BUILD_ROOT}/python/lib/holoscan/*.json"
        "${HOLOSCAN_BUILD_ROOT}/python/holoscan/*.py"
        "${HOLOSCAN_BUILD_ROOT}/python/holoscan/*.pyi"
        "${HOLOSCAN_BUILD_ROOT}/share/holoscan/*.cmake"
        "${HOLOSCAN_BUILD_ROOT}/share/holoscan/*.json"
        "${HOLOSCAN_BUILD_ROOT}/lib*/holoscan/*.cmake"
        "${HOLOSCAN_BUILD_ROOT}/lib*/holoscan/*.json"
        "${HOLOSCAN_BUILD_ROOT}/plugins/*.cmake"
        "${HOLOSCAN_BUILD_ROOT}/plugins/*.json"
        "${HOLOSCAN_BUILD_ROOT}/registry/*.json"
        "${HOLOSCAN_BUILD_ROOT}/registries/*.json"
    )
    foreach(contract_file IN LISTS build_contract_files)
        # Generated test metadata and dependency source trees contain negative
        # fixtures or third-party prose, not installable Holoscan API evidence.
        # Retain CMake export sets while excluding other CMakeFiles internals.
        if(contract_file MATCHES "/(_deps|tests|Testing)/" OR
           contract_file MATCHES "/CTestTestfile\\.cmake$" OR
           contract_file MATCHES "/cmake/CPM_[^/]*\\.cmake$" OR
           (contract_file MATCHES "/CMakeFiles/" AND
            NOT contract_file MATCHES "/CMakeFiles/Export/"))
            continue()
        endif()
        check_graph_public_contract_file("${contract_file}")
    endforeach()
endif()

file(GLOB_RECURSE HOLOSCAN_PUBLIC_HEADERS
    LIST_DIRECTORIES FALSE
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/*.h"
    "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/*.hpp"
)

foreach(header IN LISTS HOLOSCAN_PUBLIC_HEADERS)
    file(READ "${header}" contents)
    if(header STREQUAL "${HOLOSCAN_PUBLIC_INCLUDE_DIR}/core/execution_plan.hpp")
        # Execution plans record the compiler-selected READY policy as evidence;
        # this is not a public selector. Keep the exception declaration-shaped
        # and singular so similarly named fields or APIs remain forbidden.
        set(allowed_ready_policy_regex
            "AlgorithmKey[ \t\r\n]+ready_policy[ \t\r\n]*\\{[ \t\r\n]*\\}[ \t\r\n]*;"
        )
        string(REGEX REPLACE "${allowed_ready_policy_regex}"
            "HOLOSCAN_ALLOWED_READY_POLICY_EVIDENCE"
            ready_policy_scan_contents
            "${contents}"
        )
        string(REGEX MATCHALL "HOLOSCAN_ALLOWED_READY_POLICY_EVIDENCE"
            allowed_ready_policy_matches
            "${ready_policy_scan_contents}"
        )
        list(LENGTH allowed_ready_policy_matches allowed_ready_policy_count)
        if(NOT allowed_ready_policy_count EQUAL 1)
            message(FATAL_ERROR
                "execution_plan.hpp must contain exactly one compiler-selected "
                "AlgorithmKey ready_policy{} evidence field"
            )
        endif()
        string(REPLACE "HOLOSCAN_ALLOWED_READY_POLICY_EVIDENCE" ""
            ready_policy_scan_contents
            "${ready_policy_scan_contents}"
        )
        string(FIND "${ready_policy_scan_contents}" "ready_policy" unexpected_ready_policy)
        if(NOT unexpected_ready_policy EQUAL -1)
            message(FATAL_ERROR
                "Forbidden additional public ready_policy spelling found in ${header}"
            )
        endif()
        set(contents "${ready_policy_scan_contents}")
    endif()
    foreach(forbidden_class IN ITEMS Application Pipeline)
        string(REGEX MATCH
            "(class|struct)[ \t\r\n]+(HOLOSCAN_[A-Z_]+_EXPORT[ \t\r\n]+)?${forbidden_class}([ \t\r\n:{;]|$)"
            forbidden_declaration
            "${contents}"
        )
        if(forbidden_declaration)
            message(FATAL_ERROR
                "Forbidden public ${forbidden_class} declaration found in ${header}"
            )
        endif()

        string(REGEX MATCH
            "using[ \t\r\n]+${forbidden_class}[ \t\r\n]*="
            forbidden_using_alias
            "${contents}"
        )
        string(REGEX MATCH
            "typedef[ \t\r\n]+[^;]*[ \t\r\n]+${forbidden_class}[ \t\r\n]*;"
            forbidden_typedef_alias
            "${contents}"
        )
        if(forbidden_using_alias OR forbidden_typedef_alias)
            message(FATAL_ERROR
                "Forbidden public ${forbidden_class} alias found in ${header}"
            )
        endif()
    endforeach()

    foreach(forbidden_runtime_fact IN ITEMS
            ProcessLocalChannelServicePlanRecord
            ProcessLocalChannelPublisherPlanRecord
            ProcessLocalChannelSubscriberPlanRecord
            process_local_channel_services
            process_local_channel_publishers
            process_local_channel_subscribers)
        string(REGEX MATCH
            "(^|[^A-Za-z0-9_])${forbidden_runtime_fact}([^A-Za-z0-9_]|$)"
            forbidden_runtime_fact_match
            "${contents}"
        )
        if(forbidden_runtime_fact_match)
            message(FATAL_ERROR
                "Private Runtime construction fact '${forbidden_runtime_fact}' found in ${header}"
            )
        endif()
    endforeach()

    foreach(forbidden_identifier IN ITEMS
            PartitionPlanner
            PartitionAlgorithm
            PartitionAlgorithmSelector
            PartitionPlannerRegistry
            ReadyPolicy
            partition_algorithm
            partition_algorithm_selector
            partition_of
            partition_planner
            partition_planner_plugin
            ready_policy
            register_partition_planner
            set_partition_algorithm
            set_ready_policy
            serial_baseline
            source_chain
            HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM)
        string(REGEX MATCH
            "(^|[^A-Za-z0-9_])${forbidden_identifier}([^A-Za-z0-9_]|$)"
            forbidden_selector
            "${contents}"
        )
        if(forbidden_selector)
            message(FATAL_ERROR
                "Forbidden public partition/READY selector '${forbidden_identifier}' found in ${header}"
            )
        endif()
    endforeach()
endforeach()

# Production sources, public bindings, package configuration, and plugin
# manifests must not expose a second root or route around the compiler-owned
# HSDK5-009 selector. Test trees, generated build/install trees, dependency
# patches, and private conformance accessors are deliberately excluded so
# negative evidence can name forbidden spellings without making this scanner
# self-referential.
if(DEFINED HOLOSCAN_PUBLIC_SOURCE_ROOT)
    file(GLOB_RECURSE production_cpp_sources
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.c"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.cc"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.cpp"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.cxx"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.h"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/*.hpp"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.c"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.cc"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.cpp"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.cxx"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.h"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/src/*.hpp"
    )
    list(FILTER production_cpp_sources EXCLUDE REGEX
        "(_test|test_access|compiler_access)\\.(c|cc|cpp|cxx|h|hpp)$"
    )
    foreach(production_source IN LISTS production_cpp_sources)
        check_graph_production_source_file("${production_source}")
    endforeach()

    file(GLOB_RECURSE public_binding_files
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.py"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.pyi"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.cpp"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.h"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.hpp"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/holoscan/*.json"
    )
    foreach(binding_file IN LISTS public_binding_files)
        check_graph_public_contract_file("${binding_file}")
    endforeach()

    file(GLOB_RECURSE public_configuration_files
        LIST_DIRECTORIES FALSE
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/src/CMakeLists.txt"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/CMakeLists.txt"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/python/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/CMakeLists.txt"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/examples/CMakeLists.txt"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/examples/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/cmake/modules/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/cmake/modules/*.cmake.in"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/cmake/modules/*.json"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/cmake/modules/*.yaml"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/cmake/modules/*.yml"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/plugins/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/plugins/*.json"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/registry/*.json"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/registries/*.json"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/share/holoscan/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/share/holoscan/*.json"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*.cmake"
        "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/modules/*.json"
    )
    list(FILTER public_configuration_files EXCLUDE REGEX "/tests?/")
    if(EXISTS "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/CMakeLists.txt")
        list(APPEND public_configuration_files
            "${HOLOSCAN_PUBLIC_SOURCE_ROOT}/CMakeLists.txt")
    endif()
    foreach(configuration_file IN LISTS public_configuration_files)
        check_graph_public_contract_file("${configuration_file}")
    endforeach()
endif()
