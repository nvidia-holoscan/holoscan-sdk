# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ea1_expected_helpers_smoke ea1_expected_helpers_smoke.cpp)
target_link_libraries(ea1_expected_helpers_smoke PRIVATE holoscan::core)

add_test(NAME EXPECTED_HELPERS_SMOKE.ea1_expected_helpers
    COMMAND ea1_expected_helpers_smoke
)

add_executable(ea1_expected_macros_header_smoke ea1_expected_macros_header_smoke.cpp)
target_link_libraries(ea1_expected_macros_header_smoke PRIVATE holoscan::core)

add_test(NAME EXPECTED_HELPERS_SMOKE.expected_macros_header_self_contained
    COMMAND ea1_expected_macros_header_smoke
)

# REQ: SYSTEM-022
add_test(NAME SYSTEM.example_readme_tier_banners
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PUBLIC_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_example_readme_tiers.cmake
)
set_tests_properties(SYSTEM.example_readme_tier_banners
    PROPERTIES LABELS "ea2;examples;documentation"
)

add_test(NAME EXPECTED_HELPERS_SMOKE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/expected_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/expected_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_CUDA_TOOLKIT_ROOT=${CUDAToolkit_LIBRARY_ROOT}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_expected_installed_consumer.cmake
)
set_tests_properties(EXPECTED_HELPERS_SMOKE.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;installed-package"
)

if(HOLOSCAN_BUILD_EXAMPLES AND HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    if(BUILD_SHARED_LIBS)
        set(installed_examples_package_iceoryx2_target iceoryx2-cxx::shared-lib-cxx)
    else()
        set(installed_examples_package_iceoryx2_target iceoryx2-cxx::static-lib-cxx)
    endif()

    # Component installation does not build artifacts. Ensure source-provided dependency
    # libraries and tools exist before the installed-package fixture installs their component;
    # imported providers already own their artifacts and must not become build dependencies.
    add_custom_target(installed_examples_package_dependencies)
    foreach(installed_dependency_target IN ITEMS
            ${installed_examples_package_iceoryx2_target}
            spdlog::spdlog
            flatbuffers::flatbuffers
            flatbuffers::flatc)
        if(TARGET ${installed_dependency_target})
            get_target_property(installed_dependency_concrete_target
                ${installed_dependency_target} ALIASED_TARGET)
            if(NOT installed_dependency_concrete_target)
                set(installed_dependency_concrete_target ${installed_dependency_target})
            endif()
            get_target_property(installed_dependency_target_imported
                ${installed_dependency_concrete_target} IMPORTED)
            if(NOT installed_dependency_target_imported)
                add_dependencies(installed_examples_package_dependencies
                    ${installed_dependency_concrete_target})
            endif()
        endif()
    endforeach()
    unset(installed_dependency_target)
    unset(installed_dependency_concrete_target)
    unset(installed_dependency_target_imported)
    unset(installed_examples_package_iceoryx2_target)
    # REQ: EXPECTED-025
    # REQ: SYSTEM-004
    # REQ: SYSTEM-008
    # REQ: SYSTEM-012
    # REQ: SYSTEM-013
    # REQ: SYSTEM-014
    # REQ: SYSTEM-015
    # REQ: SYSTEM-021
    add_test(NAME LOGGING_POLICY.installed_examples_package
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DHOLOSCAN_EXAMPLES_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_examples_package"
            "-DHOLOSCAN_INSTALL_LIB_DIR=${HOLOSCAN_INSTALL_LIB_DIR}"
            "-DHOLOSCAN_INSTALL_BIN_DIR=${CMAKE_INSTALL_BINDIR}"
            "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
            "-DHOLOSCAN_SHARED_LIBRARY_PREFIX=${CMAKE_SHARED_LIBRARY_PREFIX}"
            "-DHOLOSCAN_SHARED_LIBRARY_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "-DHOLOSCAN_STATIC_LIBRARY_PREFIX=${CMAKE_STATIC_LIBRARY_PREFIX}"
            "-DHOLOSCAN_STATIC_LIBRARY_SUFFIX=${CMAKE_STATIC_LIBRARY_SUFFIX}"
            "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
            "-DHOLOSCAN_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
            "-DHOLOSCAN_CONFIG_COMPONENT_ENABLED=${HOLOSCAN_BUILD_CONFIG}"
            "-DHOLOSCAN_CONFIG_IO_COMPONENT_ENABLED=${HOLOSCAN_BUILD_CONFIG_IO}"
            "-DHOLOSCAN_PARAMS_COMPONENT_ENABLED=${HOLOSCAN_BUILD_PARAMS}"
            "-DHOLOSCAN_PARAMS_IO_COMPONENT_ENABLED=${HOLOSCAN_BUILD_PARAMS_IO}"
            "-DHOLOSCAN_WEB_TUNER_COMPONENT_ENABLED=${HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            "-DHOLOSCAN_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}"
            "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_installed_examples_package.cmake
    )
    set_tests_properties(LOGGING_POLICY.installed_examples_package
        PROPERTIES
            RUN_SERIAL TRUE
            LABELS "ea1;installed-package;examples"
            TIMEOUT 600
    )
endif()

# REQ: SYSTEM-008
# REQ: SYSTEM-010
# REQ: SYSTEM-011
set(invalid_artifact_modes extra_deployment nested_directory)
if(UNIX)
    list(APPEND invalid_artifact_modes non_regular_file)
endif()
foreach(invalid_artifact_mode IN LISTS invalid_artifact_modes)
    add_test(NAME GRAPH_ARTIFACT_INSTALL.rejects_${invalid_artifact_mode}
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_ARTIFACT_FIXTURE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/graph_artifact_install_fixture"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/graph_artifact_install_fixture/${invalid_artifact_mode}"
            "-DHOLOSCAN_GRAPH_ARTIFACTS_MODULE=${CMAKE_SOURCE_DIR}/cmake/HoloscanGraphArtifacts.cmake"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_MODE=${invalid_artifact_mode}"
            "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_graph_artifact_install_validation.cmake
    )
    set_tests_properties(GRAPH_ARTIFACT_INSTALL.rejects_${invalid_artifact_mode}
        PROPERTIES
            LABELS "ea1;installed-package;graph-artifacts"
            TIMEOUT 120
    )
endforeach()
unset(invalid_artifact_modes)

foreach(artifact_fixture_mode IN ITEMS
        valid_absolute_destination
        default_install_excluded
        cross_without_emulator)
    if(artifact_fixture_mode STREQUAL "valid_absolute_destination")
        set(artifact_fixture_test_name honors_absolute_destination)
    elseif(artifact_fixture_mode STREQUAL "default_install_excluded")
        set(artifact_fixture_test_name default_install_excludes_generation)
    else()
        set(artifact_fixture_test_name rejects_cross_without_emulator)
    endif()
    add_test(NAME GRAPH_ARTIFACT_INSTALL.${artifact_fixture_test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_ARTIFACT_FIXTURE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/graph_artifact_install_fixture"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/graph_artifact_install_fixture/${artifact_fixture_mode}"
            "-DHOLOSCAN_GRAPH_ARTIFACTS_MODULE=${CMAKE_SOURCE_DIR}/cmake/HoloscanGraphArtifacts.cmake"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_MODE=${artifact_fixture_mode}"
            "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_graph_artifact_install_validation.cmake
    )
    set_tests_properties(GRAPH_ARTIFACT_INSTALL.${artifact_fixture_test_name}
        PROPERTIES
            LABELS "ea1;installed-package;graph-artifacts"
            TIMEOUT 120
    )
endforeach()

add_test(NAME GRAPH_ARTIFACT_INSTALL.default_package_excludes_generation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_CPACK_CONFIG=${CMAKE_BINARY_DIR}/CPackConfig.cmake"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_default_cpack_components.cmake
)
set_tests_properties(GRAPH_ARTIFACT_INSTALL.default_package_excludes_generation
    PROPERTIES
        LABELS "ea1;installed-package;graph-artifacts"
)

add_test(NAME GRAPH_ARTIFACT_INSTALL.default_build_excludes_generation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_RUN_SCRIPT=${CMAKE_SOURCE_DIR}/run"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_default_build_install_components.cmake
)
set_tests_properties(GRAPH_ARTIFACT_INSTALL.default_build_excludes_generation
    PROPERTIES
        LABELS "ea1;installed-package;graph-artifacts"
)

add_executable(ea1_clock_registry_smoke ea1_clock_registry_smoke.cpp)
target_include_directories(ea1_clock_registry_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ea1_clock_registry_smoke PRIVATE holoscan::core)

add_test(NAME CLOCK_REGISTRY.ea1_clock_registry_smoke
    COMMAND ea1_clock_registry_smoke
)

add_test(NAME CLOCK_REGISTRY.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/time_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/time_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_time_installed_consumer.cmake
)
set_tests_properties(CLOCK_REGISTRY.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;installed-package"
)

add_executable(ea1_operator_smoke ea1_operator_smoke.cpp)
target_include_directories(ea1_operator_smoke
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests/core
)
target_link_libraries(ea1_operator_smoke PRIVATE holoscan::core)

add_test(NAME OPERATOR_SMOKE.synthetic_single_callback
    COMMAND ea1_operator_smoke
)

add_executable(ea1_graph_smoke ea1_graph_smoke.cpp)
target_link_libraries(ea1_graph_smoke PRIVATE holoscan::core)

add_test(NAME GRAPH_SMOKE.direct_and_reusable_authoring
    COMMAND ea1_graph_smoke
)
set_tests_properties(GRAPH_SMOKE.direct_and_reusable_authoring
    PROPERTIES LABELS "ea1;graph;authoring"
)

add_executable(ea1_compile_plan_smoke ea1_compile_plan_smoke.cpp)
target_link_libraries(ea1_compile_plan_smoke PRIVATE holoscan::core)

# REQ: COMPILE-001
add_test(NAME COMPILE_SMOKE.graph_only_plan_is_inspectable
    COMMAND ea1_compile_plan_smoke
)
set_tests_properties(COMPILE_SMOKE.graph_only_plan_is_inspectable
    PROPERTIES LABELS "ea1;compile;execution-plan"
)

add_executable(ea1_run_session_smoke
    ea1_run_session_smoke.cpp
    ../support/direct_port_binding.cpp
)
target_include_directories(ea1_run_session_smoke
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/tests/runtime
)
target_link_libraries(ea1_run_session_smoke
    PRIVATE
        holoscan::core
        runtime_test_graph_support
)

# REQ: RUNTIME-001
# REQ: RUNTIME-002
# REQ: RUNTIME-004
add_test(NAME RUNTIME_SMOKE.fake_input_and_synthetic_clock
    COMMAND ea1_run_session_smoke
)
set_tests_properties(RUNTIME_SMOKE.fake_input_and_synthetic_clock
    PROPERTIES LABELS "ea1;runtime;run-session"
)

add_executable(ea1_inprocess_channel_smoke ea1_inprocess_channel_smoke.cpp)
target_link_libraries(ea1_inprocess_channel_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: RUNTIME-015
# REQ: RUNTIME-019
# REQ: RUNTIME-017
# REQ: RUNTIME-018
# REQ: CHAN-065
# REQ: CHAN-066
add_test(NAME RUNTIME_SMOKE.inprocess_channel_source_filter_sink
    COMMAND ea1_inprocess_channel_smoke
)
set_tests_properties(RUNTIME_SMOKE.inprocess_channel_source_filter_sink
    PROPERTIES LABELS "ea1;runtime;channel;inprocess"
)

add_executable(ea1_tensor_stream_smoke
    ea1_tensor_stream_smoke.cpp
)
target_link_libraries(ea1_tensor_stream_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: OPERATOR-010
# REQ: COMPILE-053
# REQ: COMPILE-054
# REQ: RUNTIME-027
# REQ: TENSOR-CORE-013
add_test(NAME TENSOR_STREAM_SMOKE.direct_output_producer_filter_sink
    COMMAND ea1_tensor_stream_smoke
)
set_tests_properties(TENSOR_STREAM_SMOKE.direct_output_producer_filter_sink
    PROPERTIES
        LABELS "ea1;runtime;tensor;cuda;stream"
        SKIP_RETURN_CODE 77
)

add_executable(runtime_environment_process_probe
    runtime_environment_process_probe.cpp
    ../support/direct_port_binding.cpp
)
target_include_directories(runtime_environment_process_probe
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/tests/runtime
)
target_link_libraries(runtime_environment_process_probe
    PRIVATE
        holoscan::core
        runtime_test_graph_support
)

# Execute in a fresh process whose inherited selector conflicts with the plan
# compiled inside the probe. The accepted runtime path must consume only the
# immutable partition facts and must not reread the process environment.
# REQ: RUNTIME-002
# REQ: RUNTIME-004
add_test(NAME RUNTIME.partition_environment_process_invariance
    COMMAND ${CMAKE_COMMAND} -E env
        "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM=definitely-not-an-algorithm"
        $<TARGET_FILE:runtime_environment_process_probe>
)
set_tests_properties(RUNTIME.partition_environment_process_invariance
    PROPERTIES LABELS "ea1;runtime;run-session;environment-invariance"
)

# REQ: RUNTIME-001
# REQ: RUNTIME-002
# REQ: RUNTIME-004
# REQ: RUNTIME-035
# REQ: RUNTIME-036
# REQ: COMPILE-061
add_test(NAME RUNTIME.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/runtime_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        "-DHOLOSCAN_EXPECT_TERMINAL_SUMMARY_COUNT=1"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_time_installed_consumer.cmake
)
set_tests_properties(RUNTIME.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;runtime;run-session;installed-package"
)

# REQ: COMPILE-009
# REQ: GRAPH-026
# REQ: COMPILE-053
add_test(NAME COMPILE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/compile_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/compile_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_time_installed_consumer.cmake
)
set_tests_properties(COMPILE.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;compile;installed-package"
)

add_executable(ea1_subgraph_smoke ea1_subgraph_smoke.cpp)
target_include_directories(ea1_subgraph_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ea1_subgraph_smoke PRIVATE holoscan::core)

# REQ: GRAPH-021
# REQ: GRAPH-022
add_test(NAME SUBGRAPH_SMOKE.camera_imu_like_composition
    COMMAND ea1_subgraph_smoke
)
set_tests_properties(SUBGRAPH_SMOKE.camera_imu_like_composition
    PROPERTIES LABELS "ea1;graph;subgraph"
)

# Run the same authoring program in three fresh processes. The executable does
# not read this selector; byte-identical authored-view and private canonical-
# handoff output protects that separation at the system boundary.
# REQ: GRAPH-022
add_test(NAME SUBGRAPH_ENVIRONMENT_INVARIANCE.absent_baseline_and_source_chain_match
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE=$<TARGET_FILE:ea1_subgraph_smoke>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_subgraph_environment_invariance.cmake
)
set_tests_properties(SUBGRAPH_ENVIRONMENT_INVARIANCE.absent_baseline_and_source_chain_match
    PROPERTIES LABELS "ea1;graph;subgraph;environment-invariance"
)

add_executable(ea1_typed_params_smoke ea1_typed_params_smoke.cpp)
target_include_directories(ea1_typed_params_smoke
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests/core
)
target_link_libraries(ea1_typed_params_smoke PRIVATE holoscan::core)

add_test(NAME PARAMETER_SMOKE.typed_generation_zero
    COMMAND ea1_typed_params_smoke
)
set_tests_properties(PARAMETER_SMOKE.typed_generation_zero
    PROPERTIES LABELS "ea1;synthetic-binding;parameter"
)

add_executable(ea1_param_update_smoke ea1_param_update_smoke.cpp)
target_include_directories(ea1_param_update_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ea1_param_update_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: PARAM-019
add_test(NAME PARAMETER_UPDATE_SMOKE.next_compute_publication
    COMMAND ea1_param_update_smoke
)
set_tests_properties(PARAMETER_UPDATE_SMOKE.next_compute_publication
    PROPERTIES LABELS "ea1;parameter;runtime-update"
)
set_tests_properties(OPERATOR_SMOKE.synthetic_single_callback
    PROPERTIES LABELS "ea1;synthetic-binding"
)

# This scan protects the EA1/EA2 product boundary independently of C++ name
# lookup: the forbidden names must not appear in installed Core headers even
# when no test translation unit attempts to use them.
add_test(NAME OPERATOR_SMOKE.public_surface_boundary
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PUBLIC_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_operator_public_surface.cmake
)
set_tests_properties(OPERATOR_SMOKE.public_surface_boundary
    PROPERTIES LABELS "ea1;installed-surface"
)

find_program(HOLOSCAN_DUMPBIN_EXECUTABLE NAMES dumpbin)

# REQ: SYSTEM-016
foreach(build_tree_find_mode IN ITEMS direct exact_version version_range component)
    add_test(NAME PACKAGE_BOUNDARY.raw_build_tree_${build_tree_find_mode}_rejected
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DHOLOSCAN_BUILD_TREE_FIXTURE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/build_tree_package_consumer"
            "-DHOLOSCAN_BUILD_TREE_FIXTURE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/build_tree_package_consumer"
            "-DHOLOSCAN_BUILD_TREE_FIND_MODE=${build_tree_find_mode}"
            "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_build_tree_package_rejection.cmake
    )
    set_tests_properties(PACKAGE_BOUNDARY.raw_build_tree_${build_tree_find_mode}_rejected
        PROPERTIES
            LABELS "ea1;installed-package;package-boundary"
            TIMEOUT 120
    )
endforeach()
unset(build_tree_find_mode)

# REQ: SYSTEM-016
# REQ: SYSTEM-017
# REQ: COMPILE-059
add_test(NAME OPERATOR_SMOKE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/operator_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/operator_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        "-DHOLOSCAN_PACKAGE_DISCOVERY_MATRIX=ON"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_operator_installed_consumer.cmake
)
set_tests_properties(OPERATOR_SMOKE.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;installed-package;package-boundary"
)

add_executable(ea1_temporal_contract_smoke ea1_temporal_contract_smoke.cpp)
target_include_directories(ea1_temporal_contract_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ea1_temporal_contract_smoke PRIVATE holoscan::core)

add_test(NAME TEMPORAL_CONTRACT_SMOKE.public_and_private_contract
    COMMAND ea1_temporal_contract_smoke
)
set_tests_properties(TEMPORAL_CONTRACT_SMOKE.public_and_private_contract
    PROPERTIES LABELS "ea1;temporal-contract"
)

add_test(NAME TEMPORAL_CONTRACT.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/temporal_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/temporal_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_operator_installed_consumer.cmake
)
set_tests_properties(TEMPORAL_CONTRACT.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;installed-package;temporal-contract"
)

add_executable(ea1_lifecycle_smoke
    ea1_lifecycle_smoke.cpp
    ${CMAKE_SOURCE_DIR}/src/core/lifecycle_kernel.cpp
)
holoscan_apply_logging_policy(ea1_lifecycle_smoke)
target_include_directories(ea1_lifecycle_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ea1_lifecycle_smoke PRIVATE holoscan::core)

# REQ: LIFECYCLE-001
# REQ: LIFECYCLE-003
# REQ: LIFECYCLE-004
add_test(NAME LIFECYCLE_SMOKE.stage_trace_is_deterministic
    COMMAND ea1_lifecycle_smoke
)
set_tests_properties(LIFECYCLE_SMOKE.stage_trace_is_deterministic
    PROPERTIES LABELS "ea1;lifecycle;deterministic"
)

add_test(NAME LIFECYCLE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/lifecycle_installed_consumer"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_operator_installed_consumer.cmake
)
set_tests_properties(LIFECYCLE.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;installed-package;lifecycle"
)

add_executable(ea1_interprocess_runtime_child ea1_interprocess_runtime_child.cpp)
target_link_libraries(ea1_interprocess_runtime_child PRIVATE holoscan::core)

add_executable(ea1_interprocess_runtime_smoke ea1_interprocess_runtime_smoke.cpp)
target_link_libraries(ea1_interprocess_runtime_smoke PRIVATE holoscan::core)
target_compile_definitions(ea1_interprocess_runtime_smoke PRIVATE
    HOLOSCAN_EA1_INTERPROCESS_RUNTIME_CHILD_EXE="$<TARGET_FILE:ea1_interprocess_runtime_child>"
)
add_dependencies(ea1_interprocess_runtime_smoke ea1_interprocess_runtime_child)

# REQ: CHAN-086
# REQ: CHAN-089
add_test(NAME RUNTIME_SMOKE.interprocess_graph_boundary_both_directions
    COMMAND ea1_interprocess_runtime_smoke
)
set_tests_properties(RUNTIME_SMOKE.interprocess_graph_boundary_both_directions
    PROPERTIES LABELS "ea1;runtime;channel;interprocess"
)

add_executable(ea1_holomq_schema_smoke ea1_holomq_schema_smoke.cpp)
target_link_libraries(ea1_holomq_schema_smoke PRIVATE holoscan::core holoscan::schema)

# REQ: CHAN-001
# REQ: CHAN-003
# REQ: CHAN-012
# REQ: CHAN-074
# REQ: CHAN-075
# REQ: EXTERNAL-001
# REQ: EXTERNAL-002
# REQ: EXTERNAL-003
# REQ: EXTERNAL-004
add_test(NAME CHANNEL_SCHEMA_SMOKE.catalog_pod_and_generated_schema
    COMMAND ea1_holomq_schema_smoke
)
set_tests_properties(CHANNEL_SCHEMA_SMOKE.catalog_pod_and_generated_schema
    PROPERTIES LABELS "ea1;channel;holomq;graph;pod;schema;tensor;external-client"
)

function(holoscan_add_installed_consumer_test test_name source_dir work_dir)
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${source_dir}"
            "-DHOLOSCAN_CONSUMER_WORK_DIR=${work_dir}"
            "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_time_installed_consumer.cmake
    )
endfunction()

# REQ: EXTERNAL-005
holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_package_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/channel_schema_installed_consumer"
    "${CMAKE_CURRENT_BINARY_DIR}/channel_schema_installed_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;holomq;pod;schema;installed-package"
)

# REQ: CHAN-120
holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_third_party_native_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party_schema_installed_consumer"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party_schema_installed_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_third_party_native_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;schema;third-party;installed-package"
)

holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_schema_tools_scope_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/schema_tools_scope_installed_consumer"
    "${CMAKE_CURRENT_BINARY_DIR}/schema_tools_scope_installed_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_schema_tools_scope_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;schema;installed-package"
)

add_test(NAME CHANNEL_SCHEMA.predeclared_flatbuffers_provider_install
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SDK_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DHOLOSCAN_FLATC_EXECUTABLE=$<TARGET_FILE:flatbuffers::flatc>"
        "-DHOLOSCAN_FLATBUFFERS_INCLUDE_DIR=${Flatbuffers_SOURCE_DIR}/include"
        "-DHOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE=${CMAKE_SOURCE_DIR}/cmake/deps/flatbuffers.cmake"
        "-DHOLOSCAN_PROVIDER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers_predeclared_provider"
        "-DHOLOSCAN_PROVIDER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/flatbuffers_predeclared_provider"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_flatbuffers_predeclared_provider.cmake
)
set_tests_properties(CHANNEL_SCHEMA.predeclared_flatbuffers_provider_install
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;schema;flatbuffers;installed-package"
)

# REQ: CHAN-120
add_test(NAME CHANNEL_SCHEMA.schema_tools_validation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SCHEMA_TOOLS_MODULE=${CMAKE_SOURCE_DIR}/cmake/HoloscanSchemaTools.cmake"
        "-DHOLOSCAN_SCHEMA_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/schema_tools_validation"
        "-DHOLOSCAN_SCHEMA_TEST_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/schema_tools_validation"
        "-DHOLOSCAN_FLATC_EXECUTABLE=$<TARGET_FILE:flatbuffers::flatc>"
        "-DHOLOSCAN_FLATC_VERSION=${Flatbuffers_VERSION}"
        "-DHOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA=${HOLOSCAN_SCHEMA_DIR}/flatbuffers/reflection.fbs"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_schema_tools_validation.cmake
)
set_tests_properties(CHANNEL_SCHEMA.schema_tools_validation
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;schema;cmake"
)

if(TARGET custom_schema_flatbuffers_generated)
    get_target_property(public_custom_schema_generated_dir
        custom_schema_flatbuffers_generated HOLOSCAN_FLATBUFFER_OUTPUT_DIR)
    add_executable(public_custom_schema_validation public_custom_schema_validation.cpp)
    add_dependencies(public_custom_schema_validation custom_schema_flatbuffers_generated)
    target_compile_features(public_custom_schema_validation PRIVATE cxx_std_20)
    target_include_directories(public_custom_schema_validation
        PRIVATE "${public_custom_schema_generated_dir}")
    target_link_libraries(public_custom_schema_validation
        PRIVATE holoscan::core holoscan::schema)
    set_target_properties(public_custom_schema_validation PROPERTIES CXX_EXTENSIONS OFF)

    # REQ: CHAN-077
    # REQ: CHAN-120
    add_test(NAME CHANNEL_SCHEMA.public_custom_schema_detailed_validation
        COMMAND public_custom_schema_validation)
    set_tests_properties(CHANNEL_SCHEMA.public_custom_schema_detailed_validation
        PROPERTIES LABELS "ea1;channel;schema;third-party;examples")
endif()

# REQ: CHAN-120
# REQ: OPERATOR-035
holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_third_party_descriptor_tensor_consumer
    "${CMAKE_SOURCE_DIR}/examples/flatbuffer_schemas/descriptor_tensor_schema/cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party_descriptor_tensor_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_third_party_descriptor_tensor_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;channel;schema;tensor;third-party;installed-package"
)

set(descriptor_schema_codegen_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/descriptor_tensor_schema_codegen")

foreach(generation_suffix IN ITEMS a b)
    set(descriptor_schema_generated_dir
        "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/generated-${generation_suffix}")
    holoscan_add_flatbuffer_schema(
        TARGET descriptor_tensor_schema_codegen_${generation_suffix}
        SCHEMAS descriptor_tensor_schema_codegen/descriptor_frame_multi_order_test.fbs
        INCLUDE_DIRS "${descriptor_schema_codegen_source_dir}"
        OUTPUT_DIR "${descriptor_schema_generated_dir}"
        BFBS_EMBED
        DESCRIPTOR_TENSOR
        TRAITS_HEADER descriptor_frame_multi_order_traits.hpp
        TRAITS_ROOT_TYPE app3p::MultiDescriptorFrame
        TRAITS_NATIVE_TYPE app3p::MultiDescriptorFrameT
        TRAITS_BINARY_SCHEMA_TYPE app3p::MultiDescriptorFrameBinarySchema
        TRAITS_PACKAGE_NAMESPACE app3p::multi_order_package
        TRAITS_CANONICAL_NAME app3p.MultiDescriptorFrame
        TRAITS_SCHEMA_VERSION 1
        TRAITS_MAX_SERIALIZED_SIZE 1024
        TRAITS_DESCRIPTOR_METADATA_CAPACITY 512
    )
endforeach()

set(descriptor_schema_determinism_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/traits.deterministic")
add_custom_command(
    OUTPUT "${descriptor_schema_determinism_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E compare_files
        "${descriptor_tensor_schema_codegen_a_TRAITS_HEADER}"
        "${descriptor_tensor_schema_codegen_b_TRAITS_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
        "-DGENERATED_TRAITS_A=${descriptor_tensor_schema_codegen_a_TRAITS_HEADER}"
        "-DGENERATED_TRAITS_B=${descriptor_tensor_schema_codegen_b_TRAITS_HEADER}"
        -P "${descriptor_schema_codegen_source_dir}/VerifyDescriptorSchemaIntermediates.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${descriptor_schema_determinism_stamp}"
    DEPENDS
        "${descriptor_tensor_schema_codegen_a_TRAITS_HEADER}"
        "${descriptor_tensor_schema_codegen_b_TRAITS_HEADER}"
        "${descriptor_schema_codegen_source_dir}/VerifyDescriptorSchemaIntermediates.cmake"
    COMMENT "Comparing independently generated descriptor traits headers"
    VERBATIM
)

set(descriptor_schema_multi_order_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/field_order.verified")
add_custom_command(
    OUTPUT "${descriptor_schema_multi_order_stamp}"
    COMMAND "${CMAKE_COMMAND}"
        "-DGENERATED_TRAITS=${descriptor_tensor_schema_codegen_a_TRAITS_HEADER}"
        -P "${descriptor_schema_codegen_source_dir}/VerifyDescriptorTensorFieldOrder.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${descriptor_schema_multi_order_stamp}"
    DEPENDS
        "${descriptor_tensor_schema_codegen_a_TRAITS_HEADER}"
        "${descriptor_schema_codegen_source_dir}/VerifyDescriptorTensorFieldOrder.cmake"
    COMMENT "Verifying canonical reflected descriptor Tensor field ordering"
    VERBATIM
)
add_custom_target(descriptor_tensor_schema_codegen_validation
    DEPENDS
        "${descriptor_schema_determinism_stamp}"
        "${descriptor_schema_multi_order_stamp}"
)

# REQ: CHAN-124
# REQ: CHAN-125
# REQ: OPERATOR-035
add_executable(multi_tensor_schema_codec_validation
    descriptor_tensor_schema_codegen/multi_tensor_schema_codec_validation.cpp)
add_dependencies(multi_tensor_schema_codec_validation
    descriptor_tensor_schema_codegen_a
    descriptor_tensor_schema_codegen_validation)
target_compile_features(multi_tensor_schema_codec_validation PRIVATE cxx_std_20)
target_include_directories(multi_tensor_schema_codec_validation
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/generated-a")
target_link_libraries(multi_tensor_schema_codec_validation
    PRIVATE
        holoscan::core
        holoscan::schema)
set_target_properties(multi_tensor_schema_codec_validation PROPERTIES CXX_EXTENSIONS OFF)

add_test(NAME CHANNEL_SCHEMA.multi_tensor_schema_codec_validation
    COMMAND multi_tensor_schema_codec_validation)
set_tests_properties(CHANNEL_SCHEMA.multi_tensor_schema_codec_validation
    PROPERTIES LABELS "ea2;channel;schema;tensor;third-party;codegen")

# REQ: OPERATOR-036
# REQ: OPERATOR-037
# REQ: CHAN-123
# REQ: CHAN-125
# REQ: CHAN-126
# REQ: CHAN-127
add_executable(multi_tensor_schema_runtime_validation
    descriptor_tensor_schema_codegen/multi_tensor_schema_runtime_validation.cpp)
add_dependencies(multi_tensor_schema_runtime_validation
    descriptor_tensor_schema_codegen_a
    descriptor_tensor_schema_codegen_validation)
target_compile_features(multi_tensor_schema_runtime_validation PRIVATE cxx_std_20)
target_include_directories(multi_tensor_schema_runtime_validation
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/generated-a")
target_link_libraries(multi_tensor_schema_runtime_validation
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::schema)
set_target_properties(multi_tensor_schema_runtime_validation PROPERTIES CXX_EXTENSIONS OFF)

foreach(validation_case IN ITEMS
        contracts
        process-local
        inline-partition
        inline-partition-cuda
        over-bound
        bundle-invalid
        bundle-restart
        cross-process
        fanout
        bundle-cross-process
        bundle-fanout)
    add_test(NAME CHANNEL_SCHEMA.multi_tensor_schema_${validation_case}
        COMMAND multi_tensor_schema_runtime_validation ${validation_case})
    set_tests_properties(CHANNEL_SCHEMA.multi_tensor_schema_${validation_case}
        PROPERTIES
            LABELS "ea2;channel;schema;tensor;third-party"
            SKIP_RETURN_CODE 77
            TIMEOUT 30)
endforeach()

# Keep the generated multi-Tensor capacity proof in the same 1/4/8 smoke sweep that protects
# singular Runtime graph pool sizing.
foreach(capacity IN ITEMS 1 4 8)
    add_test(NAME BENCHMARK_MULTI_TENSOR_SCHEMA_CAPACITY_${capacity}
        COMMAND multi_tensor_schema_runtime_validation capacity ${capacity})
    set_tests_properties(BENCHMARK_MULTI_TENSOR_SCHEMA_CAPACITY_${capacity}
        PROPERTIES
            LABELS "benchmark;ea2;channel;schema;tensor;capacity"
            TIMEOUT 30)
endforeach()

add_test(NAME CHANNEL_SCHEMA.descriptor_tensor_schema_codegen_diagnostics
    COMMAND "${CMAKE_COMMAND}"
        "-DDIAGNOSTIC_SOURCE_DIR=${descriptor_schema_codegen_source_dir}/diagnostics"
        "-DDIAGNOSTIC_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/diagnostics"
        "-DHOLOSCAN_FLATC_EXECUTABLE=$<TARGET_FILE:flatbuffers::flatc>"
        "-DHOLOSCAN_FLATC_VERSION=$<TARGET_PROPERTY:holoscan::fbs_tools,HOLOSCAN_FLATC_VERSION>"
        "-DHOLOSCAN_SCHEMA_DIR=$<TARGET_PROPERTY:holoscan::fbs_tools,HOLOSCAN_SCHEMA_DIR>"
        "-DHOLOSCAN_REFLECTION_SCHEMA=$<TARGET_PROPERTY:holoscan::fbs_tools,HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA>"
        "-DHOLOSCAN_SCHEMA_TOOLS_DIR=${holoscan_SCHEMA_TOOLS_DIR}"
        -P "${descriptor_schema_codegen_source_dir}/VerifyDescriptorSchemaDiagnostics.cmake"
)
set_tests_properties(CHANNEL_SCHEMA.descriptor_tensor_schema_codegen_diagnostics
    PROPERTIES
        LABELS "ea1;channel;schema;tensor;third-party;codegen"
        TIMEOUT 30
)

if(TARGET descriptor_tensor_schema_generated)
    get_target_property(public_descriptor_schema_generated_dir
        descriptor_tensor_schema_generated HOLOSCAN_FLATBUFFER_OUTPUT_DIR)
    add_executable(public_descriptor_schema_validation public_descriptor_schema_validation.cpp)
    add_dependencies(public_descriptor_schema_validation
        descriptor_tensor_schema_generated
        descriptor_tensor_schema_codegen_validation)
    target_compile_features(public_descriptor_schema_validation PRIVATE cxx_std_20)
    target_include_directories(public_descriptor_schema_validation
        PRIVATE "${public_descriptor_schema_generated_dir}")
    target_link_libraries(public_descriptor_schema_validation
        PRIVATE holoscan::core holoscan::schema)
    set_target_properties(public_descriptor_schema_validation PROPERTIES CXX_EXTENSIONS OFF)

    # REQ: CHAN-120
    # REQ: OPERATOR-035
    add_test(NAME CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation
        COMMAND public_descriptor_schema_validation)
    set_tests_properties(CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation
        PROPERTIES LABELS "ea1;channel;schema;tensor;third-party;examples")
endif()

# REQ: CHAN-077
# REQ: CHAN-120
holoscan_add_installed_consumer_test(
    EXAMPLE_CPP_CUSTOM_SCHEMA.installed_package_consumer
    "${CMAKE_SOURCE_DIR}/examples/flatbuffer_schemas/custom_schema/cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/custom_schema_installed_consumer"
)
set_tests_properties(EXAMPLE_CPP_CUSTOM_SCHEMA.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "ea1;examples;schema;installed-package"
)

# The exec-child arm uses POSIX pipes/fork/exec intentionally. Native Windows
# remains a release qualification target, but does not compile this Linux
# process-boundary fixture.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(ea1_external_client_matrix ea1_external_client_matrix.cpp)
    target_link_libraries(ea1_external_client_matrix
        PRIVATE
            holoscan::core
            holoscan::schema
            Threads::Threads
    )

    function(holoscan_add_ea1_external_matrix_test test_name cell_index)
        # REQ: SYSTEM-001
        # REQ: SYSTEM-002
        # REQ: SYSTEM-003
        add_test(NAME ${test_name}
            COMMAND ea1_external_client_matrix --cell ${cell_index}
        )
        set_tests_properties(${test_name}
            PROPERTIES
                RUN_SERIAL TRUE
                TIMEOUT 45
                LABELS "ea1;system;external-client;matrix;linux"
        )
    endfunction()

    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.same_process.custom_pod 0)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.same_process.custom_pod 1)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.same_process.schema 2)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.same_process.schema 3)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.same_process.host_tensor 4)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.same_process.host_tensor 5)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.exec_child.custom_pod 6)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.exec_child.custom_pod 7)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.exec_child.schema 8)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.exec_child.schema 9)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.external_publisher_to_graph_import.exec_child.host_tensor 10)
    holoscan_add_ea1_external_matrix_test(
        EA1_SYSTEM.graph_export_to_external_subscriber.exec_child.host_tensor 11)
endif()
