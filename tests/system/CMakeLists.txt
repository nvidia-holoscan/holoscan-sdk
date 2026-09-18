# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(runtime_failure_report_probe test_runtime_failure_report.cpp)
holoscan_configure_test_clang_tidy(runtime_failure_report_probe)
holoscan_apply_logging_policy(runtime_failure_report_probe)
target_link_libraries(runtime_failure_report_probe PRIVATE holoscan::core)
foreach(mode IN ITEMS start compute compute-error advanced advanced-success cleanup success embedded worker async)
    add_test(NAME SYSTEM.RuntimeFailureReport.${mode}
        COMMAND ${CMAKE_COMMAND}
            "-DPROBE=$<TARGET_FILE:runtime_failure_report_probe>"
            "-DMODE=${mode}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_runtime_failure_report.cmake
    )
    set_tests_properties(SYSTEM.RuntimeFailureReport.${mode} PROPERTIES
        TIMEOUT 45
        ENVIRONMENT "HOLOSCAN_LOG_LEVEL=INFO"
    )
endforeach()

add_test(NAME SYSTEM.RuntimeFailureReport.rejects_unexpected_exit
    COMMAND ${CMAKE_COMMAND}
        "-DPROBE=$<TARGET_FILE:runtime_failure_report_probe>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_runtime_failure_report_rejection.cmake
)
set_tests_properties(SYSTEM.RuntimeFailureReport.rejects_unexpected_exit PROPERTIES
    TIMEOUT 45
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=INFO"
)

add_executable(expected_helpers_smoke expected_helpers_smoke.cpp)
target_link_libraries(expected_helpers_smoke PRIVATE holoscan::core)

add_test(NAME EXPECTED_HELPERS_SMOKE.expected_helpers
    COMMAND expected_helpers_smoke
)

add_executable(expected_macros_header_smoke expected_macros_header_smoke.cpp)
target_link_libraries(expected_macros_header_smoke PRIVATE holoscan::core)

add_test(NAME EXPECTED_HELPERS_SMOKE.expected_macros_header_self_contained
    COMMAND expected_macros_header_smoke
)

# REQ: SYSTEM-022
add_test(NAME SYSTEM.example_readme_tier_banners
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PUBLIC_SOURCE_DIR=${holoscan_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_example_readme_tiers.cmake
)
set_tests_properties(SYSTEM.example_readme_tier_banners
    PROPERTIES LABELS "ea2;examples;documentation"
)

if(NOT HOLOSCAN_BUILD_PARAMS)
    # REQ: PARAM-024
    add_test(NAME CMAKE.params_disabled_excludes_schema_codec_compile_fail_tests
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
            "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_params_disabled_test_registration.cmake
    )
    set_tests_properties(CMAKE.params_disabled_excludes_schema_codec_compile_fail_tests
        PROPERTIES
            LABELS "cmake;params"
            TIMEOUT 30
    )
endif()

# Configures Holoscan under a parent project and then inspects the install rules that
# configuration generated. The second half matters because an install rule rooted at
# CMAKE_SOURCE_DIR configures without complaint from here and only resolves into the parent tree
# once a component is installed.
add_test(NAME CMAKE.subproject_configure
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SOURCE_DIR=${holoscan_SOURCE_DIR}"
        "-DHOLOSCAN_SUBPROJECT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/subproject"
        "-DHOLOSCAN_SUBPROJECT_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/subproject"
        "-DHOLOSCAN_VERIFY_INSTALL_SOURCES_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify_subproject_install_sources.cmake"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_subproject_configure.cmake
)
set_tests_properties(CMAKE.subproject_configure
    PROPERTIES LABELS "cmake;subproject"
)

if(HOLOSCAN_BUILD_BENCHMARKS)
    add_test(NAME CMAKE.standalone_benchmark_configure
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_BENCHMARK_SOURCE_DIR=${holoscan_SOURCE_DIR}/benchmark"
            "-DHOLOSCAN_BENCHMARK_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/standalone_benchmark"
            "-DHOLOSCAN_INSTALL_LIB_DIR=${HOLOSCAN_INSTALL_LIB_DIR}"
            "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_CUDA_TOOLKIT_ROOT=${CUDAToolkit_LIBRARY_ROOT}"
            "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            "-DHOLOSCAN_GOOGLE_BENCHMARK_SOURCE_DIR=${benchmark_SOURCE_DIR}"
            "-DHOLOSCAN_STANDALONE_BENCHMARK_VERIFY=${CMAKE_CURRENT_SOURCE_DIR}/standalone_benchmark/verify.cmake"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_standalone_benchmark_configure.cmake
    )
    set_tests_properties(CMAKE.standalone_benchmark_configure
        PROPERTIES
            RUN_SERIAL TRUE
            LABELS "benchmark;cmake;installed-package"
            TIMEOUT 600
    )
endif()

add_test(NAME EXPECTED_HELPERS_SMOKE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/expected_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/expected_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
        LABELS "installed-package"
)

if(HOLOSCAN_BUILD_EXAMPLES AND HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    if(WIN32)
        set(installed_examples_package_install_dir
            "${CMAKE_INSTALL_BINDIR}/holoscan/examples")
    else()
        set(installed_examples_package_install_dir
            "${CMAKE_INSTALL_LIBDIR}/holoscan/examples")
    endif()

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
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_EXAMPLES_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_examples_package"
            "-DHOLOSCAN_INSTALL_LIB_DIR=${HOLOSCAN_INSTALL_LIB_DIR}"
            "-DHOLOSCAN_INSTALL_DATA_DIR=${CMAKE_INSTALL_DATADIR}/holoscan"
            "-DHOLOSCAN_EXAMPLES_INSTALL_DIR=${installed_examples_package_install_dir}"
            "-DHOLOSCAN_INSTALL_BIN_DIR=${CMAKE_INSTALL_BINDIR}"
            "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
            "-DHOLOSCAN_SHARED_LIBRARY_PREFIX=${CMAKE_SHARED_LIBRARY_PREFIX}"
            "-DHOLOSCAN_SHARED_LIBRARY_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "-DHOLOSCAN_STATIC_LIBRARY_PREFIX=${CMAKE_STATIC_LIBRARY_PREFIX}"
            "-DHOLOSCAN_STATIC_LIBRARY_SUFFIX=${CMAKE_STATIC_LIBRARY_SUFFIX}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DHOLOSCAN_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
            "-DHOLOSCAN_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
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
            LABELS "installed-package;examples"
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
            "-DHOLOSCAN_GRAPH_ARTIFACTS_MODULE=${holoscan_SOURCE_DIR}/cmake/HoloscanGraphArtifacts.cmake"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_MODE=${invalid_artifact_mode}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
            LABELS "installed-package;graph-artifacts"
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
            "-DHOLOSCAN_GRAPH_ARTIFACTS_MODULE=${holoscan_SOURCE_DIR}/cmake/HoloscanGraphArtifacts.cmake"
            "-DHOLOSCAN_ARTIFACT_FIXTURE_MODE=${artifact_fixture_mode}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
            LABELS "installed-package;graph-artifacts"
            TIMEOUT 120
    )
endforeach()

add_test(NAME GRAPH_ARTIFACT_INSTALL.default_package_excludes_generation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_CPACK_CONFIG=${holoscan_BINARY_DIR}/CPackConfig.cmake"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_default_cpack_components.cmake
)
set_tests_properties(GRAPH_ARTIFACT_INSTALL.default_package_excludes_generation
    PROPERTIES
        LABELS "installed-package;graph-artifacts"
)

add_test(NAME CONFIG.cpack_granular_configuration
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_CPACK_CONFIG=${holoscan_BINARY_DIR}/CPackConfig.cmake"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_granular_cpack_configuration.cmake
)
set_tests_properties(CONFIG.cpack_granular_configuration
    PROPERTIES
        LABELS "installed-package;cpack"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(HOLOSCAN_DPKG_DEB_COMMAND dpkg-deb)
    if(HOLOSCAN_DPKG_DEB_COMMAND)
        add_test(NAME CONFIG.cpack_granular_debian_packages
            COMMAND ${CMAKE_COMMAND}
                "-DHOLOSCAN_CPACK_COMMAND=${CMAKE_CPACK_COMMAND}"
                "-DHOLOSCAN_CPACK_CONFIG=${holoscan_BINARY_DIR}/CPackConfig.cmake"
                "-DHOLOSCAN_DPKG_DEB_COMMAND=${HOLOSCAN_DPKG_DEB_COMMAND}"
                "-DHOLOSCAN_INSTALL_LIB_DIR=${HOLOSCAN_INSTALL_LIB_DIR}"
                "-DHOLOSCAN_PACKAGE_VERSION=${PROJECT_VERSION}"
                "-DHOLOSCAN_BUILD_VISION=${HOLOSCAN_BUILD_VISION}"
                "-DHOLOSCAN_BUILD_INFERENCE=${HOLOSCAN_BUILD_INFERENCE}"
                "-DHOLOSCAN_BUILD_VISUALIZATION=${HOLOSCAN_BUILD_VISUALIZATION}"
                "-DHOLOSCAN_CPACK_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cpack_granular_debian_packages"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/check_granular_cpack_debian_packages.cmake
        )
        set_tests_properties(CONFIG.cpack_granular_debian_packages
            PROPERTIES
                LABELS "installed-package;cpack;debian"
                RUN_SERIAL "TRUE"
                TIMEOUT 300
        )
    else()
        message(STATUS "Skipping CONFIG.cpack_granular_debian_packages: dpkg-deb not found")
    endif()
endif()
add_test(NAME GRAPH_ARTIFACT_INSTALL.default_build_excludes_generation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_RUN_SCRIPT=${holoscan_SOURCE_DIR}/run"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_default_build_install_components.cmake
)
set_tests_properties(GRAPH_ARTIFACT_INSTALL.default_build_excludes_generation
    PROPERTIES
        LABELS "installed-package;graph-artifacts"
)

add_executable(clock_registry_smoke clock_registry_smoke.cpp)
target_include_directories(clock_registry_smoke PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(clock_registry_smoke PRIVATE holoscan::core)

add_test(NAME CLOCK_REGISTRY.clock_registry_smoke
    COMMAND clock_registry_smoke
)

add_test(NAME CLOCK_REGISTRY.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/time_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/time_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
        LABELS "installed-package"
)

add_executable(operator_smoke operator_smoke.cpp)
target_include_directories(operator_smoke
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests/core
)
target_link_libraries(operator_smoke PRIVATE holoscan::core)

add_test(NAME OPERATOR_SMOKE.synthetic_single_callback
    COMMAND operator_smoke
)

add_executable(graph_smoke graph_smoke.cpp)
target_link_libraries(graph_smoke PRIVATE holoscan::core)

add_test(NAME GRAPH_SMOKE.direct_and_reusable_authoring
    COMMAND graph_smoke
)
set_tests_properties(GRAPH_SMOKE.direct_and_reusable_authoring
    PROPERTIES LABELS "graph;authoring"
)

add_executable(compile_plan_smoke compile_plan_smoke.cpp)
target_link_libraries(compile_plan_smoke PRIVATE holoscan::core)

# REQ: COMPILE-001
add_test(NAME COMPILE_SMOKE.graph_only_plan_is_inspectable
    COMMAND compile_plan_smoke
)
set_tests_properties(COMPILE_SMOKE.graph_only_plan_is_inspectable
    PROPERTIES LABELS "compile;execution-plan"
)

add_executable(run_session_smoke
    run_session_smoke.cpp
    ../support/direct_port_binding.cpp
)
target_include_directories(run_session_smoke
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        ${holoscan_SOURCE_DIR}/tests/runtime
)
target_link_libraries(run_session_smoke
    PRIVATE
        holoscan::core
        runtime_test_graph_support
)

# REQ: RUNTIME-001
# REQ: RUNTIME-002
# REQ: RUNTIME-004
add_test(NAME RUNTIME_SMOKE.fake_input_and_synthetic_clock
    COMMAND run_session_smoke
)
set_tests_properties(RUNTIME_SMOKE.fake_input_and_synthetic_clock
    PROPERTIES LABELS "runtime;run-session"
)

add_executable(inprocess_channel_smoke inprocess_channel_smoke.cpp)
target_link_libraries(inprocess_channel_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: RUNTIME-015
# REQ: RUNTIME-019
# REQ: RUNTIME-017
# REQ: RUNTIME-018
# REQ: CHAN-065
# REQ: CHAN-066
add_test(NAME RUNTIME_SMOKE.inprocess_channel_source_filter_sink
    COMMAND inprocess_channel_smoke
)
set_tests_properties(RUNTIME_SMOKE.inprocess_channel_source_filter_sink
    PROPERTIES LABELS "runtime;channel;inprocess"
)

add_executable(tensor_stream_smoke
    tensor_stream_smoke.cpp
)
target_link_libraries(tensor_stream_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: OPERATOR-010
# REQ: COMPILE-053
# REQ: COMPILE-054
# REQ: RUNTIME-027
# REQ: TENSOR-CORE-013
add_test(NAME TENSOR_STREAM_SMOKE.direct_output_producer_filter_sink
    COMMAND tensor_stream_smoke
)
set_tests_properties(TENSOR_STREAM_SMOKE.direct_output_producer_filter_sink
    PROPERTIES
        LABELS "runtime;tensor;cuda;stream"
        SKIP_RETURN_CODE 77
)

add_executable(runtime_environment_process_probe
    runtime_environment_process_probe.cpp
    ../support/direct_port_binding.cpp
)
target_include_directories(runtime_environment_process_probe
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        ${holoscan_SOURCE_DIR}/tests/runtime
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
    PROPERTIES LABELS "runtime;run-session;environment-invariance"
)

# REQ: RUNTIME-001
# REQ: RUNTIME-002
# REQ: RUNTIME-004
# REQ: RUNTIME-035
# REQ: RUNTIME-036
# REQ: RUNTIME-047
# REQ: RUNTIME-049
# REQ: RUNTIME-054
# REQ: COMPILE-061
add_test(NAME RUNTIME.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/runtime_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
        LABELS "runtime;run-session;installed-package"
)

# REQ: COMPILE-009
# REQ: GRAPH-026
# REQ: COMPILE-053
# REQ: MEM-GREENCTX-014
# REQ: COMPILE-063
# REQ: COMPILE-068
add_test(NAME COMPILE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/compile_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/compile_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
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
        LABELS "compile;installed-package"
)

add_executable(subgraph_smoke subgraph_smoke.cpp)
target_include_directories(subgraph_smoke PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(subgraph_smoke PRIVATE holoscan::core)

# REQ: GRAPH-021
# REQ: GRAPH-022
add_test(NAME SUBGRAPH_SMOKE.camera_imu_like_composition
    COMMAND subgraph_smoke
)
set_tests_properties(SUBGRAPH_SMOKE.camera_imu_like_composition
    PROPERTIES LABELS "graph;subgraph"
)

# Run the same authoring program with the retired variable absent and present.
# Graph authoring never reads it; compile-time migration rejection is covered by
# the compiler fresh-process probe.
# REQ: GRAPH-022
add_test(NAME SUBGRAPH_ENVIRONMENT_INVARIANCE.retired_selector_does_not_leak_into_authoring
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE=$<TARGET_FILE:subgraph_smoke>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_subgraph_environment_invariance.cmake
)
set_tests_properties(SUBGRAPH_ENVIRONMENT_INVARIANCE.retired_selector_does_not_leak_into_authoring
    PROPERTIES LABELS "graph;subgraph;environment-invariance"
)

add_executable(typed_params_smoke typed_params_smoke.cpp)
target_include_directories(typed_params_smoke
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests/core
)
target_link_libraries(typed_params_smoke PRIVATE holoscan::core)

add_test(NAME PARAMETER_SMOKE.typed_generation_zero
    COMMAND typed_params_smoke
)
set_tests_properties(PARAMETER_SMOKE.typed_generation_zero
    PROPERTIES LABELS "synthetic-binding;parameter"
)

add_executable(param_update_smoke param_update_smoke.cpp)
target_include_directories(param_update_smoke PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(param_update_smoke PRIVATE holoscan::core Threads::Threads)

# REQ: PARAM-019
add_test(NAME PARAMETER_UPDATE_SMOKE.next_compute_publication
    COMMAND param_update_smoke
)
set_tests_properties(PARAMETER_UPDATE_SMOKE.next_compute_publication
    PROPERTIES LABELS "parameter;runtime-update"
)
set_tests_properties(OPERATOR_SMOKE.synthetic_single_callback
    PROPERTIES LABELS "synthetic-binding"
)

# This scan protects the installed public boundary independently of C++ name
# lookup: the forbidden names must not appear in installed Core headers even
# when no test translation unit attempts to use them.
add_test(NAME OPERATOR_SMOKE.public_surface_boundary
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PUBLIC_INCLUDE_DIR=${holoscan_SOURCE_DIR}/include"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_operator_public_surface.cmake
)
set_tests_properties(OPERATOR_SMOKE.public_surface_boundary
    PROPERTIES LABELS "installed-surface"
)

find_program(HOLOSCAN_DUMPBIN_EXECUTABLE NAMES dumpbin)

# REQ: OPERATOR-013
# REQ: GRAPH-001
add_test(NAME CORE.installed_package_boundary
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_BOUNDARY_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/core_installed_package_boundary"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        "-DHOLOSCAN_INSTALL_LIB_DIR=${HOLOSCAN_INSTALL_LIB_DIR}"
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_READELF_EXECUTABLE=${CMAKE_READELF}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_core_installed_package_boundary.cmake
)
set_tests_properties(CORE.installed_package_boundary
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "core;installed-package;package-boundary"
)

# REQ: SYSTEM-016
foreach(build_tree_find_mode IN ITEMS direct exact_version version_range component)
    add_test(NAME PACKAGE_BOUNDARY.raw_build_tree_${build_tree_find_mode}_rejected
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_BUILD_TREE_FIXTURE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/build_tree_package_consumer"
            "-DHOLOSCAN_BUILD_TREE_FIXTURE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/build_tree_package_consumer"
            "-DHOLOSCAN_BUILD_TREE_FIND_MODE=${build_tree_find_mode}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
            LABELS "installed-package;package-boundary"
            TIMEOUT 120
    )
endforeach()
unset(build_tree_find_mode)

add_executable(temporal_contract_smoke temporal_contract_smoke.cpp)
target_include_directories(temporal_contract_smoke PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(temporal_contract_smoke PRIVATE holoscan::core)

add_test(NAME TEMPORAL_CONTRACT_SMOKE.public_and_private_contract
    COMMAND temporal_contract_smoke
)
set_tests_properties(TEMPORAL_CONTRACT_SMOKE.public_and_private_contract
    PROPERTIES LABELS "temporal-contract"
)

add_test(NAME TEMPORAL_CONTRACT.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/temporal_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/temporal_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
set_tests_properties(TEMPORAL_CONTRACT.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "installed-package;temporal-contract"
)

add_executable(lifecycle_smoke
    lifecycle_smoke.cpp
    ${holoscan_SOURCE_DIR}/src/core/lifecycle_kernel.cpp
)
holoscan_apply_logging_policy(lifecycle_smoke)
target_include_directories(lifecycle_smoke PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(lifecycle_smoke PRIVATE holoscan::core)

# REQ: LIFECYCLE-001
# REQ: LIFECYCLE-003
# REQ: LIFECYCLE-004
add_test(NAME LIFECYCLE_SMOKE.stage_trace_is_deterministic
    COMMAND lifecycle_smoke
)
set_tests_properties(LIFECYCLE_SMOKE.stage_trace_is_deterministic
    PROPERTIES LABELS "lifecycle;deterministic"
)

add_test(NAME LIFECYCLE.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/lifecycle_installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/lifecycle_installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
set_tests_properties(LIFECYCLE.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "installed-package;lifecycle"
)

add_executable(interprocess_runtime_child interprocess_runtime_child.cpp)
target_link_libraries(interprocess_runtime_child PRIVATE holoscan::core)

add_executable(interprocess_runtime_smoke interprocess_runtime_smoke.cpp)
target_link_libraries(interprocess_runtime_smoke PRIVATE holoscan::core)
target_compile_definitions(interprocess_runtime_smoke PRIVATE
    HOLOSCAN_INTERPROCESS_RUNTIME_CHILD_EXE="$<TARGET_FILE:interprocess_runtime_child>"
)
add_dependencies(interprocess_runtime_smoke interprocess_runtime_child)

# REQ: CHAN-086
# REQ: CHAN-089
add_test(NAME RUNTIME_SMOKE.interprocess_graph_boundary_both_directions
    COMMAND interprocess_runtime_smoke
)
set_tests_properties(RUNTIME_SMOKE.interprocess_graph_boundary_both_directions
    PROPERTIES LABELS "runtime;channel;interprocess"
)

add_executable(holomq_schema_smoke holomq_schema_smoke.cpp)
target_link_libraries(holomq_schema_smoke PRIVATE holoscan::core holoscan::schema)

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
    COMMAND holomq_schema_smoke
)
set_tests_properties(CHANNEL_SCHEMA_SMOKE.catalog_pod_and_generated_schema
    PROPERTIES LABELS "channel;holomq;graph;pod;schema;tensor;external-client"
)

function(holoscan_add_installed_consumer_test test_name source_dir work_dir)
    add_test(NAME ${test_name}
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
            "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${source_dir}"
            "-DHOLOSCAN_CONSUMER_WORK_DIR=${work_dir}"
            "-DCMAKE_BUILD_TYPE=$<CONFIG>"
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
        LABELS "channel;holomq;pod;schema;installed-package"
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
        LABELS "channel;schema;third-party;installed-package"
)

holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_schema_tools_scope_consumer
    "${CMAKE_CURRENT_SOURCE_DIR}/schema_tools_scope_installed_consumer"
    "${CMAKE_CURRENT_BINARY_DIR}/schema_tools_scope_installed_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_schema_tools_scope_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "channel;schema;installed-package"
)

add_test(NAME CHANNEL_SCHEMA.predeclared_flatbuffers_provider_install
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SDK_SOURCE_DIR=${holoscan_SOURCE_DIR}"
        "-DHOLOSCAN_FLATC_EXECUTABLE=$<TARGET_FILE:flatbuffers::flatc>"
        "-DHOLOSCAN_FLATBUFFERS_INCLUDE_DIR=${Flatbuffers_SOURCE_DIR}/include"
        "-DHOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE=${holoscan_SOURCE_DIR}/cmake/deps/flatbuffers.cmake"
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
        LABELS "channel;schema;flatbuffers;installed-package"
)

# REQ: CHAN-120
add_test(NAME CHANNEL_SCHEMA.schema_tools_validation
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SCHEMA_TOOLS_MODULE=${holoscan_SOURCE_DIR}/cmake/HoloscanSchemaTools.cmake"
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
        LABELS "channel;schema;cmake"
)

# REQ: SCHEMA-004
# REQ: SCHEMA-006
add_test(NAME CHANNEL_SCHEMA.schema_traits_generator
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_generate_schema_traits.py"
)
set_tests_properties(CHANNEL_SCHEMA.schema_traits_generator
    PROPERTIES
        ENVIRONMENT
            "HOLOSCAN_SCHEMA_TRAITS_GENERATOR=${holoscan_SOURCE_DIR}/scripts/generate_schema_traits.py;HOLOSCAN_TEST_FLATC=$<TARGET_FILE:flatbuffers::flatc>;HOLOSCAN_TEST_FLATC_VERSION=${Flatbuffers_VERSION};HOLOSCAN_TEST_REFLECTION_SCHEMA=${HOLOSCAN_SCHEMA_DIR}/flatbuffers/reflection.fbs;HOLOSCAN_TEST_SCHEMA_DIR=${holoscan_SOURCE_DIR}/schemas;HOLOSCAN_TEST_SCHEMA_DIAGNOSTICS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/descriptor_tensor_schema_codegen/diagnostics"
        LABELS "channel;schema;flatbuffers;codegen"
)

set(schema_evolution_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer_schema_evolution")
set(schema_evolution_generated_dir
    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffer_schema_evolution/generated")
set(schema_evolution_generated_headers
    "${schema_evolution_generated_dir}/evolution_v1_generated.h"
    "${schema_evolution_generated_dir}/evolution_v2_generated.h"
    "${schema_evolution_generated_dir}/collision_a_generated.h"
    "${schema_evolution_generated_dir}/collision_b_generated.h")
set(schema_evolution_bfbs_files
    "${schema_evolution_generated_dir}/evolution_v1.bfbs"
    "${schema_evolution_generated_dir}/evolution_renamed.bfbs"
    "${schema_evolution_generated_dir}/incompatible_required_field.bfbs")

add_custom_command(
    OUTPUT
        ${schema_evolution_generated_headers}
        ${schema_evolution_bfbs_files}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${schema_evolution_generated_dir}"
    COMMAND $<TARGET_FILE:flatbuffers::flatc>
        --cpp
        --gen-object-api
        --require-explicit-ids
        -o "${schema_evolution_generated_dir}"
        "${schema_evolution_source_dir}/evolution_v1.fbs"
        "${schema_evolution_source_dir}/evolution_v2.fbs"
        "${schema_evolution_source_dir}/collision_a.fbs"
        "${schema_evolution_source_dir}/collision_b.fbs"
    COMMAND $<TARGET_FILE:flatbuffers::flatc>
        -b
        --schema
        --require-explicit-ids
        -o "${schema_evolution_generated_dir}"
        "${schema_evolution_source_dir}/evolution_v1.fbs"
        "${schema_evolution_source_dir}/evolution_renamed.fbs"
        "${schema_evolution_source_dir}/incompatible_required_field.fbs"
    DEPENDS
        flatbuffers::flatc
        "${schema_evolution_source_dir}/evolution_v1.fbs"
        "${schema_evolution_source_dir}/evolution_v2.fbs"
        "${schema_evolution_source_dir}/evolution_renamed.fbs"
        "${schema_evolution_source_dir}/incompatible_required_field.fbs"
        "${schema_evolution_source_dir}/collision_a.fbs"
        "${schema_evolution_source_dir}/collision_b.fbs"
    COMMENT "Generating FlatBuffers evolution characterization artifacts"
    VERBATIM)
add_custom_target(flatbuffer_schema_evolution_artifacts
    DEPENDS
        ${schema_evolution_generated_headers}
        ${schema_evolution_bfbs_files})

foreach(revision IN ITEMS v1 v2)
    add_executable(flatbuffer_schema_evolution_${revision}_codec
        flatbuffer_schema_evolution/evolution_codec.cpp)
    add_dependencies(flatbuffer_schema_evolution_${revision}_codec
        flatbuffer_schema_evolution_artifacts)
    target_compile_features(flatbuffer_schema_evolution_${revision}_codec PRIVATE cxx_std_20)
    target_compile_definitions(flatbuffer_schema_evolution_${revision}_codec
        PRIVATE
            "EVOLUTION_SCHEMA_HEADER=\"evolution_${revision}_generated.h\"")
    target_include_directories(flatbuffer_schema_evolution_${revision}_codec
        PRIVATE "${schema_evolution_generated_dir}")
    target_link_libraries(flatbuffer_schema_evolution_${revision}_codec
        PRIVATE flatbuffers::flatbuffers)
    set_target_properties(flatbuffer_schema_evolution_${revision}_codec
        PROPERTIES CXX_EXTENSIONS OFF)
endforeach()
target_compile_definitions(flatbuffer_schema_evolution_v1_codec
    PRIVATE EVOLUTION_HAS_NOTE=0)
target_compile_definitions(flatbuffer_schema_evolution_v2_codec
    PRIVATE EVOLUTION_HAS_NOTE=1)

holoscan_add_flatbuffer_schema(
    TARGET flatbuffer_schema_evolution_v2_package
    SCHEMAS flatbuffer_schema_evolution/evolution_v2.fbs
    INCLUDE_DIRS "${schema_evolution_source_dir}"
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffer_schema_evolution/package-v2"
    TRAITS_HEADER flatbuffer_schema_evolution_v2_traits.hpp
    COMPATIBILITY_EPOCH 3
    MAX_SERIALIZED_SIZE 1024)
get_target_property(schema_evolution_package_dir
    flatbuffer_schema_evolution_v2_package HOLOSCAN_FLATBUFFER_OUTPUT_DIR)

add_executable(flatbuffer_schema_evolution_characterization
    flatbuffer_schema_evolution/schema_evolution_characterization.cpp)
add_dependencies(flatbuffer_schema_evolution_characterization
    flatbuffer_schema_evolution_artifacts
    flatbuffer_schema_evolution_v2_package)
target_compile_features(flatbuffer_schema_evolution_characterization PRIVATE cxx_std_20)
target_compile_definitions(flatbuffer_schema_evolution_characterization
    PRIVATE
        "EVOLUTION_V1_BFBS=\"${schema_evolution_generated_dir}/evolution_v1.bfbs\""
        "EVOLUTION_RENAMED_BFBS=\"${schema_evolution_generated_dir}/evolution_renamed.bfbs\""
        "EVOLUTION_REQUIRED_BFBS=\"${schema_evolution_generated_dir}/incompatible_required_field.bfbs\"")
target_include_directories(flatbuffer_schema_evolution_characterization
    PRIVATE
        "${schema_evolution_generated_dir}"
        "${schema_evolution_package_dir}")
target_link_libraries(flatbuffer_schema_evolution_characterization
    PRIVATE
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema)
set_target_properties(flatbuffer_schema_evolution_characterization
    PROPERTIES CXX_EXTENSIONS OFF)

# REQ: SCHEMA-002
add_test(NAME CHANNEL_SCHEMA.flatbuffer_evolution_cross_read
    COMMAND "${CMAKE_COMMAND}"
        "-DV1_CODEC=$<TARGET_FILE:flatbuffer_schema_evolution_v1_codec>"
        "-DV2_CODEC=$<TARGET_FILE:flatbuffer_schema_evolution_v2_codec>"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/flatbuffer_schema_evolution/cross-read"
        -P "${schema_evolution_source_dir}/RunEvolutionCrossRead.cmake")
set_tests_properties(CHANNEL_SCHEMA.flatbuffer_evolution_cross_read
    PROPERTIES LABELS "ea2;channel;schema;flatbuffers;evolution")

# REQ: SCHEMA-002
add_test(NAME CHANNEL_SCHEMA.flatbuffer_evolution_conformance
    COMMAND "${CMAKE_COMMAND}"
        "-DFLATC=$<TARGET_FILE:flatbuffers::flatc>"
        "-DSOURCE_DIR=${schema_evolution_source_dir}"
        -P "${schema_evolution_source_dir}/VerifyEvolutionConformance.cmake")
set_tests_properties(CHANNEL_SCHEMA.flatbuffer_evolution_conformance
    PROPERTIES LABELS "ea2;channel;schema;flatbuffers;evolution")

# REQ: SCHEMA-001
# REQ: SCHEMA-002
# REQ: SCHEMA-003
add_test(NAME CHANNEL_SCHEMA.flatbuffer_evolution_characterization
    COMMAND flatbuffer_schema_evolution_characterization)
set_tests_properties(CHANNEL_SCHEMA.flatbuffer_evolution_characterization
    PROPERTIES LABELS "ea2;channel;schema;flatbuffers;evolution")

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
        PROPERTIES LABELS "channel;schema;third-party;examples")
endif()

# REQ: CHAN-120
# REQ: CHAN-143
# REQ: OPERATOR-035
# REQ: SCHEMA-004
holoscan_add_installed_consumer_test(
    CHANNEL_SCHEMA.installed_third_party_descriptor_tensor_consumer
    "${holoscan_SOURCE_DIR}/examples/flatbuffer_schemas/descriptor_tensor_schema/cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/third_party_descriptor_tensor_consumer"
)
set_tests_properties(CHANNEL_SCHEMA.installed_third_party_descriptor_tensor_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "channel;schema;tensor;third-party;installed-package"
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
        TRAITS_HEADER descriptor_frame_multi_order_traits.hpp
        MAX_SERIALIZED_SIZE 512
    )
endforeach()

get_target_property(descriptor_tensor_schema_codegen_a_traits_header
    descriptor_tensor_schema_codegen_a HOLOSCAN_FLATBUFFER_TRAITS_HEADER)
get_target_property(descriptor_tensor_schema_codegen_b_traits_header
    descriptor_tensor_schema_codegen_b HOLOSCAN_FLATBUFFER_TRAITS_HEADER)

set(descriptor_schema_determinism_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/traits.deterministic")
add_custom_command(
    OUTPUT "${descriptor_schema_determinism_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E compare_files
        "${descriptor_tensor_schema_codegen_a_traits_header}"
        "${descriptor_tensor_schema_codegen_b_traits_header}"
    COMMAND "${CMAKE_COMMAND}"
        "-DGENERATED_TRAITS_A=${descriptor_tensor_schema_codegen_a_traits_header}"
        "-DGENERATED_TRAITS_B=${descriptor_tensor_schema_codegen_b_traits_header}"
        -P "${descriptor_schema_codegen_source_dir}/VerifyDescriptorSchemaIntermediates.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${descriptor_schema_determinism_stamp}"
    DEPENDS
        "${descriptor_tensor_schema_codegen_a_traits_header}"
        "${descriptor_tensor_schema_codegen_b_traits_header}"
        "${descriptor_schema_codegen_source_dir}/VerifyDescriptorSchemaIntermediates.cmake"
    COMMENT "Comparing independently generated descriptor traits headers"
    VERBATIM
)

set(descriptor_schema_multi_order_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/field_order.verified")
add_custom_command(
    OUTPUT "${descriptor_schema_multi_order_stamp}"
    COMMAND "${CMAKE_COMMAND}"
        "-DGENERATED_TRAITS=${descriptor_tensor_schema_codegen_a_traits_header}"
        -P "${descriptor_schema_codegen_source_dir}/VerifyDescriptorTensorFieldOrder.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${descriptor_schema_multi_order_stamp}"
    DEPENDS
        "${descriptor_tensor_schema_codegen_a_traits_header}"
        "${descriptor_schema_codegen_source_dir}/VerifyDescriptorTensorFieldOrder.cmake"
    COMMENT "Verifying canonical reflected descriptor Tensor field ordering"
    VERBATIM
)
add_custom_target(descriptor_tensor_schema_codegen_validation
    DEPENDS
        "${descriptor_schema_determinism_stamp}"
        "${descriptor_schema_multi_order_stamp}"
)

# REQ: CHAN-137
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
# REQ: CHAN-137
# REQ: CHAN-126
# REQ: CHAN-127
# REQ: EXTERNAL-010
add_executable(multi_tensor_schema_runtime_validation
    descriptor_tensor_schema_codegen/multi_tensor_schema_runtime_validation.cpp)
add_dependencies(multi_tensor_schema_runtime_validation
    descriptor_tensor_schema_codegen_a
    descriptor_tensor_schema_codegen_validation)
target_compile_features(multi_tensor_schema_runtime_validation PRIVATE cxx_std_20)
target_include_directories(multi_tensor_schema_runtime_validation
    PRIVATE
        "${holoscan_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/descriptor_tensor_schema_codegen/generated-a")
target_link_libraries(multi_tensor_schema_runtime_validation
    PRIVATE
        CUDA::cudart
        holoscan::core
        holoscan::schema)
set_target_properties(multi_tensor_schema_runtime_validation PROPERTIES CXX_EXTENSIONS OFF)

foreach(validation_case IN ITEMS
        contracts
        rate-fields
        rate-fields-fixed
        rate-fields-configure
        rate-fields-restart
        rate-fields-restart-reject
        rate-fields-warm-up
        rate-fields-arm
        process-local
        process-local-fanout
        inline-partition
        inline-partition-cuda
        over-bound
        bundle-invalid
        bundle-restart
        external-topic-import
        external-topic-import-fanout
        external-topic-export
        external-topic-export-cross-process
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
        PROPERTIES LABELS "channel;schema;tensor;third-party;examples")

    # Key setup: use the singular descriptor-schema output API with one CUDA-device Tensor field.
    add_test(NAME CHANNEL_SCHEMA.public_descriptor_schema_cuda_validation
        COMMAND public_descriptor_schema_validation device)
    # Key assertion: compiler-owned CUDA storage binds without adding a dummy schema Tensor field.
    set_tests_properties(CHANNEL_SCHEMA.public_descriptor_schema_cuda_validation
        PROPERTIES
            LABELS "channel;schema;tensor;third-party;cuda"
            SKIP_RETURN_CODE 77
            TIMEOUT 30)

    # Key setup: fan out one singular CUDA descriptor-schema publication through HoloMQ.
    add_test(NAME CHANNEL_SCHEMA.public_descriptor_schema_cuda_fanout_validation
        COMMAND public_descriptor_schema_validation device_fanout)
    # Key assertion: both subscribers bind a CUDA stream and observe the same retained backing.
    set_tests_properties(CHANNEL_SCHEMA.public_descriptor_schema_cuda_fanout_validation
        PROPERTIES
            LABELS "channel;schema;tensor;third-party;cuda;fanout;holomq"
            SKIP_RETURN_CODE 77
            TIMEOUT 30)

    # Key setup: emit external CUDA Tensors by value through the singular descriptor HoloMQ path.
    add_test(NAME CHANNEL_SCHEMA.public_descriptor_schema_cuda_fanout_by_value_validation
        COMMAND public_descriptor_schema_validation device_fanout_by_value)
    # Key assertion: dtype, rank, and span drift are rejected before one matching fanout succeeds.
    set_tests_properties(
        CHANNEL_SCHEMA.public_descriptor_schema_cuda_fanout_by_value_validation
        PROPERTIES
            LABELS "channel;schema;tensor;third-party;cuda;fanout;holomq"
            SKIP_RETURN_CODE 77
            TIMEOUT 30)
endif()

# REQ: CHAN-077
# REQ: CHAN-120
holoscan_add_installed_consumer_test(
    EXAMPLE_CPP_CUSTOM_SCHEMA.installed_package_consumer
    "${holoscan_SOURCE_DIR}/examples/flatbuffer_schemas/custom_schema/cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/custom_schema_installed_consumer"
)
set_tests_properties(EXAMPLE_CPP_CUSTOM_SCHEMA.installed_package_consumer
    PROPERTIES
        RUN_SERIAL TRUE
        LABELS "examples;schema;installed-package"
)

# The exec-child arm uses POSIX pipes/fork/exec intentionally. Native Windows
# remains a release qualification target, but does not compile this Linux
# process-boundary fixture.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(external_client_matrix external_client_matrix.cpp)
    target_link_libraries(external_client_matrix
        PRIVATE
            holoscan::core
            holoscan::schema
            Threads::Threads
    )

    function(holoscan_add_external_matrix_test test_name cell_index)
        # REQ: CHAN-132
        # REQ: CHAN-136
        # REQ: SYSTEM-001
        # REQ: SYSTEM-002
        # REQ: SYSTEM-003
        add_test(NAME ${test_name}
            COMMAND external_client_matrix --cell ${cell_index}
        )
        set_tests_properties(${test_name}
            PROPERTIES
                RUN_SERIAL TRUE
                TIMEOUT 45
                LABELS "system;external-client;matrix;linux"
        )
    endfunction()

    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.same_process.custom_pod 0)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.same_process.custom_pod 1)
    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.same_process.schema 2)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.same_process.schema 3)
    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.same_process.host_tensor 4)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.same_process.host_tensor 5)
    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.exec_child.custom_pod 6)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.exec_child.custom_pod 7)
    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.exec_child.schema 8)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.exec_child.schema 9)
    holoscan_add_external_matrix_test(
        SYSTEM.external_publisher_to_graph_import.exec_child.host_tensor 10)
    holoscan_add_external_matrix_test(
        SYSTEM.graph_export_to_external_subscriber.exec_child.host_tensor 11)

    # REQ: CHAN-132
    # REQ: CHAN-136
    # REQ: RUNTIME-017
    holoscan_add_installed_consumer_test(
        CHANNEL.external_eos.installed_package_consumer
        "${CMAKE_CURRENT_SOURCE_DIR}/external_eos_installed_consumer"
        "${CMAKE_CURRENT_BINARY_DIR}/external_eos_installed_consumer"
    )
    # The watchdog covers SDK build/install and consumer compilation as well as execution.
    # Allow bounded build time in source/header-tidy configurations before running the EOS check.
    set_tests_properties(CHANNEL.external_eos.installed_package_consumer
        PROPERTIES
            RUN_SERIAL TRUE
            TIMEOUT 180
            LABELS "channel;external-client;eos;installed-package;linux"
    )
endif()
