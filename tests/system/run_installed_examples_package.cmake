# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_EXAMPLES_WORK_DIR
        HOLOSCAN_INSTALL_LIB_DIR
        HOLOSCAN_EXPECT_SHARED
        HOLOSCAN_GENERATOR
        HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(install_prefix "${HOLOSCAN_EXAMPLES_WORK_DIR}/install")
set(examples_build_dir "${HOLOSCAN_EXAMPLES_WORK_DIR}/build")
file(REMOVE_RECURSE "${install_prefix}" "${examples_build_dir}")

# Build every shipped example before component installation so a focused CTest
# invocation exercises the same complete source tree as the container image.
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}"
    --target
        installed_examples_package_dependencies
        hello_world
        hello_core
        hello_logger
        hello_tensor
        hello_schema
        hello_matx
        hello_cccl
        ea1_slow_fast_fusion
        ea1_camera_imu_control
        ea1_sensor_daemon
        ea1_result_recorder
        first_pipeline
        first_operator
        tensors_101
        typed_parameters
        contract_on_any_input
        contract_on_each
        contract_on_clock
        contract_on_aligned
        contract_on_window
        contract_on_predicate
        contract_selectors
        contract_missing_data
        contract_deadlines
        errors_as_values
        time_and_clocks
        compile_and_check
        running_and_stopping
        retained_tensor_compositor
        embedded_device_tensor
        notification_bridge
        operator_anatomy
        parameter_schemas
        lifecycle_callbacks
        graph_composition
        reusable_graphs
        pod_payloads
        schema_payloads
        tensors_in_depth
        plan_artifacts
        restart_control
        runtime_params
        determinism_replay
        custom_schema
        descriptor_tensor_schema
        multi_tensor_schema
)
if(HOLOSCAN_CONFIG_COMPONENT_ENABLED AND
   HOLOSCAN_CONFIG_IO_COMPONENT_ENABLED AND
   HOLOSCAN_PARAMS_COMPONENT_ENABLED AND
   HOLOSCAN_PARAMS_IO_COMPONENT_ENABLED)
    list(APPEND sdk_build_command config_file)
endif()
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND sdk_build_command web_tuner)
endif()
if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
    list(APPEND sdk_build_command --config "${HOLOSCAN_BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${sdk_build_command}
    RESULT_VARIABLE sdk_build_result
    TIMEOUT 600
)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the installed examples")
endif()

# Seed the explicit build-target output with a stale graph ID. The component installer must ignore
# this tree, generate a fresh bundle from the already-built executable, and install only that bundle.
set(sdk_artifact_output
    "${HOLOSCAN_BUILD_DIR}/examples/ea1_camera_imu_control/cpp/holoscan-artifacts"
)
file(REMOVE_RECURSE "${sdk_artifact_output}")
file(MAKE_DIRECTORY "${sdk_artifact_output}/stale-graph-id/developer")
file(WRITE "${sdk_artifact_output}/stale-graph-id/developer/stale-marker.txt" "stale\n")

set(installed_example_components
    holoscan
    holoscan-dependencies
    holomq-iceoryx2-libraries
    dev
    holomq
    holoscan-core
)
# On Windows, a shared iceoryx2 library's DLL is installed in the bin component.
if(WIN32 AND HOLOSCAN_EXPECT_SHARED)
    list(APPEND installed_example_components bin)
endif()
# Optional tooling is shipped as independently selectable install components.
# Install each enabled dependency before its consumers so this fixture proves
# the same component closure expected from a package assembled with those
# features, rather than relying on a build-tree target that is absent from the
# isolated prefix.
if(HOLOSCAN_CONFIG_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-config)
endif()
if(HOLOSCAN_CONFIG_IO_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-config-io)
endif()
if(HOLOSCAN_PARAMS_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-params)
endif()
if(HOLOSCAN_PARAMS_IO_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-params-io)
endif()
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-experimental-params-web-tuner)
endif()
list(APPEND installed_example_components
    holoscan-examples
    holoscan-graph-artifacts
)

foreach(component IN LISTS installed_example_components)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND install_command --config "${HOLOSCAN_BUILD_CONFIG}")
    endif()
    execute_process(
        COMMAND ${install_command}
        RESULT_VARIABLE install_result
        OUTPUT_VARIABLE install_output
        ERROR_VARIABLE install_error_output
        TIMEOUT 300
    )
    if(NOT install_result EQUAL 0)
        set(install_diagnostic "${install_error_output}")
        if(install_diagnostic STREQUAL "")
            set(install_diagnostic "${install_output}")
        endif()
        if(install_diagnostic STREQUAL "")
            set(install_diagnostic "<no command output>")
        endif()
        string(LENGTH "${install_diagnostic}" install_diagnostic_length)
        if(install_diagnostic_length GREATER 16384)
            math(EXPR install_diagnostic_start "${install_diagnostic_length} - 16384")
            string(SUBSTRING "${install_diagnostic}" ${install_diagnostic_start} -1
                install_diagnostic)
            string(PREPEND install_diagnostic "<earlier command output omitted>\n")
        endif()
        message(FATAL_ERROR
            "Failed to install SDK component ${component} (result ${install_result}):\n"
            "${install_diagnostic}")
    endif()
endforeach()

if(HOLOSCAN_EXPECT_SHARED)
    set(iceoryx2_cxx_selected_target "iceoryx2-cxx::shared-lib-cxx")
    set(iceoryx2_cxx_unselected_target "iceoryx2-cxx::static-lib-cxx")
    set(iceoryx2_cxx_shared_library_dir "${HOLOSCAN_INSTALL_LIB_DIR}")
    if(WIN32)
        set(iceoryx2_cxx_shared_library_dir "${HOLOSCAN_INSTALL_BIN_DIR}")
    endif()
    set(iceoryx2_cxx_selected_artifact_glob
        "${install_prefix}/${iceoryx2_cxx_shared_library_dir}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}holomq_iceoryx2_cxx${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
    set(iceoryx2_ffi_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}holomq_iceoryx2_ffi_c${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
    set(iceoryx2_cxx_unselected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}holomq_iceoryx2_cxx${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
    set(iceoryx2_ffi_unselected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}holomq_iceoryx2_ffi_c${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
else()
    set(iceoryx2_cxx_selected_target "iceoryx2-cxx::static-lib-cxx")
    set(iceoryx2_cxx_unselected_target "iceoryx2-cxx::shared-lib-cxx")
    set(iceoryx2_cxx_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}holomq_iceoryx2_cxx${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
    set(iceoryx2_ffi_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}holomq_iceoryx2_ffi_c${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
    set(iceoryx2_cxx_unselected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}holomq_iceoryx2_cxx${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
    set(iceoryx2_ffi_unselected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}holomq_iceoryx2_ffi_c${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
endif()

set(iceoryx2_cxx_targets_file
    "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/iceoryx2-cxx/iceoryx2-cxxTargets.cmake")
if(NOT EXISTS "${iceoryx2_cxx_targets_file}")
    message(FATAL_ERROR "Missing installed iceoryx2 C++ targets file: ${iceoryx2_cxx_targets_file}")
endif()
file(READ "${iceoryx2_cxx_targets_file}" iceoryx2_cxx_targets_contents)
string(FIND "${iceoryx2_cxx_targets_contents}" "${iceoryx2_cxx_selected_target}"
    iceoryx2_cxx_selected_target_index)
if(iceoryx2_cxx_selected_target_index EQUAL -1)
    message(FATAL_ERROR
        "Installed iceoryx2 C++ package does not export ${iceoryx2_cxx_selected_target}")
endif()
string(FIND "${iceoryx2_cxx_targets_contents}" "${iceoryx2_cxx_unselected_target}"
    iceoryx2_cxx_unselected_target_index)
if(NOT iceoryx2_cxx_unselected_target_index EQUAL -1)
    message(FATAL_ERROR
        "Installed iceoryx2 C++ package unexpectedly exports ${iceoryx2_cxx_unselected_target}")
endif()

foreach(iceoryx2_selected_artifact_glob IN ITEMS
        "${iceoryx2_cxx_selected_artifact_glob}"
        "${iceoryx2_ffi_selected_artifact_glob}")
    file(GLOB iceoryx2_selected_artifacts LIST_DIRECTORIES FALSE
        "${iceoryx2_selected_artifact_glob}")
    if(NOT iceoryx2_selected_artifacts)
        message(FATAL_ERROR
            "Missing selected iceoryx2 artifact matching ${iceoryx2_selected_artifact_glob}")
    endif()
endforeach()
foreach(iceoryx2_unselected_artifact_glob IN ITEMS
        "${iceoryx2_cxx_unselected_artifact_glob}"
        "${iceoryx2_ffi_unselected_artifact_glob}")
    file(GLOB iceoryx2_unselected_artifacts LIST_DIRECTORIES FALSE
        "${iceoryx2_unselected_artifact_glob}")
    if(iceoryx2_unselected_artifacts)
        message(FATAL_ERROR
            "Unexpected unselected iceoryx2 artifacts: ${iceoryx2_unselected_artifacts}")
    endif()
endforeach()

set(installed_artifact_root
    "${install_prefix}/share/ea1_camera_imu_control/holoscan"
)
file(GLOB installed_artifact_graph_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${installed_artifact_root}"
    "${installed_artifact_root}/*"
)
list(LENGTH installed_artifact_graph_entries installed_artifact_graph_entry_count)
if(NOT installed_artifact_graph_entry_count EQUAL 1)
    message(FATAL_ERROR
        "Expected exactly one installed camera/IMU graph-id directory, found: "
        "${installed_artifact_graph_entries}")
endif()
list(GET installed_artifact_graph_entries 0 installed_artifact_graph_id)
set(installed_artifact_graph_directory
    "${installed_artifact_root}/${installed_artifact_graph_id}"
)
if(NOT IS_DIRECTORY "${installed_artifact_graph_directory}" OR
   IS_SYMLINK "${installed_artifact_graph_directory}")
    message(FATAL_ERROR
        "Installed artifact graph-id entry is not a regular directory: "
        "${installed_artifact_graph_directory}")
endif()

file(GLOB installed_artifact_deployment_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${installed_artifact_graph_directory}"
    "${installed_artifact_graph_directory}/*"
)
if(NOT "${installed_artifact_deployment_entries}" STREQUAL "developer")
    message(FATAL_ERROR
        "Expected exactly the developer deployment directory, found: "
        "${installed_artifact_deployment_entries}")
endif()
set(installed_artifact_bundle "${installed_artifact_graph_directory}/developer")
if(NOT IS_DIRECTORY "${installed_artifact_bundle}" OR
   IS_SYMLINK "${installed_artifact_bundle}")
    message(FATAL_ERROR
        "Installed artifact deployment entry is not a regular directory: "
        "${installed_artifact_bundle}")
endif()

string(CONCAT stale_installed_artifact
    "${installed_artifact_root}/"
    "stale-graph-id/developer/stale-marker.txt"
)
if(EXISTS "${stale_installed_artifact}")
    message(FATAL_ERROR "Artifact component installed stale build-target output")
endif()
file(GLOB installed_artifact_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${installed_artifact_bundle}"
    "${installed_artifact_bundle}/*"
)
set(expected_artifact_files
    artifact-manifest.json
    endpoint-catalog.json
    execution-plan.json
    profile-manifest.json
)
list(SORT installed_artifact_entries)
list(SORT expected_artifact_files)
if(NOT "${installed_artifact_entries}" STREQUAL "${expected_artifact_files}")
    message(FATAL_ERROR
        "Installed artifact bundle does not have the exact canonical hierarchy: "
        "${installed_artifact_entries}")
endif()
foreach(expected_artifact_file IN LISTS expected_artifact_files)
    set(installed_artifact_file "${installed_artifact_bundle}/${expected_artifact_file}")
    if(NOT EXISTS "${installed_artifact_file}" OR
       IS_DIRECTORY "${installed_artifact_file}" OR
       IS_SYMLINK "${installed_artifact_file}")
        message(FATAL_ERROR
            "Installed canonical artifact is not a regular file: ${installed_artifact_file}")
    endif()
endforeach()

set(holoscan_package_dir
    "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
)
set(required_install_paths
    "${holoscan_package_dir}/holoscan-config.cmake"
    "${holoscan_package_dir}/HoloscanLoggingPolicy.cmake"
    "${holoscan_package_dir}/HoloscanGraphArtifacts.cmake"
    "${install_prefix}/examples/CMakeLists.txt"
)
if(HOLOSCAN_CONFIG_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-config-targets.cmake")
endif()
if(HOLOSCAN_CONFIG_IO_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-config-io-targets.cmake")
endif()
if(HOLOSCAN_PARAMS_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-params-targets.cmake")
endif()
if(HOLOSCAN_PARAMS_IO_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-params-io-targets.cmake")
endif()
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-experimental-params-web-tuner-targets.cmake")
endif()
foreach(required_install_path IN LISTS required_install_paths)
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed examples prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

# REQ: SYSTEM-014
# Execute the packaged target itself with inherited loader search paths removed.
# The later installed-source consumer build proves that the example remains
# buildable; this run independently proves the shipped target's INSTALL_RPATH.
set(installed_pod_payloads_executable
    "${install_prefix}/examples/experimental/pod_payloads/cpp/pod_payloads${HOLOSCAN_EXECUTABLE_SUFFIX}"
)
if(NOT EXISTS "${installed_pod_payloads_executable}")
    message(FATAL_ERROR
        "Installed POD payloads executable is missing: ${installed_pod_payloads_executable}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH
        --unset=DYLD_LIBRARY_PATH
        "${installed_pod_payloads_executable}"
    RESULT_VARIABLE installed_pod_payloads_result
    OUTPUT_VARIABLE installed_pod_payloads_output
    ERROR_VARIABLE installed_pod_payloads_error_output
    TIMEOUT 30
)
string(CONCAT installed_pod_payloads_log
    "${installed_pod_payloads_output}"
    "${installed_pod_payloads_error_output}"
)
if(NOT installed_pod_payloads_result EQUAL 0 OR
   NOT installed_pod_payloads_log MATCHES
       "pod_payloads: decoded 6 FusionControl samples \\(last frame=6, size=32 bytes, zero padding\\) termination=stopped")
    message(FATAL_ERROR
        "Failed to execute the installed POD payloads target with sanitized loader paths: "
        "${installed_pod_payloads_log}")
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${install_prefix}/examples"
    -B "${examples_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-Dholoscan_DIR:PATH=${holoscan_package_dir}"
    "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
    "-DBUILD_TESTING:BOOL=ON"
    "-DHOLOSCAN_INSTALL_EXAMPLE_SOURCE:BOOL=OFF"
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_CUDA_ARCHITECTURES AND NOT HOLOSCAN_CUDA_ARCHITECTURES STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_CUDA_ARCHITECTURES:STRING=${HOLOSCAN_CUDA_ARCHITECTURES}"
    )
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
# Let Visual Studio or an explicit toolchain select both host and CUDA compilers. For the ordinary
# no-toolchain path, forward nonempty parent compilers so the isolated examples exercise the same
# frontend pair as the SDK build.
set(toolchain_selects_compilers FALSE)
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_selects_compilers TRUE)
endif()
if(NOT HOLOSCAN_GENERATOR MATCHES "^Visual Studio" AND NOT toolchain_selects_compilers)
    if(DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
        list(APPEND configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
    endif()
    if(DEFINED HOLOSCAN_CUDA_COMPILER AND NOT HOLOSCAN_CUDA_COMPILER STREQUAL "")
        list(APPEND configure_command "-DCMAKE_CUDA_COMPILER:FILEPATH=${HOLOSCAN_CUDA_COMPILER}")
    endif()
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}"
    )
endif()
if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${HOLOSCAN_BUILD_CONFIG}")
endif()

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    TIMEOUT 300
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the complete isolated installed examples tree")
endif()

set(examples_build_command "${CMAKE_COMMAND}" --build "${examples_build_dir}")
if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
    list(APPEND examples_build_command --config "${HOLOSCAN_BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${examples_build_command}
    RESULT_VARIABLE examples_build_result
    TIMEOUT 600
)
if(NOT examples_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the complete isolated installed examples tree")
endif()

# A successful isolated compile is insufficient evidence for the reference
# applications: execute the bounded, self-contained Graphs, the lifecycle-owned
# notification bridge, and the camera/IMU graph-artifact build target from that
# consumer build as well. The latter proves that installed sources consume only
# the installed artifact helper and preserve cold, write-if-changed publication.
# CTest resolves executable locations correctly for both single- and multi-config
# generators. Enumerate every required row before executing the aggregate so a
# partial installed test inventory cannot satisfy the phase accidentally.
set(installed_example_tests
    EXAMPLE_EA1_SLOW_FAST_FUSION_TEST
    EXAMPLE_EA1_CAMERA_IMU_RUNTIME_TEST
    EXAMPLE_EA1_CAMERA_IMU_ARTIFACT_TARGET_TEST
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_HOST
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_CUDA
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_INVALID_DEVICE
    EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_TEST
    EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_ARTIFACT_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_FAILURE_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_UNSUPPORTED_TEST
)
# The experimental raw-POD example is not one of the eleven reference
# applications, but execute it from the relocated installed tree to verify its
# deeper install path resolves the SDK libraries.
set(installed_example_smoke_tests
    EXAMPLE_CPP_POD_PAYLOADS_TEST
)
set(installed_example_execution_tests
    ${installed_example_tests}
    ${installed_example_smoke_tests}
)
foreach(installed_example_test IN LISTS installed_example_execution_tests)
    set(examples_test_list_command
        "${HOLOSCAN_CTEST_COMMAND}"
        --test-dir "${examples_build_dir}"
        --show-only=json-v1
        -R "^${installed_example_test}$"
    )
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND examples_test_list_command -C "${HOLOSCAN_BUILD_CONFIG}")
    endif()
    execute_process(
        COMMAND ${examples_test_list_command}
        RESULT_VARIABLE examples_test_list_result
        OUTPUT_VARIABLE examples_test_list_json
        ERROR_VARIABLE examples_test_list_error_output
        TIMEOUT 30
    )
    string(JSON examples_test_list_count
        ERROR_VARIABLE examples_test_list_json_error
        LENGTH "${examples_test_list_json}" tests)
    if(NOT examples_test_list_result EQUAL 0 OR
       NOT examples_test_list_json_error STREQUAL "NOTFOUND" OR
       NOT examples_test_list_count EQUAL 1)
        message(FATAL_ERROR
            "Installed example test ${installed_example_test} was not registered exactly once: "
            "${examples_test_list_error_output}${examples_test_list_json_error}")
    endif()
    string(JSON examples_test_list_name
        ERROR_VARIABLE examples_test_list_name_error
        GET "${examples_test_list_json}" tests 0 name)
    if(NOT examples_test_list_name_error STREQUAL "NOTFOUND" OR
       NOT examples_test_list_name STREQUAL installed_example_test)
        message(FATAL_ERROR
            "Installed example test inventory returned ${examples_test_list_name} "
            "while checking ${installed_example_test}: ${examples_test_list_name_error}")
    endif()
endforeach()

list(JOIN installed_example_execution_tests "|" installed_example_test_alternatives)
set(installed_example_test_regex "^(${installed_example_test_alternatives})$")
set(examples_test_command
    "${HOLOSCAN_CTEST_COMMAND}"
    --test-dir "${examples_build_dir}"
    --verbose
    --output-on-failure
    --timeout 30
    --no-tests=error
    -R "${installed_example_test_regex}"
)
if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
    list(APPEND examples_test_command -C "${HOLOSCAN_BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${examples_test_command}
    RESULT_VARIABLE examples_test_result
    OUTPUT_VARIABLE examples_test_output
    ERROR_VARIABLE examples_test_error_output
    TIMEOUT 480
)
if(NOT examples_test_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to execute EA1 reference applications, installed smoke tests, notification "
        "bridge, and artifact target "
        "from the installed example sources: "
        "${examples_test_output}${examples_test_error_output}")
endif()

# The aggregate CTest result is necessary but insufficient for the fresh-prefix
# README replay. Its individual rows own exact output assertions; retain verbose
# output here and independently require one stable witness from every Task 25
# README mode family so a weakened or disconnected test cannot pass silently.
string(CONCAT examples_test_log "${examples_test_output}" "${examples_test_error_output}")
set(installed_readme_output_witnesses
    "retained Tensor compositor complete"
    "artifact-only publication complete"
    "SKIP: notification_bridge requires Linux eventfd"
)
foreach(installed_readme_output_witness IN LISTS installed_readme_output_witnesses)
    string(FIND
        "${examples_test_log}"
        "${installed_readme_output_witness}"
        installed_readme_output_offset)
    if(installed_readme_output_offset EQUAL -1)
        message(FATAL_ERROR
            "Installed README replay omitted output witness: "
            "${installed_readme_output_witness}")
    endif()
endforeach()

if(NOT examples_test_log MATCHES "embedded_device_tensor PASS: flat-equivalent reusable plan replayed twice|embedded_device_tensor skipped: no (usable )?CUDA device")
    message(FATAL_ERROR
        "Installed embedded-device README replay omitted its success-or-skip output witness")
endif()

if(NOT examples_test_log MATCHES "retained_tensor_compositor: skipped: CUDA device unavailable|route=holomq://retained-tensor-compositor\\.overlay-[^ \r\n]+ frame_memory=cuda_device device=0")
    message(FATAL_ERROR
        "Installed retained-Tensor CUDA README replay omitted its success-or-skip output witness")
endif()

if(NOT examples_test_log MATCHES "authored CUDA device ordinal 2147483647 is outside|retained_tensor_compositor: skipped: CUDA device unavailable")
    message(FATAL_ERROR
        "Installed invalid-device README replay omitted its rejection-or-skip output witness")
endif()

if(NOT examples_test_log MATCHES "notification_bridge: eventfd bridged 4 notifications|SKIP: notification_bridge requires Linux eventfd")
    message(FATAL_ERROR
        "Installed notification README replay omitted its normal success-or-skip output witness")
endif()

if(NOT examples_test_log MATCHES "notification_bridge: worker failure propagated through RUNTIME_OPERATOR_CALLBACK_FAILED|SKIP: notification_bridge requires Linux eventfd")
    message(FATAL_ERROR
        "Installed notification README replay omitted its failure success-or-skip output witness")
endif()
