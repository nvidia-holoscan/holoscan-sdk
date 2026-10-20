# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_EXAMPLES_WORK_DIR
        HOLOSCAN_INSTALL_LIB_DIR
        HOLOSCAN_INSTALL_DATA_DIR
        HOLOSCAN_EXAMPLES_INSTALL_DIR
        HOLOSCAN_EXPECT_SHARED
        HOLOSCAN_GENERATOR
        HOLOSCAN_PYTHON_COMPONENT_ENABLED
        HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED AND
   (NOT DEFINED HOLOSCAN_PYTHON_EXECUTABLE OR HOLOSCAN_PYTHON_EXECUTABLE STREQUAL ""))
    message(FATAL_ERROR "HOLOSCAN_PYTHON_EXECUTABLE is required when Python support is enabled")
endif()

set(install_destination "/usr")
set(install_staging_root "${HOLOSCAN_EXAMPLES_WORK_DIR}/stage")
set(install_prefix "${install_staging_root}${install_destination}")
set(examples_build_dir "${HOLOSCAN_EXAMPLES_WORK_DIR}/build")
set(custom_examples_data_dir "custom-example-data")
file(REMOVE_RECURSE "${install_staging_root}" "${examples_build_dir}")

# Build every shipped example before component installation so a focused CTest
# invocation exercises the same complete source tree as the container image.
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --parallel 4
    --target
        installed_examples_package_dependencies
        hello_world
        hello_core
        hello_logger
        hello_tensor
        hello_schema
        hello_matx
        hello_cccl
        slow_fast_fusion
        camera_imu_control
        sensor_daemon
        result_recorder
        external_client_relay
        holomq_main
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
        contract_max_start_delay
        errors_as_values
        time_and_clocks
        compile_and_check
        diagnostic_reporting
        running_and_stopping
        retained_tensor_compositor
        borrowed_image_transform
        embedded_device_tensor
        notification_bridge
        green_context
        operator_anatomy
        parameter_schemas
        derived_tensor_pool_capacity
        rate_aware_ports
        lifecycle_callbacks
        graph_composition
        automatic_partitioning
        cuda_graph_region
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
        descriptor_tensor_external_topic_graph
        descriptor_tensor_external_topic_client
        config_file
)
if(HOLOSCAN_TESTING_COMPONENT_ENABLED)
    list(APPEND sdk_build_command
        operator_testing operator_testing_tensor operator_testing_image operator_testing_device
        operator_testing_schema operator_testing_lifecycle operator_testing_clock operator_testing_restart)
endif()
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND sdk_build_command web_tuner)
endif()
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    list(APPEND sdk_build_command
        holoscan-python
        descriptor_tensor_external_topic_generated__python_codec)
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
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
    "${HOLOSCAN_BUILD_DIR}/examples/camera_imu_control/cpp/holoscan-artifacts"
)
file(REMOVE_RECURSE "${sdk_artifact_output}")
file(MAKE_DIRECTORY "${sdk_artifact_output}/stale-graph-id/developer")
file(WRITE "${sdk_artifact_output}/stale-graph-id/developer/stale-marker.txt" "stale\n")

set(installed_example_components
    holoscan
    holoscan-dependencies
    iceoryx2-lib
    dev
    holoscan-holomq
    holoscan-core
    holoscan-sensor-io
    holoscan-core-examples
    holoscan-config
    holoscan-config-io
)
if(HOLOSCAN_TESTING_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-testing holoscan-testing-examples)
endif()
# On Windows, a shared iceoryx2 library's DLL is installed in the bin component.
if(WIN32 AND HOLOSCAN_EXPECT_SHARED)
    list(APPEND installed_example_components bin)
endif()
# Optional tooling is shipped as an independently selectable install component. Install it after
# its dependencies so this fixture proves the same component closure expected from a package
# assembled with that feature, rather than relying on a build-tree target that is absent from the
# isolated prefix.
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-experimental-params-web-tuner)
endif()
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    list(APPEND installed_example_components holoscan-python_libs)
endif()
list(APPEND installed_example_components holoscan-graph-artifacts)

foreach(component IN LISTS installed_example_components)
    set(install_command
        "${CMAKE_COMMAND}" -E env "DESTDIR=${install_staging_root}"
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_destination}"
        --component "${component}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
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

set(installed_example_runner
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/testing/run_example_tests")
if(NOT EXISTS "${installed_example_runner}")
    message(FATAL_ERROR "Missing installed example runner: ${installed_example_runner}")
endif()
if(UNIX AND NOT IS_EXECUTABLE "${installed_example_runner}")
    message(FATAL_ERROR "Installed example runner is not executable: ${installed_example_runner}")
endif()

if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    set(installed_external_clients_source
        "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients")
    set(installed_external_clients_test
        "${installed_external_clients_source}/python/test_run_clients.py")
    if(NOT EXISTS "${installed_external_clients_test}")
        message(FATAL_ERROR
            "Missing installed external-client test helper: ${installed_external_clients_test}")
    endif()

    # Configure the component-installed source inventory independently in both consumer modes.
    # The external-clients target lists every Python source even when tests are disabled, while
    # BUILD_TESTING=ON additionally resolves the pytest command that uses this helper.
    foreach(external_clients_testing IN ITEMS ON OFF)
        string(TOLOWER "${external_clients_testing}" external_clients_testing_lower)
        set(external_clients_build_dir
            "${HOLOSCAN_EXAMPLES_WORK_DIR}/external-clients-${external_clients_testing_lower}")
        file(REMOVE_RECURSE "${external_clients_build_dir}")
        set(external_clients_configure_command
            "${CMAKE_COMMAND}"
            -S "${CMAKE_CURRENT_LIST_DIR}/installed_external_clients_source"
            -B "${external_clients_build_dir}"
            -G "${HOLOSCAN_GENERATOR}"
            "-DEXAMPLE_SOURCE:PATH=${installed_external_clients_source}"
            "-DBUILD_TESTING:BOOL=${external_clients_testing}"
        )
        if(external_clients_testing)
            list(APPEND external_clients_configure_command
                "-DPython3_EXECUTABLE:FILEPATH=${HOLOSCAN_PYTHON_EXECUTABLE}")
        endif()
        if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND
           NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
            list(APPEND external_clients_configure_command
                -A "${HOLOSCAN_GENERATOR_PLATFORM}")
        endif()
        if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND
           NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
            list(APPEND external_clients_configure_command
                -T "${HOLOSCAN_GENERATOR_TOOLSET}")
        endif()
        if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
            list(APPEND external_clients_configure_command
                "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
        endif()
        execute_process(
            COMMAND ${external_clients_configure_command}
            RESULT_VARIABLE external_clients_configure_result
            OUTPUT_VARIABLE external_clients_configure_output
            ERROR_VARIABLE external_clients_configure_error_output
            TIMEOUT 60
        )
        if(NOT external_clients_configure_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to configure installed external-client sources with "
                "BUILD_TESTING=${external_clients_testing}:\n"
                "${external_clients_configure_output}"
                "${external_clients_configure_error_output}")
        endif()
    endforeach()
endif()

# The upstream iceoryx2 install exports both the static and the shared C++ targets and
# ships both library flavors, so only the flavor HoloMQ links against is asserted here.
if(HOLOSCAN_EXPECT_SHARED)
    set(iceoryx2_cxx_selected_target "iceoryx2-cxx::shared-lib-cxx")
    set(iceoryx2_cxx_shared_library_dir "${HOLOSCAN_INSTALL_LIB_DIR}")
    if(WIN32)
        set(iceoryx2_cxx_shared_library_dir "${HOLOSCAN_INSTALL_BIN_DIR}")
    endif()
    set(iceoryx2_cxx_selected_artifact_glob
        "${install_prefix}/${iceoryx2_cxx_shared_library_dir}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}iceoryx2_cxx${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
    set(iceoryx2_ffi_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}iceoryx2_ffi_c${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*")
else()
    set(iceoryx2_cxx_selected_target "iceoryx2-cxx::static-lib-cxx")
    set(iceoryx2_cxx_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}iceoryx2_cxx${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
    set(iceoryx2_ffi_selected_artifact_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}iceoryx2_ffi_c${HOLOSCAN_STATIC_LIBRARY_SUFFIX}")
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

set(installed_artifact_root
    "${install_prefix}/${HOLOSCAN_INSTALL_DATA_DIR}/camera_imu_control/holoscan"
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
    "${install_prefix}/include/holoscan/holoscan.hpp"
    "${holoscan_package_dir}/holoscan-config.cmake"
    "${holoscan_package_dir}/holoscan-config-targets.cmake"
    "${holoscan_package_dir}/holoscan-config-io-targets.cmake"
    "${holoscan_package_dir}/HoloscanLoggingPolicy.cmake"
    "${holoscan_package_dir}/HoloscanGraphArtifacts.cmake"
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/CMakeLists.txt"
)
if(HOLOSCAN_WEB_TUNER_COMPONENT_ENABLED)
    list(APPEND required_install_paths
        "${holoscan_package_dir}/holoscan-experimental-params-web-tuner-targets.cmake")
endif()
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    file(GLOB installed_python_initializers LIST_DIRECTORIES FALSE
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/python*/site-packages/holoscan/__init__.py")
    file(GLOB installed_python_core_modules LIST_DIRECTORIES FALSE
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/python*/site-packages/holoscan/_core*.so"
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/python*/site-packages/holoscan/_core*.pyd"
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/python*/site-packages/holoscan/_core*.dylib")
    list(LENGTH installed_python_initializers installed_python_initializer_count)
    if(NOT installed_python_initializer_count EQUAL 1 OR NOT installed_python_core_modules)
        message(FATAL_ERROR
            "Installed Python example prerequisites are incomplete: initializers="
            "${installed_python_initializers}; Core extensions=${installed_python_core_modules}")
    endif()
    list(GET installed_python_initializers 0 installed_python_initializer)
    get_filename_component(installed_python_holoscan_dir
        "${installed_python_initializer}" DIRECTORY)
    get_filename_component(installed_python_package_dir
        "${installed_python_holoscan_dir}" DIRECTORY)
endif()
foreach(required_install_path IN LISTS required_install_paths)
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed examples prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

if(HOLOSCAN_EXPECT_SHARED)
    set(installed_core_library_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_SHARED_LIBRARY_PREFIX}holoscan_core${HOLOSCAN_SHARED_LIBRARY_SUFFIX}*"
    )
else()
    set(installed_core_library_glob
        "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/${HOLOSCAN_STATIC_LIBRARY_PREFIX}holoscan_core${HOLOSCAN_STATIC_LIBRARY_SUFFIX}"
    )
endif()
file(GLOB installed_core_libraries "${installed_core_library_glob}")
if(NOT installed_core_libraries)
    message(FATAL_ERROR
        "Missing Holoscan core library in the staged GNU library directory: "
        "${installed_core_library_glob}")
endif()

set(legacy_install_root "${install_staging_root}/opt/nvidia/holoscan")
if(EXISTS "${legacy_install_root}")
    message(FATAL_ERROR "Unexpected legacy install payload: ${legacy_install_root}")
endif()

file(GLOB holoscan_package_files "${holoscan_package_dir}/*.cmake")
foreach(holoscan_package_file IN LISTS holoscan_package_files)
    file(READ "${holoscan_package_file}" holoscan_package_contents)
    if(holoscan_package_contents MATCHES "/opt/nvidia/holoscan")
        message(FATAL_ERROR
            "Installed package metadata retains a legacy install prefix: ${holoscan_package_file}")
    endif()
endforeach()

# REQ: SYSTEM-014
# Execute packaged targets from both the standard and nested example layouts
# with inherited loader search paths removed. The later installed-source consumer
# build proves the examples remain buildable; these runs independently prove the
# shipped targets' INSTALL_RPATH values.
set(installed_hello_world_executable
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/hello_world/cpp/hello_world${HOLOSCAN_EXECUTABLE_SUFFIX}"
)
if(NOT EXISTS "${installed_hello_world_executable}")
    message(FATAL_ERROR
        "Installed hello world executable is missing: ${installed_hello_world_executable}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH
        --unset=DYLD_LIBRARY_PATH
        "${installed_hello_world_executable}"
    RESULT_VARIABLE installed_hello_world_result
    OUTPUT_VARIABLE installed_hello_world_output
    ERROR_VARIABLE installed_hello_world_error_output
    TIMEOUT 30
)
if(NOT installed_hello_world_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to execute the installed hello world target with sanitized loader paths: "
        "${installed_hello_world_output}${installed_hello_world_error_output}")
endif()

# REQ: COMPILE-068
set(installed_automatic_partitioning_executable
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/automatic_partitioning/cpp/automatic_partitioning${HOLOSCAN_EXECUTABLE_SUFFIX}"
)
if(NOT EXISTS "${installed_automatic_partitioning_executable}")
    message(FATAL_ERROR
        "Installed automatic partitioning executable is missing: "
        "${installed_automatic_partitioning_executable}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH
        --unset=DYLD_LIBRARY_PATH
        "${installed_automatic_partitioning_executable}"
        --mode cpu
    RESULT_VARIABLE installed_automatic_partitioning_result
    OUTPUT_VARIABLE installed_automatic_partitioning_output
    ERROR_VARIABLE installed_automatic_partitioning_error_output
    TIMEOUT 30
)
string(CONCAT installed_automatic_partitioning_log
    "${installed_automatic_partitioning_output}"
    "${installed_automatic_partitioning_error_output}"
)
if(NOT installed_automatic_partitioning_result EQUAL 0 OR
   NOT installed_automatic_partitioning_log MATCHES
       "automatic_partitioning cpu example complete")
    message(FATAL_ERROR
        "Failed to execute the installed automatic partitioning target with sanitized loader "
        "paths: ${installed_automatic_partitioning_log}")
endif()

set(installed_pod_payloads_executable
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/pod_payloads/cpp/pod_payloads${HOLOSCAN_EXECUTABLE_SUFFIX}"
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

# REQ: PY-API-019
# REQ: PY-CHANNEL-008
# The installed descriptor-Tensor Python client imports its generated schema module and native codec
# from its own installed directory. Load them, then run the client against the installed graph,
# before the installed-source consumer build below. In-tree tests put the build-tree generated
# directory on PYTHONPATH, and the consumer rebuild can regenerate these bindings and mask missing
# prebuilt artifacts, so only this pre-rebuild check proves the shipped tree is self-contained.
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    set(installed_descriptor_topic_dir
        "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_external_topic")
    set(installed_descriptor_topic_python_dir "${installed_descriptor_topic_dir}/python")
    set(installed_descriptor_topic_graph
        "${installed_descriptor_topic_dir}/cpp/descriptor_tensor_external_topic_graph${HOLOSCAN_EXECUTABLE_SUFFIX}")
    foreach(installed_descriptor_topic_file IN ITEMS
            "${installed_descriptor_topic_python_dir}/external_topic_client.py"
            "${installed_descriptor_topic_python_dir}/external_frame.py"
            "${installed_descriptor_topic_dir}/cpp/run_example.py"
            "${installed_descriptor_topic_graph}")
        if(NOT EXISTS "${installed_descriptor_topic_file}")
            message(FATAL_ERROR
                "Installed descriptor-Tensor external-topic file is missing: "
                "${installed_descriptor_topic_file}")
        endif()
    endforeach()
    file(GLOB installed_descriptor_topic_codecs LIST_DIRECTORIES FALSE
        "${installed_descriptor_topic_python_dir}/_external_frame_holoscan*")
    if(NOT installed_descriptor_topic_codecs)
        message(FATAL_ERROR
            "Installed descriptor-Tensor external-topic codec is missing from "
            "${installed_descriptor_topic_python_dir}")
    endif()

    set(installed_descriptor_topic_python_env
        "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH
        --unset=DYLD_LIBRARY_PATH
        --unset=PYTHONHOME
        "PYTHONPATH=${installed_python_package_dir}"
        "PYTHONNOUSERSITE=1"
        "PYTHONUNBUFFERED=1"
        "${HOLOSCAN_PYTHON_EXECUTABLE}"
    )
    string(CONCAT installed_descriptor_topic_import_script
        "import sys\n"
        "from pathlib import Path\n"
        "expected = Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(expected))\n"
        "import external_frame\n"
        "import _external_frame_holoscan\n"
        "for module in (external_frame, _external_frame_holoscan):\n"
        "    actual = Path(module.__file__).resolve().parent\n"
        "    if actual != expected:\n"
        "        raise SystemExit(f'{module.__name__} loaded from {actual}, expected {expected}')\n"
        "print('descriptor external-topic bindings loaded from installed tree')\n"
    )
    execute_process(
        COMMAND ${installed_descriptor_topic_python_env}
            -c "${installed_descriptor_topic_import_script}"
            "${installed_descriptor_topic_python_dir}"
        RESULT_VARIABLE installed_descriptor_topic_import_result
        OUTPUT_VARIABLE installed_descriptor_topic_import_output
        ERROR_VARIABLE installed_descriptor_topic_import_error_output
        TIMEOUT 60
    )
    string(CONCAT installed_descriptor_topic_import_log
        "${installed_descriptor_topic_import_output}"
        "${installed_descriptor_topic_import_error_output}"
    )
    if(NOT installed_descriptor_topic_import_result EQUAL 0 OR
       NOT installed_descriptor_topic_import_log MATCHES
           "descriptor external-topic bindings loaded from installed tree")
        message(FATAL_ERROR
            "Failed to import the installed descriptor-Tensor external-topic Python bindings "
            "with sanitized loader paths: ${installed_descriptor_topic_import_log}")
    endif()

    execute_process(
        COMMAND ${installed_descriptor_topic_python_env}
            "${installed_descriptor_topic_dir}/cpp/run_example.py"
            --graph "${installed_descriptor_topic_graph}"
            --client "${installed_descriptor_topic_python_dir}/external_topic_client.py"
        RESULT_VARIABLE installed_descriptor_topic_result
        OUTPUT_VARIABLE installed_descriptor_topic_output
        ERROR_VARIABLE installed_descriptor_topic_error_output
        TIMEOUT 90
    )
    string(CONCAT installed_descriptor_topic_log
        "${installed_descriptor_topic_output}"
        "${installed_descriptor_topic_error_output}"
    )
    if(NOT installed_descriptor_topic_result EQUAL 0 OR
       NOT installed_descriptor_topic_log MATCHES
           "descriptor Tensor external-topic round trip complete")
        message(FATAL_ERROR
            "Failed to execute the installed descriptor-Tensor Python client against the installed "
            "graph with sanitized loader paths: ${installed_descriptor_topic_log}")
    endif()
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}"
    -B "${examples_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-Dholoscan_DIR:PATH=${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
    "-DCMAKE_INSTALL_PREFIX:PATH=${install_prefix}"
    "-DCMAKE_INSTALL_DATADIR:STRING=${custom_examples_data_dir}"
    "-DBUILD_TESTING:BOOL=ON"
    "-DHOLOSCAN_INSTALL_EXAMPLE_SOURCE:BOOL=OFF"
    "-DHOLOSCAN_BUILD_PYTHON_EXAMPLES:BOOL=${HOLOSCAN_PYTHON_COMPONENT_ENABLED}"
)
if(DEFINED HOLOSCAN_CONSUMER_CACHE AND NOT HOLOSCAN_CONSUMER_CACHE STREQUAL "")
    list(APPEND configure_command -C "${HOLOSCAN_CONSUMER_CACHE}")
endif()
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    list(APPEND configure_command
        "-DPython3_EXECUTABLE:FILEPATH=${HOLOSCAN_PYTHON_EXECUTABLE}"
    )
endif()
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
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    TIMEOUT 300
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the complete isolated installed examples tree")
endif()

# An SDK that advertises Python support must reject an explicitly requested Python example when
# its native extension is absent. Temporarily hide only the staged copy, restore it regardless of
# the configure result, and require the helper's bounded diagnostic.
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    set(hidden_python_core_modules)
    foreach(installed_python_core_module IN LISTS installed_python_core_modules)
        set(hidden_python_core_module "${installed_python_core_module}.incomplete")
        file(RENAME "${installed_python_core_module}" "${hidden_python_core_module}")
        list(APPEND hidden_python_core_modules "${hidden_python_core_module}")
    endforeach()
    set(incomplete_python_build_dir "${HOLOSCAN_EXAMPLES_WORK_DIR}/incomplete-python")
    file(REMOVE_RECURSE "${incomplete_python_build_dir}")
    set(incomplete_python_configure_command
        "${CMAKE_COMMAND}"
        -S "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/first_pipeline/python"
        -B "${incomplete_python_build_dir}"
        -G "${HOLOSCAN_GENERATOR}"
        "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
        "-DBUILD_TESTING:BOOL=OFF"
        "-DHOLOSCAN_EXAMPLES_IN_SDK_BUILD:BOOL=OFF"
        "-DHOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR:PATH=${installed_python_package_dir}"
        "-DPython3_EXECUTABLE:FILEPATH=${HOLOSCAN_PYTHON_EXECUTABLE}"
    )
    if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND incomplete_python_configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND incomplete_python_configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
    endif()
    if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
        list(APPEND incomplete_python_configure_command
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
    endif()
    execute_process(
        COMMAND ${incomplete_python_configure_command}
        RESULT_VARIABLE incomplete_python_result
        OUTPUT_VARIABLE incomplete_python_output
        ERROR_VARIABLE incomplete_python_error_output
        TIMEOUT 60
    )
    foreach(hidden_python_core_module IN LISTS hidden_python_core_modules)
        string(REGEX REPLACE "\\.incomplete$" "" installed_python_core_module
            "${hidden_python_core_module}")
        file(RENAME "${hidden_python_core_module}" "${installed_python_core_module}")
    endforeach()
    string(CONCAT incomplete_python_log
        "${incomplete_python_output}" "${incomplete_python_error_output}")
    if(incomplete_python_result EQUAL 0 OR
       NOT incomplete_python_log MATCHES
           "installed Holoscan Python package does not include its Core extension")
        message(FATAL_ERROR
            "An explicitly requested incomplete Python package was not rejected as expected: "
            "${incomplete_python_log}")
    endif()

    # Configure this example as an independent installed consumer without injecting the
    # interpreter or Python package directory. Its two CTests must use the values discovered in
    # the same directory scope that registers the additional rejection test.
    set(standalone_derived_tensor_build_dir
        "${HOLOSCAN_EXAMPLES_WORK_DIR}/standalone-derived-tensor-python")
    file(REMOVE_RECURSE "${standalone_derived_tensor_build_dir}")
    set(standalone_derived_tensor_configure_command
        "${CMAKE_COMMAND}"
        -S
        "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/derived_tensor_pool_capacity/python"
        -B "${standalone_derived_tensor_build_dir}"
        -G "${HOLOSCAN_GENERATOR}"
        "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
        "-DBUILD_TESTING:BOOL=ON"
    )
    if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND standalone_derived_tensor_configure_command
            -A "${HOLOSCAN_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND standalone_derived_tensor_configure_command
            -T "${HOLOSCAN_GENERATOR_TOOLSET}")
    endif()
    if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
        list(APPEND standalone_derived_tensor_configure_command
            "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
    endif()
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND standalone_derived_tensor_configure_command
            "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${standalone_derived_tensor_configure_command}
        RESULT_VARIABLE standalone_derived_tensor_configure_result
        OUTPUT_VARIABLE standalone_derived_tensor_configure_output
        ERROR_VARIABLE standalone_derived_tensor_configure_error_output
        TIMEOUT 60
    )
    if(NOT standalone_derived_tensor_configure_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to configure the standalone installed Python Tensor example: "
            "${standalone_derived_tensor_configure_output}"
            "${standalone_derived_tensor_configure_error_output}")
    endif()

    set(standalone_derived_tensor_test_command
        "${HOLOSCAN_CTEST_COMMAND}"
        --test-dir "${standalone_derived_tensor_build_dir}"
        --output-on-failure
        --no-tests=error
        -R "^EXAMPLE_PYTHON_DERIVED_TENSOR_POOL_CAPACITY(_REJECTS_NON_RGBA)?_TEST$"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND standalone_derived_tensor_test_command -C "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${standalone_derived_tensor_test_command}
        RESULT_VARIABLE standalone_derived_tensor_test_result
        OUTPUT_VARIABLE standalone_derived_tensor_test_output
        ERROR_VARIABLE standalone_derived_tensor_test_error_output
        TIMEOUT 90
    )
    if(NOT standalone_derived_tensor_test_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to run the standalone installed Python Tensor example tests: "
            "${standalone_derived_tensor_test_output}"
            "${standalone_derived_tensor_test_error_output}")
    endif()
endif()

set(examples_build_command "${CMAKE_COMMAND}" --build "${examples_build_dir}" --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND examples_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${examples_build_command}
    RESULT_VARIABLE examples_build_result
    TIMEOUT 600
)
if(NOT examples_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the complete isolated installed examples tree")
endif()

set(examples_install_command
    "${CMAKE_COMMAND}" --install "${examples_build_dir}"
    --prefix "${install_prefix}"
    --component holoscan-core-examples
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND examples_install_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${examples_install_command}
    RESULT_VARIABLE examples_install_result
    TIMEOUT 300
)
if(NOT examples_install_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to install the isolated examples with a custom data directory")
endif()

set(custom_installed_automatic_partitioning_executable
    "${install_prefix}/${HOLOSCAN_EXAMPLES_INSTALL_DIR}/automatic_partitioning/cpp/automatic_partitioning${HOLOSCAN_EXECUTABLE_SUFFIX}"
)
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        --unset=LD_LIBRARY_PATH
        --unset=DYLD_LIBRARY_PATH
        "${custom_installed_automatic_partitioning_executable}"
        --mode cpu
    RESULT_VARIABLE custom_installed_automatic_partitioning_result
    OUTPUT_VARIABLE custom_installed_automatic_partitioning_output
    ERROR_VARIABLE custom_installed_automatic_partitioning_error_output
    TIMEOUT 30
)
string(CONCAT custom_installed_automatic_partitioning_log
    "${custom_installed_automatic_partitioning_output}"
    "${custom_installed_automatic_partitioning_error_output}"
)
if(NOT custom_installed_automatic_partitioning_result EQUAL 0 OR
   NOT custom_installed_automatic_partitioning_log MATCHES
       "automatic_partitioning cpu example complete")
    message(FATAL_ERROR
        "Failed to execute the custom-data-directory reinstalled automatic partitioning target with "
        "sanitized loader paths: ${custom_installed_automatic_partitioning_log}")
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
    EXAMPLE_CPP_AUTOMATIC_PARTITIONING_CPU_TEST
    EXAMPLE_CPP_AUTOMATIC_PARTITIONING_HYBRID_TEST
    EXAMPLE_SLOW_FAST_FUSION_TEST
    EXAMPLE_CAMERA_IMU_RUNTIME_TEST
    EXAMPLE_CAMERA_IMU_ARTIFACT_TARGET_TEST
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_HOST
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_CUDA
    EXAMPLE_RETAINED_TENSOR_COMPOSITOR_INVALID_DEVICE
    EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_TEST
    EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_ARTIFACT_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_FAILURE_TEST
    EXAMPLE_CPP_NOTIFICATION_BRIDGE_UNSUPPORTED_TEST
)
if(HOLOSCAN_PYTHON_COMPONENT_ENABLED)
    list(APPEND installed_example_tests
        EXAMPLE_PYTHON_FIRST_PIPELINE_TEST
        EXAMPLE_PYTHON_COMPILE_AND_CHECK_TEST)
endif()
# These smoke tests are not among the reference applications, but execute the experimental POD
# example and any available descriptor-Tensor external-topic example from the relocated installed
# tree to verify their deeper install paths resolve the SDK libraries.
# REQ: EXPECTED-033
# REQ: EXPECTED-032
set(installed_example_smoke_tests
    EXAMPLE_CPP_COMPILE_AND_CHECK_TEST
    EXAMPLE_CPP_DIAGNOSTIC_REPORTING_TEST
    EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_TEST
    EXAMPLE_CPP_RATE_AWARE_PORTS_TEST
    EXAMPLE_CPP_POD_PAYLOADS_TEST
)
if(HOLOSCAN_TESTING_COMPONENT_ENABLED)
    list(APPEND installed_example_smoke_tests
        EXAMPLE_CPP_OPERATOR_TESTING_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_TENSOR_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_IMAGE_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_DEVICE_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_GREEN_CONTEXT_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_GREEN_CONTEXT_SHARED_PARTITION_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_SCHEMA_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_LIFECYCLE_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_CLOCK_TEST)
endif()
set(optional_installed_example_smoke_test EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_TEST)

# The descriptor-Tensor driver is registered only when the isolated consumer finds Python 3.
# Inspect that consumer's CTest inventory so a Python-less package image skips only this optional
# smoke row instead of failing every installed-example test.
set(optional_example_test_list_command
    "${HOLOSCAN_CTEST_COMMAND}"
    --test-dir "${examples_build_dir}"
    --show-only=json-v1
    -R "^${optional_installed_example_smoke_test}$"
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND optional_example_test_list_command -C "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${optional_example_test_list_command}
    RESULT_VARIABLE optional_example_test_list_result
    OUTPUT_VARIABLE optional_example_test_list_json
    ERROR_VARIABLE optional_example_test_list_error_output
    TIMEOUT 30
)
string(JSON optional_example_test_list_count
    ERROR_VARIABLE optional_example_test_list_json_error
    LENGTH "${optional_example_test_list_json}" tests)
if(NOT optional_example_test_list_result EQUAL 0 OR
   NOT optional_example_test_list_json_error STREQUAL "NOTFOUND" OR
   optional_example_test_list_count GREATER 1)
    message(FATAL_ERROR
        "Optional installed example test ${optional_installed_example_smoke_test} has an invalid "
        "inventory: ${optional_example_test_list_error_output}"
        "${optional_example_test_list_json_error}")
endif()
if(optional_example_test_list_count EQUAL 1)
    list(APPEND installed_example_smoke_tests ${optional_installed_example_smoke_test})
endif()
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
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND examples_test_list_command -C "${CMAKE_BUILD_TYPE}")
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
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND examples_test_command -C "${CMAKE_BUILD_TYPE}")
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
        "Failed to execute baseline reference applications, installed smoke tests, notification "
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

# On Unix, examples are private SDK programs rather than data files. Reconfigure
# the installed source tree with a nondefault libdir and inspect its generated
# install rule to ensure the source-and-binary bundle follows that libdir.
if(NOT WIN32)
    set(custom_examples_lib_dir "custom-example-lib")
    set(custom_libdir_configure_command ${configure_command})
    list(APPEND custom_libdir_configure_command
        "-DCMAKE_INSTALL_LIBDIR:STRING=${custom_examples_lib_dir}")
    execute_process(
        COMMAND ${custom_libdir_configure_command}
        RESULT_VARIABLE custom_libdir_configure_result
        TIMEOUT 300
    )
    if(NOT custom_libdir_configure_result EQUAL 0)
        message(FATAL_ERROR "Failed to configure installed examples with a custom libdir")
    endif()

    set(custom_libdir_install_script
        "${examples_build_dir}/automatic_partitioning/cpp/cmake_install.cmake")
    if(NOT EXISTS "${custom_libdir_install_script}")
        message(FATAL_ERROR
            "Missing generated automatic partitioning install script: ${custom_libdir_install_script}")
    endif()
    file(READ "${custom_libdir_install_script}" custom_libdir_install_contents)
    string(FIND "${custom_libdir_install_contents}"
        "${custom_examples_lib_dir}/holoscan/examples/automatic_partitioning/cpp"
        custom_libdir_install_offset)
    if(custom_libdir_install_offset EQUAL -1)
        message(FATAL_ERROR
            "Installed examples do not follow CMAKE_INSTALL_LIBDIR=${custom_examples_lib_dir}")
    endif()
endif()
