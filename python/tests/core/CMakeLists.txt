# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

set(HOLOSCAN_PYTHON_CORE_TEST_FILES
    test_channel.py
    test_channel_asyncio.py
    test_channel_catalog.py
    test_channel_tensor.py
    test_channel_torch.py
    test_public_namespace.py
    test_tensor.py
    test_params.py
    test_graph_authoring.py
    test_module_discovery.py
    test_config.py
    test_constructor_handles.py
    test_constructor_callbacks.py
    test_builtin_schema.py
    test_deployment.py
    test_experience_gallery.py
    test_first_pipeline.py
    test_native_lifecycle.py
    test_native_operator.py
    test_native_selection_publication.py
    test_native_tensor_authoring.py
    test_pod_payload.py
    test_py_object_payload.py
    test_native_tensor_io.py
    test_native_cupy.py
    test_native_temporal_contract.py
    test_plan.py
    test_policy_authoring.py
    test_reusable_graph.py
    test_registry.py
    test_runtime_control.py
    test_schema_payload.py
    test_schema_bytes.py
    test_sensor_io.py
    test_session_misuse.py
    test_shipped_examples.py
    test_topics.py
)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND HOLOSCAN_PYTHON_CORE_TEST_FILES test_stubs.py)
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.SABIModule)

if(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE)
    nanobind_add_module(holoscan_python_test_support
        BACKEND_MODULE nanobind_backend
        STABLE_ABI_VERSION 3.12
        operator_test_support.cpp
        phase_counter_support.cpp
    )
else()
    nanobind_add_module(holoscan_python_test_support STABLE_ABI
        operator_test_support.cpp
        phase_counter_support.cpp)
endif()
target_link_libraries(holoscan_python_test_support PRIVATE
    holoscan::core holoscan::params holomq::holomq CUDA::cudart CUDA::cuda_driver)
# The phase-counter seam reads private Runtime headers, as the C++ benchmarks do.
target_include_directories(holoscan_python_test_support PRIVATE "${holoscan_SOURCE_DIR}/src")
set_target_properties(holoscan_python_test_support PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_ROOT}"
    OUTPUT_NAME "holoscan_test_support"
    BUILD_RPATH "\$ORIGIN/../../lib"
)
add_dependencies(
    holoscan-python-tests
    holoscan-python-init
    holoscan_core_python
    holoscan_python_test_support
)

# A synthetic constructor-handle provider built like an independent package: its own nanobind
# domain, so it shares no binding-library type registry with Holoscan's extension.
nanobind_add_module(holoscan_python_constructor_handle_provider
    STABLE_ABI
    NB_DOMAIN holoscan_test_constructor_handle_provider
    constructor_handle_provider.cpp
)
target_link_libraries(holoscan_python_constructor_handle_provider PRIVATE holoscan::core)
set_target_properties(holoscan_python_constructor_handle_provider PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_ROOT}"
    OUTPUT_NAME "holoscan_constructor_handle_provider"
    BUILD_RPATH "\$ORIGIN/../../lib"
)
add_dependencies(holoscan-python-tests holoscan_python_constructor_handle_provider)

include("${holoscan_SOURCE_DIR}/tests/cmake/consumer_test.cmake")
# REQ: PY-API-024
holoscan_add_installed_consumer_test(
    python-core.installed_constructor_handle_provider
    SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/installed_handle_provider"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installed_handle_provider"
    CMAKE_ARGS
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=holoscan-python"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-python_libs"
    LABELS python
)
# REQ: SCHEMA-PY-007
holoscan_add_installed_consumer_test(
    python-core.installed_byte_provider
    SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/installed_byte_provider"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installed_byte_provider"
    CMAKE_ARGS
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=holoscan-python"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-python_libs"
    LABELS python
)

get_property(_holoscan_builtin_schema_targets GLOBAL
    PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
foreach(_holoscan_builtin_schema_target IN LISTS _holoscan_builtin_schema_targets)
    add_dependencies(holoscan-python-tests "${_holoscan_builtin_schema_target}")
    get_target_property(_holoscan_builtin_schema_codec_target
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
    if(_holoscan_builtin_schema_codec_target AND
       NOT _holoscan_builtin_schema_codec_target MATCHES "-NOTFOUND$")
        add_dependencies(holoscan-python-tests "${_holoscan_builtin_schema_codec_target}")
    endif()
endforeach()
unset(_holoscan_builtin_schema_codec_target)
unset(_holoscan_builtin_schema_target)
unset(_holoscan_builtin_schema_targets)

holoscan_add_flatbuffer_schema(
    TARGET python_core_byte_channel_payload_generated
    SCHEMA installed_byte_provider/byte_channel_payload.fbs
    MAX_SERIALIZED_SIZE 1024
    PYTHON
)
add_dependencies(holoscan-python python_core_byte_channel_payload_generated)
add_dependencies(holoscan-python-tests python_core_byte_channel_payload_generated)

holoscan_add_flatbuffer_schema(
    TARGET python_core_byte_tensor_payload_generated
    SCHEMA schema_payload/byte_tensor_payload.fbs
    MAX_SERIALIZED_SIZE 512
    PYTHON
)
add_dependencies(holoscan-python python_core_byte_tensor_payload_generated)
add_dependencies(holoscan-python-tests python_core_byte_tensor_payload_generated)

holoscan_add_flatbuffer_schema(
    TARGET python_core_byte_payload_generated
    SCHEMA schema_payload/byte_payload.fbs
    MAX_SERIALIZED_SIZE 2097152
    PYTHON
)
nanobind_add_module(python_core_byte_schema_support STABLE_ABI byte_schema_support.cpp)
target_link_libraries(python_core_byte_schema_support PRIVATE
    holoscan::core python_core_byte_payload_generated)
set_target_properties(python_core_byte_schema_support PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_ROOT}"
    OUTPUT_NAME "byte_schema_support"
    BUILD_RPATH "\$ORIGIN/../../lib"
)
add_dependencies(holoscan-python python_core_byte_payload_generated python_core_byte_schema_support)
add_dependencies(holoscan-python-tests python_core_byte_payload_generated python_core_byte_schema_support)

holoscan_add_flatbuffer_schema(
    TARGET python_core_control_payload_generated
    SCHEMA schema_payload/control_payload.fbs
    MAX_SERIALIZED_SIZE 512
    PYTHON
)
holoscan_add_flatbuffer_schema(
    TARGET python_core_vector_payload_generated
    SCHEMA schema_payload/vector_payload.fbs
    MAX_SERIALIZED_SIZE 1048576
    PYTHON
)
holoscan_add_flatbuffer_schema(
    TARGET python_core_stereo_payload_generated
    SCHEMA schema_payload/stereo_payload.fbs
    MAX_SERIALIZED_SIZE 512
    PYTHON
)
add_dependencies(
    holoscan-python
    python_core_control_payload_generated
    python_core_vector_payload_generated
    python_core_stereo_payload_generated
)
add_dependencies(
    holoscan-python-tests
    python_core_control_payload_generated
    python_core_vector_payload_generated
    python_core_stereo_payload_generated
)
get_target_property(
    _HOLOSCAN_CONTROL_SCHEMA_PYTHON_DIR
    python_core_control_payload_generated
    HOLOSCAN_FLATBUFFER_OUTPUT_DIR
)
get_target_property(
    _HOLOSCAN_STEREO_SCHEMA_PYTHON_DIR
    python_core_stereo_payload_generated
    HOLOSCAN_FLATBUFFER_OUTPUT_DIR
)

add_executable(python_core_schema_payload_peer schema_payload_peer.cpp)
target_compile_features(python_core_schema_payload_peer PRIVATE cxx_std_20)
target_link_libraries(
    python_core_schema_payload_peer
    PRIVATE
        holoscan::core
        python_core_control_payload_generated
        python_core_stereo_payload_generated
)
set_target_properties(python_core_schema_payload_peer PROPERTIES CXX_EXTENSIONS OFF)
add_dependencies(holoscan-python python_core_schema_payload_peer)
add_dependencies(holoscan-python-tests python_core_schema_payload_peer)

set(_HOLOSCAN_PYTHON_TEST_PYTHONPATH "${HOLOSCAN_PYTHON_MODULE_ROOT}")

add_executable(python_core_config_plan config_plan.cpp)
target_link_libraries(
    python_core_config_plan
    PRIVATE
        holoscan::config_io
        holoscan::core
        holoscan::params
)
add_dependencies(holoscan-python python_core_config_plan)

add_executable(python_core_first_pipeline_plan first_pipeline_plan.cpp)
target_link_libraries(
    python_core_first_pipeline_plan
    PRIVATE
        holoscan::core
)
add_dependencies(holoscan-python python_core_first_pipeline_plan)

add_executable(python_core_policy_plan policy_plan.cpp)
target_link_libraries(
    python_core_policy_plan
    PRIVATE
        holoscan::core
)
add_dependencies(holoscan-python python_core_policy_plan)

add_executable(python_core_temporal_plan temporal_plan.cpp)
target_link_libraries(
    python_core_temporal_plan
    PRIVATE
        holoscan::core
)
add_dependencies(holoscan-python python_core_temporal_plan)

add_executable(python_core_reusable_graph_plan reusable_graph_plan.cpp)
target_link_libraries(
    python_core_reusable_graph_plan
    PRIVATE
        holoscan::core
)
add_dependencies(holoscan-python python_core_reusable_graph_plan)

add_executable(python_core_topic_plan topic_plan.cpp)
target_link_libraries(
    python_core_topic_plan
    PRIVATE
        holoscan::core
)
add_dependencies(holoscan-python python_core_topic_plan)

add_executable(python_core_topic_peer topic_peer.cpp)
target_link_libraries(python_core_topic_peer PRIVATE holoscan::core)
add_dependencies(holoscan-python python_core_topic_peer)

add_library(python_core_sdk_descriptor_topic_fixture SHARED
    sdk_descriptor_topic_fixture.cpp)
target_compile_features(python_core_sdk_descriptor_topic_fixture PRIVATE cxx_std_20)
target_link_libraries(python_core_sdk_descriptor_topic_fixture
    PRIVATE holoscan::core holoscan::schema)
set_target_properties(python_core_sdk_descriptor_topic_fixture PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden)
add_dependencies(holoscan-python python_core_sdk_descriptor_topic_fixture)

if(TARGET descriptor_tensor_external_topic_generated
   AND TARGET descriptor_tensor_external_topic_client)
    holoscan_add_flatbuffer_schema(
        TARGET python_core_descriptor_topic_mismatch_generated
        SCHEMA descriptor_topic_mismatch.fbs
        MAX_SERIALIZED_SIZE 512
    )
    add_library(python_core_descriptor_topic_fixture SHARED descriptor_topic_fixture.cpp)
    target_compile_features(python_core_descriptor_topic_fixture PRIVATE cxx_std_20)
    target_include_directories(python_core_descriptor_topic_fixture PRIVATE
        "${holoscan_SOURCE_DIR}/examples/flatbuffer_schemas/descriptor_tensor_external_topic/cpp")
    target_link_libraries(python_core_descriptor_topic_fixture
        PRIVATE holoscan::core descriptor_tensor_external_topic_generated)
    set_target_properties(python_core_descriptor_topic_fixture PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden)

    add_executable(python_core_descriptor_topic_mismatch_peer
        descriptor_topic_mismatch_peer.cpp)
    target_compile_features(python_core_descriptor_topic_mismatch_peer PRIVATE cxx_std_20)
    target_link_libraries(python_core_descriptor_topic_mismatch_peer
        PRIVATE holoscan::core python_core_descriptor_topic_mismatch_generated)
    set_target_properties(python_core_descriptor_topic_mismatch_peer PROPERTIES
        CXX_EXTENSIONS OFF)

    add_dependencies(holoscan-python
        python_core_descriptor_topic_fixture
        python_core_descriptor_topic_mismatch_peer
        descriptor_tensor_external_topic_client)
    add_dependencies(holoscan-python-tests
        python_core_descriptor_topic_fixture
        python_core_descriptor_topic_mismatch_peer
        descriptor_tensor_external_topic_client)
    set(_HOLOSCAN_DESCRIPTOR_TOPIC_TEST_ENVIRONMENT
        "HOLOSCAN_DESCRIPTOR_TOPIC_FIXTURE=$<TARGET_FILE:python_core_descriptor_topic_fixture>"
        "HOLOSCAN_DESCRIPTOR_TOPIC_CLIENT=$<TARGET_FILE:descriptor_tensor_external_topic_client>"
        "HOLOSCAN_DESCRIPTOR_TOPIC_MISMATCH_PEER=$<TARGET_FILE:python_core_descriptor_topic_mismatch_peer>")
endif()

add_executable(
    python_core_session_reaper_test
    test_session_reaper.cpp
    ../../holoscan/core/session_reaper.cpp
)
target_link_libraries(
    python_core_session_reaper_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)
add_dependencies(holoscan-python python_core_session_reaper_test)
add_dependencies(
    holoscan-python-tests
    python_core_config_plan
    python_core_first_pipeline_plan
    python_core_policy_plan
    python_core_reusable_graph_plan
    python_core_session_reaper_test
    python_core_sdk_descriptor_topic_fixture
    python_core_temporal_plan
    python_core_topic_peer
    python_core_topic_plan
)

if(TARGET slow_fast_fusion
   AND TARGET graph_composition
   AND TARGET camera_imu_control
   AND TARGET derived_tensor_pool_capacity)
    add_dependencies(
        holoscan-python-tests
        slow_fast_fusion
        graph_composition
        camera_imu_control
        derived_tensor_pool_capacity
    )
    list(APPEND _HOLOSCAN_PYTHON_TEST_ENVIRONMENT
        "HOLOSCAN_CPP_SLOW_FAST_FUSION=$<TARGET_FILE:slow_fast_fusion>"
        "HOLOSCAN_CPP_GRAPH_COMPOSITION=$<TARGET_FILE:graph_composition>"
        "HOLOSCAN_CPP_CAMERA_IMU_CONTROL=$<TARGET_FILE:camera_imu_control>"
        "HOLOSCAN_CPP_DERIVED_TENSOR_POOL_CAPACITY=$<TARGET_FILE:derived_tensor_pool_capacity>"
        "HOLOSCAN_SHIPPED_EXAMPLE_SOURCE_DIR=${holoscan_SOURCE_DIR}/examples"
    )
else()
    list(REMOVE_ITEM HOLOSCAN_PYTHON_CORE_TEST_FILES test_shipped_examples.py)
endif()
gtest_discover_tests(python_core_session_reaper_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX python-core.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
    PROPERTIES TIMEOUT 30
)

list(APPEND _HOLOSCAN_PYTHON_TEST_ENVIRONMENT
    "HOLOSCAN_CONFIG_PLAN_HELPER=$<TARGET_FILE:python_core_config_plan>"
    "HOLOSCAN_FIRST_PIPELINE_PLAN_HELPER=$<TARGET_FILE:python_core_first_pipeline_plan>"
    "HOLOSCAN_OPERATOR_STUB_GENERATOR=${holoscan_SOURCE_DIR}/python/holoscan/_tools/generate_operator_stubs.py"
    "HOLOSCAN_POLICY_PLAN_HELPER=$<TARGET_FILE:python_core_policy_plan>"
    "HOLOSCAN_REUSABLE_GRAPH_PLAN_HELPER=$<TARGET_FILE:python_core_reusable_graph_plan>"
    "HOLOSCAN_SCHEMA_PAYLOAD_PEER=$<TARGET_FILE:python_core_schema_payload_peer>"
    "HOLOSCAN_SDK_DESCRIPTOR_TOPIC_FIXTURE=$<TARGET_FILE:python_core_sdk_descriptor_topic_fixture>"
    "HOLOSCAN_TEMPORAL_PLAN_HELPER=$<TARGET_FILE:python_core_temporal_plan>"
    "HOLOSCAN_TOPIC_PEER_HELPER=$<TARGET_FILE:python_core_topic_peer>"
    "HOLOSCAN_TOPIC_PLAN_HELPER=$<TARGET_FILE:python_core_topic_plan>"
    ${_HOLOSCAN_DESCRIPTOR_TOPIC_TEST_ENVIRONMENT}
)

if(_HOLOSCAN_CONTROL_SCHEMA_PYTHON_DIR AND _HOLOSCAN_STEREO_SCHEMA_PYTHON_DIR)
    list(APPEND _HOLOSCAN_PYTHON_TEST_PYTHONPATH
        "${_HOLOSCAN_CONTROL_SCHEMA_PYTHON_DIR}"
        "${_HOLOSCAN_STEREO_SCHEMA_PYTHON_DIR}"
    )
endif()
list(REMOVE_DUPLICATES _HOLOSCAN_PYTHON_TEST_PYTHONPATH)
list(JOIN _HOLOSCAN_PYTHON_TEST_PYTHONPATH ":" _HOLOSCAN_PYTHON_TEST_PYTHONPATH_VALUE)
list(PREPEND _HOLOSCAN_PYTHON_TEST_ENVIRONMENT
    "PYTHONPATH=${_HOLOSCAN_PYTHON_TEST_PYTHONPATH_VALUE}"
)

if(TARGET holoscan_testing_python)
    add_dependencies(holoscan-python-tests holoscan_testing_python)
    add_test(NAME python-testing.operators COMMAND "${Python3_EXECUTABLE}" -m pytest -q
        "${holoscan_SOURCE_DIR}/python/tests/testing")
    set_tests_properties(python-testing.operators PROPERTIES
        ENVIRONMENT "${_HOLOSCAN_PYTHON_TEST_ENVIRONMENT}"
        LABELS "python;testing;unit" TIMEOUT 90)
endif()

holoscan_add_pytest_tests(
    FILES ${HOLOSCAN_PYTHON_CORE_TEST_FILES}
    TEST_DIRECTORY "${HOLOSCAN_PYTHON_TEST_BINARY_DIR}/core"
    WORKING_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_ROOT}"
    PREFIX python-core
    PYTEST_ARGS -v --durations=0
    ENVIRONMENT ${_HOLOSCAN_PYTHON_TEST_ENVIRONMENT}
)

add_test(
    NAME python-core.prepare-installed-package
    COMMAND
        "${CMAKE_COMMAND}"
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/installed-python"
        "-DHOLOSCAN_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
        "-DHOLOSCAN_PYTHON_VERSION=${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        "-DHOLOSCAN_REQUIRE_TYPING=${HOLOSCAN_BUILD_PYTHON_STUBS}"
        -DHOLOSCAN_INSTALL_SENSOR_IO=ON
        -P "${CMAKE_CURRENT_SOURCE_DIR}/prepare_installed_python.cmake"
)
set_tests_properties(
    python-core.prepare-installed-package
    PROPERTIES FIXTURES_SETUP python_core_installed_package
)
set(_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES
    "${CMAKE_CURRENT_BINARY_DIR}/installed-python/${CMAKE_INSTALL_LIBDIR}")
string(APPEND _HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES
    "/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

add_test(
    NAME python-core.prepare-core-only-installed-package
    COMMAND
        "${CMAKE_COMMAND}"
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/installed-python-core-only"
        "-DHOLOSCAN_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
        "-DHOLOSCAN_PYTHON_VERSION=${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        "-DHOLOSCAN_REQUIRE_TYPING=${HOLOSCAN_BUILD_PYTHON_STUBS}"
        -DHOLOSCAN_INSTALL_SENSOR_IO=OFF
        -P "${CMAKE_CURRENT_SOURCE_DIR}/prepare_installed_python.cmake"
)
set_tests_properties(
    python-core.prepare-core-only-installed-package
    PROPERTIES FIXTURES_SETUP python_core_only_installed_package
)
set(_HOLOSCAN_PYTHON_CORE_ONLY_SITE_PACKAGES
    "${CMAKE_CURRENT_BINARY_DIR}/installed-python-core-only/${CMAKE_INSTALL_LIBDIR}")
string(APPEND _HOLOSCAN_PYTHON_CORE_ONLY_SITE_PACKAGES
    "/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

add_test(NAME python-core.core-only-installed-runtime
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_core_only_native_install.py"
)
set_tests_properties(
    python-core.core-only-installed-runtime
    PROPERTIES
        FIXTURES_REQUIRED python_core_only_installed_package
        ENVIRONMENT
            "PYTHONPATH=${_HOLOSCAN_PYTHON_CORE_ONLY_SITE_PACKAGES};LD_LIBRARY_PATH=${holoscan_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
)

add_test(NAME python-core.installed-runtime
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_installed_runtime.py"
        --package-root "${_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES}"
)
set_tests_properties(
    python-core.installed-runtime
    PROPERTIES
        FIXTURES_REQUIRED python_core_installed_package
        ENVIRONMENT
            "PYTHONPATH=${_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES};LD_LIBRARY_PATH=${holoscan_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    add_test(
        NAME python-core.type_completeness
        COMMAND
            "${Python3_EXECUTABLE}"
            "${holoscan_SOURCE_DIR}/python/holoscan/_tools/check_type_completeness.py"
            holoscan
            --minimum 100
    )
    add_test(
        NAME python-core.stubtest
        COMMAND
            "${Python3_EXECUTABLE}"
            -m mypy.stubtest
            holoscan
            --allowlist "${holoscan_SOURCE_DIR}/python/stubtest_allowlist.txt"
            --ignore-unused-allowlist
    )
    add_test(
        NAME python-core.ty
        COMMAND
            ty check
            --extra-search-path "${_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES}"
            "${holoscan_SOURCE_DIR}/python/tests/typing/public_surface.py"
    )
    add_test(
        NAME python-core.typing
        COMMAND
            "${Python3_EXECUTABLE}"
            "${holoscan_SOURCE_DIR}/python/tests/typing/check_registry_typing.py"
            --package-root "${_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES}"
            --provider-root "${HOLOSCAN_PYTHON_MODULE_ROOT}"
            --operator-generator "${holoscan_SOURCE_DIR}/python/holoscan/_tools/generate_operator_stubs.py"
    )
    set_tests_properties(
        python-core.type_completeness
        python-core.stubtest
        python-core.ty
        PROPERTIES
            FIXTURES_REQUIRED python_core_installed_package
            ENVIRONMENT
                "PYTHONPATH=${_HOLOSCAN_PYTHON_INSTALLED_SITE_PACKAGES};LD_LIBRARY_PATH=${holoscan_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
    )
    set_tests_properties(
        python-core.typing
        PROPERTIES
            FIXTURES_REQUIRED python_core_installed_package
            ENVIRONMENT
                "PYTHONPATH=${_HOLOSCAN_PYTHON_TEST_PYTHONPATH_VALUE};LD_LIBRARY_PATH=${holoscan_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
    )
endif()

if(TARGET holoscan_testing_python)
    set(_testing_prefix "${CMAKE_CURRENT_BINARY_DIR}/installed-python-testing")
    set(_testing_package "${_testing_prefix}/${CMAKE_INSTALL_LIBDIR}")
    string(APPEND _testing_package
        "/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
    add_test(NAME python-testing.prepare-installed-package COMMAND "${CMAKE_COMMAND}"
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_INSTALL_PREFIX=${_testing_prefix}"
        "-DHOLOSCAN_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
        "-DHOLOSCAN_PYTHON_VERSION=${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        "-DHOLOSCAN_REQUIRE_TYPING=${HOLOSCAN_BUILD_PYTHON_STUBS}"
        -DHOLOSCAN_INSTALL_TESTING=ON
        -P "${CMAKE_CURRENT_SOURCE_DIR}/prepare_installed_python.cmake")
    set_tests_properties(python-testing.prepare-installed-package PROPERTIES
        FIXTURES_SETUP python_testing_installed_package)
    add_test(NAME python-testing.installed-examples COMMAND "${Python3_EXECUTABLE}" -m pytest -q
        "${holoscan_SOURCE_DIR}/examples/operator_testing/python")
    set(_testing_installed_tests python-testing.installed-examples)
    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        add_test(NAME python-testing.stubtest COMMAND "${Python3_EXECUTABLE}" -m mypy.stubtest
            holoscan.testing --allowlist
            "${holoscan_SOURCE_DIR}/python/tests/testing/stubtest_allowlist.txt")
        add_test(NAME python-testing.ty COMMAND ty check --extra-search-path "${_testing_package}"
            --extra-search-path "${holoscan_SOURCE_DIR}/examples/operator_testing/python"
            "${holoscan_SOURCE_DIR}/python/tests/testing/typing_contract.py"
            "${holoscan_SOURCE_DIR}/examples/operator_testing/python/test_operator_testing.py"
            "${holoscan_SOURCE_DIR}/examples/operator_testing/python/test_advanced_testing.py"
            "${holoscan_SOURCE_DIR}/examples/operator_testing/python/test_tensor_and_schema.py")
        list(APPEND _testing_installed_tests python-testing.stubtest python-testing.ty)
    endif()
    set_tests_properties(${_testing_installed_tests} PROPERTIES
        FIXTURES_REQUIRED python_testing_installed_package
        ENVIRONMENT "PYTHONPATH=${_testing_package};LD_LIBRARY_PATH=${holoscan_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
        LABELS "python;testing;installed" TIMEOUT 90)
endif()

# Count calls with a preload shim, avoiding instrumentation in the SDK's submitted hot paths.
add_library(python_cuda_event_record_counter SHARED cuda_event_record_counter.cpp)
target_link_libraries(python_cuda_event_record_counter PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
add_dependencies(holoscan-python-tests python_cuda_event_record_counter)
# REQ: TENSOR-STORAGE-010
foreach(policy IN ITEMS shared per_partition)
    # Append so an existing preload, such as a sanitizer runtime, stays first.
    add_test(NAME PYTHON_CUPY_SDK_EVENT_COUNTS_${policy}
        COMMAND ${CMAKE_COMMAND} -E env
            --modify "LD_PRELOAD=path_list_append:$<TARGET_FILE:python_cuda_event_record_counter>"
            "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cuda_event_record_probe.py"
            --policy ${policy} --cudart "$<TARGET_FILE:CUDA::cudart>")
    set_tests_properties(PYTHON_CUPY_SDK_EVENT_COUNTS_${policy} PROPERTIES
        LABELS "python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 30)
endforeach()
# REQ: TENSOR-STORAGE-010
add_test(NAME PYTHON_CUPY_IMPORT_EVENT_REUSE
    COMMAND ${CMAKE_COMMAND} -E env
        --modify "LD_PRELOAD=path_list_append:$<TARGET_FILE:python_cuda_event_record_counter>"
        "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cuda_event_reuse_probe.py"
        --cudart "$<TARGET_FILE:CUDA::cudart>")
set_tests_properties(PYTHON_CUPY_IMPORT_EVENT_REUSE PROPERTIES
    LABELS "python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 30)
# REQ: TENSOR-STORAGE-010
add_test(NAME PYTHON_CUPY_IMPORT_EVENT_REUSE_LIVE_CONTEXTS
    COMMAND ${CMAKE_COMMAND} -E env
        --modify "LD_PRELOAD=path_list_append:$<TARGET_FILE:python_cuda_event_record_counter>"
        "PYTHONPATH=${holoscan_BINARY_DIR}/python/lib"
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cuda_event_reuse_probe.py"
        --cudart "$<TARGET_FILE:CUDA::cudart>" --live-contexts 10)
set_tests_properties(PYTHON_CUPY_IMPORT_EVENT_REUSE_LIVE_CONTEXTS PROPERTIES
    LABELS "python;cuda;cupy" SKIP_RETURN_CODE 77 TIMEOUT 60)

# The native producer fixture is shared with the schema interoperability tests.
if(HOLOSCAN_BUILD_BENCHMARKS)
    add_custom_target(python_schema_byte_vector_benchmark SOURCES "${holoscan_SOURCE_DIR}/benchmark/python_schema_io/byte_vector_benchmark.py")
    add_dependencies(python_schema_byte_vector_benchmark holoscan-python)
    add_dependencies(holoscan-python-benchmark-targets python_schema_byte_vector_benchmark)
    if(BUILD_TESTING)
        get_target_property(byte_schema_dir python_core_byte_payload_generated
            HOLOSCAN_FLATBUFFER_OUTPUT_DIR)
        add_test(NAME BENCHMARK_PYTHON_SCHEMA_BYTES
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}:${byte_schema_dir}"
                ${Python3_EXECUTABLE} "${holoscan_SOURCE_DIR}/benchmark/python_schema_io/byte_vector_benchmark.py"
                --sizes 0 4096 --iterations 4 --warmup 2
                --output "${CMAKE_CURRENT_BINARY_DIR}/byte-vector-smoke.json")
        set_tests_properties(BENCHMARK_PYTHON_SCHEMA_BYTES PROPERTIES LABELS benchmark TIMEOUT 120)
    endif()
endif()
