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
    test_builtin_schema.py
    test_deployment.py
    test_experience_gallery.py
    test_first_pipeline.py
    test_native_lifecycle.py
    test_native_operator.py
    test_native_selection_publication.py
    test_native_tensor_authoring.py
    test_py_object_payload.py
    test_native_tensor_io.py
    test_native_temporal_contract.py
    test_plan.py
    test_policy_authoring.py
    test_reusable_graph.py
    test_registry.py
    test_runtime_control.py
    test_schema_payload.py
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
    )
else()
    nanobind_add_module(holoscan_python_test_support STABLE_ABI operator_test_support.cpp)
endif()
target_link_libraries(holoscan_python_test_support PRIVATE holoscan::core holoscan::params)
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
    TARGET python_core_control_payload_generated
    SCHEMA schema_payload/control_payload.fbs
    MAX_SERIALIZED_SIZE 512
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
    python_core_stereo_payload_generated
)
add_dependencies(
    holoscan-python-tests
    python_core_control_payload_generated
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
            "${holoscan_SOURCE_DIR}/python/check_type_completeness.py"
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
            --operator-generator "${holoscan_SOURCE_DIR}/python/generate_operator_stubs.py"
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
