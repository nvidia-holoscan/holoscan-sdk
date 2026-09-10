# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_executable(config_test
    test_codec.cpp
    test_document.cpp
    test_sha256.cpp
)
holoscan_configure_test_clang_tidy(config_test)
target_include_directories(config_test PRIVATE ${holoscan_SOURCE_DIR}/src)

add_executable(config_no_exceptions_compile no_exceptions_codec.cpp)
holoscan_configure_cxx20_target(config_no_exceptions_compile PRIVATE)
target_include_directories(config_no_exceptions_compile
    PRIVATE
        ${holoscan_SOURCE_DIR}/include
        ${holoscan_BINARY_DIR}/include
)
target_link_libraries(config_no_exceptions_compile PRIVATE holoscan::config)
if(MSVC)
  target_compile_options(config_no_exceptions_compile PRIVATE /EHs-c-)
else()
  target_compile_options(config_no_exceptions_compile PRIVATE -fno-exceptions)
endif()

target_link_libraries(config_test
    PRIVATE
        holoscan::config
        GTest::gtest_main
)

if(HOLOSCAN_BUILD_CONFIG_IO)
  add_executable(config_io_test
      test_io.cpp
      ${holoscan_SOURCE_DIR}/src/config/reflect_backend.cpp
      $<TARGET_OBJECTS:holoscan_config_reflect_private_objects>
      $<TARGET_OBJECTS:holoscan_config_yyjson_private_objects>
  )
  holoscan_configure_test_clang_tidy(config_io_test)
  target_compile_definitions(config_io_test PRIVATE yyjson_api=)
  target_include_directories(config_io_test
      PRIVATE ${holoscan_SOURCE_DIR}/src
      SYSTEM PRIVATE "${reflectcpp_SOURCE_DIR}/include"
  )
  target_link_libraries(config_io_test
      PRIVATE
          holoscan::config_io
          GTest::gtest_main
  )
  gtest_discover_tests(config_io_test
      DISCOVERY_MODE PRE_TEST
      TEST_PREFIX CONFIG_IO.
      WORKING_DIRECTORY ${holoscan_BINARY_DIR}
  )
endif()

gtest_discover_tests(config_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CONFIG.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

set(_holoscan_tooling_component_modes config)
if(TARGET config_io)
  list(APPEND _holoscan_tooling_component_modes config_io)
endif()
if(TARGET params)
  list(APPEND _holoscan_tooling_component_modes params)
endif()
if(TARGET params_io)
  list(APPEND _holoscan_tooling_component_modes params_io)
endif()
if(TARGET experimental_params_web_tuner)
  list(APPEND _holoscan_tooling_component_modes tuner)
endif()
list(APPEND _holoscan_tooling_component_modes full)
list(JOIN _holoscan_tooling_component_modes "," _holoscan_tooling_component_modes_csv)
# REQ: CONFIG-015
# REQ: PARAM-035
# REQ: SYSTEM-012
# This component-consumer matrix is the reciprocal execution evidence for the optional exports
# installed by the current-tree examples package fixture. It deliberately does not depend on
# examples introduced by a later examples-only change set.
add_test(NAME CONFIG.installed_package_components
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_consumer"
        "-DHOLOSCAN_COMPONENT_MODES=${_holoscan_tooling_component_modes_csv}"
        "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_installed_consumer.cmake
)
set_tests_properties(CONFIG.installed_package_components PROPERTIES
    RUN_SERIAL TRUE
    LABELS "installed-package;config;params"
)
unset(_holoscan_tooling_component_modes_csv)
unset(_holoscan_tooling_component_modes)
