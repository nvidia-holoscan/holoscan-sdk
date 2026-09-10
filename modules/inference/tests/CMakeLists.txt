# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET GTest::gtest_main)
  find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
  holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)
endif()

include(GoogleTest)

function(inference_configure_test_clang_tidy target_name)
  if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
    set(_inference_test_clang_tidy_checks
        "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity")
    set_target_properties(${target_name}
      PROPERTIES
        CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_inference_test_clang_tidy_checks}")
    unset(_inference_test_clang_tidy_checks)
  endif()
endfunction()

# Copy test data files into the build tree so tests can reference them via paths
# relative to their WORKING_DIRECTORY (CMAKE_BINARY_DIR), matching upstream's convention.
file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/test_models
  DESTINATION ${CMAKE_BINARY_DIR}/modules/inference/tests
)

# ##################################################################################################
# * holoinfer tests (TRT-only, module-level; no legacy 4.x execution-graph runtime dependency)
add_executable(HOLOINFER_TEST
  inference/test_core.cpp
  inference/test_inference.cpp
  inference/test_parameters.cpp
  inference/test_green_context_build.cpp
  inference/test_fast_path_dispatch.cpp
  inference/test_multi_op_concurrency.cpp
  inference/test_fallback_chain.cpp
  inference/test_yaml_config.cpp
  inference/test_core.hpp
  inference/test_infer_settings.hpp
)
inference_configure_test_clang_tidy(HOLOINFER_TEST)

set_target_properties(HOLOINFER_TEST
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/gtests>"
)

target_include_directories(HOLOINFER_TEST
  PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/inference/src/include
    ${CMAKE_SOURCE_DIR}/modules/inference/src/infer/trt
    # fallback_runtime.hpp — internal (non-installed) runtime half of the fallback ladder
    # (see test_fallback_chain.cpp).
    ${CMAKE_SOURCE_DIR}/modules/inference/src
    # fast_path_eligibility.hpp (header-only, no link dependency) is under the top-level
    # public include tree rather than the inference module tree.
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(HOLOINFER_TEST
  PRIVATE
    GTest::gmock_main
    GTest::gtest_main
    holoinfer
    holoscan::holoinfer_utils
    holoscan::core
    holoscan::op_inference
    yaml-cpp::yaml-cpp
    CUDA::cuda_driver
)

gtest_discover_tests(HOLOINFER_TEST
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT
    "LD_LIBRARY_PATH=${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
)

add_test(NAME INFERENCE.installed_package_consumer
  COMMAND ${CMAKE_COMMAND}
    "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
    "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer"
    "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_consumer"
    "-DHOLOSCAN_BUILD_CONFIG=$<CONFIG>"
    "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
    "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
    "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
    "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/run_installed_consumer.cmake
)
set_tests_properties(INFERENCE.installed_package_consumer
  PROPERTIES
    RUN_SERIAL TRUE
    LABELS "installed-package;inference;package-boundary"
    TIMEOUT 600
)
