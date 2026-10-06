# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS
        HOLOSCAN_SDK_SOURCE_DIR
        HOLOSCAN_FLATC_EXECUTABLE
        HOLOSCAN_FLATBUFFERS_INCLUDE_DIR
        HOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE
        HOLOSCAN_PROVIDER_SOURCE_DIR
        HOLOSCAN_PROVIDER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_C_COMPILER
        HOLOSCAN_CXX_COMPILER)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

set(build_dir "${HOLOSCAN_PROVIDER_WORK_DIR}/build")
set(install_dir "${HOLOSCAN_PROVIDER_WORK_DIR}/install")
file(REMOVE_RECURSE "${HOLOSCAN_PROVIDER_WORK_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_PROVIDER_SOURCE_DIR}"
        -B "${build_dir}"
        "-DCMAKE_INSTALL_PREFIX=${install_dir}"
        "-DHOLOSCAN_FLATC_EXECUTABLE=${HOLOSCAN_FLATC_EXECUTABLE}"
        "-DHOLOSCAN_FLATBUFFERS_INCLUDE_DIR=${HOLOSCAN_FLATBUFFERS_INCLUDE_DIR}"
        "-DHOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE=${HOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE}"
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "Predeclared FlatBuffers provider did not configure:\n"
        "${configure_stdout}\n${configure_stderr}"
    )
endif()

set(sdk_build_dir "${HOLOSCAN_PROVIDER_WORK_DIR}/sdk-build")
set(provider_init
    "${HOLOSCAN_PROVIDER_SOURCE_DIR}/predeclare_flatbuffers_provider.cmake")
set(sdk_configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_SDK_SOURCE_DIR}"
    -B "${sdk_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES:FILEPATH=${provider_init}"
    "-DHOLOSCAN_PROVIDER_FLATC_EXECUTABLE:FILEPATH=${HOLOSCAN_FLATC_EXECUTABLE}"
    "-DHOLOSCAN_PROVIDER_FLATBUFFERS_INCLUDE_DIR:PATH=${HOLOSCAN_FLATBUFFERS_INCLUDE_DIR}"
    "-DCMAKE_C_COMPILER:FILEPATH=${HOLOSCAN_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}"
    -DHOLOSCAN_BUILD_BENCHMARKS:BOOL=OFF
    -DHOLOSCAN_BUILD_EXAMPLES:BOOL=OFF
    -DHOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER:BOOL=OFF
    -DHOLOSCAN_BUILD_PYTHON:BOOL=OFF
    -DHOLOSCAN_BUILD_TESTS:BOOL=OFF
    -DHOLOSCAN_DOWNLOAD_DATASETS:BOOL=OFF
    -DHOLOSCAN_USE_SCCACHE:BOOL=OFF
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND sdk_configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND sdk_configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND sdk_configure_command
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_CUDA_COMPILER AND NOT HOLOSCAN_CUDA_COMPILER STREQUAL "")
    list(APPEND sdk_configure_command
        "-DCMAKE_CUDA_COMPILER:FILEPATH=${HOLOSCAN_CUDA_COMPILER}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND sdk_configure_command
        "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
execute_process(
    COMMAND ${sdk_configure_command}
    RESULT_VARIABLE sdk_configure_result
    OUTPUT_VARIABLE sdk_configure_stdout
    ERROR_VARIABLE sdk_configure_stderr
)
if(NOT sdk_configure_result EQUAL 0)
    message(FATAL_ERROR
        "SDK did not configure with the predeclared FlatBuffers provider:\n"
        "${sdk_configure_stdout}\n${sdk_configure_stderr}"
    )
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${sdk_build_dir}"
        --target holoscan_flatbuffer_schemas_generated --parallel 8
    RESULT_VARIABLE sdk_build_result
    OUTPUT_VARIABLE sdk_build_stdout
    ERROR_VARIABLE sdk_build_stderr
)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR
        "SDK schema generation failed with the predeclared FlatBuffers provider:\n"
        "${sdk_build_stdout}\n${sdk_build_stderr}"
    )
endif()
if(NOT EXISTS "${sdk_build_dir}/generated/holoscan/schemas/tensor_generated.h")
    message(FATAL_ERROR "Predeclared-provider SDK build did not generate Tensor schema headers")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${build_dir}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_stdout
    ERROR_VARIABLE install_stderr
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "Predeclared FlatBuffers provider did not install:\n"
        "${install_stdout}\n${install_stderr}"
    )
endif()
if(NOT EXISTS "${install_dir}/bin/flatc")
    message(FATAL_ERROR "Predeclared provider did not install bin/flatc")
endif()
if(NOT EXISTS "${install_dir}/include/flatbuffers/flatbuffers.h")
    message(FATAL_ERROR "Predeclared provider did not install FlatBuffers public headers")
endif()
execute_process(
    COMMAND "${install_dir}/bin/flatc" --version
    RESULT_VARIABLE version_result
    OUTPUT_VARIABLE installed_flatc_version
    ERROR_VARIABLE version_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT version_result EQUAL 0 OR
   NOT installed_flatc_version MATCHES "25\\.12\\.19")
    message(FATAL_ERROR
        "Installed flatc has the wrong version: ${installed_flatc_version} ${version_stderr}"
    )
endif()
