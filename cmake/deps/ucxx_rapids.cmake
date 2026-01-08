# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(ucxx 0.46.0
    GLOBAL_TARGETS ucxx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY rapidsai/ucxx
    GIT_TAG v0.46.00
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cpp
    DOWNLOAD_ONLY TRUE
)

if(ucxx_ADDED)

    set(UCXX_CXX_FLAGS -Wall -Wattributes -Werror -Wextra -Wsign-conversion
                       -Wno-missing-field-initializers -Wno-unused-parameter
    )
    set(UCXX_CXX_DEFINITIONS "")
    # Set RMM logging level
    set(RMM_LOGGING_LEVEL
        "INFO"
        CACHE STRING "Choose the logging level."
    )
    set_property(
        CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
    )
    message(VERBOSE "UCXX: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

    if(NOT TARGET holoscan::ucp)
        find_package(ucx QUIET)

        # Create an alias for ucx::ucp to support downstream holoscan::ucxx export set
        if(NOT TARGET ucx::ucp)
            message(FATAL_ERROR "Failed to import required ucx::ucp target for holoscan::ucxx!")
        endif()
        get_target_property(UCP_IMPORTED_LOCATION ucx::ucp IMPORTED_LOCATION)
        if(NOT UCP_IMPORTED_LOCATION)
            get_target_property(UCP_IMPORTED_LOCATION ucx::ucp IMPORTED_LOCATION_${CMAKE_BUILD_TYPE})
        endif()
        get_target_property(UCP_INTERFACE_INCLUDE_DIRS ucx::ucp INTERFACE_INCLUDE_DIRECTORIES)

        add_library(holoscan::ucp SHARED IMPORTED)
        set_target_properties(holoscan::ucp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${UCP_INTERFACE_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${UCP_IMPORTED_LOCATION}"
            IMPORTED_SONAME "libucp.so"
        )

    endif()

    if(NOT TARGET Threads::Threads)
        find_package(Threads REQUIRED)
    endif()

    # Build main library
    add_library(
        ucxx
        ${ucxx_SOURCE_DIR}/cpp/src/address.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/buffer.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/component.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/config.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/context.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/delayed_submission.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/endpoint.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/header.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/inflight_requests.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/internal/request_am.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/listener.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/log.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/memory_handle.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/remote_key.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_am.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_data.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_endpoint_close.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_flush.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_helper.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_mem.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_stream.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_tag.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/request_tag_multi.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/worker.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/worker_progress_thread.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/utils/callback_notifier.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/utils/file_descriptor.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/utils/python.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/utils/sockaddr.cpp
        ${ucxx_SOURCE_DIR}/cpp/src/utils/ucx.cpp
    )

    set_target_properties(
        ucxx
        PROPERTIES BUILD_RPATH "\$ORIGIN:${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/lib/gxf_extensions"
                   INSTALL_RPATH "\$ORIGIN:\$ORIGIN/gxf_extensions"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
    )

    target_compile_options(ucxx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UCXX_CXX_FLAGS}>")

    # Specify include paths for the current target and dependents
    target_include_directories(
      ucxx
      PUBLIC "$<BUILD_INTERFACE:${ucxx_SOURCE_DIR}/cpp/include>"
      PRIVATE "$<BUILD_INTERFACE:${ucxx_SOURCE_DIR}/cpp/src>"
      INTERFACE "$<INSTALL_INTERFACE:include/3rdparty/ucxx>"
    )

    target_link_libraries(ucxx PUBLIC GXF::rmm)
    # Define spdlog level
    target_compile_definitions(
        ucxx PUBLIC UCXX_ENABLE_RMM "RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LEVEL_${RMM_LOGGING_LEVEL}"
    )

    # Specify the target module library dependencies
    target_link_libraries(ucxx PUBLIC holoscan::ucp)

    add_library(ucxx::ucxx ALIAS ucxx)

    # Install the headers needed for development with the SDK
    install(DIRECTORY ${ucxx_SOURCE_DIR}/cpp/include/ucxx
        DESTINATION include/3rdparty/ucxx
        COMPONENT "holoscan-dependencies"
        )
endif()
