# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

function(grpc_generate_cpp SRCS HDRS)
    # Expect:
    # - PROTOC_EXECUTABLE: path to protoc
    # - GRPC_CPP_EXECUTABLE: path to grpc_cpp_plugin
    if(NOT ARGN)
        message(SEND_ERROR "Error: grpc_generate_cpp() called without any .proto files")
        return()
    endif()

    foreach(PROTO_FILE ${ARGN})
        # Get the full path to the proto file
        get_filename_component(_abs_proto_file "${CMAKE_CURRENT_LIST_DIR}/${PROTO_FILE}" ABSOLUTE)
        # Get the name of the proto file without extension
        get_filename_component(_proto_name_we ${PROTO_FILE} NAME_WE)
        # Get the parent directory of the proto file
        get_filename_component(_proto_parent_dir ${_abs_proto_file} DIRECTORY)
        # Get the parent directory of the parent directory
        get_filename_component(_parent_dir ${_proto_parent_dir} DIRECTORY)
        # Append 'generated' to the parent directory
        set(_generated_dir "${_parent_dir}/generated")

        set(_protobuf_include_path -I ${_proto_parent_dir})

        set(_proto_srcs "${_generated_dir}/${_proto_name_we}.pb.cc")
        set(_proto_hdrs "${_generated_dir}/${_proto_name_we}.pb.h")
        set(_grpc_srcs "${_generated_dir}/${_proto_name_we}.grpc.pb.cc")
        set(_grpc_hdrs "${_generated_dir}/${_proto_name_we}.grpc.pb.h")

        add_custom_command(
            OUTPUT "${_proto_srcs}" "${_proto_hdrs}" "${_grpc_srcs}" "${_grpc_hdrs}"
            COMMAND ${PROTOC_EXECUTABLE}
            ARGS --grpc_out=${_generated_dir}
            --cpp_out=${_generated_dir}
            --plugin=protoc-gen-grpc=${GRPC_CPP_EXECUTABLE}
            ${_protobuf_include_path} ${_abs_proto_file}
            DEPENDS ${_abs_proto_file}
            COMMENT "Running gRPC C++ protocol buffer compiler on ${PROTO_FILE}"
            VERBATIM
        )

        list(APPEND ${SRCS} "${_proto_srcs}")
        list(APPEND ${HDRS} "${_proto_hdrs}")
        list(APPEND ${SRCS} "${_grpc_srcs}")
        list(APPEND ${HDRS} "${_grpc_hdrs}")
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
    set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
endfunction()

# Based on https://github.com/grpc/grpc/blob/v1.54.2/examples/cpp/cmake/common.cmake
# which is under Apache 2.0 license

# We assume that the user has already installed gRPC and all its dependencies.
# The following variables are exposed to the user to help find gRPC-related binaries and libraries:
#
# PROTOC_EXECUTABLE - the path to the protoc executable
# GRPC_CPP_EXECUTABLE - the path to the grpc_cpp_plugin executable

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

if(CMAKE_CROSSCOMPILING)
    find_program(PROTOC_EXECUTABLE protoc)
else()
    get_target_property(PROTOC_EXECUTABLE protobuf::protoc LOCATION)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

if(CMAKE_CROSSCOMPILING)
    find_program(GRPC_CPP_EXECUTABLE grpc_cpp_plugin)
else()
    set(GRPC_CPP_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()


# Expose variables with PARENT_SCOPE so that
# root project can use it for including headers and using executables
set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
set(GRPC_CPP_EXECUTABLE ${GRPC_CPP_EXECUTABLE} PARENT_SCOPE)
