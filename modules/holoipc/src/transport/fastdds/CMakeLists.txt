# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fast DDS transport for holo_ipc. Included by parent when HOLOSCAN_IPC_TRANSPORT_FASTDDS is ON.
# IDL -> C++ is generated at build time into ${PROJECT_BINARY_DIR}/generated/ (see fast_dds_idl_gen.cmake).
# Uses PROJECT_SOURCE_DIR / PROJECT_BINARY_DIR from project(holoipc) in the parent CMakeLists.
# Sets IPC_FASTDDS_SOURCES, IPC_FASTDDS_OBJECT_LIB, IPC_FASTDDS_LIBS, IPC_FASTDDS_PUBLIC_GEN_BUILD_DIR.

cmake_minimum_required(VERSION 3.26.4)

include("${CMAKE_CURRENT_SOURCE_DIR}/fast_dds_idl_gen.cmake")

find_package(fastdds CONFIG REQUIRED)

# TinyXML2: Fast DDS dependency (FindTinyXML2.cmake in public/cmake/modules when no Config).
find_package(TinyXML2 REQUIRED)

add_library(ipc_fastdds_deps INTERFACE)
target_link_libraries(ipc_fastdds_deps INTERFACE fastdds TinyXML2::TinyXML2)

# Build-tree layout: public headers under .../include/holoscan/... for angle-bracket includes;
# private gen artifacts in a flat directory (same as former src/transport/fastdds/gen).
set(_ipc_fd_out_public_gen "${PROJECT_BINARY_DIR}/generated/include/holoscan/ipc/transport/fastdds/gen")
set(_ipc_fd_out_private "${PROJECT_BINARY_DIR}/generated/fastdds_private")

set(_ipc_fd_idl_dir "${PROJECT_SOURCE_DIR}/idl")
set(_ipc_fd_cmake_regen "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake_regen_fastdds_idl.py")
set(_ipc_fd_post "${CMAKE_CURRENT_SOURCE_DIR}/scripts/postprocess_idl_gen.py")

set(_ipc_fd_gen_outputs
  "${_ipc_fd_out_public_gen}/control_message.hpp"
  "${_ipc_fd_out_public_gen}/control_messagePubSubTypes.hpp"
  "${_ipc_fd_out_public_gen}/pointer_descriptor.hpp"
  "${_ipc_fd_out_public_gen}/pointer_descriptorPubSubTypes.hpp"
  "${_ipc_fd_out_public_gen}/pointer_descriptorTypeObjectSupport.hpp"
  "${_ipc_fd_out_private}/control_messageCdrAux.hpp"
  "${_ipc_fd_out_private}/control_messageCdrAux.ipp"
  "${_ipc_fd_out_private}/control_messageTypeObjectSupport.hpp"
  "${_ipc_fd_out_private}/control_messageTypeObjectSupport.cxx"
  "${_ipc_fd_out_private}/control_messagePubSubTypes.cxx"
  "${_ipc_fd_out_private}/pointer_descriptorCdrAux.hpp"
  "${_ipc_fd_out_private}/pointer_descriptorCdrAux.ipp"
  "${_ipc_fd_out_private}/pointer_descriptorPubSubTypes.cxx"
  "${_ipc_fd_out_private}/pointer_descriptorTypeObjectSupport.cxx"
)

add_custom_command(
  OUTPUT ${_ipc_fd_gen_outputs}
  COMMAND ${Python3_EXECUTABLE} "${_ipc_fd_cmake_regen}" ipc-transport
    --fastddsgen "${HOLOSCAN_FASTDDSGEN_EXECUTABLE}"
    --idl-dir "${_ipc_fd_idl_dir}"
    --idl pointer_descriptor.idl control_message.idl
    --out-public-gen "${_ipc_fd_out_public_gen}"
    --out-private "${_ipc_fd_out_private}"
    --postprocess "${_ipc_fd_post}"
  DEPENDS
    "${_ipc_fd_idl_dir}/pointer_descriptor.idl"
    "${_ipc_fd_idl_dir}/control_message.idl"
    "${_ipc_fd_cmake_regen}"
    "${_ipc_fd_post}"
  COMMENT "Generating Fast DDS C++ from holoipc IDL (pointer_descriptor, control_message)"
  VERBATIM
)

add_custom_target(holoipc_fastdds_transport_idl DEPENDS ${_ipc_fd_gen_outputs})

set(_ipc_fd_gen_cxx
  "${_ipc_fd_out_private}/control_messagePubSubTypes.cxx"
  "${_ipc_fd_out_private}/control_messageTypeObjectSupport.cxx"
  "${_ipc_fd_out_private}/pointer_descriptorPubSubTypes.cxx"
  "${_ipc_fd_out_private}/pointer_descriptorTypeObjectSupport.cxx"
)

add_library(ipc_fastdds_gen_objs OBJECT ${_ipc_fd_gen_cxx})
add_dependencies(ipc_fastdds_gen_objs holoipc_fastdds_transport_idl)

set_source_files_properties(${_ipc_fd_gen_outputs} ${_ipc_fd_gen_cxx} PROPERTIES GENERATED TRUE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
  set_source_files_properties(${_ipc_fd_gen_cxx} PROPERTIES SKIP_LINTING ON)
endif()

set_target_properties(ipc_fastdds_gen_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ipc_fastdds_gen_objs PUBLIC cxx_std_17)
# Generated .cxx use #include "FooPubSubTypes.hpp" etc.; those headers live in out_public_gen, not
# next to the .cxx files in fastdds_private.
target_include_directories(ipc_fastdds_gen_objs PRIVATE
  "${_ipc_fd_out_public_gen}"
  "${PROJECT_BINARY_DIR}/generated/include"
  "${_ipc_fd_out_private}"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src"
  $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
)
target_link_libraries(ipc_fastdds_gen_objs PRIVATE ipc_fastdds_deps)

set(IPC_FASTDDS_SOURCES
  src/transport/fastdds/fast_dds_transport.cpp
  src/transport/fastdds/control_channel.cpp
  PARENT_SCOPE
)
set(IPC_FASTDDS_OBJECT_LIB
  ipc_fastdds_gen_objs
  PARENT_SCOPE
)
set(IPC_FASTDDS_LIBS
  ipc_fastdds_deps
  PARENT_SCOPE
)
# For consumers: angle-bracket includes under holoscan/ipc/transport/fastdds/gen/
set(IPC_FASTDDS_PUBLIC_GEN_BUILD_DIR
  "${PROJECT_BINARY_DIR}/generated/include"
  PARENT_SCOPE
)
