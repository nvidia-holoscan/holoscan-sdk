# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# IPC (holo_ipc) library - shared pointer / descriptor exchange.
# Core is transport-agnostic. Optional transport sub-features: HOLOSCAN_IPC_TRANSPORT_FASTDDS.
# Each transport is configured in its own CMakeLists.txt under this module's src/transport/<name>/.

cmake_minimum_required(VERSION 3.26.4)

project(holoipc
  DESCRIPTION "Holoscan IPC library (holo_ipc)"
  VERSION ${HOLOSCAN_BUILD_VERSION}
  LANGUAGES CXX
)

# HOLOSCAN_PACKAGE_NAME is set in the Holoscan SDK root (public/CMakeLists.txt) before
# FetchContent loads this module. holoviz hardcodes holoscan:: targets; holoinfer uses this
# variable for install(EXPORT). Default here so ALIAS/OUTPUT_NAME work if configured alone.
if(NOT DEFINED HOLOSCAN_PACKAGE_NAME)
  set(HOLOSCAN_PACKAGE_NAME holoscan)
endif()

find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Core sources (transport-agnostic)
# -----------------------------------------------------------------------------
set(IPC_CORE_SOURCES
  src/ipc_core.cpp
  src/cuda_ipc_backend.cpp
  src/io_context.cpp
)

# -----------------------------------------------------------------------------
# Transport subdirectories (each sets *_SOURCES, *_LIBS, *_INCLUDE_DIRS in parent scope).
# Subdirs use PROJECT_SOURCE_DIR / PROJECT_BINARY_DIR from project(holoipc) above (same
# roots as FetchContent holoipc_SOURCE_DIR / holoipc_BINARY_DIR when built via Holoscan).
# -----------------------------------------------------------------------------
set(IPC_TRANSPORT_SOURCES)

if(HOLOSCAN_IPC_TRANSPORT_FASTDDS)
  add_subdirectory(src/transport/fastdds)
  list(APPEND IPC_TRANSPORT_SOURCES ${IPC_FASTDDS_SOURCES})
endif()

set(IPC_SOURCES ${IPC_CORE_SOURCES} ${IPC_TRANSPORT_SOURCES})
if(HOLOSCAN_IPC_TRANSPORT_FASTDDS AND IPC_FASTDDS_OBJECT_LIB)
  list(APPEND IPC_SOURCES $<TARGET_OBJECTS:${IPC_FASTDDS_OBJECT_LIB}>)
endif()

add_library(holo_ipc SHARED ${IPC_SOURCES})
add_library(${HOLOSCAN_PACKAGE_NAME}::ipc ALIAS holo_ipc)

set_target_properties(holo_ipc PROPERTIES
  OUTPUT_NAME ${HOLOSCAN_PACKAGE_NAME}_holo_ipc
  EXPORT_NAME ipc
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION}
)

target_compile_features(holo_ipc PUBLIC cxx_std_17)

target_include_directories(holo_ipc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
)

target_link_libraries(holo_ipc
  PUBLIC
    CUDA::cuda_driver
  PRIVATE
    holoscan_security_flags
    holoscan::logger
)

if(HOLOSCAN_IPC_TRANSPORT_FASTDDS)
  target_link_libraries(holo_ipc
    PUBLIC
      ${IPC_FASTDDS_LIBS}
  )
  # Generated IDL headers (build tree): <holoscan/ipc/transport/fastdds/gen/*.hpp>
  target_include_directories(holo_ipc
    PUBLIC
      $<BUILD_INTERFACE:${IPC_FASTDDS_PUBLIC_GEN_BUILD_DIR}>
  )
  add_dependencies(holo_ipc holoipc_fastdds_transport_idl)
endif()

add_subdirectory(examples)

# Install library (component handled by top-level via HOLOSCAN_INSTALL_TARGETS)
# Hand-written public headers under include/holoscan/
install(
  DIRECTORY include/holoscan
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT holoscan-core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
)
# Fast DDS transport: IDL-generated public API headers (build tree only)
if(HOLOSCAN_IPC_TRANSPORT_FASTDDS)
  install(
    DIRECTORY ${PROJECT_BINARY_DIR}/generated/include/holoscan/ipc/transport/fastdds/gen/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/ipc/transport/fastdds/gen
    COMPONENT holoscan-core
    FILES_MATCHING
    PATTERN "*.hpp"
  )
endif()
