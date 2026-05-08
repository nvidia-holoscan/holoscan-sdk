# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# fastdds_benchmark - uses the SDK's IPC library (holoscan::ipc) with Fast DDS transport.
# Only included when HOLOSCAN_IPC_TRANSPORT_FASTDDS is ON (see modules/holoipc/examples/CMakeLists.txt).

include("${PROJECT_SOURCE_DIR}/src/transport/fastdds/fast_dds_idl_gen.cmake")

set(_bm_fd_regen "${PROJECT_SOURCE_DIR}/src/transport/fastdds/scripts/cmake_regen_fastdds_idl.py")
set(_bm_fd_post "${PROJECT_SOURCE_DIR}/examples/postprocess_example_fastdds_idl.py")
set(_bm_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/gen")

set(FASTDDS_BENCHMARK_GEN_OUT
  "${_bm_gen_dir}/Buffer.hpp"
  "${_bm_gen_dir}/BufferPubSubTypes.hpp"
  "${_bm_gen_dir}/BufferPubSubTypes.cxx"
  "${_bm_gen_dir}/BufferCdrAux.hpp"
  "${_bm_gen_dir}/BufferCdrAux.ipp"
  "${_bm_gen_dir}/BufferTypeObjectSupport.hpp"
  "${_bm_gen_dir}/BufferTypeObjectSupport.cxx"
)

add_custom_command(
  OUTPUT ${FASTDDS_BENCHMARK_GEN_OUT}
  COMMAND ${Python3_EXECUTABLE} "${_bm_fd_regen}" flat
    --fastddsgen "${HOLOSCAN_FASTDDSGEN_EXECUTABLE}"
    --idl-file "${CMAKE_CURRENT_SOURCE_DIR}/idl/Buffer.idl"
    --ipc-idl-include "${PROJECT_SOURCE_DIR}/idl"
    --out-dir "${_bm_gen_dir}"
    --postprocess "${_bm_fd_post}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/idl/Buffer.idl"
    "${PROJECT_SOURCE_DIR}/idl/pointer_descriptor.idl"
    "${_bm_fd_regen}"
    "${_bm_fd_post}"
  COMMENT "Generating Fast DDS C++ from fastdds_benchmark IDL (Buffer)"
  VERBATIM
)

set(FASTDDS_BENCHMARK_CPP
  main.cpp
  Application.cpp
  PublisherApp.cpp
  ListenerSubscriberApp.cpp
  CudaDataWriterListener.cpp
)

add_executable(fastdds_benchmark
  ${FASTDDS_BENCHMARK_CPP}
  "${_bm_gen_dir}/BufferPubSubTypes.cxx"
  "${_bm_gen_dir}/BufferTypeObjectSupport.cxx"
)

# Sources use #include <gen/...>; include binary dir so gen/ is ${CMAKE_CURRENT_BINARY_DIR}/gen
target_include_directories(fastdds_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(fastdds_benchmark holoipc_fastdds_transport_idl)

target_link_libraries(fastdds_benchmark
  PRIVATE
    holoscan::ipc
    holoscan::logger
    CUDA::cudart
)

set_property(TARGET fastdds_benchmark PROPERTY CXX_STANDARD 17)
set_property(TARGET fastdds_benchmark PROPERTY CXX_STANDARD_REQUIRED ON)

target_compile_definitions(fastdds_benchmark PRIVATE
  $<$<AND:$<NOT:$<BOOL:${WIN32}>>,$<STREQUAL:"${CMAKE_BUILD_TYPE}","Debug">>:__DEBUG>
  $<$<BOOL:${INTERNAL_DEBUG}>:__INTERNALDEBUG>
  $<$<BOOL:${SHM_TRANSPORT_DEFAULT}>:SHM_TRANSPORT_BUILTIN>
  HAVE_LOG_NO_INFO=0
  FASTDDS_ENFORCE_LOG_INFO
)

# Install RPATH so the binary finds SDK libs
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(fastdds_benchmark PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the SDK examples pattern
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS fastdds_benchmark
  RUNTIME DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

# Install XML config files next to the binary
file(GLOB_RECURSE XML_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.xml)
foreach(XML_FILE_COMPLETE_PATH ${XML_FILES})
  get_filename_component(XML_FILE ${XML_FILE_COMPLETE_PATH} NAME)
  install(FILES ${XML_FILE_COMPLETE_PATH}
    DESTINATION ${app_relative_dest_path}
    COMPONENT holoscan-examples)
endforeach()

# Copy XML to build dir so the app can find them when run from build tree
foreach(XML_FILE_COMPLETE_PATH ${XML_FILES})
  get_filename_component(XML_FILE ${XML_FILE_COMPLETE_PATH} NAME)
  configure_file(
    ${XML_FILE_COMPLETE_PATH}
    ${CMAKE_CURRENT_BINARY_DIR}/${XML_FILE}
    COPYONLY)
endforeach()

install(FILES README.md
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

install(FILES images/latency_sweep.png
  DESTINATION ${app_relative_dest_path}/images
  COMPONENT holoscan-examples
)

install(PROGRAMS scripts/run_latency_benchmark.sh
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

install(FILES scripts/sweep_latency_benchmark.py
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

# Copy latency script next to the built binary (source stays under scripts/ in the repo)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_latency_benchmark.sh"
  "${CMAKE_CURRENT_BINARY_DIR}/run_latency_benchmark.sh"
  COPYONLY)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/run_latency_benchmark.sh"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/sweep_latency_benchmark.py"
  "${CMAKE_CURRENT_BINARY_DIR}/sweep_latency_benchmark.py"
  COPYONLY)
