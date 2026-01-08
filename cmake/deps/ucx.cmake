# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(ucx 1.17.0 REQUIRED)

# Install UCX shared libs under <prefix>/lib/libuc*.so* and libgdrapi*.so*
# Core UCX libraries: libucp.so, libucs.so, libuct.so, libucm.so, libucs_signal.so
# gdrcopy library: libgdrapi.so (copied to UCX lib dir during Docker build)
file(GLOB TOP_UCX_LIB_FILES "${UCX_LIBRARIES}/libuc*.so*")
file(GLOB GDRCOPY_LIB_FILES "${UCX_LIBRARIES}/libgdrapi*.so*")
install(
  FILES ${TOP_UCX_LIB_FILES} ${GDRCOPY_LIB_FILES}
  DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
  COMPONENT "holoscan-dependencies"
)

# Set RPATH on installed UCX and gdrcopy libraries so they can find each other
# Note: CPack sets and requires the DESTDIR prefix. See:
# https://gitlab.kitware.com/cmake/cmake/-/issues/24212
install(CODE "
  file(GLOB UCX_INSTALLED_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/libuc*.so*\")
  file(GLOB GDRCOPY_INSTALLED_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/libgdrapi*.so*\")
  list(LENGTH UCX_INSTALLED_LIBS UCX_INSTALLED_LIBS_COUNT)
  if(UCX_INSTALLED_LIBS_COUNT LESS_EQUAL 0)
    message(WARNING \"No UCX libraries found in \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/\")
  endif()
  foreach(ucx_lib \${UCX_INSTALLED_LIBS} \${GDRCOPY_INSTALLED_LIBS})
    if(NOT IS_SYMLINK \${ucx_lib})
      execute_process(COMMAND patchelf --set-rpath \$ORIGIN \${ucx_lib})
    endif()
  endforeach()
" COMPONENT "holoscan-dependencies")

# Install UCX shared libs under <prefix>/lib/ucx
install(
  DIRECTORY ${UCX_LIBRARIES}/ucx
  DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
  COMPONENT "holoscan-dependencies"
  FILES_MATCHING PATTERN "*.so*"
)

# Set RPATH on installed UCX extension libraries so they can find base UCX libraries
# Note: CPack sets and requires the DESTDIR prefix. See:
# https://gitlab.kitware.com/cmake/cmake/-/issues/24212
install(CODE "
  file(GLOB UCX_EXT_INSTALLED_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/ucx/*.so*\")
  foreach(ucx_ext_lib \${UCX_EXT_INSTALLED_LIBS})
    if(NOT IS_SYMLINK \${ucx_ext_lib})
      execute_process(COMMAND patchelf --set-rpath \$ORIGIN:\$ORIGIN/.. \${ucx_ext_lib})
    endif()
  endforeach()
" COMPONENT "holoscan-dependencies")

# Install essential UCX binaries for diagnostics and benchmarking:
# - ucx_info: Display UCX configuration, available transports, and devices
# - ucx_perftest: Benchmark UCX performance (latency, bandwidth)
# These binaries are built against Holoscan's UCX version to avoid version conflicts
# with HPC-X UCX that may be present in base images.
get_filename_component(UCX_ROOT "${UCX_LIBRARIES}" DIRECTORY)
set(UCX_BIN_DIR "${UCX_ROOT}/bin")
if(EXISTS "${UCX_BIN_DIR}")
  set(UCX_ESSENTIAL_BINS
    "${UCX_BIN_DIR}/ucx_info"
    "${UCX_BIN_DIR}/ucx_perftest"
  )
  foreach(ucx_bin ${UCX_ESSENTIAL_BINS})
    if(EXISTS "${ucx_bin}")
      install(
        PROGRAMS "${ucx_bin}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT "holoscan-dependencies"
      )
    endif()
  endforeach()

  # Set RPATH on installed UCX binaries to find libraries in ../lib
  # Note: CPack sets and requires the DESTDIR prefix. See:
  # https://gitlab.kitware.com/cmake/cmake/-/issues/24212
  install(CODE "
    set(UCX_BIN_NAMES ucx_info ucx_perftest)
    foreach(ucx_bin_name \${UCX_BIN_NAMES})
      set(ucx_bin \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/\${ucx_bin_name}\")
      if(EXISTS \${ucx_bin} AND NOT IS_SYMLINK \${ucx_bin})
        execute_process(COMMAND patchelf --set-rpath \$ORIGIN/../lib \${ucx_bin})
      endif()
    endforeach()
  " COMPONENT "holoscan-dependencies")
endif()

# Install UCX headers
foreach(ucx_target ucm ucp ucs uct)
  install(
    DIRECTORY ${UCX_INCLUDE_DIRS}/${ucx_target}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/ucx"
    COMPONENT "holoscan-dependencies"
  )
endforeach()

install(
  FILES ${ucx_DIR}/ucx-config-version.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ucx"
  COMPONENT "holoscan-dependencies"
)
install(
  FILES ${CMAKE_CURRENT_LIST_DIR}/configs/ucx-config.cmake.in
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ucx"
  COMPONENT "holoscan-dependencies"
  RENAME "ucx-config.cmake"
)
