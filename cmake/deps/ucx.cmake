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

# Install UCX shared libs under <prefix>/lib/libuc*.so*
file(GLOB TOP_UCX_LIB_FILES "${UCX_LIBRARIES}/libuc*.so*")
install(
  FILES ${TOP_UCX_LIB_FILES}
  DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
  COMPONENT "holoscan-dependencies"
)

# Set RPATH on installed UCX libraries so they can find each other
# Note: CPack sets and requires the DESTDIR prefix. See:
# https://gitlab.kitware.com/cmake/cmake/-/issues/24212
install(CODE "
  file(GLOB UCX_INSTALLED_LIBS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/libuc*.so*\")
  list(LENGTH UCX_INSTALLED_LIBS UCX_INSTALLED_LIBS_COUNT)
  if(UCX_INSTALLED_LIBS_COUNT LESS_EQUAL 0)
    message(WARNING \"No UCX libraries found in \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/\")
  endif()
  foreach(ucx_lib \${UCX_INSTALLED_LIBS})
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

# Install UCX headers
foreach(ucx_target ucm ucp ucs uct)
  install(
    DIRECTORY ${UCX_INCLUDE_DIRS}/${ucx_target}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/ucx"
    COMPONENT "holoscan-dependencies"
  )
endforeach()
