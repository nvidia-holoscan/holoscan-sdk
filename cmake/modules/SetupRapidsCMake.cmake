# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(rapids-cmake-version 26.02)

# https://github.com/rapidsai/rapids-cmake#installation
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/refs/heads/release/${rapids-cmake-version}/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/RAPIDS.cmake
       TIMEOUT 120
       TLS_VERIFY ON
       EXPECTED_HASH SHA256=7690a95285e02b4f04dd23b327da0ad46e1b7c19d85207e9ca7ac17faa1947ec
       STATUS _rapids_dl_status
       LOG _rapids_dl_log)
  list(GET _rapids_dl_status 0 _rapids_dl_code)
  list(GET _rapids_dl_status 1 _rapids_dl_message)
  if(NOT _rapids_dl_code EQUAL 0)
    file(REMOVE ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    message(FATAL_ERROR
            "rapids-cmake RAPIDS.cmake download failed (code ${_rapids_dl_code}): ${_rapids_dl_message}\n"
            "${_rapids_dl_log}")
  endif()
endif()

include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
