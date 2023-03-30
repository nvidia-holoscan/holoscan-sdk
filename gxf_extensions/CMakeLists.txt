# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# For install_gxf_extension
# TODO: move that function to its own CMake module/file
include(GenerateGXEApp)

list(APPEND CMAKE_INSTALL_RPATH_LIST
    \$ORIGIN                                    # load lib${extension}_lib.so
    \$ORIGIN/..                                 # load 3rd-party dependency libs
    \$ORIGIN/../../../nvidia/cuda_runtime/lib   # for python wheels
    \$ORIGIN/../../../tensorrt                  # for python wheels
    )
list(JOIN CMAKE_INSTALL_RPATH_LIST ":" CMAKE_INSTALL_RPATH)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/gxf_extensions)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/gxf_extensions)

add_subdirectory(opengl)
add_subdirectory(stream_playback)
add_subdirectory(tensor_rt)
add_subdirectory(v4l2)
add_subdirectory(bayer_demosaic)
add_subdirectory(gxf_holoscan_wrapper)
