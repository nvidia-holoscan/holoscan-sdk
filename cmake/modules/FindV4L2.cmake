# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Creates the V4L2 imported target
#
# This module defines V4L2_FOUND if the libv4l library is found:
# https://github.com/philips/libv4l
#
# A new imported target is created under the V4L2 namespace.

find_path(V4L2_INCLUDE_DIR
  NAMES libv4l2.h
  PATH_SUFFIXES v4l2 video4linux
  DOC "The Video4Linux Version 2 (v4l2) include directory"
)

find_library(V4L2_LIBRARY
  NAMES v4l2
  DOC "The Video4Linux Version 2 (v4l2) library"
  REQUIRED
)

mark_as_advanced(V4L2)
add_library(V4L2 IMPORTED GLOBAL SHARED)
set_target_properties(V4L2 PROPERTIES
  IMPORTED_LOCATION ${V4L2_LIBRARY}
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${V4L2_INCLUDE_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(V4L2
    FOUND_VAR V4L2_FOUND
    REQUIRED_VARS V4L2_INCLUDE_DIR
)
