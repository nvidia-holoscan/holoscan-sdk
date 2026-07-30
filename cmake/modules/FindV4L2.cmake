# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
