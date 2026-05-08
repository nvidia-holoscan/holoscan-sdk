# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Find TinyXML2 (https://github.com/leethomason/tinyxml2) when no config package is installed.
# Debian/Ubuntu libtinyxml2-dev provides headers and libtinyxml2 (.so or .a) but not TinyXML2Config.cmake.
#
# Sets TinyXML2_FOUND and defines imported target TinyXML2::TinyXML2 (same as upstream config).

find_path(TinyXML2_INCLUDE_DIR
  NAMES tinyxml2.h
  DOC "TinyXML2 include directory"
)

find_library(TinyXML2_LIBRARY
  NAMES tinyxml2
  DOC "TinyXML2 library"
)

mark_as_advanced(TinyXML2_INCLUDE_DIR TinyXML2_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyXML2
  FOUND_VAR TinyXML2_FOUND
  REQUIRED_VARS TinyXML2_LIBRARY TinyXML2_INCLUDE_DIR
)

if(TinyXML2_FOUND AND NOT TARGET TinyXML2::TinyXML2)
  add_library(TinyXML2::TinyXML2 UNKNOWN IMPORTED)
  set_target_properties(TinyXML2::TinyXML2 PROPERTIES
    IMPORTED_LOCATION "${TinyXML2_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TinyXML2_INCLUDE_DIR}"
  )
endif()
