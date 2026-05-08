# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Locate Fast-DDS-Gen (fastddsgen) and Python for CMake-driven IDL -> C++ generation.
# Included by CMakeLists.txt in this directory when HOLOSCAN_IPC_TRANSPORT_FASTDDS is ON.
#
# Holoscan SDK Docker: FAST_DDS_GEN=/opt/fastdds-gen/<ver> with bin/fastddsgen (java -jar wrapper).
# See public/Dockerfile and modules/holoipc/docs/IDL_TYPES.md.

find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)

set(_holoscan_fastddsgen_hints "")
if(DEFINED ENV{FAST_DDS_GEN})
  list(APPEND _holoscan_fastddsgen_hints "$ENV{FAST_DDS_GEN}/bin")
endif()

find_program(HOLOSCAN_FASTDDSGEN_EXECUTABLE
  NAMES fastddsgen fastddsgen.bat
  HINTS ${_holoscan_fastddsgen_hints}
  DOC "eProsima Fast DDS IDL compiler (Fast-DDS-Gen)"
)

if(NOT HOLOSCAN_FASTDDSGEN_EXECUTABLE)
  message(FATAL_ERROR
    "fastddsgen not found. Install Fast-DDS-Gen (https://github.com/eProsima/Fast-DDS-Gen), "
    "add it to PATH, or set environment FAST_DDS_GEN to the install root (see public/Dockerfile). "
    "Documentation: public/modules/holoipc/docs/IDL_TYPES.md"
  )
endif()
