# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(HoloscanTargetConfiguration)

# Define an interface library to hold the security flags
# Just need to link against this library to enable security flags
add_library(holoscan_security_flags INTERFACE)

holoscan_is_linux_gnu_like_frontend(_holoscan_linux_security_flags)
if(_holoscan_linux_security_flags)
  # Full RELRO protection. LINKER selects the driver wrapper and SHELL prevents CMake from
  # deduplicating the paired -z arguments.
  target_link_options(holoscan_security_flags INTERFACE
    "LINKER:SHELL:-z,relro"
    "LINKER:SHELL:-z,now")

  # Stack canaries and stack-clash prevention.
  target_compile_options(holoscan_security_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fstack-protector-strong>
    $<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fstack-clash-protection>
    $<$<COMPILE_LANGUAGE:CXX>:-fstack-clash-protection>
    )

  # FORTIFY_SOURCE requires optimization. Select optimized configurations at build time so Visual
  # Studio and Ninja Multi-Config do not inherit a configure-time CMAKE_BUILD_TYPE decision.
  target_compile_definitions(holoscan_security_flags INTERFACE
    $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_FORTIFY_SOURCE=2>)

  # Warning options.
  target_compile_options(holoscan_security_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wformat=2,-Wformat-security,-Werror>
    $<$<COMPILE_LANGUAGE:CXX>:-Wformat=2>
    $<$<COMPILE_LANGUAGE:CXX>:-Wformat-security>
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
    )
endif()
unset(_holoscan_linux_security_flags)

# enforce PIE/PIC
if(NOT MSVC)
  set_target_properties(holoscan_security_flags PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)
endif()
