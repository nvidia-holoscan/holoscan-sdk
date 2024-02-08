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

# Define an interface library to hold the security flags
# Just need to link against this library to enable security flags
add_library(holoscan_security_flags INTERFACE)

# full relro protection
# LINKER will be transformed into -W / -Xlinker
# SHELL makes sure CMake doesn't dedup -z
target_link_options(holoscan_security_flags INTERFACE
  "LINKER:SHELL:-z,relro"
  "LINKER:SHELL:-z,now")
# stack canaries
target_compile_options(holoscan_security_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fstack-protector-strong>
  $<$<COMPILE_LANGUAGE:CXX>:-fstack-protector-strong>
  )

# stack clash prevention
target_compile_options(holoscan_security_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fstack-clash-protection>
  $<$<COMPILE_LANGUAGE:CXX>:-fstack-clash-protection>
  )

# FORTIFY_SOURCE
# See https://man7.org/linux/man-pages/man7/feature_test_macros.7.html for more information
target_compile_definitions(holoscan_security_flags INTERFACE _FORTIFY_SOURCE=2)

# warning options
target_compile_options(holoscan_security_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wformat=2,-Wformat-security,-Werror>
  $<$<COMPILE_LANGUAGE:CXX>:-Wformat=2 -Wformat-security -Werror>
  )

# enforce PIE/PIC
set_target_properties(holoscan_security_flags PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)