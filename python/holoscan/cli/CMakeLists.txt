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

# Copy the Holoscan CLI Python source code to the build dir
# Note: installation handled by the CMakeLists.txt above for
# the whole holoscan module
file(
  COPY .
  DESTINATION ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/cli
  PATTERN "CMakeLists.txt" EXCLUDE
  PATTERN "holoscan" EXCLUDE
)

# Read the contents of requirements.txt to pass to the CLI bash script
file(READ "../../requirements.txt" HOLOSCAN_CLI_DEPENDENCIES)
string(STRIP "${HOLOSCAN_CLI_DEPENDENCIES}" HOLOSCAN_CLI_DEPENDENCIES)

# Holoscan CLI (bash)
configure_file(holoscan ${CMAKE_BINARY_DIR}/bin @ONLY)
install(
  FILES ${CMAKE_BINARY_DIR}/bin/holoscan
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT "holoscan-python_libs"
)
