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

# Holoscan CLI (bash) was migrated to a standalone project in Holoscan SDK v2.9.
# Starting from 2.9, we include a stub to instruct users to install holoscan-cli.
file(
  COPY .
  DESTINATION ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/cli
  PATTERN "CMakeLists.txt" EXCLUDE
  PATTERN "holoscan" EXCLUDE
)

# We install a stub to notify users that core support is deprecated.
configure_file(holoscan ${CMAKE_BINARY_DIR}/bin @ONLY)
install(
  FILES ${CMAKE_BINARY_DIR}/bin/holoscan
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT "holoscan-python_libs"
)
