/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/network_context.hpp"

#include "holoscan/core/fragment.hpp"

namespace holoscan {

void NetworkContext::initialize() {
  // Initialize the operator through the executor
  auto fragment_ptr = fragment();
  if (fragment_ptr) {
    auto& executor = fragment_ptr->executor();
    executor.initialize_network_context(this);
  } else {
    HOLOSCAN_LOG_WARN("NetworkContext::initialize() - Fragment is not set");
  }
}

}  // namespace holoscan
