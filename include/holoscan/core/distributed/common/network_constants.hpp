/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP
#define HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP

#include <cstdint>

namespace holoscan::distributed {

constexpr uint32_t kMinNetworkPort = 10000;
constexpr uint32_t kMaxNetworkPort = 32767;

constexpr int32_t kDefaultAppDriverPort = 8765;

}  // namespace holoscan::distributed

#endif /* HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP */
