/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP
#define HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP

#include <cstdint>

namespace holoscan::distributed {

constexpr uint32_t kMinNetworkPort = 10000;
constexpr uint32_t kMaxNetworkPort = 32767;

constexpr int32_t kDefaultAppDriverPort = 57777;

}  // namespace holoscan::distributed

#endif /* HOLOSCAN_CORE_DISTRIBUTED_COMMON_NETWORK_CONSTANTS_HPP */
