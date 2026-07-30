/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_DOMAIN_MAP_HPP
#define HOLOSCAN_CORE_DOMAIN_MAP_HPP

#include <memory>
#include <string>
#include <unordered_map>

namespace holoscan {
template <typename T>
class Map : public std::unordered_map<std::string, std::shared_ptr<T>> {
 public:
  // Define any additional member functions or variables here
};
}  // namespace holoscan
#endif /* HOLOSCAN_CORE_DOMAIN_MAP_HPP */
