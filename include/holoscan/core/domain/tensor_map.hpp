/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_DOMAIN_TENSOR_MAP_HPP
#define HOLOSCAN_CORE_DOMAIN_TENSOR_MAP_HPP

#include "map.hpp"
#include "tensor.hpp"

namespace holoscan {

class TensorMap : public Map<Tensor> {
 public:
  // Define any additional member functions or variables here
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_DOMAIN_TENSOR_MAP_HPP */
