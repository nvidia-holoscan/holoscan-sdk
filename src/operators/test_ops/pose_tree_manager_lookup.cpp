/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>

#include <holoscan/operators/test_ops/pose_tree_manager_lookup.hpp>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/pose_tree/pose_tree_manager.hpp>

namespace holoscan::ops {

void PoseTreeManagerLookupOp::setup([[maybe_unused]] OperatorSpec& spec) {}

void PoseTreeManagerLookupOp::initialize() {
  Operator::initialize();

  auto pose_tree_manager = service<holoscan::PoseTreeManager>("pose_tree_manager");
  if (!pose_tree_manager) {
    HOLOSCAN_LOG_ERROR("PoseTreeManager service 'pose_tree_manager' not found.");
    throw std::runtime_error("PoseTreeManager service lookup failed");
  }
}

}  // namespace holoscan::ops
