/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_POSE_TREE_MANAGER_LOOKUP_HPP
#define HOLOSCAN_OPERATORS_POSE_TREE_MANAGER_LOOKUP_HPP

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief PoseTreeManager service lookup operator intended for use in tests.
 *
 * During initialization, it attempts to retrieve the PoseTreeManager service using the
 * "pose_tree_manager" id. It throws if the service is not found.
 */
class PoseTreeManagerLookupOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PoseTreeManagerLookupOp)

  PoseTreeManagerLookupOp() = default;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_POSE_TREE_MANAGER_LOOKUP_HPP */
