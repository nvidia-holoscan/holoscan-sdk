/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>
#include <string>

#include "holoscan/operators/test_ops/pose_tree_manager_lookup.hpp"

#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/pose_tree/pose_tree_manager.hpp"

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
