/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/operator.hpp"

#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan {

void Operator::initialize() {
  // Set the operator type based on the base class
  if (dynamic_cast<ops::GXFOperator*>(this)) {
    operator_type_ = holoscan::Operator::OperatorType::kGXF;
  }

  // Initialize the operator through the executor
  auto fragment_ptr = fragment();
  if (fragment_ptr) {
    auto& executor = fragment_ptr->executor();
    executor.initialize_operator(this);
  } else {
    HOLOSCAN_LOG_WARN("Operator::initialize() - Fragment is not set");
  }
}

}  // namespace holoscan
