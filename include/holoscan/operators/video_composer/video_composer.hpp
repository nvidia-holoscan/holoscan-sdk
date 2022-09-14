#ifndef HOLOSCAN_OPERATORS_VIDEO_COMPOSER_VIDEO_COMPOSER_HPP
#define HOLOSCAN_OPERATORS_VIDEO_COMPOSER_VIDEO_COMPOSER_HPP

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

#include "../../core/operator.hpp"

namespace holoscan::ops {

class VideoComposerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoComposerOp)

  void start() override {
    // Initialize the operation
  }

  void stop() override {
    // Finalize the operation
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {}

 private:
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VIDEO_COMPOSER_VIDEO_COMPOSER_HPP */
