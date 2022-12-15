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

#ifndef HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP
#define HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP

#include "./common.hpp"
#include "./io_context.hpp"

namespace holoscan {

/**
 * @brief Class to hold the execution context.
 *
 * This class provides the execution context for the operator.
 */
class ExecutionContext {
 public:
  /**
   * @brief Construct a new Execution Context object.
   */
  ExecutionContext() = default;

  /**
   * @brief Get the input context.
   *
   * @return The pointer to the input context.
   */
  InputContext* input() const { return input_context_;}

  /**
   * @brief Get the output context.
   *
   * @return The pointer to the output context.
   */
  OutputContext* output() const { return output_context_;}

  /**
   * @brief Get the context.
   *
   * @return The pointer to the context.
   */
  void* context() const { return context_; }

 protected:
  InputContext* input_context_ = nullptr;    ///< The input context.
  OutputContext* output_context_ = nullptr;  ///< The output context.
  void* context_ = nullptr;                  ///< The context.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXECUTION_CONTEXT_HPP */
