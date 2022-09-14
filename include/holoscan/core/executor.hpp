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

#ifndef HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP
#define HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP

#include "./common.hpp"

#include <string>

namespace holoscan {

/**
 * @brief Base class for all executors.
 *
 * An Executor that manages the execution of a Fragment on a physical node.
 * The framework provides a default Executor that uses a GXF Scheduler to execute an Application.
 */
class Executor {
 public:
  Executor() = delete;
  /**
   * @brief Construct a new Executor object.
   *
   * @param fragment The pointer to the fragment of the executor.
   */
  explicit Executor(Fragment* fragment) : fragment_(fragment){};
  virtual ~Executor() = default;

  /**
   * @brief Run the graph.
   */
  virtual void run(Graph& graph) { (void)graph; };

  /**
   * @brief Get a pointer to Fragment object.
   *
   * @return The Pointer to Fragment object.
   */
  Fragment* fragment() { return fragment_; }

  /**
   * @brief Set the context.
   *
   * @param context The context.
   */
  void context(void* context) { context_ = context; }
  /**
   * @brief Get the context
   *
   * @return The context.
   */
  void* context() { return context_; }

 protected:
  Fragment* fragment_ = nullptr;  ///< The fragment of the executor.
  void* context_ = nullptr;       ///< The context.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP */
