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

#ifndef HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP
#define HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP

#include <gxf/core/gxf.h>

#include <memory>

#include "../execution_context.hpp"
#include "./gxf_io_context.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to hold the execution context for GXF Operator.
 *
 * This class provides the execution context for the operator using GXF.
 */
class GXFExecutionContext : public holoscan::ExecutionContext {
 public:
  /**
   * @brief Construct a new GXFExecutionContext object
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the operator.
   */
  GXFExecutionContext(gxf_context_t context, Operator* op);

  /**
   * @brief Construct a new GXFExecutionContext object
   *
   * @param context The pointer to the GXF context.
   * @param gxf_input_context The shared pointer to the GXFInputContext object.
   * @param gxf_output_context The shared pointer to the GXFOutputContext object.
   */
  GXFExecutionContext(gxf_context_t context,
                      std::shared_ptr<GXFInputContext> gxf_input_context,
                      std::shared_ptr<GXFOutputContext> gxf_output_context);

  /**
   * @brief Get the GXF input context.
   *
   * @return The pointer to the GXFInputContext object.
   */
  std::shared_ptr<GXFInputContext> gxf_input() { return gxf_input_context_; }

  /**
   * @brief Get the GXF output context.
   *
   * @return The pointer to the GXFOutputContext object.
   */
  std::shared_ptr<GXFOutputContext> gxf_output() { return gxf_output_context_; }

 protected:
  std::shared_ptr<GXFInputContext> gxf_input_context_;    ///< The GXF input context.
  std::shared_ptr<GXFOutputContext> gxf_output_context_;  ///< The GXF output context.
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_EXECUTION_CONTEXT_HPP */
