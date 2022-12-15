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

#ifndef HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP
#define HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP

#include <string>
#include <unordered_map>
#include <memory>

#include "../io_context.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to hold the input context for a GXF Operator.
 *
 * This class provides the interface to receive the input data from the operator using GXF.
 */
class GXFInputContext : public InputContext {
 public:
  /**
   * @brief Construct a new GXFInputContext object.
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFInputContext(gxf_context_t context, Operator* op);

  /**
   * @brief Construct a new GXFInputContext object
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the GXFOperator object.
   * @param inputs inputs The references to the map of the input specs.
   */
  GXFInputContext(gxf_context_t context, Operator* op,
                  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs)
      : InputContext(op, inputs), gxf_context_(context) {}

  /**
   * @brief Get GXF context.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const { return gxf_context_; }

 protected:
  std::any receive_impl(const char* name = nullptr, bool no_error_message = false) override;

 private:
  gxf_context_t gxf_context_ = nullptr;  ///< The pointer to the GXF context.
};

/**
 * @brief Class to hold the output context for a GXF Operator.
 *
 * This class provides the interface to send data to the output ports of the operator using GXF.
 */
class GXFOutputContext : public OutputContext {
 public:
  /**
   * @brief Construct a new GXFOutputContext object.
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFOutputContext(gxf_context_t context, Operator* op);

  /**
   * @brief Construct a new GXFOutputContext object
   *
   * @param context The pointer to the GXF context.
   * @param op The pointer to the GXFOperator object.
   * @param outputs outputs The references to the map of the output specs.
   */
  GXFOutputContext(gxf_context_t context, Operator* op,
                   std::unordered_map<std::string,
                   std::unique_ptr<IOSpec>>& outputs) : OutputContext(op, outputs),
                                                        gxf_context_(context) {}

  /**
   * @brief Get GXF context.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const { return gxf_context_; }

 protected:
  void emit_impl(std::any data, const char* name = nullptr,
                 OutputType out_type = OutputType::kSharedPointer) override;

 private:
  gxf_context_t gxf_context_ = nullptr;     ///< The pointer to the GXF context.
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP */
