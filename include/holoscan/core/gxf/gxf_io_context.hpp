/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <unordered_map>

#include "../io_context.hpp"

namespace holoscan::gxf {

nvidia::gxf::Receiver* get_gxf_receiver(const std::unique_ptr<IOSpec>& input_spec);

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
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFInputContext(ExecutionContext* execution_context, Operator* op);

  /**
   * @brief Construct a new GXFInputContext object
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   * @param inputs inputs The references to the map of the input specs.
   */
  GXFInputContext(ExecutionContext* execution_context, Operator* op,
                  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs);

  /**
   * @brief Get a pointer to the GXF execution runtime.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const;

 protected:
  bool empty_impl(const char* name = nullptr) override;
  std::any receive_impl(const char* name = nullptr, bool no_error_message = false) override;
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
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFOutputContext(ExecutionContext* execution_context, Operator* op);

  /**
   * @brief Construct a new GXFOutputContext object
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   * @param outputs outputs The references to the map of the output specs.
   */
  GXFOutputContext(ExecutionContext* execution_context, Operator* op,
                   std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs);

  /**
   * @brief Get pointer to the GXF execution runtime.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const;

 protected:
  void emit_impl(std::any data, const char* name = nullptr,
                 OutputType out_type = OutputType::kSharedPointer) override;
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP */
