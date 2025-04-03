/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_WRAPPER_HPP
#define HOLOSCAN_CORE_GXF_GXF_WRAPPER_HPP

#include <memory>

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/core/registrar.hpp"
#include "gxf/std/codelet.hpp"
#include "holoscan/profiler/profiler.hpp"

// Forward declarations
namespace holoscan {

class Operator;
class InputContext;
class OutputContext;

namespace gxf {
class GXFExecutionContext;
}  // namespace gxf
}  // namespace holoscan

namespace holoscan::gxf {

PROF_DEFINE_EVENT(event_start, "start", 0x76, 0xB9, 0x00);
PROF_DEFINE_EVENT(event_stop, "stop", 0xFE, 0x27, 0x12);
PROF_DEFINE_EVENT(event_compute, "compute", 0x66, 0xA1, 0xFE);

/**
 * @brief Class to wrap an Operator into a GXF Codelet.
 *
 */
class GXFWrapper : public nvidia::gxf::Codelet {
 public:
  virtual ~GXFWrapper() = default;

  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  /**
   * @brief Set the Operator object to be wrapped.
   *
   * @param op The pointer to the Operator object.
   */
  void set_operator(Operator* op);

  /**
   * @brief Get the Operator object.
   *
   * @return The pointer to the Operator object.
   */
  Operator* op() const;

  /**
   * @brief Get the ExecutionContext object.
   *
   * @return The pointer to the ExecutionContext object.
   */
  GXFExecutionContext* execution_context() const;

  /**
   * @brief Get the InputContext object.
   *
   * @return The pointer to the InputContext object.
   */
  InputContext* input_context() const;

  /**
   * @brief Get the OutputContext object.
   *
   * @return The pointer to the OutputContext object.
   */
  OutputContext* output_context() const;

 private:
  void store_exception();
  /// Initialize the local contexts (input/output/execution) for the operator.
  void initialize_contexts();

  Operator* op_{};
  /// The execution context of the operator. Copied from the operator.
  GXFExecutionContext* exec_context_{};
  /// The input context of the operator. Copied from the operator.
  InputContext* op_input_{};
  /// The output context of the operator. Copied from the operator.
  OutputContext* op_output_{};
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_WRAPPER_HPP */
