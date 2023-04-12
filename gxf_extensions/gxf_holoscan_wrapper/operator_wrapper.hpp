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

#ifndef GXF_HOLOSCAN_WRAPPER_OPERATOR_WRAPPER_HPP
#define GXF_HOLOSCAN_WRAPPER_OPERATOR_WRAPPER_HPP

#include <list>
#include <memory>

#include "holoscan/core/operator.hpp"
#include "holoscan/core/parameter.hpp"
#include "operator_wrapper_fragment.hpp"

#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"

namespace holoscan::gxf {

/**
 * @brief Class to wrap an Operator to interface with the GXF framework.
 */
class OperatorWrapper : public nvidia::gxf::Codelet {
 public:
  OperatorWrapper();
  virtual ~OperatorWrapper() = default;

  /// Get the type name of the Operator.
  virtual const char* holoscan_typename() const = 0;

  /// Create and initialize the Operator.
  gxf_result_t initialize() override;
  /// Destroy the Operator and free resources.
  gxf_result_t deinitialize() override;

  /// Register the Operator's parameters with the GXF framework.
  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;

  /// Delegate to the Operator's start() method.
  gxf_result_t start() override;
  /// Delegate to the Operator's compute() method.
  gxf_result_t tick() override;
  /// Delegate to the Operator's stop() method.
  gxf_result_t stop() override;

  struct GXFParameter {
    nvidia::gxf::Parameter<YAML::Node> param;  ///< The GXF parameter.
    holoscan::ArgType arg_type;                ///< The type of the parameter (in Holoscan)
    holoscan::Parameter<void*>* param_ptr;     ///< The pointer to the parameter (in Holoscan)
  };

 protected:
  std::shared_ptr<Operator> op_;        ///< The Operator to wrap.
  OperatorWrapperFragment fragment_;    ///< The fragment to use for the Operator.
  std::list<GXFParameter> parameters_;  ///< The parameters to use for the GXF Codelet.
};

}  // namespace holoscan::gxf

#endif /* GXF_HOLOSCAN_WRAPPER_OPERATOR_WRAPPER_HPP */
