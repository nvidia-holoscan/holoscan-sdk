/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_GXF_CODELET_GXF_CODELET_HPP
#define HOLOSCAN_OPERATORS_GXF_CODELET_GXF_CODELET_HPP

#include "holoscan/core/gxf/gxf_operator.hpp"

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#include "holoscan/core/gxf/gxf_component_info.hpp"

/**
 * @brief Wrap a GXF Codelet as a Holoscan Operator.
 *
 * This macro is designed to simplify the creation of Holoscan operators that encapsulate GXF
 * Codelets. It defines a class derived from `holoscan::ops::GXFCodeletOp` and sets up the
 * constructor to forward arguments to the base class while automatically setting the GXF type name.
 *
 * The resulting class is intended to act as a bridge, allowing GXF Codelets to be used directly
 * within the Holoscan framework as operators, facilitating seamless integration and usage.
 *
 * Example Usage:
 *
 * ```cpp
 * // Define a Holoscan operator that wraps a GXF Codelet within a Holoscan application
 * class App : public holoscan::Application {
 *   ...
 *   HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(MyTensorOp, "nvidia::gxf::test::SendTensor")
 *
 *   void compose() override {
 *     using namespace holoscan;
 *     ...
 *     auto tx = make_operator<MyTensorOp>(
 *         "tx",
 *         make_condition<CountCondition>(15),
 *         Arg("pool") = make_resource<UnboundedAllocator>("pool"));
 *     ...
 *   }
 *   ...
 * };
 * ```
 *
 * @param class_name The name of the new Holoscan operator class.
 * @param gxf_typename The GXF type name that identifies the specific GXF Codelet being wrapped.
 */
#define HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(class_name, gxf_typename)        \
  class class_name : public ::holoscan::ops::GXFCodeletOp {                    \
   public:                                                                     \
    HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()                                       \
    explicit class_name(ArgT&& arg, ArgsT&&... args)                           \
        : ::holoscan::ops::GXFCodeletOp(gxf_typename, std::forward<ArgT>(arg), \
                                        std::forward<ArgsT>(args)...) {}       \
    class_name() : ::holoscan::ops::GXFCodeletOp(gxf_typename) {}              \
  };

namespace holoscan::ops {

class GXFCodeletOp : public holoscan::ops::GXFOperator {
 public:
  template <typename... ArgsT>
  explicit GXFCodeletOp(const char* gxf_typename, ArgsT&&... args)
      : holoscan::ops::GXFOperator(std::forward<ArgsT>(args)...) {
    gxf_typename_ = gxf_typename;
  }

  GXFCodeletOp() = default;

  const char* gxf_typename() const override;

  void setup(OperatorSpec& spec) override;

  void set_parameters() override;

 protected:
  std::shared_ptr<gxf::ComponentInfo> gxf_component_info_;  ///< The GXF component info.
  std::list<Parameter<void*>> parameters_;  ///< The fake parameters for the description.
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_GXF_CODELET_GXF_CODELET_HPP */
