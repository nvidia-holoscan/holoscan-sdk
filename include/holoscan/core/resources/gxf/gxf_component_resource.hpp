/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_GXF_COMPONENT_RESOURCE_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_GXF_COMPONENT_RESOURCE_HPP

#include "../../gxf/gxf_resource.hpp"

#include <list>
#include <memory>
#include <utility>

#include "holoscan/core/gxf/gxf_component_info.hpp"

namespace holoscan {

/**
 * @brief Wrap a GXF Component as a Holoscan Resource.
 *
 * This macro is designed to simplify the creation of Holoscan resources that encapsulate a GXF
 * Component. It defines a class derived from `holoscan::GXFResource` and sets up the
 * constructor to forward arguments to the base class while automatically setting the GXF type name.
 *
 * The resulting class is intended to act as a bridge, allowing GXF Components to be used directly
 * within the Holoscan framework as resources, facilitating seamless integration and usage.
 *
 * Example Usage:
 *
 * ```cpp
 * // Define a Holoscan resource that wraps a GXF Component within a Holoscan application
 * class App : public holoscan::Application {
 *   ...
 *   HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(MyTensorOp, "nvidia::gxf::test::SendTensor")
 *   HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(MyBlockMemoryPool, "nvidia::gxf::BlockMemoryPool")
 *
 *   void compose() override {
 *     using namespace holoscan;
 *     ...
 *     auto tx = make_operator<MyCodeletOp>(
 *         "tx",
 *         make_condition<CountCondition>(15),
 *         Arg("pool") = make_resource<MyBlockMemoryPool>(
 *             "pool",
 *             Arg("storage_type") = static_cast<int32_t>(1),
 *             Arg("block_size") = 1024UL,
 *             Arg("num_blocks") = 2UL));
 *     ...
 *   }
 *   ...
 * };
 * ```
 *
 * @param class_name The name of the new Holoscan resource class.
 * @param gxf_typename The GXF type name that identifies the specific GXF Component being wrapped.
 */
#define HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(class_name, gxf_typename)         \
  class class_name : public ::holoscan::GXFComponentResource {                    \
   public:                                                                        \
    HOLOSCAN_RESOURCE_FORWARD_TEMPLATE()                                          \
    explicit class_name(ArgT&& arg, ArgsT&&... args)                              \
        : ::holoscan::GXFComponentResource(gxf_typename, std::forward<ArgT>(arg), \
                                           std::forward<ArgsT>(args)...) {}       \
    class_name() = default;                                                       \
  };

/**
 * @brief Class that wraps a GXF Component as a Holoscan Resource.
 */
class GXFComponentResource : public gxf::GXFResource {
 public:
  // Constructor
  template <typename... ArgsT>
  explicit GXFComponentResource(const char* gxf_typename, ArgsT&&... args)
      : GXFResource(std::forward<ArgsT>(args)...) {
    gxf_typename_ = gxf_typename;
  }

  // Default constructor
  GXFComponentResource() = default;

  // Returns the type name of the GXF component
  const char* gxf_typename() const override;

  // Sets up the component spec
  void setup(ComponentSpec& spec) override;

 protected:
  // Sets the parameters of the component
  void set_parameters() override;

  std::shared_ptr<gxf::ComponentInfo> gxf_component_info_;  ///< The GXF component info.
  std::list<Parameter<void*>> parameters_;  ///< The fake parameters for the description.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_GXF_COMPONENT_RESOURCE_HPP */
