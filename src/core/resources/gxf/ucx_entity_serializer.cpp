/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"

#include <memory>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/ucx_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp"

namespace holoscan {

void UcxEntitySerializer::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxEntitySerializer::setup");
  spec.param(component_serializers_,
             "component_serializers",
             "Component serializers",
             "List of serializers for serializing and deserializing components");
  spec.param(verbose_warning_,
             "verbose_warning",
             "Verbose Warning",
             "Whether or not to print verbose warning",
             true);
}

void UcxEntitySerializer::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxEntitySerializer::initialize");
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'component_serializers'
  auto has_component_serializers = std::find_if(args().begin(), args().end(), [](const auto& arg) {
    return (arg.name() == "component_serializers");
  });
  // Create a UcxHoloscanComponentSerializer if no component_serializers argument was provided
  if (has_component_serializers == args().end()) {
    std::vector<std::shared_ptr<Resource>> component_serializers;
    component_serializers.reserve(2);
    // UcxHoloscanComponentSerializer handles Holoscan SDK types such as holoscan::gxf::GXFTensor
    component_serializers.push_back(frag->make_resource<holoscan::UcxHoloscanComponentSerializer>(
        "ucx_holoscan_component_serializer"));
    // UcxComponentSerializer handles nvidia::gxf::Tensor, nvidia::gxf::VideoBuffer, etc.
    component_serializers.push_back(
        frag->make_resource<holoscan::UcxComponentSerializer>("ucx_component_serializer"));
    add_arg(Arg("component_serializers", component_serializers));
  }
  GXFResource::initialize();
}

}  // namespace holoscan
