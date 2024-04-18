/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
             false);
}

nvidia::gxf::UcxEntitySerializer* UcxEntitySerializer::get() const {
  return static_cast<nvidia::gxf::UcxEntitySerializer*>(gxf_cptr_);
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
    // UcxHoloscanComponentSerializer handles Holoscan SDK types such as holoscan::Message
    auto ucx_holoscan_component_serializer =
        frag->make_resource<holoscan::UcxHoloscanComponentSerializer>(
            "ucx_holoscan_component_serializer");
    ucx_holoscan_component_serializer->gxf_cname(ucx_holoscan_component_serializer->name().c_str());
    component_serializers.push_back(ucx_holoscan_component_serializer);
    // UcxComponentSerializer handles nvidia::gxf::Tensor, nvidia::gxf::VideoBuffer, etc.
    auto ucx_component_serializer =
        frag->make_resource<holoscan::UcxComponentSerializer>("ucx_component_serializer");
    ucx_component_serializer->gxf_cname(ucx_component_serializer->name().c_str());
    component_serializers.push_back(ucx_component_serializer);

    // Note: Activation sequence of entities in GXF:
    // 1. System entities
    // 2. Router entities
    // 3. Connection entities
    // 4. Network entities
    // 5. Graph entities
    //
    // Holoscan Resources, a GXF component created using Fragment::make_resource, are not part of
    // any entity by default. A new entity is created unless the resource's entity id is explicitly
    // set (using gxf_eid()). Network entities like UcxContext might access serializer resources
    // before the activation of graph entities. Therefore, it's essential to assign the same entity
    // id to the serializer resource as the network entity (UcxContext) to ensure simultaneous
    // activation and initialization.
    //
    // The entity id for UcxEntitySerializer is assigned during UcxContext::initialize(). Since the
    // component serializer needs to be activated concurrently with UcxContext::initialize(), it's
    // necessary to set the entity id at this point as well.
    // (issue 4398018)
    if (gxf_eid_ != 0) {
      ucx_holoscan_component_serializer->gxf_eid(gxf_eid_);
      ucx_component_serializer->gxf_eid(gxf_eid_);
    }
    add_arg(Arg("component_serializers", component_serializers));
  }
  GXFResource::initialize();
}

}  // namespace holoscan
