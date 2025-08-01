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

#include "holoscan/core/resources/gxf/std_entity_serializer.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"

namespace holoscan {

void StdEntitySerializer::setup(ComponentSpec& spec) {
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

nvidia::gxf::StdEntitySerializer* StdEntitySerializer::get() const {
  return static_cast<nvidia::gxf::StdEntitySerializer*>(gxf_cptr_);
}

void StdEntitySerializer::initialize() {
  // Set up prerequisite parameters before calling GXFResource::initialize()
  auto frag = fragment();

  auto has_component_serializers = std::find_if(args().begin(), args().end(), [](const auto& arg) {
    return (arg.name() == "component_serializers");
  });
  if (has_component_serializers == args().end()) {
    HOLOSCAN_LOG_TRACE(
        "StdEntitySerializer: component_serializers argument not found, using default.");
    auto component_serializer =
        frag->make_resource<holoscan::StdComponentSerializer>("std_component_serializer");
    component_serializer->gxf_cname(component_serializer->name().c_str());
    if (gxf_eid_ != 0) {
      component_serializer->gxf_eid(gxf_eid_);
    }
    add_arg(Arg("component_serializers") =
                std::vector<std::shared_ptr<Resource>>{component_serializer});
  } else {
    // must set the gxf_eid for the provided buffer or GXF parameter registration will fail
    auto component_serializers_arg = *has_component_serializers;
    auto component_serializers =
        std::any_cast<std::vector<std::shared_ptr<Resource>>>(component_serializers_arg.value());
    for (const auto& component_serializer : component_serializers) {
      auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(component_serializer);
      if (gxf_eid_ != 0 && gxf_resource->gxf_eid() == 0) {
        HOLOSCAN_LOG_TRACE(
            "component_serializer '{}': setting gxf_eid({}) from StdEntitySerializer '{}'",
            component_serializer->name(),
            gxf_eid_,
            name());
        gxf_resource->gxf_eid(gxf_eid_);
      }
    }
  }

  GXFResource::initialize();
}

}  // namespace holoscan
