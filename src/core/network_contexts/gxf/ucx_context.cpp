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

#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"
#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

void UcxContext::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("UcxContext::setup");
  spec.param(entity_serializer_,
             "serializer",
             "Entity Serializer",
             "The entity serializer used by this network context.");

  // TODO: implement OperatorSpec::resource for managing nvidia::gxf:Resource types
  // spec.resource(gpu_device_, "Optional GPU device resource");
}

void UcxContext::initialize() {
  HOLOSCAN_LOG_DEBUG("UcxContext::initialize");
  // Set up prerequisite parameters before calling GXFNetworkContext::initialize()
  auto frag = fragment();

  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });
  // Create a UcxEntitySerializer if no serializer argument was provided
  if (has_serializer == args().end()) {
    // In multi-fragment scenarios, the sequence_number of messages is not guaranteed to be
    // in sync across fragments. This is because the sequence_number is incremented by the
    // fragment that sends the message and the sequence_number is not synchronized across
    // fragments.
    auto entity_serializer = frag->make_resource<holoscan::UcxEntitySerializer>(
        "ucx_entity_serializer", Arg("verbose_warning") = false);

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
    // (issue 4398018)
    if (gxf_eid_ != 0) { entity_serializer->gxf_eid(gxf_eid_); }

    add_arg(Arg("serializer") = entity_serializer);
  }
  GXFNetworkContext::initialize();
}

}  // namespace holoscan
