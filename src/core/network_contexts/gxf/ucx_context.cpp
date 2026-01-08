/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdlib>

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
  spec.param(reconnect_,
             "reconnect",
             "Reconnect",
             "Try to reconnect if a connection is closed during run (default = true).",
             true);
  spec.param(cpu_data_only_,
             "cpu_data_only",
             "CPU data only",
             "If true, the UCX context will only support transmission of CPU (host) data "
             "(default = false).",
             false);
  spec.param(enable_async_,
             "enable_async",
             "enable async mode",
             "Enable asynchronous transmit/receive. This parameter is deprecated in Holoscan "
             "v3.7 and will be removed in v4.0. The new behavior will be equivalent to a "
             "value of `false` here.",
             false);
  spec.param(shutdown_timeout_ms_,
             "shutdown_timeout_ms",
             "Shutdown Timeout (ms)",
             "Timeout in milliseconds for shutdown operations such as thread joins and pending "
             "request cancellation (default = 2000).",
             static_cast<uint64_t>(2000));
  // spec.resource(gpu_device_, "Optional GPU device resource");
}

nvidia::gxf::UcxContext* UcxContext::get() const {
  return static_cast<nvidia::gxf::UcxContext*>(gxf_cptr_);
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
        "ucx_context_ucxentity_serializer", Arg("verbose_warning") = false);
    entity_serializer->gxf_cname(entity_serializer->name().c_str());

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
    if (gxf_eid_ != 0) {
      entity_serializer->gxf_eid(gxf_eid_);
    }

    add_arg(Arg("serializer") = entity_serializer);
  }

  // Allow environment variable override for shutdown_timeout_ms.
  // Note: add_arg appends to args_, and the last arg with a given name takes precedence
  // during parameter initialization, so this will override any user-provided value.
  const char* env_timeout = std::getenv("HOLOSCAN_UCX_SHUTDOWN_TIMEOUT_MS");
  if (env_timeout != nullptr && env_timeout[0] != '\0') {
    try {
      uint64_t timeout_ms = std::stoull(env_timeout);
      add_arg(Arg("shutdown_timeout_ms") = timeout_ms);
      HOLOSCAN_LOG_DEBUG("UcxContext: shutdown_timeout_ms set to {} from environment variable",
                         timeout_ms);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN(
          "UcxContext: Invalid HOLOSCAN_UCX_SHUTDOWN_TIMEOUT_MS value '{}', using default",
          env_timeout);
    }
  }

  GXFNetworkContext::initialize();
}

void UcxContext::initiate_shutdown() {
  shutting_down_ = true;
  if (auto* gxf_ucx_context = get()) {
    gxf_ucx_context->initiate_shutdown();
  }
}

bool UcxContext::is_shutting_down() const {
  return shutting_down_;
}

}  // namespace holoscan
