/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/network_contexts/gxf/pubsub_context.hpp"

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "holoscan/core/clock.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

void PubSubContext::setup(ComponentSpec& spec) {
  HOLOSCAN_LOG_DEBUG("PubSubContext::setup");
  spec.param(node_name_,
             "node_name",
             "Node Name",
             "Name of this node/fragment for identification in discovery.",
             std::string{});
  spec.param(clock_,
             "clock",
             "Clock",
             "The clock used for pub/sub timestamps (creation_timestamp_ns, source_timestamp_ns). "
             "If not set, a RealtimeClock is created automatically.",
             ParameterFlag::kOptional);
}

nvidia::gxf::PubSubContext* PubSubContext::get() const {
  return static_cast<nvidia::gxf::PubSubContext*>(gxf_cptr_);
}

std::shared_ptr<Clock> PubSubContext::clock() {
  if (clock_.has_value()) {
    // Expose a generic Clock wrapper, matching Scheduler::clock() behavior.
    return std::make_shared<Clock>(std::static_pointer_cast<ClockInterface>(clock_.get()));
  }
  return nullptr;
}

void PubSubContext::initialize() {
  HOLOSCAN_LOG_DEBUG("PubSubContext::initialize");

  // Set node_name from fragment name if not explicitly provided
  auto* frag = fragment();
  if (frag != nullptr) {
    // Check if node_name was provided as an argument
    auto has_node_name = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "node_name"); });

    if (has_node_name == args().end()) {
      // Use fragment name as default node_name
      add_arg(Arg("node_name") = frag->name());
      HOLOSCAN_LOG_DEBUG("PubSubContext: node_name set to fragment name '{}'", frag->name());
    }

    // Auto-create RealtimeClock if no clock argument was provided
    auto has_clock = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "clock"); });
    if (has_clock == args().end()) {
      clock_ = frag->make_resource<holoscan::RealtimeClock>("pubsub_context__realtime_clock");
      clock_->gxf_cname(clock_->name());
      if (gxf_eid_ != 0) {
        clock_->gxf_eid(gxf_eid_);
      }
      add_arg(clock_.get());
      HOLOSCAN_LOG_DEBUG("PubSubContext: auto-created RealtimeClock for timestamps");
    } else if (has_clock->has_value()) {
      // Arg normalizes resource-like values to std::shared_ptr<Resource>, so a
      // clock passed as RealtimeClock/SyntheticClock/ManualClock can be cast here.
      try {
        auto clock_resource = std::any_cast<std::shared_ptr<Resource>>(has_clock->value());
        if (clock_resource) {
          clock_ = std::dynamic_pointer_cast<gxf::Clock>(clock_resource);
          if (!clock_.has_value() || clock_.get() == nullptr) {
            HOLOSCAN_LOG_WARN(
                "PubSubContext: provided 'clock' resource '{}' (type: {}) is not a gxf::Clock; "
                "ignoring provided clock",
                clock_resource->name(),
                typeid(*clock_resource).name());
          }
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_WARN("PubSubContext: failed to cast 'clock' argument: {}", e.what());
      }
    }
  }

  GXFNetworkContext::initialize();

  // Set up the backend now that the GXF component exists (gxf_cptr_ is set
  // by gxf_initialize() inside initialize_network_context(), called from
  // GXFNetworkContext::initialize() above).
  // Subclasses override setup_backend() to inject their own discovery/transport/serializer.
  setup_backend();
}

void PubSubContext::setup_backend() {
  // Default: no-op.  Subclasses (e.g. DDSPubSubNetworkContext) override this
  // to create and inject backend components (discovery, transport, serializer).
  HOLOSCAN_LOG_DEBUG("PubSubContext::setup_backend: no backend configured (base class default)");
}

std::string PubSubContext::node_name() const {
  if (auto* ctx = get()) {
    return ctx->node_name();
  }
  return {};
}

std::vector<nvidia::gxf::TopicInfo> PubSubContext::get_topics() const {
  if (auto* ctx = get()) {
    return ctx->get_topics();
  }
  return {};
}

size_t PubSubContext::get_publisher_count(const std::string& topic) const {
  if (auto* ctx = get()) {
    return ctx->get_publisher_count(topic);
  }
  return 0;
}

size_t PubSubContext::get_subscriber_count(const std::string& topic) const {
  if (auto* ctx = get()) {
    return ctx->get_subscriber_count(topic);
  }
  return 0;
}

std::vector<nvidia::gxf::Gid> PubSubContext::registered_publisher_gids() const {
  if (auto* ctx = get()) {
    return ctx->registered_publisher_gids();
  }
  return {};
}

std::vector<nvidia::gxf::Gid> PubSubContext::registered_subscriber_gids() const {
  if (auto* ctx = get()) {
    return ctx->registered_subscriber_gids();
  }
  return {};
}

nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Transmitter>>
PubSubContext::get_publisher_transmitter(const nvidia::gxf::Gid& gid) const {
  auto* ctx = get();
  if (ctx == nullptr) {
    return nvidia::gxf::Unexpected{GXF_CONTEXT_INVALID};
  }
  return ctx->get_publisher_transmitter(gid);
}

nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Receiver>>
PubSubContext::get_subscriber_receiver(const nvidia::gxf::Gid& gid) const {
  auto* ctx = get();
  if (ctx == nullptr) {
    return nvidia::gxf::Unexpected{GXF_CONTEXT_INVALID};
  }
  return ctx->get_subscriber_receiver(gid);
}

}  // namespace holoscan
