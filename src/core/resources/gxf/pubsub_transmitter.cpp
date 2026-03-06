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

#include "holoscan/core/resources/gxf/pubsub_transmitter.hpp"

#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

PubSubTransmitter::PubSubTransmitter(const std::string& name,
                                     nvidia::gxf::PubSubTransmitter* component)
    : Transmitter(name, component) {
  if (component == nullptr) {
    throw std::invalid_argument("PubSubTransmitter component cannot be null");
  }
  // Note: topic_name is retrieved via topic_name() method rather than caching here
  // because it may not be available until after initialize()
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (maybe_capacity) {
    capacity_ = maybe_capacity.value();
  }
  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (maybe_policy) {
    policy_ = maybe_policy.value();
  }
}

nvidia::gxf::PubSubTransmitter* PubSubTransmitter::get() const {
  return static_cast<nvidia::gxf::PubSubTransmitter*>(gxf_cptr_);
}

void PubSubTransmitter::initialize() {
  Transmitter::initialize();

  // Forward any QoS profile set via IOSpec::qos() (or directly via qos(...))
  // to the underlying GXF PubSubTransmitter.  This must happen after the GXF
  // component is created (by Transmitter::initialize() above) but before the
  // GXF component's own initialize() lifecycle method (which calls
  // register_publisher() and uses the QoS).
  if (qos_.has_value()) {
    if (auto* tx = get()) {
      tx->set_qos(qos_.value());
      HOLOSCAN_LOG_DEBUG("PubSubTransmitter: applied user QoS (reliability={}, durability={})",
                         nvidia::gxf::to_string(qos_->reliability),
                         nvidia::gxf::to_string(qos_->durability));
    }
  }
}

void PubSubTransmitter::qos(const nvidia::gxf::QoSProfile& qos) {
  if (is_initialized_) {
    HOLOSCAN_LOG_WARN(
        "PubSubTransmitter '{}' is already initialized; qos(...) must be called before "
        "initialize(). Ignoring QoS update.",
        name());
    return;
  }
  qos_ = qos;
  // application to the underlying GXF component happens during initialize()
}

void PubSubTransmitter::setup(ComponentSpec& spec) {
  spec.param(topic_name_, "topic_name", "Topic Name", "Topic to publish to");
  spec.param(capacity_, "capacity", "Capacity", "Queue capacity", 1UL);
  auto default_policy = holoscan::gxf::get_default_queue_policy();
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", default_policy);
}

std::string PubSubTransmitter::topic_name() const {
  if (auto* transmitter = get()) {
    return transmitter->topic_name();
  }
  return {};
}

size_t PubSubTransmitter::matched_subscriber_count() const {
  if (auto* transmitter = get()) {
    return transmitter->matched_subscriber_count();
  }
  return 0;
}

bool PubSubTransmitter::has_matched_subscribers() const {
  if (auto* transmitter = get()) {
    return transmitter->has_matched_subscribers();
  }
  return false;
}

std::vector<nvidia::gxf::SubscriberGid> PubSubTransmitter::matched_subscriber_gids() const {
  if (auto* transmitter = get()) {
    return transmitter->matched_subscriber_gids();
  }
  return {};
}

}  // namespace holoscan
