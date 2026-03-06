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

#include "holoscan/core/resources/gxf/pubsub_receiver.hpp"

#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

PubSubReceiver::PubSubReceiver(const std::string& name, nvidia::gxf::PubSubReceiver* component)
    : Receiver(name, component) {
  if (component == nullptr) {
    throw std::invalid_argument("PubSubReceiver component cannot be null");
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

nvidia::gxf::PubSubReceiver* PubSubReceiver::get() const {
  return static_cast<nvidia::gxf::PubSubReceiver*>(gxf_cptr_);
}

void PubSubReceiver::initialize() {
  Receiver::initialize();

  // Forward any QoS profile set via IOSpec::qos() (or directly via qos(...))
  // to the underlying GXF PubSubReceiver.  See PubSubTransmitter::initialize()
  // for timing rationale.
  if (qos_.has_value()) {
    if (auto* rx = get()) {
      rx->set_qos(qos_.value());
      HOLOSCAN_LOG_DEBUG("PubSubReceiver: applied user QoS (reliability={}, durability={})",
                         nvidia::gxf::to_string(qos_->reliability),
                         nvidia::gxf::to_string(qos_->durability));
    }
  }
}

void PubSubReceiver::qos(const nvidia::gxf::QoSProfile& qos) {
  if (is_initialized_) {
    HOLOSCAN_LOG_WARN(
        "PubSubReceiver '{}' is already initialized; qos(...) must be called before initialize(). "
        "Ignoring QoS update.",
        name());
    return;
  }
  qos_ = qos;
  // application to the underlying GXF component happens during initialize()
}

void PubSubReceiver::setup(ComponentSpec& spec) {
  spec.param(topic_name_, "topic_name", "Topic Name", "Topic to subscribe to");
  spec.param(capacity_, "capacity", "Capacity", "Queue capacity", 1UL);
  auto default_policy = holoscan::gxf::get_default_queue_policy();
  spec.param(policy_, "policy", "Policy", "0: pop, 1: reject, 2: fault", default_policy);
}

std::string PubSubReceiver::topic_name() const {
  if (auto* receiver = get()) {
    return receiver->topic_name();
  }
  return {};
}

size_t PubSubReceiver::matched_publisher_count() const {
  if (auto* receiver = get()) {
    return receiver->matched_publisher_count();
  }
  return 0;
}

bool PubSubReceiver::has_matched_publishers() const {
  if (auto* receiver = get()) {
    return receiver->has_matched_publishers();
  }
  return false;
}

std::vector<nvidia::gxf::PublisherGid> PubSubReceiver::matched_publisher_gids() const {
  if (auto* receiver = get()) {
    return receiver->matched_publisher_gids();
  }
  return {};
}

}  // namespace holoscan
