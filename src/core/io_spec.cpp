/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/io_spec.hpp>

#include <memory>
#include <string>
#include <unordered_map>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/resources/gxf/pubsub_receiver.hpp>
#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>

using std::string_literals::operator""s;

namespace {

std::optional<std::string> topic_name_from_connector_args(
    const std::shared_ptr<holoscan::Resource>& connector) {
  if (!connector) {
    return std::nullopt;
  }

  auto& args = connector->args();
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    if (it->name() != "topic_name" || !it->has_value()) {
      continue;
    }
    try {
      return std::any_cast<std::string>(it->value());
    } catch (const std::bad_any_cast&) {
      HOLOSCAN_LOG_WARN("IOSpec: connector '{}' has a non-string topic_name argument",
                        connector->name());
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<holoscan::Arg> named_arg_from_connector_args(
    const std::shared_ptr<holoscan::Resource>& connector, const std::string& arg_name) {
  if (!connector) {
    return std::nullopt;
  }

  auto& args = connector->args();
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    if (it->name() == arg_name) {
      return *it;
    }
  }
  return std::nullopt;
}

std::optional<nvidia::gxf::QoSProfile> qos_profile_from_connector_args(
    const std::shared_ptr<holoscan::Resource>& connector) {
  if (!connector) {
    return std::nullopt;
  }

  auto qos_arg = named_arg_from_connector_args(connector, "qos_profile");
  if (!qos_arg.has_value() || !qos_arg->has_value()) {
    return std::nullopt;
  }
  try {
    return std::any_cast<nvidia::gxf::QoSProfile>(qos_arg->value());
  } catch (const std::bad_any_cast&) {
    HOLOSCAN_LOG_WARN("IOSpec: connector '{}' has a non-QoS qos_profile argument",
                      connector->name());
    return std::nullopt;
  }
}

}  // namespace

namespace holoscan {

IOSpec& IOSpec::qos(const nvidia::gxf::QoSProfile& profile) {
  switch (connector_type_) {
    case ConnectorType::kDefault:
      // Automatically switch to PubSub.
      connector_type_ = ConnectorType::kPubSub;
      break;
    case ConnectorType::kPubSub:
      break;
    default:
      HOLOSCAN_LOG_WARN(
          "qos() is ignored for non-PubSub connector on port '{}'. "
          "QoS profiles are only used with ConnectorType::kPubSub.",
          name_);
      return *this;
  }

  // Ensure the connector exists, then set the QoS profile.
  if (!connector_) {
    if (io_type_ == IOType::kInput) {
      auto rx = std::make_shared<PubSubReceiver>();
      rx->qos(profile);
      connector_ = rx;
    } else {
      auto tx = std::make_shared<PubSubTransmitter>();
      tx->qos(profile);
      connector_ = tx;
    }
  } else {
    // Connector already exists (e.g. from a prior topic() call) — set QoS on it.
    if (io_type_ == IOType::kInput) {
      auto rx = std::dynamic_pointer_cast<PubSubReceiver>(connector_);
      if (rx) {
        rx->qos(profile);
      } else {
        HOLOSCAN_LOG_ERROR("qos(): connector on input port '{}' is not a PubSubReceiver", name_);
      }
    } else {
      auto tx = std::dynamic_pointer_cast<PubSubTransmitter>(connector_);
      if (tx) {
        tx->qos(profile);
      } else {
        HOLOSCAN_LOG_ERROR("qos(): connector on output port '{}' is not a PubSubTransmitter",
                           name_);
      }
    }
  }
  return *this;
}

IOSpec& IOSpec::topic(const std::string& name, bool replace_connector) {
  switch (connector_type_) {
    case ConnectorType::kDefault:
      // Automatically switch to PubSub.
      connector_type_ = ConnectorType::kPubSub;
      break;
    case ConnectorType::kPubSub:
      break;
    default:
      if (replace_connector) {
        ArgList args;
        args.add(Arg("topic_name", name));

        if (auto capacity = named_arg_from_connector_args(connector_, "capacity");
            capacity.has_value()) {
          args.add(*capacity);
        } else if (io_type_ == IOType::kInput) {
          const auto queue_size = queue_size_.size();
          if (queue_size > 0) {
            args.add(Arg("capacity", static_cast<uint64_t>(queue_size)));
          }
        }

        if (auto policy = named_arg_from_connector_args(connector_, "policy"); policy.has_value()) {
          args.add(*policy);
        } else if (queue_policy_.has_value()) {
          args.add(Arg("policy", static_cast<uint64_t>(queue_policy_.value())));
        }

        connector_type_ = ConnectorType::kPubSub;
        if (io_type_ == IOType::kInput) {
          connector_ = std::make_shared<PubSubReceiver>(args);
        } else {
          connector_ = std::make_shared<PubSubTransmitter>(args);
        }
        return *this;
      }
      HOLOSCAN_LOG_WARN(
          "topic('{}') is ignored for non-PubSub connector on port '{}'. "
          "Topic names are only used with ConnectorType::kPubSub unless "
          "replace_connector=true is specified.",
          name,
          name_);
      return *this;
  }

  // Ensure the connector exists and set the topic_name argument.
  if (!connector_) {
    if (io_type_ == IOType::kInput) {
      connector_ = std::make_shared<PubSubReceiver>(Arg("topic_name", name));
    } else {
      connector_ = std::make_shared<PubSubTransmitter>(Arg("topic_name", name));
    }
  } else {
    auto& args = connector_->args();
    bool found_topic_name = false;
    for (auto it = args.begin(); it != args.end();) {
      if (it->name() != "topic_name") {
        ++it;
        continue;
      }
      if (!found_topic_name) {
        *it = Arg("topic_name", name);
        found_topic_name = true;
        ++it;
      } else {
        it = args.erase(it);
      }
    }
    if (!found_topic_name) {
      connector_->add_arg(Arg("topic_name", name));
    }
  }
  return *this;
}

std::optional<std::string> IOSpec::topic() const {
  if (connector_type_ != ConnectorType::kPubSub || !connector_) {
    return std::nullopt;
  }

  if (auto topic_name = topic_name_from_connector_args(connector_); topic_name.has_value()) {
    return topic_name;
  }

  if (io_type_ == IOType::kInput) {
    auto rx = std::dynamic_pointer_cast<PubSubReceiver>(connector_);
    if (!rx) {
      return std::nullopt;
    }
    const auto runtime_topic_name = rx->topic_name();
    if (runtime_topic_name.empty()) {
      return std::nullopt;
    }
    return runtime_topic_name;
  }

  auto tx = std::dynamic_pointer_cast<PubSubTransmitter>(connector_);
  if (!tx) {
    return std::nullopt;
  }
  const auto runtime_topic_name = tx->topic_name();
  if (runtime_topic_name.empty()) {
    return std::nullopt;
  }
  return runtime_topic_name;
}

YAML::Node IOSpec::to_yaml_node() const {
  YAML::Node node;

  std::unordered_map<IOType, std::string> iotype_namemap{
      {IOType::kInput, "kInput"s},
      {IOType::kOutput, "kOutput"s},
  };

  std::unordered_map<ConnectorType, std::string> connectortype_namemap{
      {ConnectorType::kDefault, "kDefault"s},
      {ConnectorType::kDoubleBuffer, "kDoubleBuffer"s},
      {ConnectorType::kAsyncBuffer, "kAsyncBuffer"s},
      {ConnectorType::kUCX, "kUCX"s},
      {ConnectorType::kPubSub, "kPubSub"s},
  };

  std::unordered_map<ConditionType, std::string> conditiontype_namemap{
      {ConditionType::kNone, "kNone"s},
      {ConditionType::kMessageAvailable, "kMessageAvailable"s},
      {ConditionType::kDownstreamMessageAffordable, "kDownstreamMessageAffordable"s},
      {ConditionType::kCount, "kCount"s},
      {ConditionType::kBoolean, "kBoolean"s},
      {ConditionType::kPeriodic, "kPeriodic"s},
      {ConditionType::kAsynchronous, "kAsynchronous"s},
  };

  node["name"] = name();
  node["io_type"] = iotype_namemap[io_type()];
  node["typeinfo_name"] = std::string{typeinfo()->name()};
  node["connector_type"] = connectortype_namemap[connector_type()];
  if (!unique_id_.empty()) {
    node["unique_id"] = unique_id_;
  }
  auto conn = connector();
  if (conn) {
    node["connector"] = conn->to_yaml_node();
  }
  node["conditions"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& c : conditions_) {
    if (c.first == ConditionType::kNone) {
      YAML::Node none_condition = YAML::Node(YAML::NodeType::Map);
      none_condition["type"] = conditiontype_namemap[c.first];
      node["conditions"].push_back(none_condition);
    } else {
      if (c.second) {
        YAML::Node condition = c.second->to_yaml_node();
        condition["type"] = conditiontype_namemap[c.first];
        node["conditions"].push_back(condition);
      }
    }
  }
  return node;
}

std::string IOSpec::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

std::optional<nvidia::gxf::QoSProfile> IOSpec::qos() const {
  if (connector_type_ != ConnectorType::kPubSub || !connector_) {
    return std::nullopt;
  }

  if (auto qos_profile = qos_profile_from_connector_args(connector_); qos_profile.has_value()) {
    return qos_profile;
  }

  if (io_type_ == IOType::kInput) {
    auto rx = std::dynamic_pointer_cast<PubSubReceiver>(connector_);
    if (!rx) {
      return std::nullopt;
    }
    return rx->qos();
  }

  auto tx = std::dynamic_pointer_cast<PubSubTransmitter>(connector_);
  if (!tx) {
    return std::nullopt;
  }
  return tx->qos();
}

}  // namespace holoscan
