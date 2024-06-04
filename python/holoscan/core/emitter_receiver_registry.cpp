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

#include "./emitter_receiver_registry.hpp"

#include <string>
#include <vector>

namespace holoscan {

EmitterReceiverRegistry& EmitterReceiverRegistry::get_instance() {
  static EmitterReceiverRegistry instance;
  return instance;
}

const EmitterReceiverRegistry::EmitterReceiver& EmitterReceiverRegistry::get_emitter_receiver(
    const std::type_index& index) const {
  auto maybe_name = index_to_name(index);
  if (!maybe_name) {
    HOLOSCAN_LOG_WARN("No emitter_receiver for type '{}' exists", index.name());
    return EmitterReceiverRegistry::none_emitter_receiver;
  }
  auto& emitter_receiver = emitter_receiver_map_.at(maybe_name.value());
  return emitter_receiver;
}

bool EmitterReceiverRegistry::has_emitter_receiver(const std::type_index& index) const {
  auto maybe_name = index_to_name(index);
  if (maybe_name) { return emitter_receiver_map_.count(maybe_name.value()) > 0 ? true : false; }
  return false;
}

const EmitterReceiverRegistry::EmitterReceiver& EmitterReceiverRegistry::get_emitter_receiver(
    const std::string& name) const {
  auto loc = emitter_receiver_map_.find(name);
  if (loc == emitter_receiver_map_.end()) {
    HOLOSCAN_LOG_WARN("No emitter_receiver for name '{}' exists", name);
    return EmitterReceiverRegistry::none_emitter_receiver;
  }
  auto& emitter_receiver = loc->second;
  return emitter_receiver;
}

const EmitterReceiverRegistry::EmitFunc& EmitterReceiverRegistry::get_emitter(
    const std::string& name) const {
  auto loc = emitter_receiver_map_.find(name);
  if (loc == emitter_receiver_map_.end()) {
    HOLOSCAN_LOG_WARN("No emitter for name '{}' exists", name);
    return EmitterReceiverRegistry::none_emit;
  }
  auto& emitter_receiver = loc->second;
  return emitter_receiver.first;
}

const EmitterReceiverRegistry::EmitFunc& EmitterReceiverRegistry::get_emitter(
    const std::type_index& index) const {
  auto maybe_name = index_to_name(index);
  if (!maybe_name) {
    HOLOSCAN_LOG_WARN("No emitter for type '{}' exists", index.name());
    return EmitterReceiverRegistry::none_emit;
  }
  auto& emitter = emitter_receiver_map_.at(maybe_name.value()).first;
  return emitter;
}

const EmitterReceiverRegistry::ReceiveFunc& EmitterReceiverRegistry::get_receiver(
    const std::string& name) const {
  auto loc = emitter_receiver_map_.find(name);
  if (loc == emitter_receiver_map_.end()) {
    HOLOSCAN_LOG_WARN("No receiver for name '{}' exists", name);
    return EmitterReceiverRegistry::none_receive;
  }
  auto& emitter_receiver = loc->second;
  return emitter_receiver.second;
}

const EmitterReceiverRegistry::ReceiveFunc& EmitterReceiverRegistry::get_receiver(
    const std::type_index& index) const {
  auto maybe_name = index_to_name(index);
  if (!maybe_name) {
    HOLOSCAN_LOG_WARN("No receiver for type '{}' exists", index.name());
    return EmitterReceiverRegistry::none_receive;
  }
  auto& receiver = emitter_receiver_map_.at(maybe_name.value()).second;
  return receiver;
}

expected<std::type_index, RuntimeError> EmitterReceiverRegistry::name_to_index(
    const std::string& name) const {
  auto loc = name_to_index_map_.find(name);
  if (loc == name_to_index_map_.end()) {
    auto err_msg = fmt::format("No emitter_receiver for name '{}' exists", name);
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kFailure, err_msg));
  }
  return loc->second;
}

expected<std::string, RuntimeError> EmitterReceiverRegistry::index_to_name(
    const std::type_index& index) const {
  auto loc = index_to_name_map_.find(index);
  if (loc == index_to_name_map_.end()) {
    auto err_msg = fmt::format("No emitter_receiver for type '{}' exists", index.name());
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kFailure, err_msg));
  }
  return loc->second;
}

std::vector<std::string> EmitterReceiverRegistry::registered_types() const {
  std::vector<std::string> names;
  names.reserve(emitter_receiver_map_.size());
  for (auto& [key, _] : emitter_receiver_map_) { names.emplace_back(key); }
  return names;
}

}  // namespace holoscan
