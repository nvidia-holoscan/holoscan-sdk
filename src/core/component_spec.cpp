/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <holoscan/core/component_spec.hpp>

#include <string>
#include <unordered_map>

#include <holoscan/core/fragment.hpp>

using std::string_literals::operator""s;

namespace holoscan {

YAML::Node ComponentSpec::to_yaml_node() const {
  YAML::Node node;

  std::unordered_map<ParameterFlag, std::string> parameterflag_namemap{
      {ParameterFlag::kNone, "kNone"s},
      {ParameterFlag::kOptional, "kOptional"s},
      {ParameterFlag::kDynamic, "kDynamic"s},
  };

  if (fragment_) {
    node["fragment"] = fragment_->name();
  } else {
    node["fragment"] = YAML::Null;
  }
  node["params"] = YAML::Node(YAML::NodeType::Sequence);
  for (auto& [name, wrapper] : params_) {
    const std::string& type = wrapper.arg_type().to_string();
    auto param = static_cast<Parameter<void*>*>(wrapper.storage_ptr());
    YAML::Node param_node;
    param_node["name"] = name;
    param_node["type"] = type;
    param_node["description"] = param ? param->description() : "";
    param_node["flag"] = parameterflag_namemap[param ? param->flag() : ParameterFlag::kNone];
    node["params"].push_back(param_node);
  }
  return node;
}

std::string ComponentSpec::description() const {
  YAML::Emitter emitter;
  emitter << to_yaml_node();
  return emitter.c_str();
}

}  // namespace holoscan
