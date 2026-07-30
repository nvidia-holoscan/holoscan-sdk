/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/config.hpp>

#include <yaml-cpp/yaml.h>

#include <string>
#include <unordered_set>
#include <vector>

#include <holoscan/core/arg.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {

namespace {  // anonymous namespace for helper functions

std::unordered_set<std::string> nested_yaml_map_keys(YAML::Node yaml_node) {
  std::unordered_set<std::string> keys;
  for (auto it = yaml_node.begin(); it != yaml_node.end(); ++it) {
    const auto key = it->first.as<std::string>();
    // Copy to extend lifetime; iterator returns a temporary YAML::Node.
    // Using a reference here can trigger dangling-pointer warnings on arm64.
    const YAML::Node value = it->second;
    keys.emplace(key);
    if (value.IsMap()) {
      std::unordered_set<std::string> inner_keys = nested_yaml_map_keys(value);
      for (const auto& inner_key : inner_keys) {
        keys.emplace(fmt::format("{}.{}", key, inner_key));
      }
    }
  }
  return keys;
}

}  // namespace

void Config::parse_file(const std::string& config_file) {
  try {
    yaml_nodes_ = YAML::LoadAllFromFile(config_file);
  } catch (const YAML::Exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to load config file: '{}' ({})", config_file, e.what());
  }
}

ArgList Config::from_config(const std::string& key) {
  ArgList args;

  std::vector<std::string> key_parts;

  size_t pos = 0;
  while (pos != std::string::npos) {
    size_t next_pos = key.find_first_of('.', pos);
    if (next_pos == std::string::npos) {
      break;
    }
    key_parts.push_back(key.substr(pos, next_pos - pos));
    pos = next_pos + 1;
  }
  key_parts.push_back(key.substr(pos));

  size_t key_parts_size = key_parts.size();

  for (const auto& yaml_node : yaml_nodes_) {
    if (yaml_node.IsMap()) {
      auto yaml_map = yaml_node.as<YAML::Node>();
      size_t key_index = 0;
      for (const auto& key_part : key_parts) {
        (void)key_part;
        if (yaml_map.IsMap()) {
          yaml_map.reset(yaml_map[key_part]);
          ++key_index;
        } else {
          break;
        }
      }
      if (!yaml_map || key_index < key_parts_size) {
        HOLOSCAN_LOG_ERROR("Unable to find the parameter item/map with key '{}'", key);
        continue;
      }

      const auto& parameters = yaml_map;

      if (parameters.IsScalar()) {
        const std::string& param_key = key_parts[key_parts_size - 1];
        auto& value = parameters;
        args.add(Arg(param_key) = value);
        continue;
      }

      for (const auto& p : parameters) {
        const std::string param_key = p.first.as<std::string>();
        auto& value = p.second;
        args.add(Arg(param_key) = value);
      }
    }
  }

  return args;
}

std::unordered_set<std::string> Config::config_keys() {
  std::unordered_set<std::string> all_keys;
  for (const auto& yaml_node : yaml_nodes_) {
    if (yaml_node.IsMap()) {
      auto node_keys = nested_yaml_map_keys(yaml_node);
      for (const auto& k : node_keys) {
        all_keys.insert(k);
      }
    }
  }
  return all_keys;
}

}  // namespace holoscan
