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

#ifndef HOLOSCAN_CORE_CONFIG_HPP
#define HOLOSCAN_CORE_CONFIG_HPP

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "./common.hpp"

namespace holoscan {

// Forward declaration
class ArgList;

/**
 * @brief Class to get the configuration of the application.
 */
class Config {
 public:
  Config() = default;
  /**
   * @brief Construct a new Config object
   *
   * @param config_file The path to the configuration file.
   * @param prefix The prefix string that is prepended to the key of the configuration. (not
   * implemented yet)
   * @throws RuntimeError if the config_file is non-empty and the file doesn't exist.
   */
  explicit Config(const std::string& config_file, const std::string& prefix = "")
      : config_file_(config_file), prefix_(prefix) {
    if (std::filesystem::exists(config_file)) {
      parse_file(config_file);
    } else if (config_file != "") {
      throw RuntimeError(ErrorCode::kNotFound,
                         fmt::format("Config file '{}' doesn't exist", config_file));
    }
  }
  virtual ~Config() = default;

  // Delete the copy constructor and assignment operator to prevent copying.
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;

  /**
   * @brief Get the path to the configuration file.
   *
   * @return The path to the configuration file.
   */
  const std::string& config_file() const { return config_file_; }
  /**
   * @brief Get the prefix string that is prepended to the key of the configuration.
   *
   * @return The prefix string that is prepended to the key of the configuration.
   */
  const std::string& prefix() const { return prefix_; }
  /**
   * @brief Get the YAML::Node objects that contains YAML document data.
   *
   * @return The reference to the vector of YAML::Node objects.
   */
  const std::vector<YAML::Node>& yaml_nodes() const { return yaml_nodes_; }

  /**
   * @brief Get the value of a configuration key as an ArgList.
   *
   * This method retrieves the value from the configuration for the given key.
   * You can use '.' (dot) to access nested fields.
   *
   * @param key The key of the configuration.
   * @return The argument list of the configuration for the key.
   */
  ArgList from_config(const std::string& key);

  /**
   * @brief Determine the set of keys present in the config.
   *
   * Returns all keys including nested keys using dot notation (e.g., "parent.child").
   *
   * @return The set of valid keys.
   */
  std::unordered_set<std::string> config_keys();

 private:
  void parse_file(const std::string& config_file);

  std::string config_file_;
  std::string prefix_;
  std::vector<YAML::Node> yaml_nodes_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONFIG_HPP */
