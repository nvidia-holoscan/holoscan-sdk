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

#ifndef HOLOSCAN_CORE_EXTENSION_MANAGER_HPP
#define HOLOSCAN_CORE_EXTENSION_MANAGER_HPP

#include <yaml-cpp/yaml.h>

#include <set>
#include <string>

#include "holoscan/core/common.hpp"

namespace holoscan {

/**
 * @brief Class to manage extensions.
 *
 * This class is a helper class to manage extensions.
 *
 */
class ExtensionManager {
 public:
  /**
   * @brief Construct a new ExtensionManager object.
   *
   * @param context The context.
   */
  explicit ExtensionManager(void* context) : context_(context) {}

  /**
   * @brief Destroy the ExtensionManager object.
   *
   */
  virtual ~ExtensionManager() = default;

  /**
   * @brief Refresh the extension list.
   *
   * Based on the current context, construct the internal extension list.
   */
  virtual void refresh() {}

  /**
   * @brief Load an extension.
   *
   * This method loads an extension and stores the extension handler so that it can be
   * unloaded when the class is destroyed.
   *
   * @param file_name The file name of the extension (e.g. libmyextension.so).
   * @param no_error_message If true, no error message will be printed if the extension is not
   * found.
   * @param search_path_envs The environment variable names that contains the search paths for the
   * extension. The environment variable names are separated by a comma (,). (default:
   * "HOLOSCAN_LIB_PATH").
   * @return true if the extension is loaded successfully, false otherwise.
   */
  virtual bool load_extension(const std::string& file_name, bool no_error_message = false,
                              const std::string& search_path_envs = "HOLOSCAN_LIB_PATH") {
    (void)file_name;
    (void)no_error_message;
    (void)search_path_envs;
    return false;
  }

  /**
   * @brief Load extensions from a yaml file.
   *
   * The yaml file should contain a list of extension file names under the key "extensions".
   *
   * For example:
   *
   * ```yaml
   * extensions:
   * - /path/to/extension1.so
   * - /path/to/extension2.so
   * - /path/to/extension3.so
   * ```
   *
   * @param node The yaml node.
   * @param no_error_message  If true, no error message will be printed if the extension is not
   * found.
   * @param search_path_envs The environment variable names that contains the search paths for the
   * extension. The environment variable names are separated by a comma (,). (default:
   * "HOLOSCAN_LIB_PATH").
   * @param key The key in the yaml node that contains the extension file names (default:
   * "extensions").
   * @return true if the extension is loaded successfully, false otherwise.
   */
  virtual bool load_extensions_from_yaml(const YAML::Node& node, bool no_error_message = false,
                                         const std::string& search_path_envs = "HOLOSCAN_LIB_PATH",
                                         const std::string& key = "extensions") {
    (void)node;
    (void)no_error_message;
    (void)search_path_envs;
    (void)key;
    return false;
  }

 protected:
  void* context_ = nullptr;  ///< The context
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXTENSION_MANAGER_HPP */
