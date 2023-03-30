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

#ifndef INCLUDE_HOLOSCAN_CORE_GXF_GXF_EXTENSION_MANAGER_HPP
#define INCLUDE_HOLOSCAN_CORE_GXF_GXF_EXTENSION_MANAGER_HPP

#include <yaml-cpp/yaml.h>

#include <set>
#include <string>
#include <vector>

#include "gxf/core/gxf.h"
#include "gxf/std/extension.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/extension_manager.hpp"

namespace holoscan::gxf {

namespace {
// Method name to get the GXF extension factory
constexpr const char* kGxfExtensionFactoryName = "GxfExtensionFactory";
// Max size of extensions
constexpr int kGXFExtensionsMaxSize = 1024;
// Method signature for the GXF extension factory
using GxfExtensionFactory = gxf_result_t(void**);
}  // namespace

/**
 * @brief Class to manage GXF extensions.
 *
 * This class is a helper class to manage GXF extensions.
 *
 * Since GXF API doesn't provide a way to ignore duplicate extensions, this class is used to
 * manage the extensions, prevent duplicate extensions from being loaded, and unload the
 * extension handlers when the class is destroyed.
 *
 * Example:
 *
 * ```cpp
 * #include <yaml-cpp/yaml.h>
 * #include "holoscan/core/gxf/gxf_extension_manager.hpp"
 * ...
 *
 * holoscan::gxf::GXFExtensionManager extension_manager(reinterpret_cast<gxf_context_t>(context));
 * // Load extensions from a file
 * for (const auto& extension_filename : extension_filenames) {
 *   extension_manager.load_extension(extension_filename);
 * }
 * // Load extensions from a yaml node (YAML::Node object)
 * for (const auto& manifest_filename : manifest_filenames) {
 *   auto node = YAML::LoadFile(manifest_filename);
 *   extension_manager.load_extensions_from_yaml(node);
 * }
 * ```
 */
class GXFExtensionManager : public ExtensionManager {
 public:
  /**
   * @brief Construct a new GXFExtensionManager object.
   *
   * @param context The GXF context
   */
  explicit GXFExtensionManager(gxf_context_t context);

  /**
   * @brief Destroy the GXFExtensionManager object.
   *
   * This method closes all the extension handles that are loaded by this class.
   * Note that the shared library is opened with `RTLD_NODELETE` flag, so the library is not
   * unloaded when the handle is closed.
   */
  ~GXFExtensionManager() override;

  /**
   * @brief Refresh the extension list.
   *
   * Based on the current GXF context, it gets the list of extensions and stores the type IDs
   * of the extensions so that duplicate extensions can be ignored.
   */
  void refresh() override;

  /**
   * @brief Load an extension.
   *
   * This method loads an extension and stores the extension handler so that it can be
   * unloaded when the class is destroyed.
   *
   * @param file_name The file name of the extension (e.g. libgxf_std.so).
   * @param no_error_message If true, no error message will be printed if the extension is not
   * found.
   * @param search_path_envs The environment variable names that contains the search paths for the
   * extension. The environment variable names are separated by a comma (,). (default:
   * "HOLOSCAN_LIB_PATH").
   * @return true if the extension is loaded successfully, false otherwise.
   */
  bool load_extension(const std::string& file_name, bool no_error_message = false,
                      const std::string& search_path_envs = "HOLOSCAN_LIB_PATH") override;

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
  bool load_extensions_from_yaml(const YAML::Node& node, bool no_error_message = false,
                                 const std::string& search_path_envs = "HOLOSCAN_LIB_PATH",
                                 const std::string& key = "extensions") override;

  /**
   * @brief Load an extension from a pointer.
   *
   * `GxfLoadExtensionFromPointer()` API is used to register the extension programmatically.
   *
   * @param extension The extension pointer to load.
   * @param handle The handle of the extension library.
   * @return true if the extension is loaded successfully.
   */
  bool load_extension(nvidia::gxf::Extension* extension, void* handle = nullptr);

  /**
   * @brief Check if the extension is loaded.
   *
   * @param tid The type ID of the extension.
   * @return true if the extension is loaded. false otherwise.
   */
  bool is_extension_loaded(gxf_tid_t tid);

  /**
   * @brief Tokenize a string.
   *
   * @param str The string to tokenize.
   * @param delimiters The delimiters.
   * @return The vector of tokens.
   */
  static std::vector<std::string> tokenize(const std::string& str, const std::string& delimiters);

 protected:
  /// Storage for the extension TIDs
  gxf_tid_t extension_tid_list_[kGXFExtensionsMaxSize] = {};
  /// request/response structure for the runtime info
  gxf_runtime_info runtime_info_{nullptr, kGXFExtensionsMaxSize, extension_tid_list_};

  std::set<gxf_tid_t> extension_tids_;  ///< Set of extension TIDs
  std::set<void*> extension_handles_;   ///< Set of extension handles
};

}  // namespace holoscan::gxf

#endif /* INCLUDE_HOLOSCAN_CORE_GXF_GXF_EXTENSION_MANAGER_HPP */
