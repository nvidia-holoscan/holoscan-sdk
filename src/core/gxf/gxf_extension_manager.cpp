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

#include "holoscan/core/gxf/gxf_extension_manager.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>

namespace holoscan::gxf {

// Prefix list to search for the extension
static const std::vector<std::string> kExtensionSearchPrefixes{"", "gxf_extensions"};

GXFExtensionManager::GXFExtensionManager(gxf_context_t context) : ExtensionManager(context) {
  refresh();
}

GXFExtensionManager::~GXFExtensionManager() {
  for (auto& handle : extension_handles_) { dlclose(handle); }
}

void GXFExtensionManager::refresh() {
  if (context_ == nullptr) { return; }

  // Configure request data
  runtime_info_ = {nullptr, kGXFExtensionsMaxSize, extension_tid_list_};

  gxf_result_t result = GxfRuntimeInfo(context_, &runtime_info_);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to get runtime info: {}", GxfResultStr(result));
    return;
  }
  auto& num_extensions = runtime_info_.num_extensions;
  auto& extensions = runtime_info_.extensions;

  // Clear the extension handles
  extension_tids_.clear();

  // Add the extension tids to the set
  for (uint64_t i = 0; i < num_extensions; ++i) { extension_tids_.insert(extensions[i]); }
}

bool GXFExtensionManager::load_extension(const std::string& file_name, bool no_error_message,
                                         const std::string& search_path_envs) {
  // Skip if file name is empty
  if (file_name.empty() || file_name == "null") {
    HOLOSCAN_LOG_DEBUG("Extension filename is empty. Skipping extension loading.");
    return true;  // return true to avoid breaking the pipeline
  }

  HOLOSCAN_LOG_DEBUG("Loading extension from '{}'", file_name);

  // We set RTLD_NODELETE to avoid unloading the extension when the library handle is closed.
  // This is because it can cause a crash when the extension is used by another library or it
  // uses thread_local variables.
  void* handle = dlopen(file_name.c_str(), RTLD_LAZY | RTLD_NODELETE);
  if (handle == nullptr) {
    // Try to load the extension from the environment variable (search_path_env) indicating the
    // folder where the extension is located.
    if (!search_path_envs.empty()) {
      bool found_extension = false;
      const auto search_path_env_list = tokenize(search_path_envs, ",");
      std::filesystem::path file_path = std::filesystem::path(file_name);
      for (const auto& search_path_env : search_path_env_list) {
        auto env_value = std::getenv(search_path_env.c_str());
        if (env_value != nullptr) {
          const auto search_paths_str = std::string(env_value);
          HOLOSCAN_LOG_DEBUG(
              "Extension search path found in the env({}): {}", search_path_env, search_paths_str);
          // Tokenize the search path and try to load the extension from each
          // path in the search path. We stop when the extension is loaded successfully or when
          // we run out of paths.
          const auto search_paths = tokenize(search_paths_str, ":");

          std::vector<std::string> base_names;
          // Try to load the extension from the full path of the file (if it is relative)
          if (file_path.is_relative() &&
              (file_path.filename().lexically_normal() != file_path.lexically_normal())) {
            base_names.push_back(file_path);
          }
          // Try to load the extension from the base name of the file
          base_names.push_back(file_path.filename());

          for (const auto& search_path : search_paths) {
            for (const auto& prefix : kExtensionSearchPrefixes) {
              const std::filesystem::path prefix_path{prefix};
              for (const auto& base_name : base_names) {
                const std::filesystem::path candidate_parent_path = search_path / prefix_path;
                const std::filesystem::path candidate_path = candidate_parent_path / base_name;
                if (std::filesystem::exists(candidate_path)) {
                  HOLOSCAN_LOG_DEBUG("Trying extension {} found in search path {}",
                                     base_name.c_str(),
                                     candidate_parent_path.c_str());
                  handle = dlopen(candidate_path.c_str(), RTLD_LAZY | RTLD_NODELETE);
                  if (handle != nullptr) {
                    HOLOSCAN_LOG_DEBUG("Loaded extension {} from search path '{}'",
                                       base_name.c_str(),
                                       candidate_parent_path.c_str());
                    found_extension = true;
                    break;
                  }
                }
              }
            }
            if (found_extension) { break; }
          }
        }
        if (found_extension) { break; }
      }
    }
    if (handle == nullptr) {
      if (!no_error_message) {
        HOLOSCAN_LOG_WARN("Unable to load extension from '{}' (error: {})", file_name, dlerror());
      }
      return false;
    }
  } else {
    HOLOSCAN_LOG_DEBUG("Loaded extension {}", file_name);
  }
  void* func_ptr = dlsym(handle, kGxfExtensionFactoryName);
  if (func_ptr == nullptr) {
    if (!no_error_message) {
      HOLOSCAN_LOG_ERROR(
          "Unable to find extension factory in '{}' (error: {})", file_name, dlerror());
    }
    dlclose(handle);
    return false;
  }

  // Get the factory function
  const auto factory_func = reinterpret_cast<GxfExtensionFactory*>(func_ptr);

  // Get the extension from the factory
  void* result = nullptr;
  gxf_result_t code = factory_func(&result);
  if (code != GXF_SUCCESS) {
    if (!no_error_message) {
      HOLOSCAN_LOG_ERROR(
          "Failed to create extension from '{}' (error: {})", file_name, GxfResultStr(code));
    }
    dlclose(handle);
    return false;
  }
  nvidia::gxf::Extension* extension = static_cast<nvidia::gxf::Extension*>(result);

  // Load the extension from the pointer
  const bool is_loaded = load_extension(extension, handle);
  if (!is_loaded) {
    HOLOSCAN_LOG_ERROR("Unable to load extension from '{}'", file_name);
    dlclose(handle);
    return false;
  }
  return true;
}

bool GXFExtensionManager::load_extensions_from_yaml(const YAML::Node& node, bool no_error_message,
                                                    const std::string& search_path_envs,
                                                    const std::string& key) {
  try {
    for (const auto& entry : node[key.c_str()]) {
      auto file_name = entry.as<std::string>();
      auto result = load_extension(file_name, no_error_message, search_path_envs);
      if (!result) { return false; }
    }
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error loading extension from yaml: {}", e.what());
    return false;
  }
  return true;
}

bool GXFExtensionManager::load_extension(nvidia::gxf::Extension* extension, void* handle) {
  if (extension == nullptr) {
    HOLOSCAN_LOG_DEBUG("Extension pointer is null. Skipping extension loading.");
    return true;  // return true to avoid breaking the pipeline
  }

  // Check if the extension is already loaded
  gxf_extension_info_t info;
  info.num_components = 0;
  auto info_result = extension->getInfo(&info);
  if (!info_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to get extension info");
    return false;
  }

  // Check and ignore if the extension is already loaded
  if (extension_tids_.find(info.id) != extension_tids_.end()) {
    HOLOSCAN_LOG_DEBUG("Extension '{}' is already loaded. Skipping extension loading.", info.name);
    return true;  // return true to avoid breaking the pipeline
  }

  extension_tids_.insert(info.id);

  // Load the extension
  gxf_result_t result = GxfLoadExtensionFromPointer(context_, extension);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Unable to load extension: {}", GxfResultStr(result));
    return false;
  }

  // Add the extension handle to the set of handles to be closed when the manager is destroyed
  if (handle != nullptr) { extension_handles_.insert(handle); }

  return true;
}

bool GXFExtensionManager::is_extension_loaded(gxf_tid_t tid) {
  return extension_tids_.find(tid) != extension_tids_.end();
}

std::vector<std::string> GXFExtensionManager::tokenize(const std::string& str,
                                                       const std::string& delimiters) {
  std::vector<std::string> search_paths;
  if (str.size() == 0) { return search_paths; }

  // Find the number of possible paths based on str and delimiters
  const auto possible_path_count =
      std::count_if(str.begin(),
                    str.end(),
                    [&delimiters](char value) {
                      return delimiters.find_first_of(value) != std::string::npos;
                    }) +
      1;
  search_paths.reserve(possible_path_count);

  size_t start = 0;
  size_t end = 0;
  while ((start = str.find_first_not_of(delimiters, end)) != std::string::npos) {
    end = str.find_first_of(delimiters, start);
    search_paths.push_back(str.substr(start, end - start));
  }
  return search_paths;
}

}  // namespace holoscan::gxf
