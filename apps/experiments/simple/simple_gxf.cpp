/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sstream>

#include <gxf/core/gxf.h>
#include <common/assert.hpp>
#include <common/logger.hpp>

// Split strings
std::vector<std::string> SplitStrings(const std::string& list_of_files) {
  std::vector<std::string> filenames;
  char delimiter = ',';
  std::istringstream stream(list_of_files);
  std::string item;
  while (std::getline(stream, item, delimiter)) { filenames.push_back(item); }

  return filenames;
}

// Loads application graph file(s)
gxf_result_t LoadApplication(gxf_context_t context, const std::string& list_of_files) {
  const auto filenames = SplitStrings(list_of_files);

  if (filenames.empty()) {
    GXF_LOG_ERROR("Atleast one application file has to be specified using -app");
    return GXF_FILE_NOT_FOUND;
  }

  for (const auto& filename : filenames) {
    GXF_LOG_INFO("Loading app: '%s'", filename.c_str());
    const gxf_result_t code = GxfGraphLoadFile(context, filename.c_str());
    if (code != GXF_SUCCESS) { return code; }
  }

  return GXF_SUCCESS;
}

int main() {
  gxf_context_t context;
  GXF_LOG_INFO("Creating context");
  GXF_ASSERT_SUCCESS(GxfContextCreate(&context));

  const char* manifest_filename = "./apps/endoscopy_tool_tracking_gxf/tracking_replayer_manifest.yaml";
  const char* graph_file = "./apps/endoscopy_tool_tracking_gxf/tracking_replayer.yaml";
  const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &manifest_filename, 1, nullptr};
  GXF_LOG_INFO("Loading extensions");
  GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_ext_info));
  GXF_LOG_INFO("Loading graph file %s", graph_file);
  GXF_ASSERT_SUCCESS(LoadApplication(context, graph_file));  // Load application graph file(s)
  GXF_LOG_INFO("Initializing...");
  GXF_ASSERT_SUCCESS(GxfGraphActivate(context));
  GXF_LOG_INFO("Running...");
  GXF_ASSERT_SUCCESS(GxfGraphRunAsync(context));
  GXF_ASSERT_SUCCESS(GxfGraphWait(context));
  GXF_LOG_INFO("Deinitializing...");
  GXF_ASSERT_SUCCESS(GxfGraphDeactivate(context));
  GXF_LOG_INFO("Destroying context");
  GXF_ASSERT_SUCCESS(GxfContextDestroy(context));

  return 0;
}
