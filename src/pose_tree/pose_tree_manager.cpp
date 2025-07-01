/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/pose_tree/pose_tree_manager.hpp"

#include <memory>

#include "holoscan/pose_tree/pose_tree.hpp"

namespace holoscan {

std::shared_ptr<Resource> PoseTreeManager::resource() const {
  return resource_.lock();
}

void PoseTreeManager::resource(const std::shared_ptr<Resource>& resource) {
  resource_ = resource;
}

void PoseTreeManager::initialize() {
  Resource::initialize();  // Call base class initialize
  pose_tree_instance_ = std::make_shared<PoseTree>();
  // Initialize the underlying PoseTree with configured parameters
  pose_tree_instance_->init(number_frames_.get(),
                            number_edges_.get(),
                            history_length_.get(),
                            default_number_edges_.get(),
                            default_history_length_.get(),
                            edges_chunk_size_.get(),
                            history_chunk_size_.get());
  HOLOSCAN_LOG_DEBUG("PoseTree initialized with the following parameters:");
  HOLOSCAN_LOG_DEBUG("  number_frames: {}", number_frames_.get());
  HOLOSCAN_LOG_DEBUG("  number_edges: {}", number_edges_.get());
  HOLOSCAN_LOG_DEBUG("  history_length: {}", history_length_.get());
  HOLOSCAN_LOG_DEBUG("  default_number_edges: {}", default_number_edges_.get());
  HOLOSCAN_LOG_DEBUG("  default_history_length: {}", default_history_length_.get());
  HOLOSCAN_LOG_DEBUG("  edges_chunk_size: {}", edges_chunk_size_.get());
  HOLOSCAN_LOG_DEBUG("  history_chunk_size: {}", history_chunk_size_.get());
}

void PoseTreeManager::setup(holoscan::ComponentSpec& spec) {
  spec.param(number_frames_,
             "number_frames",
             "Maximum Number of Frames",
             "Maximum number of frames to support.",
             1024);
  spec.param(number_edges_,
             "number_edges",
             "Maximum Number of Edges",
             "Maximum number of edges to support.",
             16384);
  spec.param(history_length_,
             "history_length",
             "Maximum History Length",
             "Maximum history length.",
             1048576);
  spec.param(default_number_edges_,
             "default_number_edges",
             "Default Number of Edges per Frame",
             "Default number of edges per frame.",
             16);
  spec.param(default_history_length_,
             "default_history_length",
             "Default History Length per Edge",
             "Default history length per edge.",
             1024);
  spec.param(edges_chunk_size_,
             "edges_chunk_size",
             "Edges Chunk Size",
             "Chunk size for edge allocation.",
             4);
  spec.param(history_chunk_size_,
             "history_chunk_size",
             "History Chunk Size",
             "Chunk size for history allocation.",
             64);
}

std::shared_ptr<PoseTree> PoseTreeManager::tree() {
  return pose_tree_instance_;
}

std::shared_ptr<PoseTree> PoseTreeManager::tree() const {
  return pose_tree_instance_;
}

}  // namespace holoscan
