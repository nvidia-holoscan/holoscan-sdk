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

#ifndef HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP
#define HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP

#include <memory>
#include <utility>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment_service.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"

namespace holoscan {

class PoseTree;  // Forward declaration
class ComponentSpec;

/**
 * @brief Manage a shared PoseTree instance as a FragmentService.
 *
 * This resource creates and holds a `holoscan::PoseTree` instance, making it accessible to
 * multiple components (like operators) within the same fragment. It simplifies the management of
 * pose data by providing a centralized, configurable point of access.
 *
 * To use it, register an instance of `PoseTreeManager` with the fragment:
 *
 * ```cpp
 * // In Application::compose()
 * auto pose_tree_manager = make_resource<PoseTreeManager>("pose_tree_manager",
 *     from_config("my_pose_tree"));
 * register_service(pose_tree_manager);
 * ```
 *
 * Then, operators can access the underlying `PoseTree` instance via the `service()` method:
 *
 * ```cpp
 * // In Operator::initialize()
 * pose_tree_ = service<holoscan::PoseTreeManager>("pose_tree_manager")->tree();
 * ```
 *
 * The parameters for the underlying `PoseTree` can be configured via the application's YAML
 * configuration file or directly when creating the resource.
 *
 * **YAML-based configuration:**
 * ```yaml
 * my_pose_tree:
 *   number_frames: 256
 *   number_edges: 4096
 *   history_length: 16384
 *   default_number_edges: 32
 *   default_history_length: 1024
 *   edges_chunk_size: 16
 *   history_chunk_size: 128
 * ```
 *
 * **Experimental Feature**
 * The Pose Tree feature, including this manager, is experimental. The API may change in future
 * releases.
 */
class PoseTreeManager : public holoscan::Resource, public holoscan::FragmentService {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(PoseTreeManager, holoscan::Resource)

  PoseTreeManager() = default;

  /**
   * @brief Return a shared pointer to this resource, part of the FragmentService interface.
   * @return A `std::shared_ptr<Resource>` pointing to this instance.
   */
  std::shared_ptr<Resource> resource() const override;

  /**
   * @brief Set the internal weak pointer to this resource, part of the FragmentService interface.
   * @param resource A `std::shared_ptr<Resource>` that must point to this instance.
   */
  void resource(const std::shared_ptr<Resource>& resource) override;

  /**
   * @brief Initialize the resource and creates the underlying `PoseTree` instance.
   *
   * This method is called by the framework after the resource is created and its parameters
   * have been set. It allocates and initializes the `PoseTree` with the configured capacity
   * parameters.
   */
  void initialize() override;

  /**
   * @brief Define the parameters for configuring the `PoseTree` instance.
   *
   * This method registers the following parameters:
   * - `number_frames`: Maximum number of coordinate frames.
   * - `number_edges`: Maximum number of edges (direct transformations) between frames.
   * - `history_length`: Total capacity for storing historical pose data across all edges.
   * - `default_number_edges`: Default number of edges allocated per new frame.
   * - `default_history_length`: Default history capacity allocated per new edge.
   * - `edges_chunk_size`: Allocation chunk size for a frame's edge list.
   * - `history_chunk_size`: Allocation chunk size for an edge's history buffer.
   *
   * @param spec The component specification to which the parameters are added.
   */
  void setup(holoscan::ComponentSpec& spec) override;

  /**
   * @brief Get a shared pointer to the managed `PoseTree` instance.
   *
   * This is the primary method for accessing the pose tree from other components.
   *
   * @return A `std::shared_ptr<PoseTree>` to the underlying pose tree.
   */
  std::shared_ptr<PoseTree> tree();

  /**
   * @brief Get a shared pointer to the managed `PoseTree` instance from a const context.
   *
   * @return A `std::shared_ptr<PoseTree>` to the underlying pose tree.
   */
  std::shared_ptr<PoseTree> tree() const;

 private:
  // Parameters for PoseTree::init
  Parameter<int32_t> number_frames_;
  Parameter<int32_t> number_edges_;
  Parameter<int32_t> history_length_;
  Parameter<int32_t> default_number_edges_;
  Parameter<int32_t> default_history_length_;
  Parameter<int32_t> edges_chunk_size_;
  Parameter<int32_t> history_chunk_size_;

  std::shared_ptr<PoseTree> pose_tree_instance_;

  std::weak_ptr<Resource> resource_;  ///< Weak reference to the managed resource (self)
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_POSE_TREE_MANAGER_HPP */
