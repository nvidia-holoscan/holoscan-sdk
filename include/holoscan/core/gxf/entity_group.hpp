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

#ifndef HOLOSCAN_CORE_GXF_ENTITY_GROUP_HPP
#define HOLOSCAN_CORE_GXF_ENTITY_GROUP_HPP

#include <gxf/core/gxf.h>

#include <memory>
#include <string>

#include "./gxf_condition.hpp"
#include "./gxf_operator.hpp"
#include "../operator.hpp"

namespace holoscan::gxf {

/**
 * @brief GXF entity group.
 *
 * Define an entity group for the underlying GXF runtime. Entity groups are used to associate
 * components with resources inheriting from nvidia::gxf::ResourceBase. The components of
 * this type exposed in Holoscan SDK's API are GPUDevice and ThreadPool.
 *
 */
class EntityGroup {
 public:
  EntityGroup() = delete;

  EntityGroup(gxf_context_t context, const std::string& name);

  /**
   * @brief Get the group id of the entity group.
   *
   * @return The GXF group id of the entity group.
   */
  gxf_uid_t gxf_gid() const { return gxf_gid_; }

  /**
   * @brief Get the GXF context of the entity group.
   *
   * @return The GXF context of the entity group.
   */
  gxf_context_t gxf_context() const { return gxf_context_; }

  /**
   * @brief Get the name of the entity group.
   *
   * @return The name of the entity group.
   */
  std::string name() const { return name_; }

  /**
   * @brief Add a GXF entity to the entity group.
   *
   * If the entity is already a member of a different entity group, it will be removed from that
   * group and added to this one.
   *
   * Will raise a runtime_error if the entity is already a member of this entity group.
   *
   *
   * @param eid The GXF unique id corresponding to the entity.
   */
  void add(gxf_uid_t eid);

  /**
   * @brief Add a GXFComponent to the entity group.
   *
   * If the component is already a member of a different entity group, it will be removed from that
   * group and the entity it belongs to will be added to this one.
   *
   * Will raise a runtime_error if the entity is already a member of this entity group.
   *
   * @param component The component to add to the entity group.
   */
  void add(const GXFComponent& component);

  /**
   * @brief Add an Operator to the entity group.
   *
   * If the operator is already a member of a different entity group, it will be removed from that
   * group and added to this one.
   *
   * Will raise a runtime_error if the entity is already a member of this entity group.
   *
   * @param op The operator to add to the entity group.
   * @param entity_prefix A string prefix that can be used to indicate the entity the operator
   *                      belongs to.
   */
  void add(std::shared_ptr<Operator> op, const std::string& entity_prefix = "");

  // TODO:
  //   There is also the following related runtime GXF method
  //     gxf_result_t Runtime::GxfEntityGroupFindResources(gxf_uid_t eid,
  //                                                       uint64_t* num_resource_cids,
  //                                                       gxf_uid_t* resource_cids)
  //   It takes an entity's eid, determines the corresponding group id and then returns all of the
  //   component ids associated with resource_components for that group.
  //
  //   should this find_resources API be supported as a static method of EntityGroup?
  //     static std::vector<gxf_uid_t> find_resources(gxf_uid_t eid);
  //
  //   or perhaps even better if we could return the actual SystemResource objects
  //     static std::vector<SystemResource> find_resources(gxf_uid_t eid);

 private:
  std::string name_;              ///< The name of the entity group.
  gxf_context_t gxf_context_;     ///< The GXF context
  gxf_uid_t gxf_gid_ = kNullUid;  ///< The GXF group id.
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_ENTITY_GROUP_HPP */
