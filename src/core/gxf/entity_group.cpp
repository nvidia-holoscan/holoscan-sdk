/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/gxf/entity_group.hpp>

#include <gxf/core/gxf.h>
#include <gxf/core/gxf_ext.h>

#include <memory>
#include <string>

namespace holoscan::gxf {

EntityGroup::EntityGroup(gxf_context_t context, const std::string& name) {
  name_ = name;
  gxf_context_ = context;
  HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntityGroup(context, name.c_str(), &gxf_gid_));
}

void EntityGroup::add(gxf_uid_t eid) {
  HOLOSCAN_GXF_CALL_FATAL(GxfUpdateEntityGroup(gxf_context_, gxf_gid_, eid));
}

void EntityGroup::add(const GXFComponent& component) {
  HOLOSCAN_GXF_CALL_FATAL(GxfUpdateEntityGroup(gxf_context_, gxf_gid_, component.gxf_eid()));
}

void EntityGroup::add(const std::shared_ptr<Operator>& op, const std::string& entity_prefix) {
  gxf_uid_t op_eid = kNullUid;
  if (op->operator_type() == Operator::OperatorType::kGXF) {
    op_eid = std::dynamic_pointer_cast<holoscan::ops::GXFOperator>(op)->gxf_eid();
  } else {
    // get the GXF entity ID corresponding to the native operator's GXF Codelet
    const std::string op_entity_name = fmt::format("{}{}", entity_prefix, op->name());
    HOLOSCAN_GXF_CALL_FATAL(GxfEntityFind(gxf_context_, op_entity_name.c_str(), &op_eid));
  }
  add(op_eid);
}
}  // namespace holoscan::gxf
