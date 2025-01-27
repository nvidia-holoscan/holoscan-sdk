/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <common/fixed_vector.hpp>

#include "holoscan/core/gxf/gxf_utils.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"

namespace fmt {

format_context::iterator formatter<nvidia::gxf::Tensor>::format(const nvidia::gxf::Tensor& t,
                                                                fmt::format_context& ctx) const {
  const auto& shape = t.shape();
  std::vector<int32_t> dimensions(shape.rank());
  for (uint32_t index = 0; index < shape.rank(); ++index) {
    dimensions[index] = shape.dimension(index);
  }
  std::vector<uint64_t> strides(t.rank());
  for (uint32_t index = 0; index < t.rank(); ++index) { strides[index] = t.stride(index); }
  fmt::format_to(
      ctx.out(),
      "storage_type: {}, shape: ( {} ), strides: ( {} ), element_type: {}, element_count: {}",
      magic_enum::enum_name(t.storage_type()),
      fmt::join(dimensions, ", "),
      fmt::join(strides, ", "),
      magic_enum::enum_name(t.element_type()),
      t.element_count());
  return ctx.out();
}

format_context::iterator formatter<nvidia::gxf::ColorPlane>::format(
    const nvidia::gxf::ColorPlane& c, fmt::format_context& ctx) const {
  return fmt::format_to(ctx.out(),
                        "color_space: {}, bytes_per_pixel: {}, stride: {}, offset: {}, width: "
                        "{}, height: {}, size: {}",
                        c.color_space,
                        c.bytes_per_pixel,
                        c.stride,
                        c.offset,
                        c.width,
                        c.height,
                        c.bytes_per_pixel,
                        c.size);
}

format_context::iterator formatter<nvidia::gxf::VideoBufferInfo>::format(
    const nvidia::gxf::VideoBufferInfo& v, fmt::format_context& ctx) const {
  return fmt::format_to(
      ctx.out(),
      "width: {}, height: {}, color_format: {}, color_planes: ( {} ), surface_layout: {}\n",
      v.width,
      v.height,
      magic_enum::enum_name(v.color_format),
      fmt::join(v.color_planes, "), ("),
      magic_enum::enum_name(v.surface_layout));
}

format_context::iterator formatter<nvidia::gxf::VideoBuffer>::format(
    const nvidia::gxf::VideoBuffer& v, fmt::format_context& ctx) const {
  return fmt::format_to(ctx.out(),
                        "storage_type: {}, {}",
                        magic_enum::enum_name(v.storage_type()),
                        v.video_frame_info());
}

}  // namespace fmt

namespace holoscan::gxf {

gxf_uid_t get_component_eid(gxf_context_t context, gxf_uid_t cid) {
  gxf_uid_t eid;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentEntity(context, cid, &eid));
  return eid;
}

std::string get_full_component_name(gxf_context_t context, gxf_uid_t cid) {
  const char* cname;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(context, cid, &cname));
  gxf_uid_t eid;
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentEntity(context, cid, &eid));
  const char* ename;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGetName(context, eid, &ename));

  std::stringstream sstream;
  sstream << ename << "/" << cname;
  return sstream.str();
}

std::string create_name(const char* prefix, int index) {
  std::stringstream sstream;
  sstream << prefix << "_" << index;
  return sstream.str();
}

std::string create_name(const char* prefix, const std::string& name) {
  std::stringstream sstream;
  sstream << prefix << "_" << name;
  return sstream.str();
}

bool has_component(gxf_context_t context, gxf_uid_t eid, gxf_tid_t tid, const char* name,
                   int32_t* offset, gxf_uid_t* cid) {
  gxf_uid_t temp_cid = 0;
  auto result = GxfComponentFind(context, eid, tid, name, offset, cid ? cid : &temp_cid);
  if (result == GXF_SUCCESS) {
    return true;
  } else {
    return false;
  }
}

gxf_uid_t add_entity_group(void* context, std::string name) {
  gxf_uid_t entity_group_gid = kNullUid;
  HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntityGroup(context, name.c_str(), &entity_group_gid));
  return entity_group_gid;
}

uint64_t get_default_queue_policy() {
  const char* env_value = std::getenv("HOLOSCAN_QUEUE_POLICY");
  if (env_value) {
    std::string value{env_value};
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    if (value == "pop") {
      return 0UL;
    } else if (value == "reject") {
      return 1UL;
    } else if (value == "fail") {
      return 2UL;
    } else {
      HOLOSCAN_LOG_ERROR(
          "Unrecognized HOLOSCAN_QUEUE_POLICY: {}. It should be 'pop', 'reject' or 'fail'. Falling "
          "back to default policy of 'fail'",
          value);
      return 2UL;
    }
  }
  return 2UL;  // fail
}

std::optional<int32_t> gxf_device_id(gxf_context_t context, gxf_uid_t eid) {
  // Get handle to entity
  auto maybe = nvidia::gxf::Entity::Shared(context, eid);
  if (!maybe) {
    HOLOSCAN_LOG_ERROR("Failed to create shared Entity for eid {}", eid);
    return std::nullopt;
  }
  auto entity = maybe.value();
  // Find all GPUDevice components
  auto maybe_resources = entity.findAllHeap<nvidia::gxf::GPUDevice>();
  if (!maybe_resources) {
    HOLOSCAN_LOG_ERROR("Failed to find resources in entity");
    return std::nullopt;
  }
  auto resources = std::move(maybe_resources.value());
  if (resources.empty()) { return std::nullopt; }
  if (resources.size() > 1) {
    HOLOSCAN_LOG_WARN(
        "Multiple ({}) GPUDevice resources found in entity {}.", resources.size(), eid);
  }

  int32_t device_id = resources.at(0).value()->device_id();
  // Loop over any additional device ID(s), warning if there are multiple conflicting IDs
  for (size_t i = 1; i < resources.size(); i++) {
    int32_t this_dev_id = resources.at(i).value()->device_id();
    if (this_dev_id != device_id) {
      HOLOSCAN_LOG_WARN(
          "Additional GPUDevice resources with conflicting CUDA device ID {} found in entity "
          "{}. The CUDA device ID of the first device found ({}) will be returned.",
          this_dev_id,
          eid,
          device_id);
    }
  }
  return device_id;
}

std::string gxf_entity_group_name(gxf_context_t context, gxf_uid_t eid) {
  const char* name;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGroupName(context, eid, &name));
  return std::string{name};
}

gxf_uid_t gxf_entity_group_id(gxf_context_t context, gxf_uid_t eid) {
  gxf_uid_t gid;
  HOLOSCAN_GXF_CALL_FATAL(GxfEntityGroupId(context, eid, &gid));
  return gid;
}

}  // namespace holoscan::gxf
