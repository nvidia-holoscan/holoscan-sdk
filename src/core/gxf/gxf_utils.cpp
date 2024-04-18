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
#include <string>
#include <utility>

#include "holoscan/core/gxf/gxf_utils.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/io_context.hpp"

#include "gxf/std/transmitter.hpp"

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
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(context, eid, &ename));

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

}  // namespace holoscan::gxf
