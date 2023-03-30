/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/entity.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/execution_context.hpp"

namespace holoscan::gxf {

Entity Entity::New(ExecutionContext* context) {
  if (context == nullptr) { throw std::runtime_error("Null context is not allowed"); }
  auto gxf_context = context->context();

  gxf_uid_t eid;
  const GxfEntityCreateInfo info{};
  const gxf_result_t code = GxfCreateEntity(gxf_context, &info, &eid);
  if (code != GXF_SUCCESS) {
    throw std::runtime_error("Unable to create entity");
  } else {
    auto result = Shared(gxf_context, eid);
    if (!result) {
      throw std::runtime_error("Unable to increment entity reference count");
    } else {
      return Entity(std::move(result.value()));
    }
  }
}

nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> get_videobuffer(Entity entity, const char* name) {
  // We should use nullptr as a default name because In GXF, 'nullptr' should be used with
  // GxfComponentFind() if we want to get the first component of the given type.

  // We first try to get holoscan::gxf::GXFTensor from GXF Entity.
  gxf_tid_t tid;
  auto tid_result = GxfComponentTypeId(
      entity.context(), nvidia::TypenameAsString<nvidia::gxf::VideoBuffer>(), &tid);
  if (tid_result != GXF_SUCCESS) {
    throw std::runtime_error(fmt::format("Unable to get component type id: {}", tid_result));
  }

  gxf_uid_t cid;
  auto cid_result = GxfComponentFind(entity.context(), entity.eid(), tid, name, nullptr, &cid);
  if (cid_result != GXF_SUCCESS) {
    std::string msg = fmt::format(
        "Unable to find nvidia::gxf::VideoBuffer component from the name '{}' (error code: {})",
        name == nullptr ? "" : name,
        cid_result);
    throw std::runtime_error(msg);
  }

  // Create a handle to the GXF VideoBuffer object.
  auto result = nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>::Create(entity.context(), cid);
  if (!result) { throw std::runtime_error("Failed to create Handle to nvidia::gxf::VideoBuffer"); }
  return result.value();
}

}  // namespace holoscan::gxf
