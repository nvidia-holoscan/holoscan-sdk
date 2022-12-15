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

}  // namespace holoscan::gxf
