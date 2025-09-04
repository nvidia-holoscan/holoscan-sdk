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

#include "holoscan/core/resources/gxf/cuda_green_context.hpp"

#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/resources/gxf/cuda_green_context_pool.hpp"

namespace holoscan {

CudaGreenContext::CudaGreenContext(const std::string& name,
                                   nvidia::gxf::CudaGreenContext* component)
    : GXFResource(name, component) {
  auto maybe_green_context_pool =
      component->getParameter<std::shared_ptr<CudaGreenContextPool>>("green_context_pool");
  if (!maybe_green_context_pool) {
    throw std::runtime_error("Failed to get green_context_pool");
  }
  cuda_green_context_pool_ = maybe_green_context_pool.value();

  auto maybe_index = component->getParameter<int32_t>("index");
  if (!maybe_index) {
    throw std::runtime_error("Failed to get index");
  }
  index_ = maybe_index.value();
}

nvidia::gxf::CudaGreenContext* CudaGreenContext::get() const {
  if (gxf_cptr_ == nullptr) {
    HOLOSCAN_LOG_ERROR("CudaGreenContext is not initialized");
    return nullptr;
  }
  return static_cast<nvidia::gxf::CudaGreenContext*>(gxf_cptr_);
}

void CudaGreenContext::setup(ComponentSpec& spec) {
  spec.param(cuda_green_context_pool_,
             "cuda_green_context_pool",
             "Green Context Pool",
             "The green context pool to use. If not provided, an error will be returned",
             static_cast<std::shared_ptr<CudaGreenContextPool>>(nullptr));
  spec.param(index_, "index", "Index", "The index of the green context to use.", -1);
}

void CudaGreenContext::initialize() {
  HOLOSCAN_LOG_DEBUG("CudaGreenContext '{}': initialize", name());

  std::shared_ptr<CudaGreenContextPool> green_context_pool_ptr;
  if (cuda_green_context_pool_.has_value()) {
    green_context_pool_ptr = cuda_green_context_pool_.get();
    HOLOSCAN_LOG_DEBUG("CudaGreenContext '{}': found green_context_pool valid pointer", name());
  }

  if (green_context_pool_ptr != nullptr) {
    if (gxf_eid_ != 0 && green_context_pool_ptr->gxf_eid() == 0) {
      green_context_pool_ptr->gxf_eid(gxf_eid_);
    }
    HOLOSCAN_LOG_DEBUG("CudaGreenContext '{}': initializing CudaGreenContextPool '{}'",
                       name(),
                       green_context_pool_ptr->name());
    green_context_pool_ptr->initialize();
  }
  GXFResource::initialize();
}

}  // namespace holoscan
