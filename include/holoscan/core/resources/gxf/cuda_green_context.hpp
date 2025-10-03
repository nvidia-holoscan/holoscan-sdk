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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_HPP

#include <string>

#include <gxf/std/cuda_green_context_pool.hpp>
#include <gxf/std/cuda_green_context.hpp>
#include "holoscan/core/resources/gxf/cuda_green_context_pool.hpp"

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief CUDA green context.
 *
 * A resource component that creates a CUDA green context.
 *
 * Internally, the green contexts created correspond to `nvidia::gxf::CudaGreenContext` objects
 * whose lifetime is managed by the underlying GXF framework.
 *
 * ==Parameters==
 *
 * - **green_context_pool** (CudaGreenContextPool): The CUDA green context pool to use.
 * - **index** (uint32_t, optional): The index of the green context to use, When not specified,
 *       the default green context will be used.
 * - **nvtx_identifier** (std::string, optional): The NVTX identifier of the green context. This
 *      will be used in NSight captures to identify the green context.
 */
class CudaGreenContext : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CudaGreenContext, gxf::GXFResource)
  CudaGreenContext() = default;
  CudaGreenContext(std::shared_ptr<CudaGreenContextPool> green_context_pool = nullptr,
                   int32_t index = -1, const std::string& nvtx_identifier = "defaultGreenContext")
      : cuda_green_context_pool_(green_context_pool),
        index_(index),
        nvtx_identifier_(nvtx_identifier) {}
  CudaGreenContext(const std::string& name, nvidia::gxf::CudaGreenContext* component);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaGreenContext"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::CudaGreenContext* get() const;

 private:
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;
  Parameter<int32_t> index_;
  Parameter<std::string> nvtx_identifier_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_HPP */
