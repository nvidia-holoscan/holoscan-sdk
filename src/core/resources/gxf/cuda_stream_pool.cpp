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

#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

namespace {
constexpr uint32_t kDefaultStreamFlags = 0;
constexpr int32_t kDefaultStreamPriority = 0;
constexpr uint32_t kDefaultReservedSize = 1;
constexpr uint32_t kDefaultMaxSize = 0;
constexpr int32_t kDefaultDeviceId = 0;
}  // namespace

CudaStreamPool::CudaStreamPool(const std::string& name, nvidia::gxf::CudaStreamPool* component)
    : Allocator(name, component) {
  int32_t dev_id = 0;
  GxfParameterGetInt32(gxf_context_, gxf_cid_, "dev_id", &dev_id);
  dev_id_ = dev_id;
  uint32_t stream_flags = 0;
  GxfParameterGetUInt32(gxf_context_, gxf_cid_, "stream_flags", &stream_flags);
  stream_flags_ = stream_flags;
  int32_t stream_priority = 0;
  GxfParameterGetInt32(gxf_context_, gxf_cid_, "stream_priority", &stream_priority);
  stream_priority_ = stream_priority;
  uint32_t reserved_size = 0;
  GxfParameterGetUInt32(gxf_context_, gxf_cid_, "reserved_size", &reserved_size);
  reserved_size_ = reserved_size;
  uint32_t max_size = 0;
  GxfParameterGetUInt32(gxf_context_, gxf_cid_, "max_size", &max_size);
  max_size_ = max_size;
}

void CudaStreamPool::setup(ComponentSpec& spec) {
  spec.param(
      dev_id_, "dev_id", "Device Id", "Create CUDA Stream on which device.", kDefaultDeviceId);
  spec.param(stream_flags_,
             "stream_flags",
             "Stream Flags",
             "Create CUDA streams with flags.",
             kDefaultStreamFlags);
  spec.param(stream_priority_,
             "stream_priority",
             "Stream Priority",
             "Create CUDA streams with priority.",
             kDefaultStreamPriority);
  spec.param(reserved_size_,
             "reserved_size",
             "Reserved Stream Size",
             "Reserve several CUDA streams before 1st request coming",
             kDefaultReservedSize);
  spec.param(max_size_,
             "max_size",
             "Maximum Stream Size",
             "The maximum stream size for the pool to allocate, unlimited by default",
             kDefaultMaxSize);
}

}  // namespace holoscan
