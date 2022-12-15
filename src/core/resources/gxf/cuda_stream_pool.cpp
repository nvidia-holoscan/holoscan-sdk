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
