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

#include "holoscan/core/resources/gxf/cuda_green_context_pool.hpp"

#include <cstdint>
#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

namespace holoscan {

namespace {
constexpr uint32_t kDefaultFlags = 0;
constexpr uint32_t kDefaultNumPartitions = 0;
constexpr uint32_t kDefaultMinSM = 2;
constexpr int32_t kDefaultDeviceId = 0;
}  // namespace

CudaGreenContextPool::CudaGreenContextPool(const std::string& name,
                                           nvidia::gxf::CudaGreenContextPool* component)
    : GXFResource(name, component) {
  auto maybe_flags = component->getParameter<uint32_t>("flags");
  if (!maybe_flags) {
    throw std::runtime_error("Failed to get flags");
  }
  flags_ = maybe_flags.value();

  auto maybe_num_partitions = component->getParameter<uint32_t>("num_partitions");
  if (!maybe_num_partitions) {
    throw std::runtime_error("Failed to get num_partitions");
  }
  num_partitions_ = maybe_num_partitions.value();

  auto maybe_min_sm_size = component->getParameter<int32_t>("min_sm_size");
  if (!maybe_min_sm_size) {
    throw std::runtime_error("Failed to get min_sm_size");
  }
  min_sm_size_ = maybe_min_sm_size.value();

  auto maybe_sms_per_partition =
      component->getParameter<std::vector<uint32_t>>("sms_per_partition");
  if (!maybe_sms_per_partition) {
    throw std::runtime_error("Failed to get sms_per_partition");
  }
  sms_per_partition_ = maybe_sms_per_partition.value();

  auto maybe_gpu_device =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>>("dev_id");
  if (!maybe_gpu_device) {
    throw std::runtime_error("Failed to get dev_id");
  }
  auto gpu_device_handle = maybe_gpu_device.value();
  dev_id_ = gpu_device_handle->device_id();

  auto maybe_default_context_index = component->getParameter<int32_t>("default_context_index");
  if (!maybe_default_context_index) {
    throw std::runtime_error("Failed to get default_context_index");
  }
  default_context_index_ = maybe_default_context_index.value();
}

nvidia::gxf::CudaGreenContextPool* CudaGreenContextPool::get() const {
  return static_cast<nvidia::gxf::CudaGreenContextPool*>(gxf_cptr_);
}

void CudaGreenContextPool::setup(ComponentSpec& spec) {
  // TODO(unknown): The dev_id parameter was removed in GXF 3.0 and replaced with a GPUDevice
  // Resource Note: We are currently working around this with special handling of the "dev_id"
  // parameter in GXFResource::initialize().
  spec.param(
      dev_id_, "dev_id", "Device Id", "Create CUDA Stream on which device.", kDefaultDeviceId);
  spec.param(flags_,
             "flags",
             "Flags",
             "Flags for CUDA green contexts in the pool. The flag value will be passed to CUDA's "
             "cuDevSmResourceSplitByCount and cuGreenCtxCreate  when creating the green contexts."
             "See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html"
             "#group__CUDA__GREEN__CONTEXTS.",
             kDefaultFlags);
  spec.param(num_partitions_,
             "num_partitions",
             "Number of Partitions",
             "Number of partitions to create for the green context pool.",
             kDefaultNumPartitions);
  spec.param(min_sm_size_,
             "min_sm_size",
             "Minimum SM Size",
             "The minimum number of SMs used for green context creation.",
             kDefaultMinSM);
  spec.param(sms_per_partition_,
             "sms_per_partition",
             "SMs per Partition",
             "The number of SMs to allocate per partition. If empty, the SMs will be distributed "
             "evenly across partitions.",
             {0});
  spec.param(default_context_index_,
             "default_context_index",
             "Default Context Index",
             "The index of the default green context to use. When index < 0, the last partition's "
             "index will be used. (Default: -1)",
             -1);
}

}  // namespace holoscan
