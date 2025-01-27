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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_STREAM_ORDERED_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_STREAM_ORDERED_ALLOCATOR_HPP

#include <string>

#include <gxf/cuda/stream_ordered_allocator.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./cuda_allocator.hpp"

namespace holoscan {

/**
 * @brief
 *
 * StreamOrderedAllocator uses `cudaMallocFromPoolAsync`/`cudaFreeAsync` dynamically without a
 * pool.
 *
 * This allocator only supports CUDA device memory. If host memory is also needed, see
 * `RMMAllocator`. This allocator does not provide bounded execution times.
 *
 * Because it is a CudaAllocator it supports both synchronous (`allocate`, `free`) and
 * asynchronous (`allocate_async`, `free_async`) APIs for memory allocation.
 *
 * The values for the memory parameters, such as `device_memory_initial_size` must be specified in
 * the form of a string containing a non-negative integer value followed by a suffix representing
 * the units. Supported units are B, KB, MB, GB and TB where the values are powers of 1024 bytes
 * (e.g. MB = 1024 * 1024 bytes). Examples of valid units are "512MB", "256 KB", "1 GB". If a
 * floating point number is specified that decimal portion will be truncated (i.e. the value is
 * rounded down to the nearest integer).
 *
 * ==Parameters==
 *
 * - **device_memory_initial_size** (std::string, optional): The initial size of the device memory
 * pool. See above for the format accepted. Defaults to "8MB" on aarch64 and "16MB" on x86_64.
 * - **device_memory_max_size** (std::string, optional): The maximum size of the device memory
 * pool. See above for the format accepted. The default is to use twice the value set for
 * `device_memory_initial_size`.
 * - **release_threshold** (std::string, optional): The amount of reserved memory to hold onto
 * before trying to release memory back to the OS.  See above for the format accepted. The default
 * value is "4MB".
 * - **dev_id** (int32_t, optional): The CUDA device id specifying which device the memory pool
 * will use. (Default: 0)
 */
class StreamOrderedAllocator : public CudaAllocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(StreamOrderedAllocator, CudaAllocator)

  StreamOrderedAllocator() = default;
  StreamOrderedAllocator(const std::string& name, nvidia::gxf::StreamOrderedAllocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::StreamOrderedAllocator"; }

  void setup(ComponentSpec& spec) override;

  nvidia::gxf::StreamOrderedAllocator* get() const;

 private:
  Parameter<std::string> release_threshold_;
  Parameter<std::string> device_memory_initial_size_;
  Parameter<std::string> device_memory_max_size_;
  Parameter<int32_t> dev_id_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_STREAM_ORDERED_ALLOCATOR_HPP */
