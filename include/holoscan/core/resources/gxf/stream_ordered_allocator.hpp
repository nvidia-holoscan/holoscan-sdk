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
 * StreamOrderedAllocator uses cudaMallocFromPoolAsync / cudaFreeAsync dynamically without a pool.
 * Does not provide bounded execution times.
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
