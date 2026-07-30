/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_STREAM_ORDERED_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_STREAM_ORDERED_ALLOCATOR_HPP

#include <string>

#include <gxf/cuda/stream_ordered_allocator.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./cuda_allocator.hpp"

namespace holoscan {

/**
 * @brief CUDA device memory allocator using stream-ordered allocation.
 *
 * StreamOrderedAllocator uses CUDA's stream-ordered memory allocator
 * (`cudaMallocAsync`/`cudaFreeAsync`) to dynamically allocate device memory. Stream-ordered
 * allocation enables memory operations to be tied to specific CUDA streams, allowing allocation
 * and deallocation without blocking the host or other streams.
 *
 * See the CUDA Programming Guide section on
 * [Stream-Ordered Memory
 * Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#stream-ordered-memory-allocator)
 * for details on the underlying CUDA feature.
 *
 * This allocator only supports CUDA device memory. If host memory is also needed, see
 * `RMMAllocator` which provides both device and pinned host memory pools.
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
