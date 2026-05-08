/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared utilities for the fastdds_benchmark example: DDS TypeSupport factory, CUDA buffer
 * helpers, and grep-friendly latency logging (see README.md).
 *
 * The CUDA device buffer helpers (`make_cuda_device_buffer`, `cuda_device_memset`,
 * `cuda_device_memcpy_device_to_device`) are derived from material
 * Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima), licensed under
 * Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_COMMON_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_COMMON_HPP

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <fastdds/dds/topic/TypeSupport.hpp>

#include <gen/BufferPubSubTypes.hpp>

namespace holoscan::ipc::fastdds_benchmark {

/**
 * Topic type support for the benchmark sample: CUDA IPC (AccelBuffer) or in-band bytes
 * (Buffer).
 */
inline eprosima::fastdds::dds::TypeSupport make_buffer_sample_typesupport(bool use_accel_buffer) {
  using eprosima::fastdds::dds::TypeSupport;
  if (use_accel_buffer) {
    return TypeSupport(new AccelBufferPubSubType());
  }
  return TypeSupport(new BufferPubSubType());
}

/** Wall-clock nanoseconds since Unix epoch (comparable across processes on a synced host). */
inline int64_t cuda_buffer_latency_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/**
 * Grep-friendly one-line markers for CUDA buffer sharing latency (see README.md).
 *
 * Example:
 *   grep CUDA_BUF_LAT run.log
 *   # Match pub/sub by index; delta_ns ≈ ts_ns(sub) - ts_ns(pub) on a time-synced host.
 *
 * @param kind  "pub" | "sub"
 * @param flow  "accel" | "host" (AccelBuffer + CUDA IPC vs plain Buffer octets)
 * @param index sample index from the DDS message (0-based)
 */
inline void cuda_buffer_latency_log(const char* kind, const char* flow, uint32_t index) {
  std::cout << "CUDA_BUF_LAT kind=" << kind << " flow=" << flow << " index=" << index
            << " ts_ns=" << cuda_buffer_latency_now_ns() << std::endl;
}

/**
 * @brief Allocate a CUDA device buffer and return it as a shared pointer.
 *
 * @param size_bytes Number of bytes to allocate (`cudaMalloc`).
 * @return Shared pointer with a deleter that calls `cudaFree` when the last reference is dropped.
 * @throws std::runtime_error if `cudaMalloc` fails.
 */
inline std::shared_ptr<void> make_cuda_device_buffer(size_t size_bytes) {
  void* raw = nullptr;
  const cudaError_t err = cudaMalloc(&raw, size_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed for device buffer: ") +
                             cudaGetErrorString(err));
  }
  if (raw == nullptr) {
    throw std::runtime_error("cudaMalloc reported success but returned null pointer");
  }
  return std::shared_ptr<void>(raw, [](void* ptr) {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  });
}

/**
 * @brief Run `cudaMemset` on a device buffer referenced by `buffer`.
 *
 * @param buffer Device allocation (must not be null).
 * @param value Byte value to set (same as `cudaMemset`).
 * @param size_bytes Number of bytes to set.
 * @throws std::runtime_error if `buffer` is empty or `cudaMemset` fails.
 */
inline void cuda_device_memset(const std::shared_ptr<void>& buffer, int value, size_t size_bytes) {
  if (buffer.get() == nullptr) {
    throw std::runtime_error("cuda_device_memset: null buffer");
  }
  const cudaError_t err = cudaMemset(buffer.get(), value, size_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemset failed: ") + cudaGetErrorString(err));
  }
}

/**
 * @brief Run `cudaMemcpy` device-to-device from `src` into `dst`.
 *
 * @param dst Destination device buffer (must not be null).
 * @param src Source device buffer (must not be null).
 * @param size_bytes Number of bytes to copy.
 * @throws std::runtime_error if either pointer is null or `cudaMemcpy` fails.
 */
inline void cuda_device_memcpy_device_to_device(const std::shared_ptr<void>& dst,
                                                const std::shared_ptr<void>& src,
                                                size_t size_bytes) {
  if (dst.get() == nullptr || src.get() == nullptr) {
    throw std::runtime_error(
        "cuda_device_memcpy_device_to_device: null destination or source buffer");
  }
  const cudaError_t err = cudaMemcpy(dst.get(), src.get(), size_bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy (device to device) failed: ") +
                             cudaGetErrorString(err));
  }
}

}  // namespace holoscan::ipc::fastdds_benchmark

#endif  // HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_COMMON_HPP
