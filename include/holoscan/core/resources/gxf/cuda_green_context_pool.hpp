/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_POOL_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_POOL_HPP

#include <string>
#include <vector>

#include <gxf/std/cuda_green_context_pool.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief CUDA green context pool.
 *
 * A resource component that creates a pool of CUDA green contexts.
 *
 * Internally, the green contexts created correspond to `nvidia::gxf::CudaGreenContext` objects
 * whose lifetime is managed by the underlying GXF framework.
 *
 * ==Parameters==
 *
 * - **dev_id** (int32_t, optional): The CUDA device id specifying which device the green context
 * pool will use (Default: 0).
 * - **green_context_flags** (uint32_t, optional): The flags passed to the underlying CUDA runtime
 * API call when the green contexts for this pool are created
 * (Default: cudaStreamNonBlocking, matching GXF).
 * Python kwarg: ``flags``.
 * - **num_partitions** (uint32_t, optional): The number of partitions to create for the green
 * context pool (Default: 1).
 * - **sms_per_partition** (std::vector<uint32_t>, optional): The number of SMs to allocate per
 * partition. If empty, and num_partitions is 0, one green context will be created using all the SMs
 * available on the device.
 * - **default_context** (int32_t, optional): The index of the default green context to use.
 * When index < 0, the last partition's index will be used. (Default: -1).
 * Python kwarg: ``default_context_index``.
 * - **min_sm_count** (uint32_t, optional): The minimum number of SMs to allocate per partition.
 * (Default: 2). Python kwarg: ``min_sm_size``.
 */
class CudaGreenContextPool : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CudaGreenContextPool, gxf::GXFResource)
  CudaGreenContextPool() = default;
  CudaGreenContextPool(int32_t dev_id, uint32_t flags, uint32_t num_partitions,
                       const std::vector<uint32_t>& sms_per_partition = {},
                       int32_t default_context_index = -1, uint32_t min_sm_size = 2) {
    if (min_sm_size < 1)
      throw std::invalid_argument("min_sm_size must be at least 1");
    if (num_partitions == 0 && !sms_per_partition.empty()) {
      throw std::invalid_argument("sms_per_partition should be empty when num_partitions is 0");
    }
    dev_id_ = dev_id;
    flags_ = flags;
    num_partitions_ = num_partitions;
    sms_per_partition_ = sms_per_partition;
    default_context_index_ = default_context_index;
    min_sm_size_ = min_sm_size;
  }
  CudaGreenContextPool(const std::string& name, nvidia::gxf::CudaGreenContextPool* component);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaGreenContextPool"; }

  // void initialize() override;
  void setup(ComponentSpec& spec) override;

  nvidia::gxf::CudaGreenContextPool* get() const;

  /**
   * @brief Check whether CUDA Green Context partitioning will succeed on a device.
   *
   * Verifies that the CUDA driver accepts the requested SM split and
   * per-partition resource generation for the given device, partition sizes,
   * and minimum SM block size.
   *
   * @param dev_id CUDA device index (typically 0).
   * @param min_sm_count Minimum SM block size for the pool.
   * @param sms_per_partition SM counts per partition (e.g. {8, 8}).
   * @return true if partitioning is supported; false if the driver rejects the
   *         configuration or the required driver API is unavailable.
   */
  static bool is_partitioning_supported(int32_t dev_id, uint32_t min_sm_count,
                                        const std::vector<uint32_t>& sms_per_partition);

 private:
  Parameter<int32_t> dev_id_;
  Parameter<uint32_t> flags_;
  Parameter<uint32_t> num_partitions_;
  Parameter<std::vector<uint32_t>> sms_per_partition_;
  Parameter<int32_t> default_context_index_;
  Parameter<uint32_t> min_sm_size_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CUDA_GREEN_CONTEXT_POOL_HPP */
