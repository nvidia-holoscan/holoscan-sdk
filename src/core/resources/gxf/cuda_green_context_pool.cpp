/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/cuda_green_context_pool.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/gxf/gxf_utils.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace holoscan {

namespace {
constexpr uint32_t kDefaultFlags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
constexpr uint32_t kDefaultNumPartitions = 0;
constexpr uint32_t kDefaultMinSM = 2;
constexpr int32_t kDefaultDeviceId = 0;
}  // namespace

CudaGreenContextPool::CudaGreenContextPool(const std::string& name,
                                           nvidia::gxf::CudaGreenContextPool* component)
    : GXFResource(name, component) {
  auto maybe_flags = component->getParameter<uint32_t>("green_context_flags");
  if (!maybe_flags) {
    throw std::runtime_error("Failed to get green_context_flags");
  }
  flags_ = maybe_flags.value();

  auto maybe_num_partitions = component->getParameter<uint32_t>("num_partitions");
  if (!maybe_num_partitions) {
    throw std::runtime_error("Failed to get num_partitions");
  }
  num_partitions_ = maybe_num_partitions.value();

  auto maybe_min_sm_count = component->getParameter<uint32_t>("min_sm_count");
  if (!maybe_min_sm_count) {
    throw std::runtime_error("Failed to get min_sm_count");
  }
  // Store as min_sm_size_ to maintain public API compatibility
  min_sm_size_ = maybe_min_sm_count.value();

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

  auto maybe_default_context = component->getParameter<int32_t>("default_context");
  if (!maybe_default_context) {
    throw std::runtime_error("Failed to get default_context");
  }
  default_context_index_ = maybe_default_context.value();
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
  // Spec keys below use the GXF-registered names so that GXFResource::set_parameters() forwards
  // them without remapping.  The Python binding still exposes user-friendly kwarg names (e.g.
  // ``flags``, ``min_sm_size``, ``default_context_index``) and maps them to these keys via Arg
  // names in the PyCudaGreenContextPool constructor.
  spec.param(flags_,
             "green_context_flags",
             "Green Context Flags",
             "Flags for CUDA green contexts in the pool. The flag value will be passed to CUDA's "
             "cuDevSmResourceSplitByCount and cuGreenCtxCreate when creating the green contexts. "
             "See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html"
             "#group__CUDA__GREEN__CONTEXTS.",
             kDefaultFlags);
  spec.param(num_partitions_,
             "num_partitions",
             "Number of Partitions",
             "Number of partitions to create for the green context pool.",
             kDefaultNumPartitions);
  spec.param(min_sm_size_,
             "min_sm_count",
             "Minimum SM Count",
             "The minimum number of SMs used for green context creation.",
             kDefaultMinSM);
  spec.param(sms_per_partition_,
             "sms_per_partition",
             "SMs per Partition",
             "The number of SMs to allocate per partition. If empty, the SMs will be distributed "
             "evenly across partitions.",
             {0});
  spec.param(default_context_index_,
             "default_context",
             "Default Context",
             "The index of the default green context to use. When index < 0, the last partition's "
             "index will be used. (Default: -1)",
             -1);
}

bool CudaGreenContextPool::is_partitioning_supported(
    int32_t dev_id, uint32_t min_sm_count, const std::vector<uint32_t>& sms_per_partition) {
  // ``min_sm_count`` is used as a divisor below.
  if (min_sm_count == 0) {
    throw std::invalid_argument(
        "CudaGreenContextPool::is_partitioning_supported: min_sm_count must be > 0");
  }

  // Resolve CUDA driver entry points via cudaGetDriverEntryPointByVersion.
  using cuDeviceGet_t = CUresult (*)(CUdevice*, int);
  using cuDeviceGetDevResource_t = CUresult (*)(CUdevice, CUdevResource*, CUdevResourceType);
  using cuDevSmResourceSplitByCount_t = CUresult (*)(
      CUdevResource*, unsigned int*, CUdevResource*, CUdevResource*, unsigned int, unsigned int);
  using cuDevResourceGenerateDesc_t =
      CUresult (*)(CUdevResourceDesc*, CUdevResource*, unsigned int);

  // Clamp the version arg to the driver's max supported version.  Mirrors the
  // GXF-side fix in ``gxf/std/cuda_green_context_pool.cpp::loadCudaDriver``.
  // Without this clamp, ``cudaGetDriverEntryPointByVersion`` rejects (with
  // ``cudaErrorInvalidValue``) entry-point lookups at any ``version >
  // cudaDriverGetVersion()`` in Minor Version Compatibility (MVC) mode --
  // even for symbols available since CUDA 3 (e.g. ``cuDeviceGet``) -- causing
  // ``is_partitioning_supported`` to give false negatives on configurations
  // where the bundled CUDA Toolkit is a newer minor version than the host
  // driver and CUDA Forward Compatibility is unavailable.
  int rt_version = 0;
  int drv_version = 0;
  HOLOSCAN_CUDA_CALL_DEBUG(cudaRuntimeGetVersion(&rt_version));
  HOLOSCAN_CUDA_CALL_DEBUG(cudaDriverGetVersion(&drv_version));
  int version = (drv_version > 0 && drv_version < rt_version) ? drv_version : rt_version;
  HOLOSCAN_LOG_DEBUG(
      "is_partitioning_supported: CUDA runtime version: {}, driver version: {}, "
      "using version: {}",
      rt_version,
      drv_version,
      version);

  cuDeviceGet_t fn_DeviceGet = nullptr;
  cuDeviceGetDevResource_t fn_GetDevResource = nullptr;
  cuDevSmResourceSplitByCount_t fn_SplitByCount = nullptr;
  cuDevResourceGenerateDesc_t fn_GenerateDesc = nullptr;
  cudaDriverEntryPointQueryResult driver_status;

  HOLOSCAN_CUDA_CALL_DEBUG(cudaGetDriverEntryPointByVersion("cuDeviceGet",
                                                            reinterpret_cast<void**>(&fn_DeviceGet),
                                                            version,
                                                            cudaEnableDefault,
                                                            &driver_status));
  HOLOSCAN_CUDA_CALL_DEBUG(
      cudaGetDriverEntryPointByVersion("cuDeviceGetDevResource",
                                       reinterpret_cast<void**>(&fn_GetDevResource),
                                       version,
                                       cudaEnableDefault,
                                       &driver_status));
  HOLOSCAN_CUDA_CALL_DEBUG(
      cudaGetDriverEntryPointByVersion("cuDevSmResourceSplitByCount",
                                       reinterpret_cast<void**>(&fn_SplitByCount),
                                       version,
                                       cudaEnableDefault,
                                       &driver_status));
  HOLOSCAN_CUDA_CALL_DEBUG(
      cudaGetDriverEntryPointByVersion("cuDevResourceGenerateDesc",
                                       reinterpret_cast<void**>(&fn_GenerateDesc),
                                       version,
                                       cudaEnableDefault,
                                       &driver_status));

  if (!fn_DeviceGet || !fn_GetDevResource || !fn_SplitByCount || !fn_GenerateDesc) {
    HOLOSCAN_LOG_DEBUG("is_partitioning_supported: CUDA driver API for green context unavailable");
    return false;
  }

  int device_count = 0;
  if (HOLOSCAN_CUDA_CALL_DEBUG(cudaGetDeviceCount(&device_count)) != cudaSuccess) {
    return false;
  }
  if (dev_id < 0 || dev_id >= device_count) {
    HOLOSCAN_LOG_DEBUG(
        "is_partitioning_supported: dev_id {} out of range [0, {})", dev_id, device_count);
    return false;
  }
  CUdevice device;
  if (fn_DeviceGet(&device, dev_id) != CUDA_SUCCESS) {
    HOLOSCAN_LOG_DEBUG("is_partitioning_supported: cuDeviceGet failed for dev_id {}", dev_id);
    return false;
  }

  CUdevResource device_resource{};
  if (fn_GetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS) {
    HOLOSCAN_LOG_DEBUG("is_partitioning_supported: cuDeviceGetDevResource failed for dev_id {}",
                       dev_id);
    return false;
  }

  int sm_count = 0;
  if (HOLOSCAN_CUDA_CALL_DEBUG(cudaDeviceGetAttribute(
          &sm_count, cudaDevAttrMultiProcessorCount, dev_id)) != cudaSuccess) {
    return false;
  }

  unsigned int expected_groups = static_cast<unsigned int>(sm_count) / min_sm_count;
  // sm_count < min_sm_count -- the GPU does not have enough SMs for a single
  // group of the requested minimum size.
  if (expected_groups == 0) {
    HOLOSCAN_LOG_DEBUG(
        "is_partitioning_supported: 0 expected groups (sm_count={}, min_sm_count={})",
        sm_count,
        min_sm_count);
    return false;
  }
  unsigned int nb_groups = expected_groups;
  CUdevResource remaining{};
  std::vector<CUdevResource> split_result(expected_groups);

  // ``cuDevSmResourceSplitByCount`` only accepts ``CU_DEV_SM_RESOURCE_SPLIT_*`` flags
  // (see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html).
  // The two valid values are 0 and ``CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING``.
  //
  // Note on ``flags=0``: the driver docs say "Zero is valid for default behavior",
  // but "default behavior" means the driver enforces the hardware-defined minimum
  // SM count and SM coscheduling alignment.
  // ``CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`` relaxes those constraints,
  // allowing finer-grained ``min_sm_count`` values than the hardware default.
  // Passing ``flags=0`` therefore *rejects* probes for ``min_sm_count`` below the
  // hardware minimum even though the resulting pool would be perfectly usable --
  // e.g. on Blackwell sm_10x/sm_11x where the hardware default is 8, ``flags=0``
  // rejects ``min_sm_count=4`` and our resolver would skip the test entirely.
  //
  // We therefore use ``CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`` here to
  // match what GXF's CudaGreenContextPool::reserveGreenContexts() ends up
  // passing to the driver.  GXF reads its ``green_context_flags_`` parameter and
  // forwards it directly to ``cuDevSmResourceSplitByCount``.  That parameter is
  // initialised with the CUDA Runtime constant ``cudaStreamNonBlocking`` -- which
  // is technically the wrong namespace (it is a stream creation flag, not a
  // resource-split flag), but its numeric value (1) coincides with
  // ``CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING``, so the driver still
  // interprets it as the IGNORE bit.  Net effect: GXF effectively always relaxes
  // SM coscheduling alignment when splitting SMs, regardless of compute
  // capability.  The probe must mirror this exactly -- using the *correct* flag
  // name -- to avoid false-negative rejections of configurations that GXF would
  // later accept at pool-creation time.
  unsigned int flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;

  if (fn_SplitByCount(
          split_result.data(), &nb_groups, &device_resource, &remaining, flags, min_sm_count) !=
      CUDA_SUCCESS) {
    HOLOSCAN_LOG_DEBUG(
        "is_partitioning_supported: cuDevSmResourceSplitByCount failed"
        " (sm_count={}, min_sm_count={}, flags={})",
        sm_count,
        min_sm_count,
        flags);
    return false;
  }
  if (nb_groups != expected_groups) {
    HOLOSCAN_LOG_DEBUG(
        "is_partitioning_supported: expected {} SM groups but driver returned {}"
        " (sm_count={}, min_sm_count={})",
        expected_groups,
        nb_groups,
        sm_count,
        min_sm_count);
    return false;
  }

  unsigned int split_idx = 0;
  uint32_t total_requested = 0;
  for (auto s : sms_per_partition) {
    total_requested += s;
  }
  bool has_remainder = total_requested < static_cast<uint32_t>(sm_count);
  auto num_partitions =
      static_cast<unsigned int>(sms_per_partition.size()) + (has_remainder ? 1 : 0);

  for (unsigned int i = 0; i < num_partitions; ++i) {
    auto requested_sms = static_cast<uint32_t>(
        (i < sms_per_partition.size()) ? sms_per_partition[i] : (sm_count - total_requested));
    unsigned int num_resources = requested_sms / min_sm_count;
    // Defensive bounds check: split_result has expected_groups slots and
    // fn_GenerateDesc will read num_resources of them starting at split_idx.
    // Reject malformed inputs (e.g. partitions whose total exceeds sm_count, or
    // partitions misaligned with min_sm_count) before constructing an
    // out-of-range pointer span.
    if (static_cast<size_t>(split_idx) + num_resources > split_result.size()) {
      HOLOSCAN_LOG_DEBUG(
          "is_partitioning_supported: partition {} would exceed split_result bounds"
          " (split_idx={}, num_resources={}, split_result.size={}, requested_sms={},"
          " min_sm_count={}, sm_count={})",
          i,
          split_idx,
          num_resources,
          split_result.size(),
          requested_sms,
          min_sm_count,
          sm_count);
      return false;
    }
    CUdevResourceDesc desc{};
    if (fn_GenerateDesc(&desc, &split_result[split_idx], num_resources) != CUDA_SUCCESS) {
      HOLOSCAN_LOG_DEBUG(
          "is_partitioning_supported: cuDevResourceGenerateDesc failed for partition {}"
          " ({} SMs, {} resources)",
          i,
          requested_sms,
          num_resources);
      return false;
    }
    split_idx += num_resources;
  }

  return true;
}

}  // namespace holoscan
