/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

#include "cuda_green_context_driver.hpp"

#include <fmt/format.h>

#include <array>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>

#include <holoscan/core/error_helpers.hpp>

#include "cuda_driver_entry_points.hpp"

namespace holoscan::cuda_execution_detail {
namespace {

struct DriverOverrideState {
  std::mutex mutex;
  std::shared_ptr<const CudaGreenContextDriver> driver;
};

[[nodiscard]] DriverOverrideState& driver_override_state() {
  static DriverOverrideState state;
  return state;
}

expected<void, Error> validate_driver(const CudaGreenContextDriver& driver) {
  const std::array<std::pair<const void*, const char*>, 16> required{{
      {reinterpret_cast<const void*>(driver.device_get), "cuDeviceGet"},
      {reinterpret_cast<const void*>(driver.device_get_uuid), "cuDeviceGetUuid[_v2]"},
      {reinterpret_cast<const void*>(driver.device_get_attribute), "cuDeviceGetAttribute"},
      {reinterpret_cast<const void*>(driver.device_get_resource), "cuDeviceGetDevResource"},
      {reinterpret_cast<const void*>(driver.sm_resource_split_by_count),
       "cuDevSmResourceSplitByCount"},
      {reinterpret_cast<const void*>(driver.resource_generate_desc), "cuDevResourceGenerateDesc"},
      {reinterpret_cast<const void*>(driver.green_context_create), "cuGreenCtxCreate"},
      {reinterpret_cast<const void*>(driver.green_context_destroy), "cuGreenCtxDestroy"},
      {reinterpret_cast<const void*>(driver.green_context_get_resource),
       "cuGreenCtxGetDevResource"},
      {reinterpret_cast<const void*>(driver.context_from_green_context), "cuCtxFromGreenCtx"},
      {reinterpret_cast<const void*>(driver.green_context_stream_create), "cuGreenCtxStreamCreate"},
      {reinterpret_cast<const void*>(driver.stream_synchronize), "cuStreamSynchronize"},
      {reinterpret_cast<const void*>(driver.stream_destroy), "cuStreamDestroy"},
      {reinterpret_cast<const void*>(driver.stream_get_green_context), "cuStreamGetGreenCtx"},
      {reinterpret_cast<const void*>(driver.get_error_name), "cuGetErrorName"},
      {reinterpret_cast<const void*>(driver.get_error_string), "cuGetErrorString"},
  }};
  for (const auto& [function, name] : required) {
    if (function == nullptr) {
      return make_unexpected(Error{
          ErrorCode::kFailure, fmt::format("CUDA driver entry point '{}' is unavailable", name)});
    }
  }
  return {};
}

expected<std::shared_ptr<const CudaGreenContextDriver>, Error> load_driver() {
  auto driver = std::make_shared<CudaGreenContextDriver>();
  const std::array<std::pair<void**, const char*>, 14> symbols{{
      {reinterpret_cast<void**>(&driver->device_get), "cuDeviceGet"},
      {reinterpret_cast<void**>(&driver->device_get_attribute), "cuDeviceGetAttribute"},
      {reinterpret_cast<void**>(&driver->device_get_resource), "cuDeviceGetDevResource"},
      {reinterpret_cast<void**>(&driver->sm_resource_split_by_count),
       "cuDevSmResourceSplitByCount"},
      {reinterpret_cast<void**>(&driver->resource_generate_desc), "cuDevResourceGenerateDesc"},
      {reinterpret_cast<void**>(&driver->green_context_create), "cuGreenCtxCreate"},
      {reinterpret_cast<void**>(&driver->green_context_destroy), "cuGreenCtxDestroy"},
      {reinterpret_cast<void**>(&driver->green_context_get_resource), "cuGreenCtxGetDevResource"},
      {reinterpret_cast<void**>(&driver->context_from_green_context), "cuCtxFromGreenCtx"},
      {reinterpret_cast<void**>(&driver->stream_synchronize), "cuStreamSynchronize"},
      {reinterpret_cast<void**>(&driver->stream_destroy), "cuStreamDestroy"},
      {reinterpret_cast<void**>(&driver->stream_get_green_context), "cuStreamGetGreenCtx"},
      {reinterpret_cast<void**>(&driver->get_error_name), "cuGetErrorName"},
      {reinterpret_cast<void**>(&driver->get_error_string), "cuGetErrorString"},
  }};
  for (const auto& [function, name] : symbols) {
    if (auto loaded = load_driver_entry_point(function, name, kCUDADriverABIVersion12040);
        !loaded) {
      return forward_error(loaded);
    }
  }
  // The Green Context resource APIs use the CUDA 12.4 ABI, but stream creation was added in 12.5.
  if (auto loaded =
          load_driver_entry_point(reinterpret_cast<void**>(&driver->green_context_stream_create),
                                  "cuGreenCtxStreamCreate",
                                  kCUDADriverABIVersion12050);
      !loaded) {
    return forward_error(loaded);
  }
  // CUDA exposes ABI-compatible UUID queries under two names across driver/toolkit generations.
  // Prefer the current spelling, but retain the unsuffixed fallback supported by older drivers.
  if (auto loaded = load_driver_entry_point(reinterpret_cast<void**>(&driver->device_get_uuid),
                                            "cuDeviceGetUuid_v2",
                                            kCUDADriverABIVersion12040);
      !loaded) {
    if (auto fallback = load_driver_entry_point(reinterpret_cast<void**>(&driver->device_get_uuid),
                                                "cuDeviceGetUuid",
                                                kCUDADriverABIVersion12040);
        !fallback) {
      return forward_error(fallback);
    }
  }
  // CUDA 13.1 added the exact heterogeneous structured-split entry point. Keep it optional so the
  // SDK's CUDA 13.0 baseline uses the equivalent equal-unit aggregation path.
  (void)load_driver_entry_point(reinterpret_cast<void**>(&driver->sm_resource_split),
                                "cuDevSmResourceSplit",
                                kCUDADriverABIVersion13010);
  if (auto valid = validate_driver(*driver); !valid) {
    return forward_error(valid);
  }
  return std::shared_ptr<const CudaGreenContextDriver>{std::move(driver)};
}

Error driver_error(const CudaGreenContextDriver& driver, const char* operation, CUresult result) {
  const char* name = nullptr;
  const char* detail = nullptr;
  (void)driver.get_error_name(result, &name);
  (void)driver.get_error_string(result, &detail);
  return Error{ErrorCode::kFailure,
               fmt::format("{}: {} - {}",
                           operation,
                           name != nullptr ? name : "unknown CUDA error",
                           detail != nullptr ? detail : "no driver detail")};
}

}  // namespace

expected<std::shared_ptr<const CudaGreenContextDriver>, Error> cuda_green_context_driver() {
  {
    auto& override = driver_override_state();
    const std::lock_guard<std::mutex> lock(override.mutex);
    if (override.driver != nullptr) {
      if (auto valid = validate_driver(*override.driver); !valid) {
        return forward_error(valid);
      }
      return override.driver;
    }
  }
  static const expected<std::shared_ptr<const CudaGreenContextDriver>, Error> production_driver =
      load_driver();
  return production_driver;
}

expected<CudaGreenContextDeviceCapability, Error> query_cuda_green_context_device_capability(
    const CudaGreenContextDriver& driver, std::int32_t device_id) {
  if (device_id < 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CUDA Green Context device id must be nonnegative"});
  }
  CUdevice device = 0;
  if (const CUresult result = driver.device_get(&device, device_id); result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(driver, "cuDeviceGet failed", result));
  }
  CUuuid uuid{};
  if (const CUresult result = driver.device_get_uuid(&uuid, device); result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(driver, "cuDeviceGetUuid failed", result));
  }
  int pci_domain = 0;
  int pci_bus = 0;
  int pci_device = 0;
  for (const auto [attribute, output] :
       {std::pair{CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, &pci_domain},
        std::pair{CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, &pci_bus},
        std::pair{CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, &pci_device}}) {
    if (const CUresult result = driver.device_get_attribute(output, attribute, device);
        result != CUDA_SUCCESS) {
      return make_unexpected(driver_error(driver, "cuDeviceGetAttribute failed", result));
    }
  }
  std::string uuid_hex;
  uuid_hex.reserve(sizeof(uuid.bytes) * 2U);
  for (const char byte : uuid.bytes) {
    fmt::format_to(std::back_inserter(uuid_hex), "{:02x}", static_cast<unsigned char>(byte));
  }
  CUdevResource resource{};
  if (const CUresult result =
          driver.device_get_resource(device, &resource, CU_DEV_RESOURCE_TYPE_SM);
      result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(driver, "cuDeviceGetDevResource failed", result));
  }
  if (resource.type != CU_DEV_RESOURCE_TYPE_SM || resource.sm.smCount == 0U ||
      resource.sm.minSmPartitionSize == 0U || resource.sm.smCoscheduledAlignment == 0U) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        fmt::format("CUDA device {} returned an invalid Green Context SM resource", device_id)});
  }
  const std::uint32_t quantum =
      std::lcm(resource.sm.minSmPartitionSize, resource.sm.smCoscheduledAlignment);
  if (quantum == 0U || quantum > resource.sm.smCount) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        fmt::format("CUDA device {} cannot form one structured Green Context group", device_id)});
  }
  // Ask the driver to simulate the equal-size structured split. The returned count can be lower
  // than smCount / quantum when the device's cluster topology imposes additional constraints.
  unsigned int structured_unit_count = 0U;
  if (const CUresult result = driver.sm_resource_split_by_count(
          nullptr, &structured_unit_count, &resource, nullptr, 0U, quantum);
      result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(driver, "cuDevSmResourceSplitByCount failed", result));
  }
  const std::uint64_t structured_sm_capacity =
      static_cast<std::uint64_t>(structured_unit_count) * quantum;
  if (structured_unit_count == 0U || structured_sm_capacity > resource.sm.smCount) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              fmt::format("CUDA device {} returned an invalid structured Green Context capacity",
                          device_id)});
  }
  return CudaGreenContextDeviceCapability{
      .device_id = device_id,
      .deployment_device_id = fmt::format(
          "cuda.uuid.{}.pci.{:04x}:{:02x}:{:02x}", uuid_hex, pci_domain, pci_bus, pci_device),
      .device_sm_count = resource.sm.smCount,
      .min_sm_partition_size = resource.sm.minSmPartitionSize,
      .sm_coscheduled_alignment = resource.sm.smCoscheduledAlignment,
      .structured_sm_quantum = quantum,
      .max_structured_context_count = structured_unit_count,
  };
}

expected<CudaGreenContextStreamIdentity, Error> query_cuda_green_context_stream_identity(
    const CudaGreenContextDriver& driver, CUstream stream, CUgreenCtx expected_context) {
  if (stream == nullptr) {
    return CudaGreenContextStreamIdentity{.kind = CudaGreenContextStreamIdentityKind::kNullStream};
  }
  if (driver.stream_get_green_context == nullptr) {
    return CudaGreenContextStreamIdentity{.kind = CudaGreenContextStreamIdentityKind::kUnsupported};
  }
  CUgreenCtx observed_context = nullptr;
  if (const CUresult result = driver.stream_get_green_context(stream, &observed_context);
      result != CUDA_SUCCESS) {
    return make_unexpected(driver_error(driver, "cuStreamGetGreenCtx failed", result));
  }
  if (observed_context == nullptr) {
    return CudaGreenContextStreamIdentity{.kind =
                                              CudaGreenContextStreamIdentityKind::kOrdinaryContext};
  }
  return CudaGreenContextStreamIdentity{
      .kind = observed_context == expected_context
                  ? CudaGreenContextStreamIdentityKind::kExpectedGreenContext
                  : CudaGreenContextStreamIdentityKind::kWrongGreenContext,
      .observed_context = observed_context,
  };
}

std::shared_ptr<const CudaGreenContextDriver> set_cuda_green_context_driver_for_testing(
    std::shared_ptr<const CudaGreenContextDriver> driver) noexcept {
  auto& override = driver_override_state();
  const std::lock_guard<std::mutex> lock(override.mutex);
  return std::exchange(override.driver, std::move(driver));
}

}  // namespace holoscan::cuda_execution_detail

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
