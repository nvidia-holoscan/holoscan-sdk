/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#include "cuda_execution/cuda_green_context_driver.hpp"

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

namespace holoscan::detail::cuda_test {

using cuda_execution_detail::CudaGreenContextDriver;

// Synthetic handle bands keep fake descriptors, contexts, and streams distinct and inspectable.
constexpr std::uintptr_t kFakeDescriptorHandleBase = 0x1000U;
constexpr std::uintptr_t kFakeGreenContextHandleBase = 0x2000U;
constexpr std::uintptr_t kFakeCudaContextHandleOffset = 0x1000U;
constexpr std::uintptr_t kFakeStreamHandleBase = 0x4000U;
constexpr std::uintptr_t kFakePrimaryContextHandleBase = 0x8000U;

// Records fake CUDA Driver handles, injected failures, stream identities, and cleanup activity for
// runtime materialization tests. Used by
// RuntimeMaterializationTest.GreenContextFailureBoundariesRollbackAndReleaseDeviceClaim.
struct FakeDriverState {
  CUdevResource device_resource{};
  std::uint8_t uuid_offset{1U};
  std::uint32_t context_create_calls{};
  std::uint32_t stream_create_calls{};
  std::uint32_t stream_synchronize_calls{};
  std::optional<std::uint32_t> fail_context_create_index;
  std::optional<std::uint32_t> fail_stream_create_index;
  CUresult stream_synchronize_result{CUDA_SUCCESS};
  std::optional<CUstream> fail_stream_synchronize;
  std::function<void()> stream_synchronize_hook;
  CUresult context_destroy_result{CUDA_SUCCESS};
  CUresult stream_destroy_result{CUDA_SUCCESS};
  std::optional<CUstream> fail_stream_destroy;
  bool return_null_stream{};
  bool observe_ordinary_stream{};
  bool alias_device_ordinals{};
  std::optional<CUgreenCtx> observed_context;
  CUresult identity_result{CUDA_SUCCESS};
  std::vector<CUgreenCtx> created_contexts;
  std::vector<CUgreenCtx> destroyed_contexts;
  std::vector<CUstream> created_streams;
  std::vector<CUstream> synchronized_streams;
  std::vector<CUstream> destroyed_streams;
  std::unordered_map<std::uintptr_t, unsigned int> descriptor_sms;
  std::unordered_map<std::uintptr_t, unsigned int> context_sms;
  std::unordered_map<std::uintptr_t, CUgreenCtx> stream_contexts;
  std::uint32_t next_descriptor{1U};
  std::uint32_t next_context{1U};
  std::uint32_t next_stream{1U};
  CUcontext current_context{};
};

inline FakeDriverState* fake_state = nullptr;

[[nodiscard]] inline std::uintptr_t handle_value(const void* handle) noexcept {
  return reinterpret_cast<std::uintptr_t>(handle);
}

inline CUresult device_get(CUdevice* device, int ordinal) {
  if (fake_state == nullptr || device == nullptr || ordinal < 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = ordinal;
  return CUDA_SUCCESS;
}

inline CUresult device_get_uuid(CUuuid* uuid, CUdevice device) {
  if (fake_state == nullptr || uuid == nullptr || device < 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  for (std::size_t index = 0U; index < sizeof(uuid->bytes); ++index) {
    const auto device_offset =
        fake_state->alias_device_ordinals ? 0U : static_cast<std::uint32_t>(device) * 32U;
    uuid->bytes[index] = static_cast<char>(fake_state->uuid_offset + device_offset + index);
  }
  return CUDA_SUCCESS;
}

inline CUresult device_get_attribute(int* value, CUdevice_attribute attribute, CUdevice device) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
      *value = 0;
      return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
      *value = 0x65 + (fake_state != nullptr && !fake_state->alias_device_ordinals ? device : 0);
      return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
      *value = 0;
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

inline CUresult synchronize_stream(CUstream stream) {
  if (fake_state == nullptr || stream == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  ++fake_state->stream_synchronize_calls;
  fake_state->synchronized_streams.push_back(stream);
  if (fake_state->stream_synchronize_hook) {
    fake_state->stream_synchronize_hook();
  }
  if (fake_state->fail_stream_synchronize == stream) {
    return CUDA_ERROR_UNKNOWN;
  }
  return fake_state->stream_synchronize_result;
}

inline CUresult device_get_resource(CUdevice, CUdevResource* resource, CUdevResourceType type) {
  if (fake_state == nullptr || resource == nullptr || type != CU_DEV_RESOURCE_TYPE_SM) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resource = fake_state->device_resource;
  return CUDA_SUCCESS;
}

inline CUresult split_by_count(CUdevResource* result, unsigned int* count, const CUdevResource*,
                               CUdevResource* remainder, unsigned int flags, unsigned int minimum) {
  if (fake_state == nullptr || count == nullptr || minimum == 0U || flags != 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const unsigned int available = fake_state->device_resource.sm.smCount / minimum;
  if (result == nullptr) {
    *count = available;
    return CUDA_SUCCESS;
  }
  const unsigned int created = std::min(*count, available);
  for (unsigned int index = 0U; index < created; ++index) {
    result[index] = fake_state->device_resource;
    result[index].sm.smCount = minimum;
  }
  *count = created;
  if (remainder != nullptr) {
    *remainder = {};
  }
  return CUDA_SUCCESS;
}

inline CUresult split(CUdevResource* result, unsigned int count, const CUdevResource*,
                      CUdevResource* remainder, unsigned int flags,
                      CudaGreenContextDriver::SmResourceGroupParams* params) {
  if (fake_state == nullptr || result == nullptr || params == nullptr || count == 0U ||
      flags != 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  unsigned int used = 0U;
  for (unsigned int index = 0U; index < count; ++index) {
    used += params[index].sm_count;
    result[index] = fake_state->device_resource;
    result[index].sm.smCount = params[index].sm_count;
  }
  if (remainder != nullptr) {
    *remainder = {};
    if (used < fake_state->device_resource.sm.smCount) {
      remainder->type = CU_DEV_RESOURCE_TYPE_SM;
      remainder->sm.smCount = fake_state->device_resource.sm.smCount - used;
    }
  }
  return used <= fake_state->device_resource.sm.smCount ? CUDA_SUCCESS
                                                        : CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION;
}

inline CUresult generate_desc(CUdevResourceDesc* descriptor, CUdevResource* resources,
                              unsigned int count) {
  if (fake_state == nullptr || descriptor == nullptr || resources == nullptr || count == 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto value = kFakeDescriptorHandleBase + fake_state->next_descriptor++;
  *descriptor = reinterpret_cast<CUdevResourceDesc>(value);
  unsigned int sms = 0U;
  for (unsigned int index = 0U; index < count; ++index) {
    sms += resources[index].sm.smCount;
  }
  fake_state->descriptor_sms.emplace(value, sms);
  return CUDA_SUCCESS;
}

inline CUresult create_context(CUgreenCtx* context, CUdevResourceDesc descriptor, CUdevice,
                               unsigned int flags) {
  if (fake_state == nullptr || context == nullptr || flags != CU_GREEN_CTX_DEFAULT_STREAM) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::uint32_t call = fake_state->context_create_calls++;
  if (fake_state->fail_context_create_index == call) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  const auto found = fake_state->descriptor_sms.find(handle_value(descriptor));
  if (found == fake_state->descriptor_sms.end()) {
    return CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION;
  }
  const auto value = kFakeGreenContextHandleBase + fake_state->next_context++;
  *context = reinterpret_cast<CUgreenCtx>(value);
  fake_state->created_contexts.push_back(*context);
  fake_state->context_sms.emplace(value, found->second);
  return CUDA_SUCCESS;
}

inline CUresult destroy_context(CUgreenCtx context) {
  if (fake_state == nullptr || context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  fake_state->destroyed_contexts.push_back(context);
  return fake_state->context_destroy_result;
}

inline CUresult context_resource(CUgreenCtx context, CUdevResource* resource,
                                 CUdevResourceType type) {
  if (fake_state == nullptr || resource == nullptr || type != CU_DEV_RESOURCE_TYPE_SM) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto found = fake_state->context_sms.find(handle_value(context));
  if (found == fake_state->context_sms.end()) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  *resource = fake_state->device_resource;
  resource->sm.smCount = found->second;
  return CUDA_SUCCESS;
}

inline CUresult context_from_green(CUcontext* context, CUgreenCtx green_context) {
  if (context == nullptr || green_context == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *context =
      reinterpret_cast<CUcontext>(handle_value(green_context) + kFakeCudaContextHandleOffset);
  return CUDA_SUCCESS;
}

inline CUresult create_stream(CUstream* stream, CUgreenCtx context, unsigned int flags, int) {
  if (fake_state == nullptr || stream == nullptr || context == nullptr ||
      flags != CU_STREAM_NON_BLOCKING) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::uint32_t call = fake_state->stream_create_calls++;
  if (fake_state->fail_stream_create_index == call) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (fake_state->return_null_stream) {
    *stream = nullptr;
    return CUDA_SUCCESS;
  }
  const auto value = kFakeStreamHandleBase + fake_state->next_stream++;
  *stream = reinterpret_cast<CUstream>(value);
  fake_state->created_streams.push_back(*stream);
  fake_state->stream_contexts.emplace(value, context);
  return CUDA_SUCCESS;
}

inline CUresult destroy_stream(CUstream stream) {
  if (fake_state == nullptr || stream == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  fake_state->destroyed_streams.push_back(stream);
  if (fake_state->fail_stream_destroy == stream) {
    return CUDA_ERROR_UNKNOWN;
  }
  return fake_state->stream_destroy_result;
}

inline CUresult stream_context(CUstream stream, CUgreenCtx* context) {
  if (fake_state == nullptr || stream == nullptr || context == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (fake_state->identity_result != CUDA_SUCCESS) {
    return fake_state->identity_result;
  }
  if (fake_state->observe_ordinary_stream) {
    *context = nullptr;
  } else if (fake_state->observed_context) {
    *context = *fake_state->observed_context;
  } else {
    const auto found = fake_state->stream_contexts.find(handle_value(stream));
    if (found == fake_state->stream_contexts.end()) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    *context = found->second;
  }
  return CUDA_SUCCESS;
}

inline CUresult error_name(CUresult, const char** text) {
  if (text != nullptr) {
    *text = "FAKE_CUDA_ERROR";
  }
  return CUDA_SUCCESS;
}

inline CUresult error_string(CUresult, const char** text) {
  if (text != nullptr) {
    *text = "injected fake driver failure";
  }
  return CUDA_SUCCESS;
}

[[nodiscard]] inline std::shared_ptr<CudaGreenContextDriver> make_driver() {
  auto driver = std::make_shared<CudaGreenContextDriver>();
  driver->device_get = device_get;
  driver->device_get_uuid = device_get_uuid;
  driver->device_get_attribute = device_get_attribute;
  driver->device_get_resource = device_get_resource;
  driver->sm_resource_split_by_count = split_by_count;
  driver->sm_resource_split = split;
  driver->resource_generate_desc = generate_desc;
  driver->green_context_create = create_context;
  driver->green_context_destroy = destroy_context;
  driver->green_context_get_resource = context_resource;
  driver->context_from_green_context = context_from_green;
  driver->green_context_stream_create = create_stream;
  driver->stream_synchronize = synchronize_stream;
  driver->stream_destroy = destroy_stream;
  driver->stream_get_green_context = stream_context;
  driver->get_error_name = error_name;
  driver->get_error_string = error_string;
  return driver;
}

// Return a handle in the fake Green Context band that the generator deliberately never emits.
[[nodiscard]] inline CUgreenCtx unrecognized_green_context() noexcept {
  return reinterpret_cast<CUgreenCtx>(kFakeGreenContextHandleBase);
}

[[nodiscard]] inline FakeDriverState make_state(unsigned int sms = 80U, unsigned int minimum = 8U,
                                                unsigned int alignment = 8U) {
  FakeDriverState state;
  state.device_resource.type = CU_DEV_RESOURCE_TYPE_SM;
  state.device_resource.sm.smCount = sms;
  state.device_resource.sm.minSmPartitionSize = minimum;
  state.device_resource.sm.smCoscheduledAlignment = alignment;
  return state;
}

// Installs a FakeDriverState-backed CUDA Driver override for one test scope and restores the
// previous global state on destruction. Used by
// RuntimeMaterializationTest.GreenContextNativeFailuresPreserveDetailsAndRollback.
class FakeDriverScope final {
 public:
  explicit FakeDriverScope(FakeDriverState& state,
                           std::shared_ptr<const CudaGreenContextDriver> driver = make_driver())
      : state_(&state) {
    previous_state_ = fake_state;
    fake_state = state_;
    previous_driver_ =
        cuda_execution_detail::set_cuda_green_context_driver_for_testing(std::move(driver));
  }

  FakeDriverScope(const FakeDriverScope&) = delete;
  FakeDriverScope& operator=(const FakeDriverScope&) = delete;

  ~FakeDriverScope() {
    static_cast<void>(cuda_execution_detail::set_cuda_green_context_driver_for_testing(
        std::move(previous_driver_)));
    fake_state = previous_state_;
  }

 private:
  FakeDriverState* state_{};
  FakeDriverState* previous_state_{};
  std::shared_ptr<const CudaGreenContextDriver> previous_driver_;
};

}  // namespace holoscan::detail::cuda_test

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
