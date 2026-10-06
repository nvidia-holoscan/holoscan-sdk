/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#include "cuda_execution/cuda_driver_entry_points.hpp"
#include "cuda_execution/cuda_green_context_driver.hpp"

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

namespace holoscan {
namespace {

using cuda_execution_detail::CudaGreenContextDriver;

TEST(CudaGreenContextDriver, SymbolABIVersionDoesNotFollowNewerRuntimeVersion) {
  // Key setup: model a CUDA 13.0-capable driver underneath a newer CUDA 13.1 Runtime. The query
  // accepts both baseline ABI generations and rejects a 13.1-only request.
  static std::uint8_t fake_symbol;
  std::int32_t observed_version = 0;
  const auto query = [&](const char*,
                         void** output,
                         std::int32_t version,
                         cudaDriverEntryPointQueryResult* status) {
    observed_version = version;
    if (version > 13000) {
      *output = nullptr;
      *status = cudaDriverEntryPointVersionNotSufficent;
      return cudaErrorInvalidValue;
    }
    *output = &fake_symbol;
    *status = cudaDriverEntryPointSuccess;
    return cudaSuccess;
  };
  void* symbol = nullptr;
  const auto baseline = cuda_execution_detail::load_driver_entry_point_with_query(
      &symbol,
      "cuDevSmResourceSplitByCount",
      cuda_execution_detail::kCUDADriverABIVersion12040,
      query);

  // Key assertion: baseline resolution succeeds with its compile-time ABI; the optional 13.1
  // symbol remains independently rejectable without disabling the baseline table.
  ASSERT_TRUE(baseline);
  EXPECT_NE(symbol, nullptr);
  EXPECT_EQ(observed_version, cuda_execution_detail::kCUDADriverABIVersion12040);
  symbol = nullptr;
  const auto stream_create = cuda_execution_detail::load_driver_entry_point_with_query(
      &symbol, "cuGreenCtxStreamCreate", cuda_execution_detail::kCUDADriverABIVersion12050, query);
  ASSERT_TRUE(stream_create);
  EXPECT_NE(symbol, nullptr);
  EXPECT_EQ(observed_version, cuda_execution_detail::kCUDADriverABIVersion12050);
  symbol = nullptr;
  const auto optional = cuda_execution_detail::load_driver_entry_point_with_query(
      &symbol, "cuDevSmResourceSplit", cuda_execution_detail::kCUDADriverABIVersion13010, query);
  EXPECT_FALSE(optional);
  EXPECT_EQ(symbol, nullptr);
  EXPECT_EQ(observed_version, cuda_execution_detail::kCUDADriverABIVersion13010);
}

// Synthetic handle bands keep generated descriptors, Green Contexts, CUDA contexts, and streams
// distinct while preserving reversible Green Context-to-stream identity.
constexpr std::uintptr_t kFakeDescriptorHandleBase = 0x1000U;
constexpr std::uintptr_t kFakeGreenContextHandleBase = 0x2000U;
constexpr std::uintptr_t kFakeCudaContextHandleOffset = 0x1000U;
constexpr std::uintptr_t kFakeStreamHandleOffset = 0x2000U;
constexpr std::uintptr_t kFirstFakeGreenContextHandle = kFakeGreenContextHandleBase + 1U;
constexpr std::uintptr_t kSecondFakeGreenContextHandle = kFakeGreenContextHandleBase + 2U;
constexpr std::uintptr_t kFirstFakeStreamHandle =
    kFirstFakeGreenContextHandle + kFakeStreamHandleOffset;

// Captures fake CUDA Driver inputs, generated handles, failure injection, and cleanup order for
// deterministic Green Context driver tests. Used by
// CudaGreenContextDriver.StructuredPoolUsesZeroSplitFlagsAndNeverUsesWeakRemainder.
struct FakeDriverState {
  CUdevResource device_resource{};
  unsigned int split_flags{999U};
  unsigned int split_by_count_flags{999U};
  std::vector<unsigned int> split_sms;
  std::vector<CUgreenCtx> created_contexts;
  std::vector<CUgreenCtx> destroyed_contexts;
  std::unordered_map<std::uintptr_t, unsigned int> descriptor_sms;
  std::unordered_map<std::uintptr_t, unsigned int> context_sms;
  std::uint32_t next_descriptor{1U};
  std::uint32_t next_context{1U};
  std::optional<std::uint32_t> fail_context_create_index;
  std::optional<unsigned int> split_available_units;
  std::optional<unsigned int> split_created_units;
  std::optional<CUgreenCtx> stream_observed_context;
  CUresult stream_identity_result{CUDA_SUCCESS};
};

FakeDriverState* fake_state = nullptr;

std::uintptr_t handle_value(const void* handle) {
  return reinterpret_cast<std::uintptr_t>(handle);
}

CUresult fake_device_get(CUdevice* device, int ordinal) {
  if (fake_state == nullptr || device == nullptr || ordinal < 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult fake_device_get_uuid(CUuuid* uuid, CUdevice device) {
  if (uuid == nullptr || device < 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  for (std::size_t index = 0U; index < sizeof(uuid->bytes); ++index) {
    uuid->bytes[index] = static_cast<char>(index + 1U);
  }
  return CUDA_SUCCESS;
}

CUresult fake_device_get_attribute(int* value, CUdevice_attribute attribute, CUdevice) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
      *value = 0;
      return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
      *value = 0x65;
      return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
      *value = 0;
      return CUDA_SUCCESS;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
}

CUresult fake_stream_synchronize(CUstream stream) {
  return stream == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

CUresult fake_device_get_resource(CUdevice, CUdevResource* resource, CUdevResourceType type) {
  if (fake_state == nullptr || resource == nullptr || type != CU_DEV_RESOURCE_TYPE_SM) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resource = fake_state->device_resource;
  return CUDA_SUCCESS;
}

CUresult fake_sm_resource_split(CUdevResource* result, unsigned int group_count,
                                const CUdevResource*, CUdevResource* remainder, unsigned int flags,
                                CudaGreenContextDriver::SmResourceGroupParams* group_params) {
  if (fake_state == nullptr || result == nullptr || group_params == nullptr || group_count == 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  fake_state->split_flags = flags;
  fake_state->split_sms.clear();
  unsigned int used = 0U;
  for (unsigned int index = 0U; index < group_count; ++index) {
    const unsigned int count = group_params[index].sm_count;
    fake_state->split_sms.push_back(count);
    used += count;
    result[index] = fake_state->device_resource;
    result[index].sm.smCount = count;
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

CUresult fake_sm_resource_split_by_count(CUdevResource* result, unsigned int* group_count,
                                         const CUdevResource*, CUdevResource* remainder,
                                         unsigned int flags, unsigned int minimum) {
  if (fake_state == nullptr || group_count == nullptr || minimum == 0U || flags != 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  fake_state->split_by_count_flags = flags;
  const unsigned int available =
      fake_state->split_available_units.value_or(fake_state->device_resource.sm.smCount / minimum);
  if (result == nullptr) {
    *group_count = available;
    return CUDA_SUCCESS;
  }
  const unsigned int created =
      std::min({*group_count, available, fake_state->split_created_units.value_or(available)});
  for (unsigned int index = 0U; index < created; ++index) {
    result[index] = fake_state->device_resource;
    result[index].sm.smCount = minimum;
  }
  *group_count = created;
  if (remainder != nullptr) {
    *remainder = {};
  }
  return CUDA_SUCCESS;
}

CUresult fake_generate_desc(CUdevResourceDesc* descriptor, CUdevResource* resources,
                            unsigned int resource_count) {
  if (fake_state == nullptr || descriptor == nullptr || resources == nullptr ||
      resource_count == 0U) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto value = kFakeDescriptorHandleBase + fake_state->next_descriptor++;
  *descriptor = reinterpret_cast<CUdevResourceDesc>(value);
  unsigned int sm_count = 0U;
  for (unsigned int index = 0U; index < resource_count; ++index) {
    sm_count += resources[index].sm.smCount;
  }
  fake_state->descriptor_sms.emplace(value, sm_count);
  return CUDA_SUCCESS;
}

CUresult fake_green_context_create(CUgreenCtx* context, CUdevResourceDesc descriptor, CUdevice,
                                   unsigned int flags) {
  if (fake_state == nullptr || context == nullptr || flags != CU_GREEN_CTX_DEFAULT_STREAM) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const std::uint32_t index = static_cast<std::uint32_t>(fake_state->created_contexts.size());
  if (fake_state->fail_context_create_index == index) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  const auto descriptor_entry = fake_state->descriptor_sms.find(handle_value(descriptor));
  if (descriptor_entry == fake_state->descriptor_sms.end()) {
    return CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION;
  }
  const auto value = kFakeGreenContextHandleBase + fake_state->next_context++;
  *context = reinterpret_cast<CUgreenCtx>(value);
  fake_state->created_contexts.push_back(*context);
  fake_state->context_sms.emplace(value, descriptor_entry->second);
  return CUDA_SUCCESS;
}

CUresult fake_green_context_destroy(CUgreenCtx context) {
  if (fake_state == nullptr || context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  fake_state->destroyed_contexts.push_back(context);
  return CUDA_SUCCESS;
}

CUresult fake_green_context_get_resource(CUgreenCtx context, CUdevResource* resource,
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

CUresult fake_context_from_green_context(CUcontext* context, CUgreenCtx green_context) {
  if (context == nullptr || green_context == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *context =
      reinterpret_cast<CUcontext>(handle_value(green_context) + kFakeCudaContextHandleOffset);
  return CUDA_SUCCESS;
}

CUresult fake_green_context_stream_create(CUstream* stream, CUgreenCtx context, unsigned int, int) {
  if (stream == nullptr || context == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *stream = reinterpret_cast<CUstream>(handle_value(context) + kFakeStreamHandleOffset);
  return CUDA_SUCCESS;
}

CUresult fake_stream_destroy(CUstream) {
  return CUDA_SUCCESS;
}

CUresult fake_stream_get_green_context(CUstream stream, CUgreenCtx* context) {
  if (context == nullptr || stream == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (fake_state->stream_identity_result != CUDA_SUCCESS) {
    return fake_state->stream_identity_result;
  }
  *context = fake_state->stream_observed_context
                 ? *fake_state->stream_observed_context
                 : reinterpret_cast<CUgreenCtx>(handle_value(stream) - kFakeStreamHandleOffset);
  return CUDA_SUCCESS;
}

CUresult fake_error_name(CUresult, const char** text) {
  if (text != nullptr) {
    *text = "FAKE_CUDA_ERROR";
  }
  return CUDA_SUCCESS;
}

CUresult fake_error_string(CUresult, const char** text) {
  if (text != nullptr) {
    *text = "injected fake driver failure";
  }
  return CUDA_SUCCESS;
}

std::shared_ptr<CudaGreenContextDriver> make_fake_driver() {
  auto driver = std::make_shared<CudaGreenContextDriver>();
  driver->device_get = fake_device_get;
  driver->device_get_uuid = fake_device_get_uuid;
  driver->device_get_attribute = fake_device_get_attribute;
  driver->device_get_resource = fake_device_get_resource;
  driver->sm_resource_split_by_count = fake_sm_resource_split_by_count;
  driver->sm_resource_split = fake_sm_resource_split;
  driver->resource_generate_desc = fake_generate_desc;
  driver->green_context_create = fake_green_context_create;
  driver->green_context_destroy = fake_green_context_destroy;
  driver->green_context_get_resource = fake_green_context_get_resource;
  driver->context_from_green_context = fake_context_from_green_context;
  driver->green_context_stream_create = fake_green_context_stream_create;
  driver->stream_synchronize = fake_stream_synchronize;
  driver->stream_destroy = fake_stream_destroy;
  driver->stream_get_green_context = fake_stream_get_green_context;
  driver->get_error_name = fake_error_name;
  driver->get_error_string = fake_error_string;
  return driver;
}

// Installs a FakeDriverState-backed driver for one lexical test scope and restores the global
// test override on destruction. Used by
// CudaGreenContextDriver.PartialCreationFailureDestroysContextsInReverseOrder.
class FakeDriverScope {
 public:
  explicit FakeDriverScope(
      FakeDriverState& state,
      std::shared_ptr<const CudaGreenContextDriver> driver = make_fake_driver())
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
  FakeDriverState* state_;
  FakeDriverState* previous_state_{};
  std::shared_ptr<const CudaGreenContextDriver> previous_driver_;
};

FakeDriverState make_state(unsigned int sm_count = 80U, unsigned int minimum = 8U,
                           unsigned int alignment = 8U) {
  FakeDriverState state;
  state.device_resource.type = CU_DEV_RESOURCE_TYPE_SM;
  state.device_resource.sm.smCount = sm_count;
  state.device_resource.sm.minSmPartitionSize = minimum;
  state.device_resource.sm.smCoscheduledAlignment = alignment;
  return state;
}

// REQ: MEM-GREENCTX-011
// REQ: MEM-GREENCTX-MIG-002
TEST(CudaGreenContextDriver, ReportsDriverCapabilityAndStructuredContextLimit) {
  // Key setup: expose 80 SMs with a 4-SM minimum and 8-SM co-scheduling alignment.
  auto state = make_state(80U, 4U, 8U);
  FakeDriverScope scope(state);
  auto driver = cuda_execution_detail::cuda_green_context_driver();
  ASSERT_TRUE(driver);

  auto capability = cuda_execution_detail::query_cuda_green_context_device_capability(**driver, 0);
  ASSERT_TRUE(capability);
  // Key assertion: capability reports the stable physical identity and ten aligned contexts.
  EXPECT_EQ(capability->device_sm_count, 80U);
  EXPECT_EQ(capability->deployment_device_id,
            "cuda.uuid.0102030405060708090a0b0c0d0e0f10.pci.0000:65:00");
  EXPECT_EQ(capability->min_sm_partition_size, 4U);
  EXPECT_EQ(capability->sm_coscheduled_alignment, 8U);
  EXPECT_EQ(capability->structured_sm_quantum, 8U);
  EXPECT_EQ(capability->max_structured_context_count, 10U);
}

// REQ: MEM-GREENCTX-011
TEST(CudaGreenContextDriver, ReportsDriverQualifiedCapacityBelowArithmeticMaximum) {
  // Key setup: model a 132-SM device whose cluster topology permits only fifteen 8-SM units.
  auto state = make_state(132U, 8U, 8U);
  state.split_available_units = 15U;
  FakeDriverScope scope(state);
  auto driver = cuda_execution_detail::cuda_green_context_driver();
  ASSERT_TRUE(driver);

  auto capability = cuda_execution_detail::query_cuda_green_context_device_capability(**driver, 0);
  ASSERT_TRUE(capability);
  // Key assertion: discovery reports the simulated driver limit, not floor(132 / 8) == 16.
  EXPECT_EQ(capability->device_sm_count, 132U);
  EXPECT_EQ(capability->structured_sm_quantum, 8U);
  EXPECT_EQ(capability->max_structured_context_count, 15U);
  EXPECT_EQ(state.split_by_count_flags, 0U);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-014
TEST(CudaGreenContextDriver, PublicCapabilityQueryUsesReadOnlyDriverEvidence) {
  // Key setup: query device ordinal two through the public API using a 96-SM fake device.
  auto state = make_state(96U, 4U, 8U);
  FakeDriverScope scope(state);

  auto capability = query_cuda_green_context_capability(DeviceId{2});
  ASSERT_TRUE(capability);
  // Key assertion: public evidence matches the driver probe without creating a Green Context.
  EXPECT_EQ(capability->device, DeviceId{2});
  EXPECT_EQ(capability->deployment_device_id,
            "cuda.uuid.0102030405060708090a0b0c0d0e0f10.pci.0000:65:00");
  EXPECT_EQ(capability->total_sm_count, 96U);
  EXPECT_EQ(capability->min_sm_partition_size, 4U);
  EXPECT_EQ(capability->sm_coscheduled_alignment, 8U);
  EXPECT_EQ(capability->structured_sm_quantum, 8U);
  EXPECT_EQ(capability->max_structured_context_count, 12U);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-014
TEST(CudaGreenContextDriver, PublicCapabilityQueryRejectsInvalidUnavailableAndUnsupportedInputs) {
  // Key setup: exercise an unresolved device, a missing resource symbol, and an invalid resource.
  auto state = make_state();
  auto driver = make_fake_driver();
  FakeDriverScope scope(state, driver);

  auto capability = query_cuda_green_context_capability(DeviceId{});
  ASSERT_FALSE(capability);
  EXPECT_EQ(capability.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(capability.error().message,
            "CUDA_GREEN_CONTEXT_INVALID_PLACEMENT reason=device_invalid");

  driver->device_get_resource = nullptr;
  capability = query_cuda_green_context_capability(DeviceId{0});
  ASSERT_FALSE(capability);
  EXPECT_NE(runtime_error_message(capability.error()).find("cuDeviceGetDevResource"),
            std::string::npos);

  driver->device_get_resource = fake_device_get_resource;
  state.device_resource.type = CU_DEV_RESOURCE_TYPE_INVALID;
  capability = query_cuda_green_context_capability(DeviceId{0});
  // Key assertion: every invalid input fails before any native Green Context is created.
  ASSERT_FALSE(capability);
  EXPECT_NE(runtime_error_message(capability.error()).find("invalid Green Context SM resource"),
            std::string::npos);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-015
TEST(CudaGreenContextDriver, StreamIdentityClassifiesEveryNonFailureDisposition) {
  // Key setup: vary null, ordinary, matching, mismatched, and unsupported stream evidence.
  auto state = make_state();
  auto driver = make_fake_driver();
  FakeDriverScope scope(state, driver);
  const auto expected = reinterpret_cast<CUgreenCtx>(kFirstFakeGreenContextHandle);
  const auto stream = reinterpret_cast<CUstream>(kFirstFakeStreamHandle);

  auto identity =
      cuda_execution_detail::query_cuda_green_context_stream_identity(*driver, nullptr, expected);
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->kind, cuda_execution_detail::CudaGreenContextStreamIdentityKind::kNullStream);

  state.stream_observed_context = nullptr;
  identity =
      cuda_execution_detail::query_cuda_green_context_stream_identity(*driver, stream, expected);
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->kind,
            cuda_execution_detail::CudaGreenContextStreamIdentityKind::kOrdinaryContext);

  state.stream_observed_context = expected;
  identity =
      cuda_execution_detail::query_cuda_green_context_stream_identity(*driver, stream, expected);
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->kind,
            cuda_execution_detail::CudaGreenContextStreamIdentityKind::kExpectedGreenContext);
  EXPECT_EQ(identity->observed_context, expected);

  state.stream_observed_context = reinterpret_cast<CUgreenCtx>(kSecondFakeGreenContextHandle);
  identity =
      cuda_execution_detail::query_cuda_green_context_stream_identity(*driver, stream, expected);
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->kind,
            cuda_execution_detail::CudaGreenContextStreamIdentityKind::kWrongGreenContext);

  driver->stream_get_green_context = nullptr;
  identity =
      cuda_execution_detail::query_cuda_green_context_stream_identity(*driver, stream, expected);
  // Key assertion: each non-error disposition receives its distinct stable identity kind.
  ASSERT_TRUE(identity);
  EXPECT_EQ(identity->kind,
            cuda_execution_detail::CudaGreenContextStreamIdentityKind::kUnsupported);
}

// REQ: MEM-GREENCTX-015
TEST(CudaGreenContextDriver, StreamIdentityPreservesDriverFailureDetails) {
  // Key setup: inject CUDA_ERROR_INVALID_CONTEXT from cuStreamGetGreenCtx.
  auto state = make_state();
  state.stream_identity_result = CUDA_ERROR_INVALID_CONTEXT;
  auto driver = make_fake_driver();
  FakeDriverScope scope(state, driver);

  auto identity = cuda_execution_detail::query_cuda_green_context_stream_identity(
      *driver,
      reinterpret_cast<CUstream>(kFirstFakeStreamHandle),
      reinterpret_cast<CUgreenCtx>(kFirstFakeGreenContextHandle));
  // Key assertion: the error retains both the failed API name and fake CUDA diagnostic.
  ASSERT_FALSE(identity);
  EXPECT_NE(runtime_error_message(identity.error()).find("cuStreamGetGreenCtx failed"),
            std::string::npos);
  EXPECT_NE(runtime_error_message(identity.error()).find("FAKE_CUDA_ERROR"), std::string::npos);
}

// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextDriver, StructuredPoolUsesZeroSplitFlagsAndNeverUsesWeakRemainder) {
  // Key setup: request three aligned groups totaling 48 SMs from an 80-SM fake device.
  auto state = make_state();
  FakeDriverScope scope(state);
  CudaGreenContextPool::Config config;
  config.num_partitions = 3U;
  config.sms_per_partition = {16, 8, 24};

  CudaGreenContextPool pool(config);
  ASSERT_TRUE(pool.initialize_pool());
  // Key assertion: the exact split uses zero flags and promotes only a structured 32-SM residual.
  EXPECT_EQ(state.split_flags, 0U);
  EXPECT_EQ(state.split_sms, (std::vector<unsigned int>{16U, 8U, 24U, 32U}));
  EXPECT_EQ(pool.num_partitions(), 4U);
  ASSERT_TRUE(pool.min_sm_partition_size());
  EXPECT_EQ(*pool.min_sm_partition_size(), 8U);
  ASSERT_TRUE(pool.sm_coscheduled_alignment());
  EXPECT_EQ(*pool.sm_coscheduled_alignment(), 8U);
}

// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextDriver, StructuredPoolExcludesDriverUnavailableTopologyTail) {
  // Key setup: request 8 SMs on a 132-SM device whose driver qualifies only 120 structured SMs.
  auto state = make_state(132U, 8U, 8U);
  state.split_available_units = 15U;
  FakeDriverScope scope(state);
  CudaGreenContextPool::Config config;
  config.num_partitions = 1U;
  config.sms_per_partition = {8};

  CudaGreenContextPool pool(config);
  ASSERT_TRUE(pool.initialize_pool());
  // Key assertion: only the 112-SM qualified residual is appended; the 12-SM tail stays unused.
  EXPECT_EQ(state.split_sms, (std::vector<unsigned int>{8U, 112U}));
  EXPECT_EQ(pool.num_partitions(), 2U);
}

// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextDriver, CudaThirteenBaselineAggregatesStructuredUnits) {
  // Key setup: remove the explicit split symbol so the CUDA 13.0 by-count fallback is selected.
  auto state = make_state();
  auto driver = make_fake_driver();
  driver->sm_resource_split = nullptr;
  FakeDriverScope scope(state, std::move(driver));
  CudaGreenContextPool::Config config;
  config.num_partitions = 3U;
  config.sms_per_partition = {16, 8, 24};

  CudaGreenContextPool pool(config);
  ASSERT_TRUE(pool.initialize_pool());
  // Key assertion: zero-flag 8-SM units aggregate into the requested groups and residual.
  EXPECT_EQ(state.split_by_count_flags, 0U);
  EXPECT_EQ(pool.num_partitions(), 4U);
  ASSERT_EQ(state.created_contexts.size(), 4U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[0])), 16U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[1])), 8U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[2])), 24U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[3])), 32U);
}

// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextDriver, CudaThirteenBaselineAcceptsFewerSufficientStructuredUnits) {
  // Key setup: qualify five units on an 82-SM device and request two 16-SM groups.
  auto state = make_state(82U);
  state.split_available_units = 5U;
  state.split_created_units = 5U;
  auto driver = make_fake_driver();
  driver->sm_resource_split = nullptr;
  FakeDriverScope scope(state, std::move(driver));
  CudaGreenContextPool::Config config;
  config.num_partitions = 2U;
  config.sms_per_partition = {16, 16};

  CudaGreenContextPool pool(config);
  // Key assertion: four units realize the requests, one forms the structured residual, and the
  // remaining 42 device SMs never enter HSDK work.
  ASSERT_TRUE(pool.initialize_pool());
  EXPECT_EQ(pool.num_partitions(), 3U);
  ASSERT_EQ(state.created_contexts.size(), 3U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[0])), 16U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[1])), 16U);
  EXPECT_EQ(state.context_sms.at(handle_value(state.created_contexts[2])), 8U);
}

// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextDriver, CudaThirteenBaselineRejectsFewerInsufficientStructuredUnits) {
  // Key setup: report ten units during sizing but return only three for two 16-SM requests.
  auto state = make_state(82U);
  state.split_available_units = 10U;
  state.split_created_units = 3U;
  auto driver = make_fake_driver();
  driver->sm_resource_split = nullptr;
  FakeDriverScope scope(state, std::move(driver));
  CudaGreenContextPool::Config config;
  config.num_partitions = 2U;
  config.sms_per_partition = {16, 16};

  CudaGreenContextPool pool(config);
  auto initialized = pool.initialize_pool();
  // Key assertion: initialization fails before creating contexts when structured units run short.
  ASSERT_FALSE(initialized);
  EXPECT_NE(runtime_error_message(initialized.error()).find("created only 3 structured units"),
            std::string::npos);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-011
TEST(CudaGreenContextDriver, MissingRequiredSymbolFailsBeforeNativeCreation) {
  // Key setup: remove the required stream-identity symbol from an otherwise complete fake driver.
  auto state = make_state();
  auto driver = make_fake_driver();
  driver->stream_get_green_context = nullptr;
  FakeDriverScope scope(state, std::move(driver));

  CudaGreenContextPool pool;
  auto initialized = pool.initialize_pool();
  // Key assertion: capability validation fails before native context creation.
  ASSERT_FALSE(initialized);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-011
TEST(CudaGreenContextDriver, UnsupportedSmResourceFailsBeforeNativeCreation) {
  // Key setup: return an invalid device resource type from the SM resource probe.
  auto state = make_state();
  state.device_resource.type = CU_DEV_RESOURCE_TYPE_INVALID;
  FakeDriverScope scope(state);

  CudaGreenContextPool pool;
  auto initialized = pool.initialize_pool();
  // Key assertion: unsupported resource evidence fails before native context creation.
  ASSERT_FALSE(initialized);
  EXPECT_TRUE(state.created_contexts.empty());
}

// REQ: MEM-GREENCTX-009
// REQ: MEM-GREENCTX-013
// REQ: MEM-GREENCTX-MIG-003
TEST(CudaGreenContextDriver, PartialCreationFailureDestroysContextsInReverseOrder) {
  // Key setup: fail creation of the third context after two contexts have been created.
  auto state = make_state();
  state.fail_context_create_index = 2U;
  FakeDriverScope scope(state);
  CudaGreenContextPool::Config config;
  config.num_partitions = 4U;
  config.sms_per_partition = {16, 16, 16, 32};

  CudaGreenContextPool pool(config);
  auto initialized = pool.initialize_pool();
  ASSERT_FALSE(initialized);
  // Key assertion: rollback destroys each created context exactly once in reverse order.
  ASSERT_EQ(state.created_contexts.size(), 2U);
  ASSERT_EQ(state.destroyed_contexts.size(), 2U);
  EXPECT_EQ(state.destroyed_contexts[0], state.created_contexts[1]);
  EXPECT_EQ(state.destroyed_contexts[1], state.created_contexts[0]);
}

// REQ: MEM-GREENCTX-009
TEST(CudaGreenContextDriver, PoolDestructionReleasesMaterializedContextsInReverseOrder) {
  // Key setup: fully initialize two structured groups, then leave the pool's owning scope.
  auto state = make_state();
  FakeDriverScope driver_scope(state);
  {
    CudaGreenContextPool::Config config;
    config.num_partitions = 2U;
    config.sms_per_partition = {16, 64};
    CudaGreenContextPool pool(config);
    ASSERT_TRUE(pool.initialize_pool());
    ASSERT_EQ(state.created_contexts.size(), 2U);
  }
  // Key assertion: normal destruction releases every context exactly once in reverse order.
  ASSERT_EQ(state.destroyed_contexts.size(), state.created_contexts.size());
  EXPECT_EQ(state.destroyed_contexts[0], state.created_contexts[1]);
  EXPECT_EQ(state.destroyed_contexts[1], state.created_contexts[0]);
}

TEST(CudaGreenContextDriver, NestedFakeDriverScopesRestorePriorState) {
  // Key setup: materialize one pool under an inner fake driver, then another after it unwinds.
  auto outer_state = make_state(80U);
  auto inner_state = make_state(40U);
  FakeDriverScope outer_scope(outer_state);
  {
    FakeDriverScope inner_scope(inner_state);
    CudaGreenContextPool inner_pool;
    ASSERT_TRUE(inner_pool.initialize_pool());
    EXPECT_EQ(inner_state.created_contexts.size(), 1U);
    EXPECT_TRUE(outer_state.created_contexts.empty());
  }
  CudaGreenContextPool outer_pool;
  ASSERT_TRUE(outer_pool.initialize_pool());
  // Key assertion: inner teardown restores both the outer fake state and its driver override.
  EXPECT_EQ(outer_state.created_contexts.size(), 1U);
  EXPECT_EQ(inner_state.created_contexts.size(), 1U);
}

}  // namespace
}  // namespace holoscan

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
