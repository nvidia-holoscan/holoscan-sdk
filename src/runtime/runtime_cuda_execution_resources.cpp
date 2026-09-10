/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_cuda_execution_resources.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/logger/logger.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
#include <nvtx3/nvToolsExtCuda.h>
#endif

#include "../compiler/cuda_execution_plan_validation.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../cuda_execution/cuda_green_context_driver.hpp"
#include "graph_exec.hpp"
#include "runtime_cuda_execution_diagnostic.hpp"
#include "runtime_tensor_resources.hpp"
#include "runtime_work_inventory.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] expected<void, Error> cuda_execution_error(std::string_view token,
                                                         std::string_view detail = {}) {
  std::string message(token);
  if (!detail.empty()) {
    message.append(": ");
    message.append(detail);
  }
  return make_unexpected(Error{ErrorCode::kFailure, std::move(message)});
}

struct DeviceClaim {
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  std::uint64_t owner_id{};
  enum class State : std::uint8_t {
    kClaiming,
    kActive,
    kQuiescing,
    kQuarantined,
  } state{State::kClaiming};
};

struct DeviceClaimSet {
  std::optional<DeviceClaim> green;
  std::vector<DeviceClaim> ordinary;
  std::string physical_identity;
};

struct DeviceClaimRegistry {
  std::mutex mutex;
  std::map<std::int32_t, DeviceClaimSet> devices;
  std::map<std::string, std::int32_t, std::less<>> physical_green_devices;
  std::atomic<std::uint64_t> next_owner_id{1U};
};

[[nodiscard]] DeviceClaimRegistry& device_claim_registry() {
  static DeviceClaimRegistry registry;
  return registry;
}

[[nodiscard]] expected<std::uint64_t, Error> reserve_device_claim_owner_id() {
  auto& next_owner_id = device_claim_registry().next_owner_id;
  std::uint64_t owner_id = next_owner_id.load(std::memory_order_relaxed);
  while (owner_id != std::numeric_limits<std::uint64_t>::max()) {
    if (next_owner_id.compare_exchange_weak(owner_id, owner_id + 1U, std::memory_order_relaxed)) {
      return owner_id;
    }
  }
  return make_unexpected(Error{ErrorCode::kResourceExhausted,
                               "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED: device claim owner "
                               "identity exhausted"});
}

[[nodiscard]] bool same_claim(const DeviceClaim& claim, PlanGenerationId plan_generation,
                              RunGenerationId run_generation, std::uint64_t owner_id) noexcept {
  return claim.plan_generation == plan_generation && claim.run_generation == run_generation &&
         claim.owner_id == owner_id;
}

[[nodiscard]] expected<void, Error> claim_green_device(std::string_view identity, DeviceId device,
                                                       PlanGenerationId plan_generation,
                                                       RunGenerationId run_generation,
                                                       std::uint64_t owner_id) {
  auto& registry = device_claim_registry();
  const std::lock_guard lock(registry.mutex);
  if (identity.empty() || !device.valid() || owner_id == 0U) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
  }
  const auto device_claims = registry.devices.find(device.value);
  if (registry.physical_green_devices.contains(identity) ||
      (device_claims != registry.devices.end() &&
       (device_claims->second.green || !device_claims->second.ordinary.empty()))) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  }
  const DeviceClaim claim{
      .plan_generation = plan_generation,
      .run_generation = run_generation,
      .owner_id = owner_id,
  };
  const auto [physical, inserted] =
      registry.physical_green_devices.emplace(std::string(identity), device.value);
  if (!inserted) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto& claims = registry.devices[device.value];
    claims.green = claim;
    claims.physical_identity = identity;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    registry.physical_green_devices.erase(physical);
    const auto partial = registry.devices.find(device.value);
    if (partial != registry.devices.end()) {
      partial->second.green.reset();
      partial->second.physical_identity.clear();
      if (partial->second.ordinary.empty()) {
        registry.devices.erase(partial);
      }
    }
    throw;
  }
#endif
  return {};
}

[[nodiscard]] expected<void, Error> claim_ordinary_device(DeviceId device,
                                                          PlanGenerationId plan_generation,
                                                          RunGenerationId run_generation,
                                                          std::uint64_t owner_id) {
  auto& registry = device_claim_registry();
  const std::lock_guard lock(registry.mutex);
  if (!device.valid() || owner_id == 0U) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
  }
  auto& claims = registry.devices[device.value];
  if (claims.green || std::ranges::any_of(claims.ordinary, [](const auto& claim) {
        return claim.state == DeviceClaim::State::kQuarantined;
      })) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    claims.ordinary.push_back(DeviceClaim{
        .plan_generation = plan_generation,
        .run_generation = run_generation,
        .owner_id = owner_id,
    });
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    if (!claims.green && claims.ordinary.empty()) {
      registry.devices.erase(device.value);
    }
    throw;
  }
#endif
  return {};
}

[[nodiscard]] DeviceClaim* find_claim(DeviceClaimSet& claims, PlanGenerationId plan_generation,
                                      RunGenerationId run_generation,
                                      std::uint64_t owner_id) noexcept {
  if (claims.green && same_claim(*claims.green, plan_generation, run_generation, owner_id)) {
    return &*claims.green;
  }
  const auto ordinary =
      std::find_if(claims.ordinary.begin(), claims.ordinary.end(), [&](const auto& claim) {
        return same_claim(claim, plan_generation, run_generation, owner_id);
      });
  return ordinary == claims.ordinary.end() ? nullptr : &*ordinary;
}

[[nodiscard]] bool transition_device_claim(DeviceId device, PlanGenerationId plan_generation,
                                           RunGenerationId run_generation, std::uint64_t owner_id,
                                           DeviceClaim::State state) noexcept {
  auto& registry = device_claim_registry();
  const std::lock_guard lock(registry.mutex);
  const auto found = registry.devices.find(device.value);
  if (found == registry.devices.end()) {
    return false;
  }
  DeviceClaim* claim = find_claim(found->second, plan_generation, run_generation, owner_id);
  if (claim == nullptr) {
    return false;
  }
  const DeviceClaim::State previous = claim->state;
  const bool valid_transition =
      (state == DeviceClaim::State::kActive && previous == DeviceClaim::State::kClaiming) ||
      (state == DeviceClaim::State::kQuiescing &&
       (previous == DeviceClaim::State::kClaiming || previous == DeviceClaim::State::kActive)) ||
      (state == DeviceClaim::State::kQuarantined && previous != DeviceClaim::State::kQuarantined);
  if (!valid_transition) {
    return false;
  }
  claim->state = state;
  return true;
}

void release_device(DeviceId device, PlanGenerationId plan_generation,
                    RunGenerationId run_generation, std::uint64_t owner_id) noexcept {
  auto& registry = device_claim_registry();
  const std::lock_guard lock(registry.mutex);
  const auto found = registry.devices.find(device.value);
  if (found == registry.devices.end()) {
    return;
  }
  DeviceClaimSet& claims = found->second;
  if (claims.green && same_claim(*claims.green, plan_generation, run_generation, owner_id)) {
    if (claims.green->state == DeviceClaim::State::kQuarantined) {
      return;
    }
    registry.physical_green_devices.erase(claims.physical_identity);
    claims.green.reset();
    claims.physical_identity.clear();
  } else {
    const auto ordinary =
        std::find_if(claims.ordinary.begin(), claims.ordinary.end(), [&](const auto& claim) {
          return same_claim(claim, plan_generation, run_generation, owner_id);
        });
    if (ordinary == claims.ordinary.end() || ordinary->state == DeviceClaim::State::kQuarantined) {
      return;
    }
    claims.ordinary.erase(ordinary);
  }
  if (!claims.green && claims.ordinary.empty()) {
    registry.devices.erase(found);
  }
}

}  // namespace

class RuntimeCudaExecutionResources::Impl final {
 public:
  struct ClaimedDevice {
    DeviceId device{};
    const CudaGreenContextPoolPlanRecord* green_pool{};
  };

  struct PoolOwner {
    const CudaGreenContextPoolPlanRecord* plan{};
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    std::unique_ptr<CudaGreenContextPool> pool;
    std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> driver;
#endif
  };

  struct StreamOwner {
    const CudaComputeStreamPlanRecord* plan{};
    std::unique_ptr<CudaStreamPool> pool;
    CudaStreamPool::StreamLease lease;
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> driver;
    CUstream green_stream{};
#endif

    StreamOwner(const CudaComputeStreamPlanRecord* stream_plan,
                std::unique_ptr<CudaStreamPool> stream_pool,
                CudaStreamPool::StreamLease stream_lease) noexcept
        : plan(stream_plan), pool(std::move(stream_pool)), lease(std::move(stream_lease)) {}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    StreamOwner(const CudaComputeStreamPlanRecord* stream_plan,
                std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> stream_driver,
                CUstream stream) noexcept
        : plan(stream_plan), driver(std::move(stream_driver)), green_stream(stream) {}
#endif

    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    StreamOwner& operator=(StreamOwner&&) = delete;

    StreamOwner(StreamOwner&& other) noexcept
        : plan(std::exchange(other.plan, nullptr)),
          pool(std::move(other.pool)),
          lease(std::move(other.lease))
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
          ,
          driver(std::move(other.driver)),
          green_stream(std::exchange(other.green_stream, nullptr))
#endif
    {
    }

    [[nodiscard]] expected<void, Error> synchronize(std::int32_t* native_result) noexcept {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
      if (green_stream != nullptr) {
        if (driver == nullptr || driver->stream_synchronize == nullptr) {
          return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                      "missing cuStreamSynchronize owner");
        }
        const CUresult result = driver->stream_synchronize(green_stream);
        if (native_result != nullptr) {
          *native_result = static_cast<std::int32_t>(result);
        }
        if (result != CUDA_SUCCESS) {
          return cuda_execution_error(
              "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
              fmt::format("cuStreamSynchronize failed for stream {} with CUresult {}",
                          plan != nullptr ? plan->stream_id : "<unknown>",
                          static_cast<int>(result)));
        }
        return {};
      }
#endif
      cudaStream_t stream = lease.native_handle();
      if (stream == nullptr) {
        return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                    "missing CUDA stream during quiesce");
      }
      const cudaError_t result = cudaStreamSynchronize(stream);
      if (native_result != nullptr) {
        *native_result = static_cast<std::int32_t>(result);
      }
      if (result != cudaSuccess) {
        return cuda_execution_error(
            "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
            fmt::format("cudaStreamSynchronize failed for stream {} with cudaError_t {}",
                        plan != nullptr ? plan->stream_id : "<unknown>",
                        static_cast<int>(result)));
      }
      return {};
    }

    [[nodiscard]] expected<void, Error> destroy() noexcept {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
      if (green_stream != nullptr && driver != nullptr && driver->stream_destroy != nullptr) {
        CUstream stream = std::exchange(green_stream, nullptr);
        if (const CUresult result = driver->stream_destroy(stream); result != CUDA_SUCCESS) {
          return cuda_execution_error(
              "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
              fmt::format("cuStreamDestroy failed for stream {} with CUresult {}",
                          plan != nullptr ? plan->stream_id : "<unknown>",
                          static_cast<int>(result)));
        }
      }
#endif
      return {};
    }

    void abandon_after_failed_quiesce() noexcept {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
      // The quarantined claim prevents reuse. Keep an uncertain native stream alive rather than
      // invoking destruction after CUDA failed to prove that submitted work completed.
      green_stream = nullptr;
#endif
      // Releasing the lease into a deliberately retained pool preserves an ordinary stream under
      // the same rule. The process restart that clears the quarantined claim reclaims both.
      [[maybe_unused]] const auto released = pool.release();
    }

    ~StreamOwner() {
      if (auto destroyed = destroy(); !destroyed) {
        try {
          HOLOSCAN_LOG_ERROR("Failed to destroy CUDA Green Context stream: {}",
                             runtime_error_message(destroyed.error()));
        } catch (...) {  // NOLINT(bugprone-empty-catch)
          // Destruction is best-effort and must not terminate during stack unwinding.
        }
      }
    }

    [[nodiscard]] cudaStream_t native_handle() const noexcept {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
      if (green_stream != nullptr) {
        return reinterpret_cast<cudaStream_t>(green_stream);
      }
#endif
      return lease.native_handle();
    }
  };

  ~Impl();

  [[nodiscard]] expected<void, Error> quiesce(RuntimeCudaExecutionDiagnostic* output = nullptr);
  [[nodiscard]] expected<void, Error> shutdown(RuntimeCudaExecutionDiagnostic* output = nullptr);

  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  ProcessGroupId process_group{};
  std::uint64_t claim_owner_id{};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control;
  std::shared_ptr<RuntimeCudaExecutionTestControl> cuda_test_control;
  std::vector<ClaimedDevice> claimed_devices;
  std::vector<PoolOwner> pools;
  std::vector<StreamOwner> streams;
  std::size_t materialized_stream_count{};
  std::optional<Error> quiesce_failure;
  bool quiesce_complete{};
  bool shutdown_complete{};
  RuntimeCudaExecutionDiagnostic diagnostic;
  RuntimeCudaExecutionDiagnostic* diagnostic_output{};
};

namespace {

[[nodiscard]] bool inject_failure(RuntimeCudaExecutionResources::Impl& impl,
                                  RuntimeCudaExecutionFailurePoint point) noexcept {
  return impl.cuda_test_control != nullptr && impl.cuda_test_control->should_fail(point);
}

void record_diagnostic(RuntimeCudaExecutionResources::Impl& impl, std::string_view operation,
                       const CudaGreenContextPoolPlanRecord* pool = nullptr,
                       const CudaGreenContextResourcePlanRecord* context = nullptr,
                       const CudaComputeStreamPlanRecord* stream = nullptr,
                       std::int32_t driver_result = -1,
                       LifecycleStage lifecycle_stage = LifecycleStage::kArm,
                       DeviceId device = {}) {
  impl.diagnostic.plan_generation = impl.plan_generation;
  impl.diagnostic.run_generation = impl.run_generation;
  impl.diagnostic.lifecycle_stage = lifecycle_stage;
  impl.diagnostic.operation = operation;
  impl.diagnostic.device = device;
  impl.diagnostic.deployment_device_id.clear();
  impl.diagnostic.pool_id.clear();
  impl.diagnostic.green_context_id.clear();
  impl.diagnostic.stream_id.clear();
  if (pool != nullptr) {
    impl.diagnostic.device = pool->device;
    impl.diagnostic.deployment_device_id = pool->deployment_device_id;
    impl.diagnostic.pool_id = pool->pool_id;
  }
  if (context != nullptr) {
    impl.diagnostic.pool_id = context->pool_id;
    impl.diagnostic.green_context_id = context->green_context_id;
  }
  if (stream != nullptr) {
    impl.diagnostic.device = stream->device;
    impl.diagnostic.green_context_id = stream->green_context_id.value_or(std::string{});
    impl.diagnostic.stream_id = stream->stream_id;
    const auto owner = std::find_if(impl.pools.begin(), impl.pools.end(), [&](const auto& row) {
      return row.plan != nullptr && row.plan->process_group == stream->process_group &&
             row.plan->device == stream->device;
    });
    if (owner != impl.pools.end()) {
      impl.diagnostic.deployment_device_id = owner->plan->deployment_device_id;
      impl.diagnostic.pool_id = owner->plan->pool_id;
    }
  }
  impl.diagnostic.driver_result = driver_result;
  if (impl.diagnostic_output != nullptr) {
    *impl.diagnostic_output = impl.diagnostic;
  }
}

[[nodiscard]] Error injected_materialization_failure(RuntimeCudaExecutionFailurePoint point) {
  return Error{ErrorCode::kFailure,
               "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED: injected failure point " +
                   std::to_string(static_cast<std::uint32_t>(point))};
}

// Validate the complete CUDA ownership graph before materializing any native resource.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_plan(
    const ExecutionPlanState& plan, std::span<const RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, ProcessGroupId process_group) {
  if (plan.plan_generation == 0U || run_generation == 0U ||
      (!process_group && has_structural_runtime_topology(plan))) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
  }
  if (auto valid = validate_cuda_execution_plan_records(plan); !valid) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID", valid.error().message);
  }

  std::set<std::string, std::less<>> pool_ids;
  std::set<std::string, std::less<>> resource_ids;
  std::set<std::string, std::less<>> stream_ids;
  std::set<std::string, std::less<>> deployment_devices;
  std::set<std::tuple<std::uint32_t, std::int32_t>> pool_keys;
  std::set<std::tuple<std::uint32_t, std::int32_t, std::uint32_t>> partition_keys;
  for (const auto& pool : plan.cuda_green_context_pools) {
    if (pool.plan_generation != plan.plan_generation || pool.run_generation != 0U ||
        !pool.process_group || !pool.device.valid() || pool.deployment_device_id.empty() ||
        pool.pool_id.empty() || pool.device_sm_count == 0U || pool.min_sm_partition_size == 0U ||
        pool.sm_coscheduled_alignment == 0U || pool.structured_sm_quantum == 0U ||
        pool.structured_sm_quantum !=
            std::lcm(pool.min_sm_partition_size, pool.sm_coscheduled_alignment) ||
        pool.materialize_before != LifecycleStage::kArm || !pool.fallback_rejected ||
        !pool_ids.insert(pool.pool_id).second ||
        !deployment_devices.insert(pool.deployment_device_id).second ||
        !pool_keys.emplace(pool.process_group.value, pool.device.value).second) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }
  for (const auto& resource : plan.cuda_green_context_resources) {
    const auto pool = std::find_if(plan.cuda_green_context_pools.begin(),
                                   plan.cuda_green_context_pools.end(),
                                   [&](const auto& row) {
                                     return row.process_group == resource.process_group &&
                                            row.device == resource.device &&
                                            row.pool_id == resource.pool_id;
                                   });
    if (resource.plan_generation != plan.plan_generation || resource.run_generation != 0U ||
        !resource.process_group || !resource.device.valid() || resource.pool_id.empty() ||
        resource.green_context_id.empty() || resource.accepted_sm_count == 0U ||
        pool == plan.cuda_green_context_pools.end() ||
        !resource_ids.insert(resource.green_context_id).second) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }
  for (const auto& stream : plan.cuda_compute_streams) {
    if (stream.plan_generation != plan.plan_generation || stream.run_generation != 0U ||
        !stream.process_group || !stream.device.valid() || stream.owner_id.empty() ||
        stream.stream_id.empty() || stream.materialize_before != LifecycleStage::kArm ||
        !stream_ids.insert(stream.stream_id).second) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
    if (stream.green_context_id && !resource_ids.contains(*stream.green_context_id)) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
    if (stream.partition &&
        !partition_keys
             .emplace(stream.process_group.value, stream.device.value, stream.partition->value)
             .second) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }

  for (const auto& pool : plan.cuda_green_context_pools) {
    std::uint64_t structured_sms = pool.unprovisioned_sm_count;
    std::uint32_t resource_count = 0U;
    for (const auto& resource : plan.cuda_green_context_resources) {
      if (resource.pool_id == pool.pool_id) {
        structured_sms += resource.accepted_sm_count;
        ++resource_count;
      }
    }
    if (structured_sms != pool.device_sm_count || resource_count == 0U) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }

  for (const auto& stream : plan.cuda_compute_streams) {
    if (stream.process_group != process_group || !stream.partition) {
      continue;
    }
    const auto* partition = find_execution_partition_by_id(plan, *stream.partition);
    const auto member_count =
        std::count_if(operators.begin(), operators.end(), [&](const auto& instance) {
          return instance.partition == *stream.partition;
        });
    const auto executable_member_count =
        partition == nullptr
            ? 0U
            : std::count_if(
                  partition->members.begin(), partition->members.end(), [&](VertexId member) {
                    return std::any_of(
                        plan.executable_factories.operators.begin(),
                        plan.executable_factories.operators.end(),
                        [&](const auto& factory) { return factory.vertex == member; });
                  });
    if (partition == nullptr || std::cmp_equal(executable_member_count, 0U) ||
        member_count != executable_member_count) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }
  for (const auto& assignment : plan.cuda_green_context_assignments) {
    if (assignment.process_group != process_group) {
      continue;
    }
    const auto* stream = find_cuda_compute_stream_by_partition(
        plan, assignment.process_group, assignment.device, assignment.partition);
    if (stream == nullptr || stream->green_context_id != assignment.green_context_id) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_PLAN_INVALID");
    }
  }
  return {};
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
[[nodiscard]] std::string native_driver_detail(
    const cuda_execution_detail::CudaGreenContextDriver& driver, std::string_view operation,
    CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  if (driver.get_error_name != nullptr) {
    static_cast<void>(driver.get_error_name(result, &name));
  }
  if (driver.get_error_string != nullptr) {
    static_cast<void>(driver.get_error_string(result, &description));
  }
  return fmt::format("{} failed with CUresult {} ({}: {})",
                     operation,
                     static_cast<int>(result),
                     name == nullptr ? "unknown" : name,
                     description == nullptr ? "unknown" : description);
}

[[nodiscard]] expected<std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver>, Error>
reprobe_pool(const CudaGreenContextPoolPlanRecord& pool_plan,
             RuntimeCudaExecutionResources::Impl& impl) {
  record_diagnostic(impl, "capability_reprobe", &pool_plan);
  if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kProbe)) {
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kProbe));
  }
  auto driver = cuda_execution_detail::cuda_green_context_driver();
  if (!driver) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED: " + runtime_error_message(driver.error())});
  }
  auto observed = cuda_execution_detail::query_cuda_green_context_device_capability(
      **driver, pool_plan.device.value);
  if (!observed) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED: " + runtime_error_message(observed.error())});
  }
  if (observed->deployment_device_id != pool_plan.deployment_device_id ||
      observed->device_sm_count != pool_plan.device_sm_count ||
      observed->min_sm_partition_size != pool_plan.min_sm_partition_size ||
      observed->sm_coscheduled_alignment != pool_plan.sm_coscheduled_alignment ||
      observed->structured_sm_quantum != pool_plan.structured_sm_quantum ||
      observed->max_structured_context_count != pool_plan.max_structured_context_count) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              fmt::format("CUDA_GREEN_CONTEXT_CAPABILITY_DRIFT: admitted_device={} "
                          "observed_device={} admitted_sms={} observed_sms={} admitted_min={} "
                          "observed_min={} admitted_alignment={} observed_alignment={} "
                          "admitted_quantum={} observed_quantum={} admitted_max_contexts={} "
                          "observed_max_contexts={}",
                          pool_plan.deployment_device_id,
                          observed->deployment_device_id,
                          pool_plan.device_sm_count,
                          observed->device_sm_count,
                          pool_plan.min_sm_partition_size,
                          observed->min_sm_partition_size,
                          pool_plan.sm_coscheduled_alignment,
                          observed->sm_coscheduled_alignment,
                          pool_plan.structured_sm_quantum,
                          observed->structured_sm_quantum,
                          pool_plan.max_structured_context_count,
                          observed->max_structured_context_count)});
  }
  return *driver;
}
#endif

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
[[nodiscard]] expected<void, Error> verify_stream_identity(
    const cuda_execution_detail::CudaGreenContextDriver& driver, CUstream stream,
    CUgreenCtx expected_context) {
  auto identity = cuda_execution_detail::query_cuda_green_context_stream_identity(
      driver, stream, expected_context);
  if (!identity) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                runtime_error_message(identity.error()));
  }
  if (identity->kind == cuda_execution_detail::CudaGreenContextStreamIdentityKind::kUnsupported) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                "cuStreamGetGreenCtx is unavailable");
  }
  if (identity->kind !=
      cuda_execution_detail::CudaGreenContextStreamIdentityKind::kExpectedGreenContext) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
  }
  return {};
}
#endif

[[nodiscard]] expected<void, Error> materialize_stream(RuntimeCudaExecutionResources::Impl& impl,
                                                       const CudaComputeStreamPlanRecord& stream) {
  record_diagnostic(impl, "stream_create", nullptr, nullptr, &stream);
  if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kStream)) {
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kStream));
  }
  CudaStreamPool::Config config{
      .device_id = stream.device.value,
      .stream_flags = cudaStreamNonBlocking,
      .reserved_size = 1U,
      .max_size = 1U,
      .nvtx_identifier = stream.stream_id,
  };
  auto pool = std::make_unique<CudaStreamPool>(std::move(config));
  if (auto initialized = pool->initialize_pool(); !initialized) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                runtime_error_message(initialized.error()));
  }
  auto lease = pool->acquire();
  if (!lease) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                runtime_error_message(lease.error()));
  }
  if (lease->native_handle() == nullptr) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                "CUDA stream pool returned a null stream");
  }
  impl.streams.emplace_back(&stream, std::move(pool), std::move(*lease));
  ++impl.materialized_stream_count;
  if (impl.test_control != nullptr) {
    impl.test_control->record_cuda_stream_created(0U);
  }
  return {};
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
[[nodiscard]] expected<void, Error> materialize_green_stream(
    RuntimeCudaExecutionResources::Impl& impl, const CudaComputeStreamPlanRecord& stream,
    CUgreenCtx green_context,
    std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> driver) {
  record_diagnostic(impl, "stream_create", nullptr, nullptr, &stream);
  if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kStream)) {
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kStream));
  }
  if (green_context == nullptr || driver == nullptr ||
      driver->green_context_stream_create == nullptr || driver->stream_destroy == nullptr) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                "Green Context stream driver entry point unavailable");
  }
  CUstream native_stream = nullptr;
  const CUresult result =
      driver->green_context_stream_create(&native_stream, green_context, CU_STREAM_NON_BLOCKING, 0);
  if (result != CUDA_SUCCESS) {
    record_diagnostic(
        impl, "stream_create", nullptr, nullptr, &stream, static_cast<std::int32_t>(result));
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                native_driver_detail(*driver, "cuGreenCtxStreamCreate", result));
  }
  if (native_stream == nullptr) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                "cuGreenCtxStreamCreate returned a null stream");
  }
  if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kIdentity)) {
    static_cast<void>(driver->stream_destroy(native_stream));
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kIdentity));
  }
  record_diagnostic(impl, "stream_identity", nullptr, nullptr, &stream);
  auto identity = verify_stream_identity(*driver, native_stream, green_context);
  if (!identity) {
    static_cast<void>(driver->stream_destroy(native_stream));
    return identity;
  }
  nvtxNameCuStreamA(native_stream, stream.stream_id.c_str());
  impl.streams.emplace_back(&stream, std::move(driver), native_stream);
  ++impl.materialized_stream_count;
  if (impl.test_control != nullptr) {
    impl.test_control->record_cuda_stream_created(0U);
  }
  return {};
}
#endif

// Split, identify, and bind one Green Context pool as an all-or-nothing lifecycle resource.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> materialize_green_pool(
    const ExecutionPlanState& plan, const CudaGreenContextPoolPlanRecord& pool_plan,
    RuntimeCudaExecutionResources::Impl& impl
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    ,
    std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver>
        driver  // NOLINT(performance-unnecessary-value-param)
#endif
) {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
  record_diagnostic(impl, "pool_split", &pool_plan);
  std::vector<const CudaGreenContextResourcePlanRecord*> resources;
  std::vector<std::int32_t> sm_counts;
  for (const auto& resource : plan.cuda_green_context_resources) {
    if (resource.process_group == pool_plan.process_group && resource.device == pool_plan.device &&
        resource.pool_id == pool_plan.pool_id) {
      resources.push_back(&resource);
      sm_counts.push_back(static_cast<std::int32_t>(resource.accepted_sm_count));
    }
  }
  CudaGreenContextPool::Config config{
      .device_id = pool_plan.device.value,
      .num_partitions = static_cast<std::uint32_t>(resources.size()),
      .sms_per_partition = std::move(sm_counts),
      .nvtx_identifier = pool_plan.pool_id,
  };
  if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kSplit)) {
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kSplit));
  }
  auto pool = std::make_unique<CudaGreenContextPool>(std::move(config));
  if (auto initialized = pool->initialize_pool(); !initialized) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED",
                                runtime_error_message(initialized.error()));
  }
  const auto total = pool->device_total_sms();
  const auto minimum = pool->min_sm_partition_size();
  const auto alignment = pool->sm_coscheduled_alignment();
  if (!total || !minimum || !alignment || *total != pool_plan.device_sm_count ||
      *minimum != pool_plan.min_sm_partition_size ||
      *alignment != pool_plan.sm_coscheduled_alignment) {
    return cuda_execution_error("CUDA_GREEN_CONTEXT_CAPABILITY_DRIFT");
  }

  CudaGreenContextPool* const borrowed_pool = pool.get();
  impl.pools.push_back(RuntimeCudaExecutionResources::Impl::PoolOwner{
      .plan = &pool_plan,
      .pool = std::move(pool),
      .driver = driver,
  });
  for (std::size_t resource_index = 0U; resource_index < resources.size(); ++resource_index) {
    record_diagnostic(impl, "green_context_create", &pool_plan, resources[resource_index]);
    if (inject_failure(impl, RuntimeCudaExecutionFailurePoint::kContext)) {
      return make_unexpected(
          injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kContext));
    }
    auto context = borrowed_pool->green_context(static_cast<std::uint32_t>(resource_index));
    if (!context || *context == nullptr) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED");
    }
    auto cuda_context = borrowed_pool->cuda_context(static_cast<std::uint32_t>(resource_index));
    if (!cuda_context || *cuda_context == nullptr) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED");
    }
    nvtxNameCuContextA(*cuda_context, resources[resource_index]->green_context_id.c_str());
    for (const auto& stream : plan.cuda_compute_streams) {
      if (stream.process_group == pool_plan.process_group && stream.device == pool_plan.device &&
          stream.green_context_id == resources[resource_index]->green_context_id) {
        auto materialized = materialize_green_stream(impl, stream, *context, driver);
        if (!materialized) {
          return materialized;
        }
      }
    }
  }
  return {};
#else
  static_cast<void>(plan);
  static_cast<void>(pool_plan);
  static_cast<void>(impl);
  return cuda_execution_error("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED");
#endif
}

// Cross-check every stream, assignment, partition, and operator against the accepted plan.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_materialized_streams(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources::Impl& impl,
    std::span<const RuntimeOperatorInstance> operators) {
  for (const auto& stream : plan.cuda_compute_streams) {
    if (stream.process_group != impl.process_group) {
      continue;
    }
    const auto owner = std::find_if(impl.streams.begin(), impl.streams.end(), [&](const auto& row) {
      return row.plan->process_group == stream.process_group && row.plan->device == stream.device &&
             row.plan->owner_id == stream.owner_id && row.plan->stream_id == stream.stream_id;
    });
    if (owner == impl.streams.end() || owner->native_handle() == nullptr) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
    }
    if (!stream.partition) {
      continue;
    }
    const auto member_count =
        std::count_if(operators.begin(), operators.end(), [&](const auto& instance) {
          return instance.partition == *stream.partition;
        });
    if (std::cmp_equal(member_count, 0U)) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
    }
  }
  for (const auto& assignment : plan.cuda_green_context_assignments) {
    if (assignment.process_group != impl.process_group) {
      continue;
    }
    cudaStream_t stream = std::invoke([&]() {
      const auto owner =
          std::find_if(impl.streams.begin(), impl.streams.end(), [&](const auto& row) {
            return row.plan->process_group == assignment.process_group &&
                   row.plan->device == assignment.device &&
                   row.plan->partition == assignment.partition &&
                   row.plan->green_context_id == assignment.green_context_id;
          });
      return owner == impl.streams.end() ? nullptr : owner->native_handle();
    });
    if (stream == nullptr) {
      return cuda_execution_error("CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
    }
    for (const auto& instance : operators) {
      if (instance.partition == assignment.partition &&
          ((instance.compute_stream != nullptr && instance.compute_stream != stream) ||
           (instance.compute_device.valid() && instance.compute_device != assignment.device))) {
        return cuda_execution_error("CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
      }
    }
  }
  return {};
}

void bind_partition_streams(const RuntimeCudaExecutionResources::Impl& impl,
                            std::span<RuntimeOperatorInstance> operators) {
  for (const auto& owner : impl.streams) {
    const auto& partition = owner.plan->partition;
    if (!partition) {
      continue;
    }
    // The immediately preceding guard proves the plan-owned optional is engaged.
    const ExecutionPartitionId partition_id =
        *partition;  // NOLINT(bugprone-unchecked-optional-access)
    cudaStream_t stream = owner.native_handle();
    for (auto& instance : operators) {
      if (instance.partition != partition_id) {
        continue;
      }
      instance.compute_stream = stream;
      instance.compute_device = owner.plan->device;
    }
  }
}

void clear_partition_streams(const RuntimeCudaExecutionResources::Impl& impl,
                             std::span<RuntimeOperatorInstance> operators) noexcept {
  for (const auto& owner : impl.streams) {
    const auto& partition = owner.plan->partition;
    if (!partition) {
      continue;
    }
    // The immediately preceding guard proves the plan-owned optional is engaged.
    const ExecutionPartitionId partition_id =
        *partition;  // NOLINT(bugprone-unchecked-optional-access)
    for (auto& instance : operators) {
      if (instance.partition == partition_id && instance.compute_stream == owner.native_handle() &&
          instance.compute_device == owner.plan->device) {
        instance.compute_stream = nullptr;
        instance.compute_device = {};
      }
    }
  }
}

void retain_first_failure(std::optional<Error>& first, Error error,
                          std::string_view cleanup_operation = {}) {
  if (!first) {
    first = std::move(error);
    return;
  }
  first->message += fmt::format("; cleanup {}: {}",
                                cleanup_operation.empty() ? "failure" : cleanup_operation,
                                runtime_error_message(error));
}

void settle_device_claims_after_failure(RuntimeCudaExecutionResources::Impl& impl,
                                        const std::set<std::int32_t>& failed_devices,
                                        bool failure_device_unknown) noexcept {
  for (const auto& claim : impl.claimed_devices) {
    if (failure_device_unknown || failed_devices.contains(claim.device.value)) {
      static_cast<void>(transition_device_claim(claim.device,
                                                impl.plan_generation,
                                                impl.run_generation,
                                                impl.claim_owner_id,
                                                DeviceClaim::State::kQuarantined));
    } else {
      release_device(claim.device, impl.plan_generation, impl.run_generation, impl.claim_owner_id);
    }
  }
}

void quarantine_all_device_claims(RuntimeCudaExecutionResources::Impl& impl) noexcept {
  for (const auto& claim : impl.claimed_devices) {
    static_cast<void>(transition_device_claim(claim.device,
                                              impl.plan_generation,
                                              impl.run_generation,
                                              impl.claim_owner_id,
                                              DeviceClaim::State::kQuarantined));
  }
}

}  // namespace

RuntimeCudaExecutionResources::Impl::~Impl() {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (auto result = shutdown(); !result) {
      if (quiesce_failure) {
        for (auto& stream : streams) {
          stream.abandon_after_failed_quiesce();
        }
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
        for (auto& pool : pools) {
          [[maybe_unused]] const auto released = pool.pool.release();
        }
#endif
      }
      HOLOSCAN_LOG_ERROR("Failed to shut down Runtime CUDA execution resources: {}",
                         runtime_error_message(result.error()));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    try {
      HOLOSCAN_LOG_ERROR("Failed to shut down Runtime CUDA execution resources: unknown exception");
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Destruction is best-effort and must not terminate while reporting another exception.
    }
  }
#endif
}

expected<void, Error> RuntimeCudaExecutionResources::Impl::quiesce(
    RuntimeCudaExecutionDiagnostic* output) {
  diagnostic_output = output;
  if (quiesce_complete) {
    if (diagnostic_output != nullptr) {
      *diagnostic_output = diagnostic;
    }
    diagnostic_output = nullptr;
    return {};
  }
  if (quiesce_failure) {
    if (diagnostic_output != nullptr) {
      *diagnostic_output = diagnostic;
    }
    diagnostic_output = nullptr;
    return make_unexpected(*quiesce_failure);
  }
  std::optional<Error> first_failure;
  std::optional<RuntimeCudaExecutionDiagnostic> first_failure_diagnostic;
  for (const ClaimedDevice& claim : claimed_devices) {
    record_diagnostic(*this,
                      "device_claim_quiesce",
                      claim.green_pool,
                      nullptr,
                      nullptr,
                      -1,
                      LifecycleStage::kRelease,
                      claim.device);
    if (!transition_device_claim(claim.device,
                                 plan_generation,
                                 run_generation,
                                 claim_owner_id,
                                 DeviceClaim::State::kQuiescing)) {
      retain_first_failure(first_failure,
                           Error{ErrorCode::kFailure, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT"});
      if (!first_failure_diagnostic) {
        first_failure_diagnostic = diagnostic;
      }
    }
  }
  // The required Green Context default stream cannot carry HSDK work: HSDK never exposes that
  // stream or makes the converted CUcontext current. Every stream to which HSDK can submit is
  // therefore owned in this inventory and synchronized explicitly before native teardown.
  const auto quiesce_started = std::chrono::steady_clock::now();
  for (auto& stream : streams) {
    record_diagnostic(
        *this, "stream_quiesce", nullptr, nullptr, stream.plan, -1, LifecycleStage::kRelease);
    std::int32_t native_result = -1;
    if (auto synchronized = stream.synchronize(&native_result); !synchronized) {
      record_diagnostic(*this,
                        "stream_quiesce",
                        nullptr,
                        nullptr,
                        stream.plan,
                        native_result,
                        LifecycleStage::kRelease);
      retain_first_failure(first_failure, std::move(synchronized.error()));
      if (!first_failure_diagnostic) {
        first_failure_diagnostic = diagnostic;
      }
    }
  }
  if (test_control != nullptr) {
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - quiesce_started);
    test_control->record_cuda_stream_quiesce(0U, static_cast<std::uint64_t>(duration.count()));
  }
  if (first_failure) {
    // RunSession retains the complete native owner closure after failed quiescence. Keep every
    // claim in that closure quarantined until process exit rather than admitting another exclusive
    // owner over a peer device whose synchronized resources remain alive.
    quarantine_all_device_claims(*this);
    if (first_failure_diagnostic) {
      diagnostic = std::move(*first_failure_diagnostic);
      if (diagnostic_output != nullptr) {
        *diagnostic_output = diagnostic;
      }
    }
    quiesce_failure = *first_failure;
    diagnostic_output = nullptr;
    return make_unexpected(std::move(*first_failure));
  }
  quiesce_complete = true;
  record_diagnostic(
      *this, "quiesce_complete", nullptr, nullptr, nullptr, -1, LifecycleStage::kRelease);
  diagnostic_output = nullptr;
  return {};
}

// Release streams, pools, and device claims in strict reverse ownership order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeCudaExecutionResources::Impl::shutdown(
    RuntimeCudaExecutionDiagnostic* output) {
  diagnostic_output = output;
  if (shutdown_complete) {
    if (diagnostic_output != nullptr) {
      *diagnostic_output = diagnostic;
    }
    diagnostic_output = nullptr;
    return {};
  }
  if (auto synchronized = quiesce(output); !synchronized) {
    return synchronized;
  }
  diagnostic_output = output;
  std::optional<Error> first_failure;
  std::optional<RuntimeCudaExecutionDiagnostic> first_failure_diagnostic;
  std::set<std::int32_t> failed_devices;
  bool failure_device_unknown = false;
  for (auto stream = streams.rbegin(); stream != streams.rend(); ++stream) {
    record_diagnostic(
        *this, "stream_destroy", nullptr, nullptr, stream->plan, -1, LifecycleStage::kRelease);
    if (auto destroyed = stream->destroy(); !destroyed) {
      retain_first_failure(first_failure, std::move(destroyed.error()), "stream destruction");
      if (!first_failure_diagnostic) {
        first_failure_diagnostic = diagnostic;
      }
      if (stream->plan != nullptr) {
        failed_devices.insert(stream->plan->device.value);
      } else {
        failure_device_unknown = true;
      }
    }
  }
  streams.clear();
  if (test_control != nullptr) {
    for (std::size_t index = 0U; index < materialized_stream_count; ++index) {
      test_control->record_cuda_stream_destroyed(0U);
    }
  }
  materialized_stream_count = 0U;
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
  for (auto pool = pools.rbegin(); pool != pools.rend(); ++pool) {
    if (pool->pool != nullptr) {
      record_diagnostic(
          *this, "pool_destroy", pool->plan, nullptr, nullptr, -1, LifecycleStage::kRelease);
      if (auto destroyed = pool->pool->deinitialize_pool(); !destroyed) {
        retain_first_failure(first_failure, std::move(destroyed.error()), "pool destruction");
        if (!first_failure_diagnostic) {
          first_failure_diagnostic = diagnostic;
        }
        if (pool->plan != nullptr) {
          failed_devices.insert(pool->plan->device.value);
        } else {
          failure_device_unknown = true;
        }
      }
    }
  }
#endif
  pools.clear();
  if (first_failure) {
    settle_device_claims_after_failure(*this, failed_devices, failure_device_unknown);
  } else {
    for (const ClaimedDevice& claim : claimed_devices) {
      release_device(claim.device, plan_generation, run_generation, claim_owner_id);
    }
  }
  claimed_devices.clear();
  shutdown_complete = true;
  if (first_failure) {
    if (first_failure_diagnostic) {
      diagnostic = std::move(*first_failure_diagnostic);
      if (diagnostic_output != nullptr) {
        *diagnostic_output = diagnostic;
      }
    }
    diagnostic_output = nullptr;
    return make_unexpected(std::move(*first_failure));
  }
  record_diagnostic(
      *this, "shutdown_complete", nullptr, nullptr, nullptr, -1, LifecycleStage::kRelease);
  diagnostic_output = nullptr;
  return {};
}

RuntimeCudaExecutionResources::RuntimeCudaExecutionResources(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeCudaExecutionResources::~RuntimeCudaExecutionResources() = default;

expected<std::unique_ptr<RuntimeCudaExecutionResources>, Error>
// Materialize all CUDA resources before exposing the owner to RunSession.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeCudaExecutionResources::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, ProcessGroupId process_group,
    std::shared_ptr<RuntimeTensorOutputTestControl> test_control,
    std::shared_ptr<RuntimeCudaExecutionTestControl> cuda_test_control,
    RuntimeCudaExecutionDiagnostic* diagnostic) {
  if (diagnostic != nullptr) {
    *diagnostic = RuntimeCudaExecutionDiagnostic{
        .plan_generation = plan.plan_generation,
        .run_generation = run_generation,
        .lifecycle_stage = LifecycleStage::kArm,
        .operation = "plan_validation",
    };
  }
  auto valid = validate_plan(plan, operators, run_generation, process_group);
  if (!valid) {
    return make_unexpected(std::move(valid.error()));
  }
  auto impl = std::make_unique<Impl>();
  impl->plan_generation = plan.plan_generation;
  impl->run_generation = run_generation;
  impl->process_group = process_group;
  const bool owns_cuda_execution =
      std::ranges::any_of(
          plan.cuda_compute_streams,
          [&](const auto& stream) { return stream.process_group == process_group; }) ||
      std::ranges::any_of(plan.cuda_green_context_pools,
                          [&](const auto& pool) { return pool.process_group == process_group; });
  if (owns_cuda_execution) {
    auto owner_id = reserve_device_claim_owner_id();
    if (!owner_id) {
      return make_unexpected(std::move(owner_id.error()));
    }
    impl->claim_owner_id = *owner_id;
  }
  if (diagnostic != nullptr) {
    impl->diagnostic = *diagnostic;
  }
  impl->diagnostic_output = diagnostic;
  impl->test_control = std::move(test_control);
  impl->cuda_test_control = std::move(cuda_test_control);

  std::set<std::int32_t> green_devices;
  for (const auto& pool : plan.cuda_green_context_pools) {
    if (pool.process_group != process_group) {
      continue;
    }
    record_diagnostic(*impl, "device_claim", &pool);
    if (inject_failure(*impl, RuntimeCudaExecutionFailurePoint::kClaim)) {
      return make_unexpected(
          injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kClaim));
    }
    auto claimed = claim_green_device(pool.deployment_device_id,
                                      pool.device,
                                      plan.plan_generation,
                                      run_generation,
                                      impl->claim_owner_id);
    if (!claimed) {
      return make_unexpected(std::move(claimed.error()));
    }
    impl->claimed_devices.push_back(
        Impl::ClaimedDevice{.device = pool.device, .green_pool = &pool});
    green_devices.insert(pool.device.value);
  }
  std::set<std::int32_t> ordinary_devices;
  for (const auto& stream : plan.cuda_compute_streams) {
    if (stream.process_group == process_group && !stream.green_context_id &&
        !green_devices.contains(stream.device.value)) {
      ordinary_devices.insert(stream.device.value);
    }
  }
  for (const std::int32_t device_ordinal : ordinary_devices) {
    const DeviceId device{device_ordinal};
    record_diagnostic(*impl,
                      "ordinary_device_claim",
                      nullptr,
                      nullptr,
                      nullptr,
                      -1,
                      LifecycleStage::kArm,
                      device);
    auto claimed =
        claim_ordinary_device(device, plan.plan_generation, run_generation, impl->claim_owner_id);
    if (!claimed) {
      return make_unexpected(std::move(claimed.error()));
    }
    impl->claimed_devices.push_back(Impl::ClaimedDevice{.device = device});
  }

  for (const auto& pool : plan.cuda_green_context_pools) {
    if (pool.process_group != process_group) {
      continue;
    }
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    auto driver = reprobe_pool(pool, *impl);
    if (!driver) {
      return make_unexpected(std::move(driver.error()));
    }
    auto materialized = materialize_green_pool(plan, pool, *impl, *driver);
#else
    auto materialized = materialize_green_pool(plan, pool, *impl);
#endif
    if (!materialized) {
      return make_unexpected(std::move(materialized.error()));
    }
  }
  for (const auto& stream : plan.cuda_compute_streams) {
    if (stream.process_group != process_group || stream.green_context_id) {
      continue;
    }
    auto materialized = materialize_stream(*impl, stream);
    if (!materialized) {
      return make_unexpected(std::move(materialized.error()));
    }
  }
  // Validate native owners and any pre-existing bindings before mutating Operator instances. The
  // second pass below verifies the bindings that bind_partition_streams() publishes.
  record_diagnostic(*impl, "stream_binding_validation");
  auto bindings = validate_materialized_streams(plan, *impl, operators);
  if (!bindings) {
    return make_unexpected(std::move(bindings.error()));
  }
  if (inject_failure(*impl, RuntimeCudaExecutionFailurePoint::kBinding)) {
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kBinding));
  }
  record_diagnostic(*impl, "stream_binding_publish");
  bind_partition_streams(*impl, operators);
  bindings = validate_materialized_streams(plan, *impl, operators);
  if (!bindings) {
    clear_partition_streams(*impl, operators);
    return make_unexpected(std::move(bindings.error()));
  }
  if (inject_failure(*impl, RuntimeCudaExecutionFailurePoint::kPublish)) {
    clear_partition_streams(*impl, operators);
    return make_unexpected(
        injected_materialization_failure(RuntimeCudaExecutionFailurePoint::kPublish));
  }
  for (const Impl::ClaimedDevice& claim : impl->claimed_devices) {
    record_diagnostic(*impl, "device_claim_activate", claim.green_pool);
    if (!transition_device_claim(claim.device,
                                 impl->plan_generation,
                                 impl->run_generation,
                                 impl->claim_owner_id,
                                 DeviceClaim::State::kActive)) {
      clear_partition_streams(*impl, operators);
      return make_unexpected(
          Error{ErrorCode::kFailure, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT"});
    }
  }
  record_diagnostic(*impl, "materialization_complete");
  impl->diagnostic_output = nullptr;
  return std::unique_ptr<RuntimeCudaExecutionResources>{
      new RuntimeCudaExecutionResources(std::move(impl))};
}

expected<void, Error> RuntimeCudaExecutionResources::shutdown(
    RuntimeCudaExecutionDiagnostic* diagnostic) {
  return impl_ != nullptr ? impl_->shutdown(diagnostic) : expected<void, Error>{};
}

expected<void, Error> RuntimeCudaExecutionResources::quiesce(
    RuntimeCudaExecutionDiagnostic* diagnostic) {
  return impl_ != nullptr ? impl_->quiesce(diagnostic) : expected<void, Error>{};
}

cudaStream_t RuntimeCudaExecutionResources::partition_stream(
    ProcessGroupId process_group, DeviceId device, ExecutionPartitionId partition) const noexcept {
  // This cold lookup is resolved once while binding Runtime endpoint/operator state.
  const auto found =
      std::find_if(impl_->streams.begin(), impl_->streams.end(), [&](const auto& row) {
        return row.plan->process_group == process_group && row.plan->device == device &&
               row.plan->partition == partition;
      });
  return found == impl_->streams.end() ? nullptr : found->native_handle();
}

cudaStream_t RuntimeCudaExecutionResources::service_stream(
    ProcessGroupId process_group, DeviceId device, std::string_view owner_id) const noexcept {
  // Service streams are resolved during Channel binding, never per message.
  const auto found =
      std::find_if(impl_->streams.begin(), impl_->streams.end(), [&](const auto& row) {
        return row.plan->process_group == process_group && row.plan->device == device &&
               row.plan->owner_id == owner_id && !row.plan->partition;
      });
  return found == impl_->streams.end() ? nullptr : found->native_handle();
}

cudaStream_t RuntimeCudaExecutionResources::ordinary_stream(ProcessGroupId process_group,
                                                            DeviceId device) const noexcept {
  // Ordinary streams are resolved once during Runtime materialization.
  const auto found =
      std::find_if(impl_->streams.begin(), impl_->streams.end(), [&](const auto& row) {
        return row.plan->process_group == process_group && row.plan->device == device &&
               !row.plan->partition && !row.plan->green_context_id;
      });
  return found == impl_->streams.end() ? nullptr : found->native_handle();
}

}  // namespace holoscan::detail
