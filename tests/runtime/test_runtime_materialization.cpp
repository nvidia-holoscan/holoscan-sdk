/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <holoscan/core/run.hpp>

#include "common/cuda_test_utils.hpp"
#include "compiler/cuda_execution_plan_validation.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/lifecycle_executor.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_fixture_observation.hpp"
#include "runtime/runtime_materialization.hpp"
#include "runtime/runtime_tensor_resource_materialization.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_cuda_green_context_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan::detail {

// White-box accessor that exposes RuntimeLifecycleExecutor::materialization_ for test
// inspection without requiring a public API. Used in LifecycleGraphFails... to inspect
// per-obligation records after a failed startup.
class RuntimeLifecycleExecutorTestAccess {
 public:
  [[nodiscard]] static const RuntimeMaterializationState* materialization(
      const RuntimeLifecycleExecutor& executor) noexcept {
    return executor.materialization_ ? &*executor.materialization_ : nullptr;
  }
};

namespace {

TEST(RuntimeMaterializationTest, ProviderReportingPreservesIndependentOptionalEvidence) {
  RuntimeProviderMaterializationResult seeded{
      .provider = ProviderAllocationPlanRecordId{4U},
      .compatibility_reason = DiagnosticId{17U},
      .stride_bytes_per_entry = 64U,
      .reserved_bytes_total = 512U,
  };

  const auto updated = with_provider_materialization(
      seeded, ProviderAllocationPlanRecordId{4U}, AllocationAction::kReusedLarger, 8U);

  EXPECT_EQ(updated.provider, ProviderAllocationPlanRecordId{4U});
  EXPECT_EQ(updated.action, AllocationAction::kReusedLarger);
  EXPECT_EQ(updated.compatibility_reason, DiagnosticId{17U});
  EXPECT_EQ(updated.materialized_entries, 8U);
  EXPECT_EQ(updated.stride_bytes_per_entry, 64U);
  EXPECT_EQ(updated.reserved_bytes_total, 512U);
  EXPECT_TRUE(updated.reported);
}

// Minimal operator that records start/stop invocations via static atomics. Used in the lifecycle
// graph test to verify that operator
// callbacks are not called when materialization fails before reaching kStart.
class LifecycleMaterializationProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(spin_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  void start() override { start_count.fetch_add(1U, std::memory_order_relaxed); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  void stop() override { stop_count.fetch_add(1U, std::memory_order_relaxed); }

  static inline std::atomic<std::uint32_t> start_count{};
  static inline std::atomic<std::uint32_t> stop_count{};

 private:
  SpinSource spin_;
};

// Compiles one direct-host Tensor output whose queue and pool geometry become concrete only after
// the kDiscover rate barrier. Runtime materialization must accept its provider template without
// requiring an immutable zero-slot allocation placeholder.
class LifecycleCandidateMaterializationTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output_;
};

class LifecycleCandidateMaterializationTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input_, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{1}})
        .input(input_, Exact{})
        .on_missing(input_, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input_;
};

class OutputFailureCapabilityTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Provides a clock-driven, Tensor-inactive operator whose assigned compute stream can be inspected
// before callbacks run. Used by
// RuntimeMaterializationTest.TensorInactiveGreenContextPartitionReceivesVerifiedStream.
class CudaGreenContextMaterializationProbeOperator final : public Operator<> {
 public:
  ~CudaGreenContextMaterializationProbeOperator() override {
    destruction_count.fetch_add(1U, std::memory_order_relaxed);
  }

  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  void start() override { start_count.fetch_add(1U, std::memory_order_relaxed); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    while (block_compute.load(std::memory_order_acquire) &&
           !release_compute.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    if (fail_compute.load(std::memory_order_acquire)) {
      return make_unexpected(Error{ErrorCode::kFailure, "injected Green Context compute failure"});
    }
    return {};
  }

  static inline std::atomic<std::uint32_t> start_count{};
  static inline std::atomic<std::uint32_t> compute_count{};
  static inline std::atomic<std::uint32_t> destruction_count{};
  static inline std::atomic_bool block_compute{};
  static inline std::atomic_bool release_compute{};
  static inline std::atomic_bool fail_compute{};
};

// Provides a minimal process-local Channel owner whose lifetime can be compared with terminal CUDA
// quiescence. Used by RuntimeMaterializationTest.ChannelMappingsOutliveGreenContextQuiesce.
class CudaGreenContextChannelSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_release);
    return output.emit(1U);
  }

  Output<std::uint32_t> output;
  static inline std::atomic<std::uint32_t> compute_count{};
};

class CudaGreenContextChannelSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive_data();
    return received ? expected<void, Error>{} : make_unexpected(std::move(received.error()));
  }

  Input<std::uint32_t> input;
};

// Provides one finite direct activation so teardown failures can be separated from the callback's
// requested terminal reason.
class CudaGreenContextCompletionProbeOperator final : public Operator<> {
 public:
  enum class TerminationRequest : std::uint8_t {
    kNone,
    kCompletion,
    kStop,
  };

  void setup(OperatorSpec& spec) override { spec.spin_source(spin_, "activation").capacity(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    const TerminationRequest request =
        termination_request.exchange(TerminationRequest::kNone, std::memory_order_acq_rel);
    consumed_termination_request.store(request, std::memory_order_release);
    switch (request) {
      case TerminationRequest::kCompletion:
        context.request_completion();
        break;
      case TerminationRequest::kStop:
        context.request_stop();
        break;
      case TerminationRequest::kNone:
        break;
    }
    return {};
  }

  static inline std::atomic<std::uint32_t> compute_count{};
  static inline std::atomic<TerminationRequest> termination_request{TerminationRequest::kNone};
  static inline std::atomic<TerminationRequest> consumed_termination_request{
      TerminationRequest::kNone};

 private:
  SpinSource spin_;
};

// Launches one stream-directed CUDA host callback without declaring Tensor ports. Used by
// RuntimeMaterializationTest.TensorInactiveGreenContextCallbackLaunchesOnAssignedStream.
class CudaGreenContextCallbackProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (submitted.exchange(true, std::memory_order_acq_rel)) {
      return {};
    }
    observed_stream.store(context.cuda_stream().get(), std::memory_order_release);
    if (observed_stream.load(std::memory_order_acquire) == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "missing assigned Green Context stream"});
    }
    const cudaError_t result =
        cudaLaunchHostFunc(observed_stream.load(std::memory_order_acquire), complete, nullptr);
    if (result != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure, "stream-directed CUDA launch failed"});
    }
    return {};
  }

  static void reset() noexcept {
    submitted.store(false, std::memory_order_release);
    completed.store(false, std::memory_order_release);
    observed_stream.store(nullptr, std::memory_order_release);
  }

  static inline std::atomic_bool submitted{};
  static inline std::atomic_bool completed{};
  static inline std::atomic<cudaStream_t> observed_stream{};

 private:
  static void CUDART_CB complete(void*) noexcept {
    completed.store(true, std::memory_order_release);
  }
};

// Captures private, pointer-free CUDA diagnostic evidence published with one failed RunSession.
// Used by RunSessionPreservesDetailedCudaMaterializationDiagnosticToken.
class CudaExecutionDiagnosticSink final : public RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic>) override {}

  void publish_runtime_details(std::span<const RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    details_.assign(details.begin(), details.end());
  }

  [[nodiscard]] std::vector<RuntimeDiagnosticDetail> details() const {
    const std::lock_guard lock(mutex_);
    return details_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<RuntimeDiagnosticDetail> details_;
};

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
// One finite POD publication gives the observed output thunk a real retired activation before
// quiescence. The depth-two connection keeps this graph on ordinary dispatch, as required by the
// existing private observation profile.
class CudaGreenContextObservedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return output.emit(37U);
  }

  Output<std::uint32_t> output;
};

class CudaGreenContextObservedSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive_data();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    return *received == 37U ? expected<void, Error>{} : make_unexpected(Error{ErrorCode::kFailure});
  }

  Input<std::uint32_t> input;
};

[[nodiscard]] ExecutionPlan fake_observed_green_context_channel_plan() {
  Graph graph{"runtime-observed-green-context-channel-lifetime"};
  const auto source = graph.op<CudaGreenContextObservedSource>("source");
  const auto sink = graph.op<CudaGreenContextObservedSink>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  graph.partition("fixture").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  CompileOptions options;
  options.deployment.bind_cuda_green_context("fixture", {.device = DeviceId{0}, .sm_count = 80U});
  return compile(graph, std::move(options));
}

// Graph topology (each probe is isolated in its own execution partition):
//   probe-0   probe-1   ...   probe-(partition_count - 1)
[[nodiscard]] ExecutionPlan fake_green_context_plan(std::size_t partition_count = 2U,
                                                    bool only_first_bound = false,
                                                    DeviceId device = DeviceId{0}) {
  Graph graph{"runtime-green-context-fake-driver"};
  std::vector<OperatorHandle<CudaGreenContextMaterializationProbeOperator>> operators;
  operators.reserve(partition_count);
  for (std::size_t index = 0U; index < partition_count; ++index) {
    const std::string name = "probe-" + std::to_string(index);
    const auto probe = graph.op<CudaGreenContextMaterializationProbeOperator>(name);
    graph.partition("partition-" + std::to_string(index)).add(probe);
    operators.push_back(probe);
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  if (only_first_bound) {
    options.deployment.bind_cuda_green_context(
        "partition-0", CudaGreenContextPlacement{.device = device, .sm_count = 16U});
  } else {
    options.deployment
        .bind_cuda_green_context("partition-0",
                                 CudaGreenContextPlacement{.device = device, .sm_count = 32U})
        .bind_cuda_green_context("partition-1",
                                 CudaGreenContextPlacement{.device = device, .sm_count = 48U});
  }
  return compile(graph, std::move(options));
}

// Graph topology:
//   completion-probe   (one finite activation in one full-device Green Context)
[[nodiscard]] ExecutionPlan fake_finite_green_context_plan(DeviceId device = DeviceId{0}) {
  Graph graph{"runtime-green-context-completion-fake-driver"};
  const auto probe = graph.op<CudaGreenContextCompletionProbeOperator>("completion-probe");
  graph.partition("completion-partition").add(probe);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("completion-partition",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_cuda_green_context("completion-partition",
                               CudaGreenContextPlacement{.device = device, .sm_count = 80U});
  return compile(graph, std::move(options));
}

// Graph topology (each Channel endpoint has a dedicated Green Context):
//   source(device 0) --> sink(device 0)
[[nodiscard]] ExecutionPlan fake_green_context_channel_plan() {
  Graph graph{"runtime-green-context-channel-lifetime"};
  const auto source = graph.op<CudaGreenContextChannelSourceOperator>("source");
  const auto sink = graph.op<CudaGreenContextChannelSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("sink-partition").add(sink);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_cuda_green_context("source-partition", {.device = DeviceId{0}, .sm_count = 32U})
      .bind_cuda_green_context("sink-partition", {.device = DeviceId{0}, .sm_count = 48U});
  return compile(graph, std::move(options));
}

// Graph topology (each probe has an exact full-device Green Context on a distinct device):
//   probe-0(device 0)   probe-1(device 1)
[[nodiscard]] ExecutionPlan fake_two_device_green_context_plan() {
  Graph graph{"runtime-green-context-two-device-fake-driver"};
  const auto device_zero = graph.op<CudaGreenContextMaterializationProbeOperator>("probe-0");
  const auto device_one = graph.op<CudaGreenContextMaterializationProbeOperator>("probe-1");
  graph.partition("partition-0").add(device_zero);
  graph.partition("partition-1").add(device_one);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_cuda_green_context("partition-0", {.device = DeviceId{0}, .sm_count = 80U})
      .bind_cuda_green_context("partition-1", {.device = DeviceId{1}, .sm_count = 80U});
  return compile(graph, std::move(options));
}

[[nodiscard]] ProcessGroupId green_process_group(const ExecutionPlanState& plan) {
  return plan.cuda_green_context_pools.empty()
             ? ProcessGroupId{}
             : plan.cuda_green_context_pools.front().process_group;
}

[[nodiscard]] std::shared_ptr<ExecutionPlanState> mutable_plan_state(const ExecutionPlan& plan) {
  const auto accepted = ExecutionPlanAccess::retain(plan);
  return accepted == nullptr ? nullptr : std::make_shared<ExecutionPlanState>(*accepted);
}

// Builds the smallest valid plan inventory that materializes one ordinary CUDA stream on device
// zero. Used by the device-ledger tests to represent an unpartitioned Tensor service owner.
[[nodiscard]] ExecutionPlanState ordinary_cuda_execution_plan() {
  ExecutionPlanState plan;
  plan.plan_generation = 61U;
  plan.tensor_device_resources.push_back(TensorDeviceResourcePlanRecord{
      .plan_generation = plan.plan_generation,
      .run_generation = 0U,
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .allocator_root_id = "tensor-allocator.61.process.1.device.0",
      .materialize_before = LifecycleStage::kArm,
      .fallback_rejected = true,
  });
  plan.cuda_compute_streams.push_back(CudaComputeStreamPlanRecord{
      .plan_generation = plan.plan_generation,
      .run_generation = 0U,
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .owner_id = plan.tensor_device_resources.front().allocator_root_id,
      .stream_id = "tensor-stream.61.process.1.device.0",
      .materialize_before = LifecycleStage::kArm,
  });
  return plan;
}

template <typename Handle>
[[nodiscard]] bool destroyed_exactly_once(const std::vector<Handle>& created,
                                          const std::vector<Handle>& destroyed) {
  return created.size() == destroyed.size() && std::ranges::all_of(created, [&](Handle handle) {
           return std::ranges::count(destroyed, handle) == 1;
         });
}
#endif

// Compiles a three-operator plan with distinct execution partitions, all bound to BusyPoll.
// Operators a, b, c are independent spin-sources with no flow edges:
//   a   b   c   (each a spin-source; no downstream consumer)
// Used in LifecycleGraphFailsInCanonicalOrder... to supply a realistic ExecutionPlan.
[[nodiscard]] ExecutionPlan three_lane_materialization_plan() {
  Graph graph{"runtime-materialization-order"};
  const auto first = graph.op<LifecycleMaterializationProbeOperator>("a");
  const auto second = graph.op<LifecycleMaterializationProbeOperator>("b");
  const auto third = graph.op<LifecycleMaterializationProbeOperator>("c");
  graph.partition("materialization-partition-a").add(first);
  graph.partition("materialization-partition-b").add(second);
  graph.partition("materialization-partition-c").add(third);
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("materialization-partition-a",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("materialization-partition-b",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("materialization-partition-c",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan lifecycle_candidate_tensor_materialization_plan() {
  Graph graph{"runtime-rate-candidate-materialization"};
  const auto source = graph.op<LifecycleCandidateMaterializationTensorSource>("source");
  const auto sink = graph.op<LifecycleCandidateMaterializationTensorSink>("sink");
  graph.add_flow(source->output_, sink->input_);
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// Re-instantiates each operator from its factory and assigns it to the correct partition.
// Mirrors what the runtime does before startup; used in the lifecycle graph test to
// build a realistic RuntimeOperatorInstance list from an ExecutionPlanState.
[[nodiscard]] expected<std::vector<RuntimeOperatorInstance>, Error> reconstruct_operators(
    const ExecutionPlanState& plan) {
  std::vector<RuntimeOperatorInstance> operators;
  operators.reserve(plan.executable_factories.operators.size());
  for (const auto& record : plan.executable_factories.operators) {
    if (record.factory == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_FACTORY_INVALID"});
    }
    auto created = record.factory->create(nullptr);
    if (!created || created.value() == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"});
    }
    auto replay = std::make_unique<OperatorSpec>();
    created.value()->setup(*replay);
    const auto partition = std::find_if(
        plan.execution_partitions.begin(),
        plan.execution_partitions.end(),
        [&](const auto& candidate) {
          return std::find(candidate.members.begin(), candidate.members.end(), record.vertex) !=
                 candidate.members.end();
        });
    if (partition == plan.execution_partitions.end()) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"});
    }
    operators.push_back(RuntimeOperatorInstance{
        .vertex = record.vertex,
        .operator_id = record.operator_id,
        .partition = partition->id,
        .execution_lane = partition->execution_lane,
        .op = std::move(created).value(),
        .replay_spec = std::move(replay),
        .tags = std::make_unique<TagAllocator>(),
    });
  }
  return operators;
}

// Returns a SchedulerPlanRecord for deterministic serial execution across `lane_count`
// lanes, defaulting to 1. Used to build ExecutionPlanState fixtures in this file.
[[nodiscard]] SchedulerPlanRecord serial_scheduler(std::uint32_t lane_count = 1U) {
  return SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = lane_count,
      .intra_path_parallelism = 1U,
  };
}

// Returns an ExecutionLanePlanRecord for the given lane index and partition with no explicit
// OS placement. Used alongside default_lane() in multi-lane materialization tests.
[[nodiscard]] ExecutionLanePlanRecord explicit_lane(std::uint32_t lane,
                                                    ExecutionPartitionId partition) {
  return ExecutionLanePlanRecord{
      .execution_lane = lane,
      .partition = partition,
      .accepted_placement = ThreadPlacement{},
  };
}

// Returns the first partition lane with no explicit OS placement.
[[nodiscard]] ExecutionLanePlanRecord default_lane() {
  return ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = ExecutionPartitionId{1U},
      .accepted_placement = ThreadPlacement{},
  };
}

// Returns a RuntimePackageCapabilityMatchReport representing a successful capability
// match. Used as the reference package across all RuntimeMaterializationState tests.
[[nodiscard]] RuntimePackageCapabilityMatchReport matching_package() {
  return RuntimePackageCapabilityMatchReport{
      .context = {.startup_attempt = 3U, .plan_generation = 5U, .run_generation = 7U},
      .package_id = "holoscan-sdk",
      .package_build_id = "test-build",
      .package_manifest_digest =
          "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      .result = RuntimePackageCapabilityMatchResult::kMatched,
  };
}

[[nodiscard]] std::optional<std::size_t> tensor_output_pool_obligation(
    const RuntimeMaterializationState& state) noexcept {
  for (std::size_t index = 0U; index < state.size(); ++index) {
    if (state.record(index).strategy_id == "holoscan.runtime.tensor_output_pool/v1") {
      return index;
    }
  }
  return std::nullopt;
}

// Returns a ProfileMaterializationObligationRecord for the execution_lane strategy on
// object "lane:0". The most common obligation fixture in this file.
[[nodiscard]] ProfileMaterializationObligationRecord lane_obligation() {
  return ProfileMaterializationObligationRecord{
      .object_id = "lane:0",
      .strategy_id = "holoscan.runtime.execution_lane/v1",
      .complete_before_stage = LifecycleStage::kStart,
      .cleanup_stage_mask = 0xC0U,
      .max_cleanup_actions = 1U,
      .evidence_id = "holoscan.compile.thread_placement.validation/v1",
  };
}

// Returns a ProfileMaterializationObligationRecord for the generalized CUDA compute-stream
// strategy, targeting the stream resource defined by cuda_compute_stream(). Due at kArm.
[[nodiscard]] ProfileMaterializationObligationRecord tensor_device_stream_obligation() {
  return ProfileMaterializationObligationRecord{
      .object_id = "tensor-stream.5.process.1.device.0",
      .strategy_id = "holoscan.runtime.cuda_compute_stream/v1",
      .complete_before_stage = LifecycleStage::kArm,
      .cleanup_stage_mask = 0xC0U,
      .max_cleanup_actions = 1U,
      .evidence_id = "holoscan.compile.cuda_compute_stream_plan/v1",
  };
}

[[nodiscard]] CudaComputeStreamPlanRecord cuda_compute_stream() {
  return CudaComputeStreamPlanRecord{
      .plan_generation = 5U,
      .run_generation = 0U,
      .process_group = ProcessGroupId{1U},
      .device = DeviceId{0},
      .owner_id = "tensor-allocator.5.process.1.device.0",
      .stream_id = "tensor-stream.5.process.1.device.0",
      .materialize_before = LifecycleStage::kArm,
  };
}

// Holds a pre-built ExecutionPlanState and a started ExecutionLaneSet together.
// Populated by started_lane() before returning to each test.
struct StartedLane {
  ExecutionPlanState plan;
  std::unique_ptr<ExecutionLaneSet> lanes;
};

// Creates a single-lane serial ExecutionPlanState, starts its ExecutionLaneSet,
// and returns the pair. Used as the base fixture for most materialization tests.
[[nodiscard]] StartedLane started_lane() {
  StartedLane result;
  result.plan.scheduler_plan = serial_scheduler();
  result.plan.execution_lanes = {default_lane()};
  auto lanes = ExecutionLaneSet::create(result.plan.scheduler_plan, result.plan.execution_lanes);
  EXPECT_TRUE(lanes) << runtime_error_message(lanes.error());
  if (!lanes) {
    return result;
  }
  result.lanes = std::move(lanes).value();
  EXPECT_TRUE(result.lanes->start());
  return result;
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CompletesAtDeadlineAndCleansExactlyOnce) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  ASSERT_EQ(state.size(), 1U);

  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCompleted);
  EXPECT_EQ(state.record(0U).context.plan_generation, 5U);
  EXPECT_EQ(state.record(0U).package_id, "holoscan-sdk");

  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kSkipped);
  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kOk);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_EQ(
      state.transition(0U, LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown),
      LifecycleStatus::kSkipped);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: CHAN-103
TEST(RuntimeMaterializationTest, HostOnlyReconstructionKeepsCudaSelectionDisabled) {
  const ExecutionPlan plan =
      RuntimeTestPlanAccess::enable_direct_once(three_lane_materialization_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);

  auto reconstructed = GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  ASSERT_EQ((*reconstructed)->operators().size(), 3U);
  for (const RuntimeOperatorInstance& instance : (*reconstructed)->operators()) {
    EXPECT_FALSE(instance.compute_device.valid());
    EXPECT_EQ(instance.compute_stream, nullptr);
  }
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeMaterializationTest, ReconstructionRetainsOutputFailureCaptureCapability) {
  // All endpoints use ordinary host placement. The input-only and portless operators cannot
  // produce output rejection evidence; both POD and Tensor producers retain the reset path.
  Graph graph{"runtime-output-failure-capability"};
  graph.op<CudaGreenContextMaterializationProbeOperator>("portless");
  const auto pod_source = graph.op<CudaGreenContextChannelSourceOperator>("pod-source");
  const auto pod_sink = graph.op<CudaGreenContextChannelSinkOperator>("pod-sink");
  graph.add_flow(pod_source->output, pod_sink->input);
  const auto tensor_source = graph.op<OutputFailureCapabilityTensorSource>("tensor-source");
  const auto tensor_sink = graph.op<LifecycleCandidateMaterializationTensorSink>("tensor-sink");
  graph.add_flow(tensor_source->output, tensor_sink->input_);
  graph.set_default_clock(graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);

  for (const auto generation : {17U, 18U}) {
    RuntimeReplayMismatchDetail mismatch;
    auto reconstructed =
        GraphExec::reconstruct(accepted, RunGenerationId{generation}, {}, &mismatch);
    ASSERT_TRUE(reconstructed) << reconstructed.error().message << "; replay category "
                               << static_cast<unsigned>(mismatch.category);
    ASSERT_EQ((*reconstructed)->operators().size(), 5U);
    for (const RuntimeOperatorInstance& instance : (*reconstructed)->operators()) {
      const auto vertex = std::ranges::find_if(
          accepted->graph.vertices(), [&](const auto& row) { return row.id == instance.vertex; });
      ASSERT_NE(vertex, accepted->graph.vertices().end());
      EXPECT_EQ(instance.has_output_ports,
                vertex->name == "pod-source" || vertex->name == "tensor-source")
          << vertex->name << " in generation " << generation;
    }
  }
}

// A pinned-host descriptor schema carries a CUDA placement identity for its BufferHandle contract,
// but its host-to-host staging does not require a CUDA service stream or zero-pool device resource.
// REQ: CHAN-126
// REQ: CHAN-134
// REQ: CHAN-135
TEST(RuntimeMaterializationTest, PinnedDescriptorSchemaWithoutDeviceResourceMaterializes) {
  // Cross-process service topology:
  //   Image source [process 1] -- pinned BufferHandle --> Image sink [process 2]
  ExecutionPlanState plan;
  plan.plan_generation = 51U;
  plan.ports = {
      PortPlanRecord{
          .id = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .local_name = "image",
      },
      PortPlanRecord{
          .id = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .local_name = "image",
      },
  };
  plan.vertex_process_placements = {
      VertexProcessPlacementPlanRecord{
          .vertex = VertexId{1U},
          .process_group = ProcessGroupId{1U},
      },
      VertexProcessPlacementPlanRecord{
          .vertex = VertexId{2U},
          .process_group = ProcessGroupId{2U},
      },
  };
  plan.cross_process_channel_services.push_back(CrossProcessChannelServicePlanRecord{
      .edge = EdgeId{1U},
      .source = PortId{1U},
      .target = PortId{2U},
      .service_id = "pinned-descriptor-schema",
      .payload_identity =
          PayloadIdentityRecord{
              .kind = PayloadIdentityKind::kSchema,
              .payload_kind = PayloadKind::kDescriptorBacked,
              .transport_profile_tag = 1U,
          },
      .tensor_payload = true,
      .tensor_max_byte_span = 1024U,
      .tensor_memory_kind = MemoryKind::kPinnedHost,
      .tensor_device = DeviceId{0},
      .tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
      .source_process_group = ProcessGroupId{1U},
      .target_process_group = ProcessGroupId{2U},
      .target_process_groups = {ProcessGroupId{2U}},
  });
  plan.cross_process_channel_subscribers.push_back(CrossProcessChannelSubscriberPlanRecord{
      .edge = EdgeId{1U},
      .source = PortId{1U},
      .target = PortId{2U},
      .service_id = "pinned-descriptor-schema",
  });

  // Key setup: materialize the compiler's corrected no-binding inventory separately for the
  // descriptor-schema publisher and subscriber, each with a real local operator instance.
  const auto materialize_endpoint = [&](VertexId vertex,
                                        OperatorInstanceId operator_id,
                                        ProcessGroupId process_group) -> expected<void, Error> {
    std::array operators{RuntimeOperatorInstance{
        .vertex = vertex,
        .operator_id = operator_id,
    }};
    auto cuda_resources =
        RuntimeCudaExecutionResources::create(plan, operators, RunGenerationId{1U}, process_group);
    if (!cuda_resources) {
      return make_unexpected(std::move(cuda_resources.error()));
    }
    auto tensor_resources = RuntimeTensorResources::create(
        plan, operators, RunGenerationId{1U}, process_group, **cuda_resources);
    if (!tensor_resources) {
      return make_unexpected(std::move(tensor_resources.error()));
    }
    if ((*tensor_resources)->device_resource_count() != 0U ||
        operators.front().compute_stream != nullptr || operators.front().compute_device.valid()) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "PINNED_DESCRIPTOR_SCHEMA_ACQUIRED_CUDA_RESOURCES"});
    }
    return expected<void, Error>{};
  };
  auto publisher = materialize_endpoint(VertexId{1U}, OperatorInstanceId{1U}, ProcessGroupId{1U});
  auto subscriber = materialize_endpoint(VertexId{2U}, OperatorInstanceId{2U}, ProcessGroupId{2U});

  // Key assertion: stream selection accepts both non-empty endpoint inventories without creating
  // a CUDA resource or reporting a publisher/subscriber device resource mismatch before arm.
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, TensorInactiveGreenContextPartitionReceivesVerifiedStream) {
  // Graph topology:
  //   probe   (isolated in green-probe partition)
  // Key setup: bind a Tensor-inactive probe to the maximum driver-qualified structured capacity.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }

  Graph graph{"runtime-green-context-materialization"};
  const auto probe = graph.op<CudaGreenContextMaterializationProbeOperator>("probe");
  graph.partition("green-probe").add(probe);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  const auto structured_sm_count =
      static_cast<std::uint64_t>(capability->max_structured_context_count) *
      capability->structured_sm_quantum;
  ASSERT_GT(structured_sm_count, 0U);
  ASSERT_LE(structured_sm_count, std::numeric_limits<std::uint32_t>::max());
  options.deployment.bind_cuda_green_context(
      "green-probe",
      CudaGreenContextPlacement{.device = DeviceId{0},
                                .sm_count = static_cast<std::uint32_t>(structured_sm_count)});
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 1U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  ASSERT_TRUE(plan.tensor_device_resources().empty());

  const auto accepted = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  const ProcessGroupId process_group = plan.cuda_compute_streams().front().process_group;
  auto reconstructed = reconstruct_operators(*accepted);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  ASSERT_EQ(reconstructed->size(), 1U);
  EXPECT_EQ(reconstructed->front().compute_stream, nullptr);

  const PlanDigest plan_digest = plan.digest();
  auto resources = RuntimeCudaExecutionResources::create(
      *accepted, *reconstructed, RunGenerationId{31U}, process_group);
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());
  // Key assertion: materialization binds a verified device stream, enforces exclusive device
  // ownership, and releases that claim so a later generation can restart.
  const RuntimeOperatorInstance& instance = reconstructed->front();
  EXPECT_EQ(instance.compute_device, DeviceId{0});
  EXPECT_NE(instance.compute_stream, nullptr);
  EXPECT_EQ((*resources)->partition_stream(process_group, DeviceId{0}, instance.partition),
            instance.compute_stream);
  EXPECT_EQ(plan.digest(), plan_digest);

  auto contender_operators = reconstruct_operators(*accepted);
  ASSERT_TRUE(contender_operators) << contender_operators.error().message;
  auto contender = RuntimeCudaExecutionResources::create(
      *accepted, *contender_operators, RunGenerationId{32U}, process_group);
  ASSERT_FALSE(contender);
  EXPECT_EQ(contender.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");

  resources->reset();
  auto restarted_operators = reconstruct_operators(*accepted);
  ASSERT_TRUE(restarted_operators) << restarted_operators.error().message;
  auto restarted = RuntimeCudaExecutionResources::create(
      *accepted, *restarted_operators, RunGenerationId{33U}, process_group);
  ASSERT_TRUE(restarted) << runtime_error_message(restarted.error());
  EXPECT_NE(restarted_operators->front().compute_stream, nullptr);
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, TensorInactiveGreenContextCallbackLaunchesOnAssignedStream) {
  // Graph topology:
  //   callback-probe   (isolated in green-callback partition)
  // Key setup: bind a Tensor-inactive clock callback to one exact Green Context and launch a
  // CUDA host function through ExecutionContext::cuda_stream().
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const auto sm_count = static_cast<std::uint64_t>(capability->max_structured_context_count) *
                        capability->structured_sm_quantum;
  if (sm_count == 0U) {
    GTEST_SKIP() << "device exposes no structured SM group";
  }
  ASSERT_LE(sm_count, std::numeric_limits<std::uint32_t>::max());

  Graph graph{"runtime-green-context-callback"};
  const auto probe = graph.op<CudaGreenContextCallbackProbeOperator>("callback-probe");
  graph.partition("green-callback").add(probe);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_cuda_green_context(
      "green-callback",
      CudaGreenContextPlacement{.device = DeviceId{0},
                                .sm_count = static_cast<std::uint32_t>(sm_count)});
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  CudaGreenContextCallbackProbeOperator::reset();
  std::optional<RunSession> session{run_async(plan)};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!CudaGreenContextCallbackProbeOperator::completed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  session->request_stop();
  session->wait();

  // Key assertion: real callback work completes on the non-null partition stream and the
  // RunSession quiesces that stream without a runtime diagnostic.
  EXPECT_TRUE(CudaGreenContextCallbackProbeOperator::completed.load(std::memory_order_acquire));
  EXPECT_NE(CudaGreenContextCallbackProbeOperator::observed_stream.load(std::memory_order_acquire),
            nullptr);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
// REQ: RUNTIME-045
// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, GreenContextFailureBoundariesRollbackAndReleaseDeviceClaim) {
  // Key setup: inject one failure at every claim-through-publication materialization boundary for
  // two isolated Green Context probe partitions.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  const ProcessGroupId process_group = green_process_group(*state);
  const std::array failures{
      std::pair{RuntimeCudaExecutionFailurePoint::kClaim, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kProbe, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kSplit, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kContext, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kContext, 1U},
      std::pair{RuntimeCudaExecutionFailurePoint::kStream, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kStream, 1U},
      std::pair{RuntimeCudaExecutionFailurePoint::kIdentity, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kIdentity, 1U},
      std::pair{RuntimeCudaExecutionFailurePoint::kBinding, 0U},
      std::pair{RuntimeCudaExecutionFailurePoint::kPublish, 0U},
  };

  CudaGreenContextMaterializationProbeOperator::start_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::compute_count.store(0U);
  // Key assertion: every failure rolls back exactly once, clears operator bindings, releases the
  // device claim for a successful retry, and occurs before operator callbacks.
  RunGenerationId run_generation{100U};
  for (const auto [point, occurrence] : failures) {
    SCOPED_TRACE(static_cast<std::uint32_t>(point));
    auto operators = reconstruct_operators(*state);
    ASSERT_TRUE(operators) << runtime_error_message(operators.error());
    const auto control = std::make_shared<RuntimeCudaExecutionTestControl>(point, occurrence);
    auto resources = RuntimeCudaExecutionResources::create(
        *state, *operators, run_generation, process_group, {}, control);
    ASSERT_FALSE(resources);
    EXPECT_TRUE(resources.error().message.starts_with("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
    EXPECT_GT(control->visits(), occurrence);
    EXPECT_TRUE(std::ranges::all_of(*operators, [](const auto& instance) {
      return instance.compute_stream == nullptr && !instance.compute_device.valid();
    }));
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));

    const std::size_t contexts_before_retry = driver_state.created_contexts.size();
    const std::size_t streams_before_retry = driver_state.created_streams.size();
    auto retry_operators = reconstruct_operators(*state);
    ASSERT_TRUE(retry_operators) << runtime_error_message(retry_operators.error());
    auto retry = RuntimeCudaExecutionResources::create(
        *state, *retry_operators, RunGenerationId{run_generation + 1U}, process_group);
    ASSERT_TRUE(retry) << runtime_error_message(retry.error());
    EXPECT_TRUE(std::ranges::all_of(*retry_operators, [](const auto& instance) {
      return instance.compute_stream != nullptr && instance.compute_device == DeviceId{0};
    }));
    ASSERT_GT(driver_state.created_contexts.size(), contexts_before_retry);
    ASSERT_GT(driver_state.created_streams.size(), streams_before_retry);
    const auto prior_contexts_end = driver_state.created_contexts.begin() + contexts_before_retry;
    EXPECT_TRUE(std::all_of(
        prior_contexts_end, driver_state.created_contexts.end(), [&](CUgreenCtx context) {
          return std::find(driver_state.created_contexts.begin(), prior_contexts_end, context) ==
                 prior_contexts_end;
        }));
    const auto prior_streams_end = driver_state.created_streams.begin() + streams_before_retry;
    EXPECT_TRUE(
        std::all_of(prior_streams_end, driver_state.created_streams.end(), [&](CUstream stream) {
          return std::find(driver_state.created_streams.begin(), prior_streams_end, stream) ==
                 prior_streams_end;
        }));
    retry->reset();
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));
    run_generation += 2U;
  }
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::start_count.load(), 0U);
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::compute_count.load(), 0U);
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, RunSessionPreservesDetailedCudaMaterializationDiagnosticToken) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: inject a context-boundary failure whose internal Error contains
  // "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED: <detail>".
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan baseline = fake_green_context_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto control =
      std::make_shared<RuntimeCudaExecutionTestControl>(RuntimeCudaExecutionFailurePoint::kContext);
  const ExecutionPlan controlled =
      RuntimeTestPlanAccess::with_cuda_execution_test_control(baseline, control);
  const auto diagnostic_sink = std::make_shared<CudaExecutionDiagnosticSink>();
  const ExecutionPlan plan =
      RuntimeTestPlanAccess::with_diagnostic_sink(controlled, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  CudaGreenContextMaterializationProbeOperator::start_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::compute_count.store(0U);

  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: the session publishes the registered CUDA token, not the generic profile
  // fallback, and fails before any Operator callback can execute.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_FALSE(test::has_runtime_diagnostic(session, "PROFILE_MATERIALIZATION_FAILED"));
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::start_count.load(), 0U);
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::compute_count.load(), 0U);
  const auto details = diagnostic_sink->details();
  ASSERT_EQ(details.size(), 1U);
  EXPECT_TRUE(details.front().present);
  EXPECT_EQ(details.front().cuda_execution.plan_generation,
            plan.cuda_green_context_pools().front().plan_generation);
  EXPECT_NE(details.front().cuda_execution.run_generation, 0U);
  EXPECT_EQ(details.front().cuda_execution.lifecycle_stage, LifecycleStage::kArm);
  EXPECT_EQ(details.front().cuda_execution.operation, "green_context_create");
  EXPECT_FALSE(details.front().cuda_execution.deployment_device_id.empty());
  EXPECT_FALSE(details.front().cuda_execution.pool_id.empty());
  EXPECT_FALSE(details.front().cuda_execution.green_context_id.empty());
  EXPECT_TRUE(details.front().cuda_execution.stream_id.empty());
}

// REQ: RUNTIME-056
TEST(RuntimeMaterializationTest, StreamBindingMismatchRetainsExpectedAndObservedContext) {
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());
  ASSERT_FALSE(operators->empty());
  ASSERT_FALSE(state->operator_cuda_stream_bindings.empty());
  const auto& binding = state->operator_cuda_stream_bindings.front();
  auto instance =
      std::ranges::find(*operators, binding.operator_vertex, &RuntimeOperatorInstance::vertex);
  ASSERT_NE(instance, operators->end());
  instance->compute_device = DeviceId{7};
  RuntimeCudaExecutionDiagnostic cuda_diagnostic;

  auto resources = RuntimeCudaExecutionResources::create(*state,
                                                         *operators,
                                                         RunGenerationId{181U},
                                                         green_process_group(*state),
                                                         {},
                                                         {},
                                                         &cuda_diagnostic);

  ASSERT_FALSE(resources);
  EXPECT_EQ(resources.error().message, "CUDA_STREAM_BINDING_STREAM_MISMATCH");
  EXPECT_EQ(cuda_diagnostic.operation, "operator_stream_binding");
  EXPECT_EQ(cuda_diagnostic.device, binding.device);
  EXPECT_EQ(cuda_diagnostic.stream_id, binding.stream_id);
  EXPECT_EQ(cuda_diagnostic.mismatch.field, "operator_cuda_stream_binding");
  EXPECT_NE(cuda_diagnostic.mismatch.provided.find("reason=operator device differs"),
            std::string::npos);
  EXPECT_NE(cuda_diagnostic.mismatch.required.find("stream=" + binding.stream_id),
            std::string::npos);

  RuntimeDiagnosticBuffer buffer{DiagnosticId{999U}, DiagnosticSource{}};
  const RuntimeDiagnosticDetail runtime_detail{
      .present = true,
      .error = resources.error(),
      .cuda_execution = cuda_diagnostic,
      .operation = DiagnosticOperation::kRuntime,
  };
  buffer.append(DiagnosticId{998U},
                "CUDA_STREAM_BINDING_STREAM_MISMATCH",
                DiagnosticSource{.graph_generation = state->graph.generation(),
                                 .vertex = binding.operator_vertex},
                {},
                &runtime_detail,
                {},
                {},
                ::holoscan::DiagnosticSeverity::kError,
                state->graph);

  ASSERT_EQ(buffer.diagnostics().size(), 1U);
  const auto& retained = buffer.diagnostics().front();
  ASSERT_TRUE(retained.detail);
  ASSERT_EQ(retained.detail->mismatches.size(), 1U);
  EXPECT_EQ(retained.detail->mismatches.front().field, "operator_cuda_stream_binding");
  const std::string formatted = format_diagnostic(retained, plan.graph());
  EXPECT_NE(formatted.find("operator_cuda_stream_binding"), std::string::npos);
  EXPECT_NE(formatted.find("operator device differs"), std::string::npos);
  EXPECT_NE(formatted.find(binding.stream_id), std::string::npos);
}

// REQ: RUNTIME-055
TEST(RuntimeMaterializationTest, TensorStartupAuthenticatesExactOperatorStreamId) {
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{182U}, green_process_group(*state));
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());

  auto forged = *state;
  ASSERT_GE(forged.operator_cuda_stream_bindings.size(), 2U);
  forged.operator_cuda_stream_bindings.front().stream_id =
      forged.operator_cuda_stream_bindings.back().stream_id;

  // Key assertion: matching process/device/partition fields cannot hide a forged retained stream
  // identity during the Tensor startup cross-check.
  const auto bound = bind_tensor_operator_streams(forged, *operators, **resources);
  ASSERT_FALSE(bound);
  EXPECT_NE(bound.error().message.find("operator CUDA stream binding mismatch"), std::string::npos);
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, ConcurrentRunSessionsRejectSecondPhysicalDeviceOwner) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: hold the first RunSession in compute after it owns the split, then start a second
  // session from the same immutable plan and process-local physical-device identity.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  CudaGreenContextMaterializationProbeOperator::start_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::compute_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::release_compute.store(false);
  CudaGreenContextMaterializationProbeOperator::block_compute.store(true);

  RunSession first = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (CudaGreenContextMaterializationProbeOperator::compute_count.load() == 0U &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  if (CudaGreenContextMaterializationProbeOperator::compute_count.load() == 0U) {
    CudaGreenContextMaterializationProbeOperator::release_compute.store(true);
    first.request_stop();
    first.wait();
    FAIL() << "first Green Context session did not reach compute";
  }

  RunSession second = run_async(plan);
  second.wait();
  CudaGreenContextMaterializationProbeOperator::release_compute.store(true);
  first.request_stop();
  first.wait();
  CudaGreenContextMaterializationProbeOperator::block_compute.store(false);

  // Key assertion: only the owning session reaches callbacks; the contender fails before arm with
  // the stable active-owner diagnostic and the first session completes ordered quiescence.
  EXPECT_EQ(second.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(second, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT"));
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::start_count.load(), 2U);
  EXPECT_EQ(first.termination(), RunTermination::kStopped);
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, GreenContextOwnersRejectEachOtherInEitherStartupOrder) {
  test::skip_without_cuda_device();
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);

  // Competing device ownership orders:
  //   two-partition Green Context plan --> one-partition Green Context plan
  //   one-partition Green Context plan --> two-partition Green Context plan
  // Key setup: materialize two distinct plans against the same fake physical device in both
  // startup orders while the first plan retains its exclusive process/device claim.
  const ExecutionPlan two_partition_plan = fake_green_context_plan();
  const ExecutionPlan one_partition_plan = fake_green_context_plan(1U, true);
  ASSERT_TRUE(two_partition_plan.ok()) << two_partition_plan.json();
  ASSERT_TRUE(one_partition_plan.ok()) << one_partition_plan.json();
  const std::array<const ExecutionPlan*, 2U> plans{&two_partition_plan, &one_partition_plan};

  for (std::size_t owner_index = 0U; owner_index < plans.size(); ++owner_index) {
    const std::size_t contender_index = 1U - owner_index;
    const auto owner_state = ExecutionPlanAccess::retain(*plans[owner_index]);
    ASSERT_TRUE(owner_state);
    auto owner_operators = reconstruct_operators(*owner_state);
    ASSERT_TRUE(owner_operators) << runtime_error_message(owner_operators.error());
    auto owner = RuntimeCudaExecutionResources::create(
        *owner_state,
        *owner_operators,
        RunGenerationId{static_cast<std::uint64_t>(171U + owner_index * 2U)},
        green_process_group(*owner_state));
    ASSERT_TRUE(owner) << runtime_error_message(owner.error());
    const std::size_t context_count = driver_state.created_contexts.size();
    const std::size_t stream_count = driver_state.created_streams.size();

    const auto contender_state = ExecutionPlanAccess::retain(*plans[contender_index]);
    ASSERT_TRUE(contender_state);
    auto contender_operators = reconstruct_operators(*contender_state);
    ASSERT_TRUE(contender_operators) << runtime_error_message(contender_operators.error());
    auto contender = RuntimeCudaExecutionResources::create(
        *contender_state,
        *contender_operators,
        RunGenerationId{static_cast<std::uint64_t>(172U + owner_index * 2U)},
        green_process_group(*contender_state));

    // Key assertion: either Green Context plan rejects the other before native context or stream
    // creation, and releasing the owner admits the reverse order in the next iteration.
    ASSERT_FALSE(contender);
    EXPECT_EQ(contender.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
    EXPECT_EQ(driver_state.created_contexts.size(), context_count);
    EXPECT_EQ(driver_state.created_streams.size(), stream_count);
    owner->reset();
  }
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, OrdinaryCudaOwnerRejectsGreenContextContender) {
  test::skip_without_cuda_device();
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);

  // Device ownership order:
  //   ordinary CUDA owner(device 0) --> Green Context contender(device 0)
  // Key setup: keep one ordinary process/device stream owner alive while materializing a Green
  // Context plan against the same process-local device ordinal and fake physical identity.
  const ExecutionPlanState ordinary_plan = ordinary_cuda_execution_plan();
  std::array<RuntimeOperatorInstance, 0U> ordinary_operators;
  auto ordinary = RuntimeCudaExecutionResources::create(
      ordinary_plan, ordinary_operators, RunGenerationId{161U}, ProcessGroupId{1U});
  ASSERT_TRUE(ordinary) << runtime_error_message(ordinary.error());

  const ExecutionPlan green_plan = fake_green_context_plan();
  ASSERT_TRUE(green_plan.ok()) << green_plan.json();
  const auto green_state = ExecutionPlanAccess::retain(green_plan);
  ASSERT_TRUE(green_state);
  auto green_operators = reconstruct_operators(*green_state);
  ASSERT_TRUE(green_operators) << runtime_error_message(green_operators.error());
  auto contender = RuntimeCudaExecutionResources::create(
      *green_state, *green_operators, RunGenerationId{162U}, green_process_group(*green_state));

  // Key assertion: an ordinary CUDA lease prevents an exclusive Green Context claim before any
  // Green Context or Green Context stream is created.
  ASSERT_FALSE(contender);
  EXPECT_EQ(contender.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  EXPECT_TRUE(driver_state.created_contexts.empty());
  EXPECT_TRUE(driver_state.created_streams.empty());
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, GreenContextOwnerRejectsOrdinaryCudaContender) {
  test::skip_without_cuda_device();
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);

  // Device ownership order:
  //   Green Context owner(device 0) --> ordinary CUDA contender(device 0)
  // Key setup: keep one exclusive Green Context owner alive, then materialize an ordinary
  // process/device stream plan for the same process-local CUDA device.
  const ExecutionPlan green_plan = fake_green_context_plan();
  ASSERT_TRUE(green_plan.ok()) << green_plan.json();
  const auto green_state = ExecutionPlanAccess::retain(green_plan);
  ASSERT_TRUE(green_state);
  auto green_operators = reconstruct_operators(*green_state);
  ASSERT_TRUE(green_operators) << runtime_error_message(green_operators.error());
  auto green = RuntimeCudaExecutionResources::create(
      *green_state, *green_operators, RunGenerationId{163U}, green_process_group(*green_state));
  ASSERT_TRUE(green) << runtime_error_message(green.error());

  const ExecutionPlanState ordinary_plan = ordinary_cuda_execution_plan();
  std::array<RuntimeOperatorInstance, 0U> ordinary_operators;
  RuntimeCudaExecutionDiagnostic diagnostic;
  auto contender = RuntimeCudaExecutionResources::create(ordinary_plan,
                                                         ordinary_operators,
                                                         RunGenerationId{164U},
                                                         ProcessGroupId{1U},
                                                         {},
                                                         {},
                                                         &diagnostic);

  // Key assertion: the exclusive Green Context claim rejects ordinary primary-context stream
  // materialization before the ordinary stream can be created.
  ASSERT_FALSE(contender);
  EXPECT_EQ(contender.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  EXPECT_EQ(diagnostic.operation, "ordinary_device_claim");
  EXPECT_EQ(diagnostic.device, DeviceId{0});
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, TerminalQuiescePrecedesOperatorDestruction) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: observe reconstructed Operator lifetime from the fake driver's terminal stream
  // synchronization boundary.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  CudaGreenContextMaterializationProbeOperator::start_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::compute_count.store(0U);
  CudaGreenContextMaterializationProbeOperator::destruction_count.store(0U);
  std::atomic_bool owners_alive_during_quiesce{true};
  driver_state.stream_synchronize_hook = [&]() {
    if (CudaGreenContextMaterializationProbeOperator::destruction_count.load(
            std::memory_order_acquire) != 0U) {
      owners_alive_during_quiesce.store(false, std::memory_order_release);
    }
  };

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (CudaGreenContextMaterializationProbeOperator::compute_count.load(
             std::memory_order_acquire) == 0U &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  session.request_stop();
  session.wait();

  // Key assertion: every stream synchronizes while both reconstructed Operators remain alive;
  // only after that barrier does terminal cleanup destroy the graph owner closure.
  EXPECT_TRUE(owners_alive_during_quiesce.load(std::memory_order_acquire));
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_EQ(CudaGreenContextMaterializationProbeOperator::destruction_count.load(), 2U);
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, ChannelMappingsOutliveGreenContextQuiesce) {
  // Graph topology (both endpoints are Green Context partitions):
  //   source --> sink
  // Key setup: retain a Channel teardown observer in the plan and inspect it from every fake
  // driver stream-synchronization boundary.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const auto channel_control = std::make_shared<RuntimeChannelTeardownTestControl>();
  const ExecutionPlan plan = RuntimeTestPlanAccess::with_channel_teardown_test_control(
      fake_green_context_channel_plan(), channel_control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  CudaGreenContextChannelSourceOperator::compute_count.store(0U);
  std::atomic_bool channel_destroyed_during_quiesce{false};
  driver_state.stream_synchronize_hook = [&]() {
    if (channel_control->runtime_destroyed()) {
      channel_destroyed_during_quiesce.store(true, std::memory_order_release);
    }
  };

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (CudaGreenContextChannelSourceOperator::compute_count.load(std::memory_order_acquire) ==
             0U &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  session.request_stop();
  session.wait();

  // Key assertion: the Channel session and its imported-mapping owners remain alive through every
  // CUDA stream barrier, then are destroyed before terminal completion is published.
  EXPECT_FALSE(channel_destroyed_during_quiesce.load(std::memory_order_acquire));
  EXPECT_TRUE(channel_control->runtime_destroyed());
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
}

void expect_observed_channel_quiesce_lifetime(bool quiesce_fails) {
  auto driver_state = cuda_test::make_state();
  // A quarantined device claim intentionally survives the test. Keep its fake UUID distinct from
  // all other fake-driver tests and from the healthy twin.
  driver_state.uuid_offset = quiesce_fails ? 0x71U : 0x72U;
  driver_state.stream_synchronize_result = quiesce_fails ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
  const cuda_test::FakeDriverScope driver_scope(driver_state);
  const auto channel_control = std::make_shared<RuntimeChannelTeardownTestControl>();
  const auto diagnostic_sink = std::make_shared<CudaExecutionDiagnosticSink>();
  const ExecutionPlan plan = RuntimeTestPlanAccess::with_diagnostic_sink(
      RuntimeTestPlanAccess::with_channel_teardown_test_control(
          fake_observed_green_context_channel_plan(), channel_control),
      diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto retained = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(retained);
  ASSERT_EQ(retained->execution_partitions.size(), 1U);
  ASSERT_TRUE(retained->closed_island_activations.empty());

  auto observation = std::make_shared<RuntimeFixtureObservation>(32U, 1U, 4U);
  const std::weak_ptr<RuntimeFixtureObservation> weak_observation = observation;
  std::atomic_bool owners_alive_during_quiesce{true};
  driver_state.stream_synchronize_hook = [&]() {
    if (channel_control->runtime_destroyed() || weak_observation.expired()) {
      owners_alive_during_quiesce.store(false, std::memory_order_release);
    }
  };

  {
    RunSession session = start_operator_test_run(plan, observation);
    observation.reset();
    // Bound the natural-completion wait so a scheduling regression becomes a failed assertion.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    const bool completed_without_stop = !session.is_running();
    if (!completed_without_stop) {
      session.request_stop();
    }
    session.wait();
    ASSERT_TRUE(completed_without_stop);
    EXPECT_EQ(session.termination(),
              quiesce_fails ? RunTermination::kFailed : RunTermination::kCompleted);

    // Prove the observed binding actually published and retired before testing its owner closure.
    // This temporary strong reference is destroyed before the RunSession wrapper below.
    const auto completed_observation = weak_observation.lock();
    ASSERT_TRUE(completed_observation);
    EXPECT_FALSE(completed_observation->invalid());
    std::size_t retired_publications = 0U;
    std::size_t retired_inputs = 0U;
    for (const auto& record : completed_observation->records()) {
      if (record.kind != FixtureObservationKind::kCandidateFinished || !record.admitted ||
          record.outcome != FixtureCandidateOutcome::kRetired) {
        continue;
      }
      for (const auto& publication : record.publications) {
        EXPECT_EQ(publication.error, ErrorCode::kSuccess);
        EXPECT_TRUE(publication.sequence_consumed);
        ++retired_publications;
      }
      for (const auto& input : record.inputs) {
        retired_inputs += input.has_sample ? 1U : 0U;
      }
    }
    EXPECT_EQ(retired_publications, 1U);
    EXPECT_EQ(retired_inputs, 1U);
    if (quiesce_fails) {
      EXPECT_TRUE(
          test::has_runtime_diagnostic(session, "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
      const auto details = diagnostic_sink->details();
      const auto cleanup = std::find_if(details.begin(), details.end(), [](const auto& row) {
        return row.cuda_execution.operation == "stream_quiesce";
      });
      ASSERT_NE(cleanup, details.end());
      EXPECT_EQ(cleanup->cuda_execution.driver_result,
                static_cast<std::int32_t>(CUDA_ERROR_UNKNOWN));
      EXPECT_EQ(cleanup->cuda_execution.lifecycle_stage, LifecycleStage::kRelease);
    } else {
      EXPECT_TRUE(session.diagnostics().empty());
    }
  }

  ASSERT_FALSE(driver_state.created_streams.empty());
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_TRUE(owners_alive_during_quiesce.load(std::memory_order_acquire));
  if (quiesce_fails) {
    EXPECT_FALSE(channel_control->runtime_destroyed());
    EXPECT_TRUE(driver_state.destroyed_streams.empty());
    EXPECT_TRUE(driver_state.destroyed_contexts.empty());
    // The RunSession and every caller-held strong observer are gone. Only the quarantined channel
    // closure can now retain the observer borrowed by its OutputState thunks. This assertion is
    // safely red for the former raw-only binding without calling through its dangling pointer.
    EXPECT_FALSE(weak_observation.expired());
  } else {
    EXPECT_TRUE(channel_control->runtime_destroyed());
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
    EXPECT_TRUE(
        destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));
    EXPECT_TRUE(weak_observation.expired());
  }
}

// REQ: LIFECYCLE-010
// REQ: RUNTIME-062
TEST(RuntimeMaterializationTest, FailedQuiesceRetainsObservationWithQuarantinedChannel) {
  expect_observed_channel_quiesce_lifetime(true);
}

// REQ: LIFECYCLE-010
// REQ: RUNTIME-062
TEST(RuntimeMaterializationTest, HealthyQuiesceReleasesChannelAndObservationAfterRunSession) {
  expect_observed_channel_quiesce_lifetime(false);
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, TeardownDiagnosticDoesNotReplacePrimaryCallbackFailure) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: fail an Operator callback and then fail its stream-quiesce barrier for the same
  // RunSession while capturing ordered public and private diagnostic evidence.
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x31U;
  driver_state.stream_synchronize_result = CUDA_ERROR_UNKNOWN;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan baseline = fake_green_context_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<CudaExecutionDiagnosticSink>();
  const ExecutionPlan plan = RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  CudaGreenContextMaterializationProbeOperator::fail_compute.store(true);

  RunSession session = run_async(plan);
  session.wait();
  CudaGreenContextMaterializationProbeOperator::fail_compute.store(false);

  // Key assertion: the callback failure remains first, cleanup adds a separate CUDA diagnostic,
  // and private evidence identifies the stable device/pool, run generations, driver result, and
  // cold lifecycle stage without retaining a native handle.
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  const auto diagnostics = session.diagnostics();
  ASSERT_GE(diagnostics.size(), 2U);
  EXPECT_EQ(diagnostics.front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  const auto cleanup = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
    return row.token == "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED";
  });
  ASSERT_NE(cleanup, diagnostics.end());
  EXPECT_GT(std::distance(diagnostics.begin(), cleanup), 0);
  const auto& cleanup_evidence = cleanup->detail;
  if (!cleanup_evidence.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(cleanup_evidence->role, DiagnosticFailureRole::kTeardown);
  EXPECT_EQ(cleanup_evidence->operation, DiagnosticOperation::kRuntime);
  EXPECT_EQ(cleanup_evidence->primary_diagnostic_index, 0U);
  const auto details = diagnostic_sink->details();
  const auto cleanup_detail = std::find_if(details.begin(), details.end(), [](const auto& row) {
    return row.cuda_execution.operation == "stream_quiesce";
  });
  ASSERT_NE(cleanup_detail, details.end());
  EXPECT_TRUE(cleanup_detail->present);
  EXPECT_NE(cleanup_detail->cuda_execution.plan_generation, 0U);
  EXPECT_NE(cleanup_detail->cuda_execution.run_generation, 0U);
  EXPECT_EQ(cleanup_detail->cuda_execution.lifecycle_stage, LifecycleStage::kRelease);
  EXPECT_FALSE(cleanup_detail->cuda_execution.deployment_device_id.empty());
  EXPECT_FALSE(cleanup_detail->cuda_execution.pool_id.empty());
  EXPECT_FALSE(cleanup_detail->cuda_execution.green_context_id.empty());
  EXPECT_FALSE(cleanup_detail->cuda_execution.stream_id.empty());
  EXPECT_EQ(cleanup_detail->cuda_execution.driver_result,
            static_cast<std::int32_t>(CUDA_ERROR_UNKNOWN));
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-046
TEST(RuntimeMaterializationTest, TeardownFailureOverridesRequestedCompletion) {
  // Graph topology:
  //   completion-probe   (one finite activation in one full-device Green Context)
  // Key setup: request whole-run success from the only callback, then inject a stream-quiesce
  // failure before the worker publishes its terminal outcome.
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x41U;
  driver_state.stream_synchronize_result = CUDA_ERROR_UNKNOWN;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan baseline = fake_finite_green_context_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan = RuntimeTestPlanAccess::enable_direct_once(baseline);
  CudaGreenContextCompletionProbeOperator::compute_count.store(0U);
  CudaGreenContextCompletionProbeOperator::consumed_termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kNone);
  CudaGreenContextCompletionProbeOperator::termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kCompletion);

  RunSession session = run_async(plan);
  session.wait();
  CudaGreenContextCompletionProbeOperator::termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kNone);

  // Key assertion: cleanup failure overrides requested success while the separate diagnostic
  // preserves the failed CUDA synchronization and no dependent native owner is destroyed.
  EXPECT_EQ(CudaGreenContextCompletionProbeOperator::compute_count.load(), 1U);
  EXPECT_EQ(CudaGreenContextCompletionProbeOperator::consumed_termination_request.load(),
            CudaGreenContextCompletionProbeOperator::TerminationRequest::kCompletion);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_TRUE(driver_state.destroyed_streams.empty());
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-046
TEST(RuntimeMaterializationTest, TeardownFailureOverridesCallbackStop) {
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x42U;
  driver_state.stream_destroy_result = CUDA_ERROR_UNKNOWN;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan baseline = fake_finite_green_context_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan = RuntimeTestPlanAccess::enable_direct_once(baseline);
  CudaGreenContextCompletionProbeOperator::compute_count.store(0U);
  CudaGreenContextCompletionProbeOperator::consumed_termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kNone);
  CudaGreenContextCompletionProbeOperator::termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kStop);

  RunSession session = run_async(plan);
  session.wait();
  CudaGreenContextCompletionProbeOperator::termination_request.store(
      CudaGreenContextCompletionProbeOperator::TerminationRequest::kNone);

  EXPECT_EQ(CudaGreenContextCompletionProbeOperator::compute_count.load(), 1U);
  EXPECT_EQ(CudaGreenContextCompletionProbeOperator::consumed_termination_request.load(),
            CudaGreenContextCompletionProbeOperator::TerminationRequest::kStop);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_TRUE(destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, GreenContextReprobeRejectsPhysicalIdentityAndCapabilityDrift) {
  // Key setup: mutate first the physical UUID and then the SM count after compiling two probes.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());

  ++driver_state.uuid_offset;
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{200U}, green_process_group(*state));
  // Key assertion: either identity or capacity drift fails reprobe before native context creation.
  ASSERT_FALSE(resources);
  EXPECT_TRUE(resources.error().message.starts_with("CUDA_GREEN_CONTEXT_CAPABILITY_DRIFT"));
  EXPECT_TRUE(driver_state.created_contexts.empty());

  --driver_state.uuid_offset;
  ++driver_state.device_resource.sm.smCount;
  resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{201U}, green_process_group(*state));
  ASSERT_FALSE(resources);
  EXPECT_TRUE(resources.error().message.starts_with("CUDA_GREEN_CONTEXT_CAPABILITY_DRIFT"));
  EXPECT_TRUE(driver_state.created_contexts.empty());
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, GreenContextNativeFailuresPreserveDetailsAndRollback) {
  // Key setup: inject native context creation and stream creation failures in separate attempts.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  const ProcessGroupId process_group = green_process_group(*state);

  driver_state.fail_context_create_index = 1U;
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{210U}, process_group);
  // Key assertion: both failures retain CUDA API diagnostics and destroy every created handle.
  ASSERT_FALSE(resources);
  EXPECT_TRUE(resources.error().message.starts_with("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_NE(resources.error().message.find("FAKE_CUDA_ERROR"), std::string::npos);
  EXPECT_TRUE(
      destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));

  driver_state.fail_context_create_index.reset();
  driver_state.fail_stream_create_index = driver_state.stream_create_calls;
  operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{211U}, process_group);
  ASSERT_FALSE(resources);
  EXPECT_NE(resources.error().message.find("cuGreenCtxStreamCreate"), std::string::npos);
  EXPECT_NE(resources.error().message.find("FAKE_CUDA_ERROR"), std::string::npos);
  EXPECT_TRUE(destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
  EXPECT_TRUE(
      destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, PhysicalIdentityRegistryAliasesOrdinalsButSeparatesDevices) {
  // Graph topology for each plan (each probe is isolated):
  //   probe-0   probe-1
  // Key setup: compile device-0 and device-1 plans first with aliased UUID/PCI evidence, then with
  // distinct physical identities, and materialize them under distinct monotonic claim owners.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const auto materialize = [](const ExecutionPlan& plan, RunGenerationId generation)
      -> expected<std::unique_ptr<RuntimeCudaExecutionResources>, Error> {
    const auto state = ExecutionPlanAccess::retain(plan);
    if (state == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "missing retained plan"});
    }
    auto operators = reconstruct_operators(*state);
    if (!operators) {
      return make_unexpected(std::move(operators.error()));
    }
    return RuntimeCudaExecutionResources::create(
        *state, *operators, generation, green_process_group(*state));
  };

  driver_state.alias_device_ordinals = true;
  const ExecutionPlan aliased_zero = fake_green_context_plan(2U, false, DeviceId{0});
  const ExecutionPlan aliased_one = fake_green_context_plan(2U, false, DeviceId{1});
  ASSERT_TRUE(aliased_zero.ok()) << aliased_zero.json();
  ASSERT_TRUE(aliased_one.ok()) << aliased_one.json();
  auto first = materialize(aliased_zero, RunGenerationId{212U});
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto aliased = materialize(aliased_one, RunGenerationId{213U});
  ASSERT_FALSE(aliased);
  EXPECT_EQ(aliased.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  first->reset();

  driver_state.alias_device_ordinals = false;
  const ExecutionPlan distinct_zero = fake_green_context_plan(2U, false, DeviceId{0});
  const ExecutionPlan distinct_one = fake_green_context_plan(2U, false, DeviceId{1});
  ASSERT_TRUE(distinct_zero.ok()) << distinct_zero.json();
  ASSERT_TRUE(distinct_one.ok()) << distinct_one.json();
  first = materialize(distinct_zero, RunGenerationId{214U});
  auto second = materialize(distinct_one, RunGenerationId{215U});

  // Key assertion: aliases conflict on the stable physical identity while distinct UUID/PCI keys
  // own independent process-local claims and can coexist.
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(second) << runtime_error_message(second.error());
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, OneRunMaterializesIndependentPoolsOnTwoPhysicalDevices) {
  // Graph topology:
  //   probe-0(device 0)   probe-1(device 1)
  // Key setup: compile and materialize one process-local run with exact full-device partitions on
  // two distinct fake physical GPUs.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_two_device_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 2U);
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());
  ASSERT_EQ(operators->size(), 2U);
  const ProcessGroupId process_group = green_process_group(*state);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{218U}, process_group);
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());

  // Key assertion: device-qualified lookups resolve two distinct native streams and every
  // Operator receives the stream and device from its own assignment.
  std::array<cudaStream_t, 2U> streams{};
  for (const auto& assignment : plan.cuda_green_context_assignments()) {
    ASSERT_LT(static_cast<std::size_t>(assignment.device.value), streams.size());
    const cudaStream_t stream =
        (*resources)->partition_stream(process_group, assignment.device, assignment.partition);
    ASSERT_NE(stream, nullptr);
    streams[static_cast<std::size_t>(assignment.device.value)] = stream;
    const auto instance = std::find_if(operators->begin(), operators->end(), [&](const auto& row) {
      return row.partition == assignment.partition;
    });
    ASSERT_NE(instance, operators->end());
    EXPECT_EQ(instance->compute_device, assignment.device);
    EXPECT_EQ(instance->compute_stream, stream);
  }
  EXPECT_NE(streams[0], streams[1]);
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, QuiesceFailureQuarantinesCompleteOwnerClosure) {
  // Graph topology:
  //   probe-0(device 0)   probe-1(device 1)
  // Key setup: materialize both physical devices in one run, then fail synchronization only for
  // the stream assigned to device 1.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_two_device_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());
  const ProcessGroupId process_group = green_process_group(*state);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{230U}, process_group);
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());
  const auto device_one_assignment =
      std::find_if(plan.cuda_green_context_assignments().begin(),
                   plan.cuda_green_context_assignments().end(),
                   [](const auto& row) { return row.device == DeviceId{1}; });
  ASSERT_NE(device_one_assignment, plan.cuda_green_context_assignments().end());
  const cudaStream_t device_one_stream =
      (*resources)->partition_stream(process_group, DeviceId{1}, device_one_assignment->partition);
  ASSERT_NE(device_one_stream, nullptr);
  driver_state.fail_stream_synchronize = reinterpret_cast<CUstream>(device_one_stream);
  ASSERT_FALSE((*resources)->shutdown());

  // Key assertion: no native owner is destroyed and every claim in the retained two-device
  // closure rejects a successor, including the peer device whose streams synchronized.
  EXPECT_TRUE(driver_state.destroyed_streams.empty());
  EXPECT_TRUE(driver_state.destroyed_contexts.empty());
  const ExecutionPlan healthy_plan = fake_green_context_plan(1U, true, DeviceId{0});
  ASSERT_TRUE(healthy_plan.ok()) << healthy_plan.json();
  const auto healthy_state = ExecutionPlanAccess::retain(healthy_plan);
  ASSERT_TRUE(healthy_state);
  auto healthy_operators = reconstruct_operators(*healthy_state);
  ASSERT_TRUE(healthy_operators) << runtime_error_message(healthy_operators.error());
  auto healthy_retry = RuntimeCudaExecutionResources::create(*healthy_state,
                                                             *healthy_operators,
                                                             RunGenerationId{231U},
                                                             green_process_group(*healthy_state));
  ASSERT_FALSE(healthy_retry);
  EXPECT_EQ(healthy_retry.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");

  const ExecutionPlan failed_plan = fake_green_context_plan(1U, true, DeviceId{1});
  ASSERT_TRUE(failed_plan.ok()) << failed_plan.json();
  const auto failed_state = ExecutionPlanAccess::retain(failed_plan);
  ASSERT_TRUE(failed_state);
  auto failed_operators = reconstruct_operators(*failed_state);
  ASSERT_TRUE(failed_operators) << runtime_error_message(failed_operators.error());
  auto failed_retry = RuntimeCudaExecutionResources::create(
      *failed_state, *failed_operators, RunGenerationId{232U}, green_process_group(*failed_state));
  ASSERT_FALSE(failed_retry);
  EXPECT_EQ(failed_retry.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  EXPECT_TRUE(driver_state.destroyed_streams.empty());
  EXPECT_TRUE(driver_state.destroyed_contexts.empty());
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, QuiesceFailureQuarantinesDeviceAndDependents) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: preserve a pre-existing current context and inject stream synchronization failure
  // into a uniquely identified device's cold shutdown transaction.
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x51U;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  driver_state.current_context =
      reinterpret_cast<CUcontext>(cuda_test::kFakePrimaryContextHandleBase + 99U);
  const CUcontext prior_context = driver_state.current_context;
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{216U}, green_process_group(*state));
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());
  driver_state.stream_synchronize_result = CUDA_ERROR_UNKNOWN;
  auto shutdown = (*resources)->shutdown();

  // Key assertion: shutdown reports the first stream-quiesce error, leaves the caller's current
  // context and every dependent native owner untouched, and quarantines ownership so a successor
  // cannot overlap work whose completion CUDA did not prove.
  ASSERT_FALSE(shutdown);
  EXPECT_TRUE(shutdown.error().message.starts_with("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_EQ(driver_state.current_context, prior_context);
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_EQ(driver_state.synchronized_streams, driver_state.created_streams);
  EXPECT_TRUE(driver_state.destroyed_streams.empty());
  EXPECT_TRUE(driver_state.destroyed_contexts.empty());
  EXPECT_FALSE((*resources)->shutdown());
  resources->reset();
  EXPECT_TRUE(driver_state.destroyed_streams.empty());
  EXPECT_TRUE(driver_state.destroyed_contexts.empty());

  driver_state.stream_synchronize_result = CUDA_SUCCESS;
  auto retry_operators = reconstruct_operators(*state);
  ASSERT_TRUE(retry_operators);
  auto retry = RuntimeCudaExecutionResources::create(
      *state, *retry_operators, RunGenerationId{217U}, green_process_group(*state));
  ASSERT_FALSE(retry);
  EXPECT_EQ(retry.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
}

// REQ: LIFECYCLE-011
TEST(RuntimeMaterializationTest, ExternalCudaOwnerFailureQuarantinesDeviceClaim) {
  // A CUDA Graph owner is outside RuntimeCudaExecutionResources, but it shares the same device
  // claim. Simulate that owner's failure on a unique fake device and retain the owner closure.
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x71U;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan(1U, true, DeviceId{37});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  const ProcessGroupId process_group = green_process_group(*state);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{233U}, process_group);
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());

  (*resources)->quarantine_after_external_failure();
  auto retry_operators = reconstruct_operators(*state);
  ASSERT_TRUE(retry_operators);
  auto retry = RuntimeCudaExecutionResources::create(
      *state, *retry_operators, RunGenerationId{234U}, process_group);

  ASSERT_FALSE(retry);
  EXPECT_EQ(retry.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
  // Match the production quarantine path: uncertain owners intentionally survive until process
  // exit. Device 37 is unique to this fake-driver regression.
  // NOLINTNEXTLINE(bugprone-unused-return-value,clang-analyzer-cplusplus.NewDeleteLeaks)
  static_cast<void>(resources->release());
}

// REQ: LIFECYCLE-010
TEST(RuntimeMaterializationTest, NativeDestroyFailureQuarantinesDeviceClaim) {
  // Graph topology (each probe is an isolated Green Context partition):
  //   probe-0   probe-1
  // Key setup: successfully quiesce a uniquely identified device, then fail stream and context
  // destruction while attempting every reverse-order cleanup boundary.
  auto driver_state = cuda_test::make_state();
  driver_state.uuid_offset = 0x61U;
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{218U}, green_process_group(*state));
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());
  driver_state.stream_destroy_result = CUDA_ERROR_UNKNOWN;
  driver_state.context_destroy_result = CUDA_ERROR_UNKNOWN;
  const auto shutdown = (*resources)->shutdown();

  // Key assertion: destruction is attempted after proven quiescence, but any native failure keeps
  // the physical-device claim quarantined and rejects a successor generation.
  ASSERT_FALSE(shutdown);
  EXPECT_EQ(driver_state.stream_synchronize_calls, driver_state.created_streams.size());
  EXPECT_TRUE(destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
  EXPECT_TRUE(
      destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));
  driver_state.stream_destroy_result = CUDA_SUCCESS;
  driver_state.context_destroy_result = CUDA_SUCCESS;
  auto retry_operators = reconstruct_operators(*state);
  ASSERT_TRUE(retry_operators);
  auto retry = RuntimeCudaExecutionResources::create(
      *state, *retry_operators, RunGenerationId{219U}, green_process_group(*state));
  ASSERT_FALSE(retry);
  EXPECT_EQ(retry.error().message, "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT");
}

// REQ: RUNTIME-045
// REQ: CHAN-134
TEST(RuntimeMaterializationTest, GreenContextRejectsNullOrdinaryAndWrongContextStreams) {
  // Key setup: inject null, ordinary-context, wrong-Green-Context, and identity-query-failure
  // stream outcomes across four materialization attempts.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  const ProcessGroupId process_group = green_process_group(*state);
  RunGenerationId generation{220U};

  const auto run = [&]() {
    auto operators = reconstruct_operators(*state);
    EXPECT_TRUE(operators);
    return RuntimeCudaExecutionResources::create(*state, *operators, generation, process_group);
  };
  driver_state.return_null_stream = true;
  auto result = run();
  // Key assertion: every invalid identity fails closed and all partially created handles roll back.
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().message.starts_with("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  driver_state.return_null_stream = false;
  ++generation;

  driver_state.observe_ordinary_stream = true;
  result = run();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
  driver_state.observe_ordinary_stream = false;
  ++generation;

  driver_state.observed_context = cuda_test::unrecognized_green_context();
  result = run();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "CUDA_GREEN_CONTEXT_STREAM_MISMATCH");
  driver_state.observed_context.reset();
  ++generation;

  driver_state.identity_result = CUDA_ERROR_INVALID_HANDLE;
  result = run();
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().message.starts_with("CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"));
  EXPECT_NE(result.error().message.find("FAKE_CUDA_ERROR"), std::string::npos);
  EXPECT_TRUE(destroyed_exactly_once(driver_state.created_streams, driver_state.destroyed_streams));
  EXPECT_TRUE(
      destroyed_exactly_once(driver_state.created_contexts, driver_state.destroyed_contexts));
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, GreenContextRejectsCorruptPlanBeforeCuda) {
  // Key setup: apply seven malformed pool/resource/stream join mutations to a valid two-probe plan.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto base = mutable_plan_state(plan);
  ASSERT_TRUE(base);
  using Mutation = std::function<void(ExecutionPlanState&)>;
  const std::array<Mutation, 8U> mutations{
      [](auto& state) { ++state.cuda_green_context_pools.front().plan_generation; },
      [](auto& state) { ++state.cuda_green_context_pools.front().structured_sm_quantum; },
      [](auto& state) {
        state.cuda_green_context_pools.front().materialize_before = LifecycleStage::kStart;
      },
      [](auto& state) { state.cuda_green_context_resources.front().pool_id += ".broken"; },
      [](auto& state) { state.cuda_compute_streams.front().green_context_id = "missing"; },
      [](auto& state) {
        state.cuda_compute_streams.front().partition = ExecutionPartitionId{999U};
      },
      [](auto& state) {
        std::swap(state.cuda_green_context_resources[0], state.cuda_green_context_resources[1]);
      },
      [](auto& state) { state.cuda_compute_streams.push_back(state.cuda_compute_streams.front()); },
  };

  RunGenerationId generation{230U};
  // Key assertion: validation rejects every mutation before the probe boundary or native creation.
  for (const auto& mutate : mutations) {
    auto state = std::make_shared<ExecutionPlanState>(*base);
    mutate(*state);
    auto operators = reconstruct_operators(*state);
    ASSERT_TRUE(operators);
    const auto control =
        std::make_shared<RuntimeCudaExecutionTestControl>(RuntimeCudaExecutionFailurePoint::kProbe);
    auto resources = RuntimeCudaExecutionResources::create(
        *state, *operators, generation, green_process_group(*base), {}, control);
    ASSERT_FALSE(resources);
    EXPECT_TRUE(resources.error().message.starts_with("CUDA_GREEN_CONTEXT_PLAN_INVALID"));
    EXPECT_EQ(control->visits(), 0U);
    ++generation;
  }
  EXPECT_TRUE(driver_state.created_contexts.empty());
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, ResidualPartitionsResolveByCompleteStreamKey) {
  // Key setup: extend a three-probe plan so two unbound partitions share one residual context but
  // receive separately keyed streams.
  auto driver_state = cuda_test::make_state();
  cuda_test::FakeDriverScope driver_scope(driver_state);
  const ExecutionPlan plan = fake_green_context_plan(3U, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto state = mutable_plan_state(plan);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->cuda_green_context_resources.size(), 2U);
  auto residual =
      std::find_if(state->cuda_green_context_resources.begin(),
                   state->cuda_green_context_resources.end(),
                   [](const auto& resource) {
                     return resource.role == CudaGreenContextResourceRole::kReservedResidual;
                   });
  ASSERT_NE(residual, state->cuda_green_context_resources.end());
  residual->role = CudaGreenContextResourceRole::kSharedServiceResidual;
  const ProcessGroupId process_group = green_process_group(*state);
  for (const ExecutionPartitionId partition :
       {ExecutionPartitionId{2U}, ExecutionPartitionId{3U}}) {
    state->cuda_compute_streams.push_back(CudaComputeStreamPlanRecord{
        .plan_generation = state->plan_generation,
        .run_generation = 0U,
        .process_group = process_group,
        .device = DeviceId{0},
        .owner_id = "partition." + std::to_string(partition.value),
        .partition = partition,
        .stream_id = residual->green_context_id + ".partition." + std::to_string(partition.value) +
                     ".stream",
        .green_context_id = residual->green_context_id,
        .materialize_before = LifecycleStage::kArm,
    });
  }
  std::sort(state->cuda_compute_streams.begin(),
            state->cuda_compute_streams.end(),
            canonical_cuda_compute_stream_order);
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  auto resources = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{240U}, process_group);
  ASSERT_TRUE(resources) << runtime_error_message(resources.error());
  // Key assertion: complete process/device/partition keys resolve distinct streams to both probes.
  const cudaStream_t second =
      (*resources)->partition_stream(process_group, DeviceId{0}, ExecutionPartitionId{2U});
  const cudaStream_t third =
      (*resources)->partition_stream(process_group, DeviceId{0}, ExecutionPartitionId{3U});
  EXPECT_NE(second, nullptr);
  EXPECT_NE(third, nullptr);
  EXPECT_NE(second, third);
  // Stream ownership alone must not recreate the retired partition-membership binding heuristic.
  // These tensor-inactive probes have no exact binding rows and therefore remain unbound.
  EXPECT_EQ((*operators)[1].compute_stream, nullptr);
  EXPECT_EQ((*operators)[2].compute_stream, nullptr);
}

// REQ: RUNTIME-045
TEST(RuntimeMaterializationTest, NoBindingDoesNotLoadGreenContextDriverOrClaimDevice) {
  // Graph topology:
  //   probe   (isolated and unbound)
  // Key setup: compile one probe without Green Context records under an intentionally empty driver.
  auto driver_state = cuda_test::make_state();
  auto driver = std::make_shared<cuda_execution_detail::CudaGreenContextDriver>();
  cuda_test::FakeDriverScope driver_scope(driver_state, driver);
  Graph graph{"runtime-no-green-context"};
  graph.op<CudaGreenContextMaterializationProbeOperator>("probe");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->cuda_green_context_pools.empty());
  ASSERT_TRUE(state->cuda_compute_streams.empty());
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators);
  const ProcessGroupId process_group = state->vertex_process_placements.front().process_group;
  const auto probe =
      std::make_shared<RuntimeCudaExecutionTestControl>(RuntimeCudaExecutionFailurePoint::kProbe);
  auto first = RuntimeCudaExecutionResources::create(
      *state, *operators, RunGenerationId{250U}, process_group, {}, probe);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  // Key assertion: repeated materialization bypasses driver probing and creates no CUDA handles.
  EXPECT_EQ(probe->visits(), 0U);
  auto second_operators = reconstruct_operators(*state);
  ASSERT_TRUE(second_operators);
  auto second = RuntimeCudaExecutionResources::create(
      *state, *second_operators, RunGenerationId{251U}, process_group);
  EXPECT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_TRUE(driver_state.created_contexts.empty());
  EXPECT_TRUE(driver_state.created_streams.empty());
}
#endif

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CleansCompletedObligationAfterPartialStartup) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();

  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  state.cleanup_remaining();
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  state.cleanup_remaining();
  EXPECT_EQ(state.record(0U).cleanup_actions, 1U);
  EXPECT_TRUE(fixture.lanes->join());
}

// A direct-host lifecycle candidate has an immutable provider template but deliberately has no
// compile-time TensorAllocationPlanRecord. The materialization verifier must recognize that exact
// one-of form before the Core kAllocate barrier creates concrete backing.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RuntimeMaterializationTest, AcceptsDirectHostLifecycleCandidateTensorProviderTemplate) {
  const ExecutionPlan plan = lifecycle_candidate_tensor_materialization_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->provider_allocation_plans.size(), 1U);
  EXPECT_EQ(state->provider_allocation_plans.front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  EXPECT_TRUE(state->tensor_allocations.empty());

  auto created = RuntimeMaterializationState::create(*state, matching_package(), nullptr);
  ASSERT_TRUE(created) << created.error().message;
  auto materialization = std::move(created).value();
  const auto obligation = tensor_output_pool_obligation(materialization);
  ASSERT_TRUE(obligation.has_value());
  EXPECT_EQ(materialization.transition(
                *obligation, LifecycleStage::kAllocate, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
}

// The template alternative is deliberately narrow. Missing or duplicated identity, field-backed
// geometry, a rejected provider, or a stage mismatch must not pass as direct-host materialization.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RuntimeMaterializationTest, RejectsMalformedLifecycleCandidateTensorProviderTemplates) {
  const ExecutionPlan plan = lifecycle_candidate_tensor_materialization_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto baseline = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(baseline, nullptr);
  ASSERT_EQ(baseline->provider_allocation_plans.size(), 1U);

  const auto expect_rejected = [&](auto mutate) {
    ExecutionPlanState candidate{*baseline};
    mutate(candidate);
    auto created = RuntimeMaterializationState::create(candidate, matching_package(), nullptr);
    ASSERT_TRUE(created) << created.error().message;
    auto materialization = std::move(created).value();
    const auto obligation = tensor_output_pool_obligation(materialization);
    ASSERT_TRUE(obligation.has_value());
    EXPECT_EQ(
        materialization.transition(
            *obligation, LifecycleStage::kAllocate, LifecycleTransitionPhase::kStartupForward),
        LifecycleStatus::kFatalFailure);
  };

  expect_rejected([](ExecutionPlanState& candidate) {
    candidate.provider_allocation_plans.front().tensor_template.reset();
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    candidate.provider_allocation_plans.front().allocation_kind =
        ProviderAllocationKind::kTensorFieldOutput;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    candidate.provider_allocation_plans.front().disposition = PlanDisposition::kRejected;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    candidate.provider_allocation_plans.front().complete_before = LifecycleStage::kArm;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    auto& tensor_template = *candidate.provider_allocation_plans.front().tensor_template;
    tensor_template.memory_kind = MemoryKind::kPinnedHost;
    tensor_template.layout->memory_kind = MemoryKind::kPinnedHost;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    ASSERT_FALSE(candidate.process_local_channel_services.empty());
    candidate.process_local_channel_services.front().delivery_kind =
        ProcessLocalChannelDeliveryKind::kHoloMQ;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    ASSERT_FALSE(candidate.process_local_channel_services.empty());
    candidate.process_local_channel_services.front().cross_process_transport = true;
  });
  expect_rejected([](ExecutionPlanState& candidate) {
    auto duplicate = candidate.provider_allocation_plans.front();
    duplicate.id = ProviderAllocationPlanRecordId{duplicate.id.value + 1U};
    candidate.provider_allocation_plans.push_back(std::move(duplicate));
  });
}

// Lifecycle-selected Tensor geometry is authenticated before output construction mutates the
// reconstructed Operators' stream bindings.
//
// Graph topology:
//   source.output -- lifecycle-sized host Tensor --> sink.input
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RuntimeMaterializationTest, RejectsLifecycleTensorInventoryBeforeStreamBindingMutation) {
  const ExecutionPlan plan = lifecycle_candidate_tensor_materialization_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_FALSE(state->vertex_process_placements.empty());
  auto operators = reconstruct_operators(*state);
  ASSERT_TRUE(operators) << runtime_error_message(operators.error());
  const ProcessGroupId process_group = state->vertex_process_placements.front().process_group;
  auto cuda_resources =
      RuntimeCudaExecutionResources::create(*state, *operators, RunGenerationId{1U}, process_group);
  ASSERT_TRUE(cuda_resources) << runtime_error_message(cuda_resources.error());

  const auto sentinel_stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1U));
  for (auto& instance : *operators) {
    instance.compute_stream = sentinel_stream;
    instance.compute_device = DeviceId{7};
  }

  // Key setup: omit the one required lifecycle allocation from the materialization view.
  const auto tensor_resources = RuntimeTensorResources::create(*state,
                                                               *operators,
                                                               RunGenerationId{1U},
                                                               process_group,
                                                               **cuda_resources,
                                                               RuntimeRateMaterializationView{});

  // Key assertion: the candidate error wins before stream/device fields are touched.
  ASSERT_FALSE(tensor_resources);
  EXPECT_EQ(tensor_resources.error().message,
            "PROFILE_MATERIALIZATION_FAILED: lifecycle candidate inventory mismatch");
  for (const auto& instance : *operators) {
    EXPECT_EQ(instance.compute_stream, sentinel_stream);
    EXPECT_EQ(instance.compute_device, DeviceId{7});
  }
}

// REQ: COMPILE-048
// REQ: COMPILE-064
// REQ: MEM-STREAM-016
// REQ: OPERATOR-025
// REQ: RUNTIME-056
TEST(RuntimeMaterializationTest, ValidatesTensorDeviceStreamObligationAtArm) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  // Key setup: plan_generation=5 links the generalized compute stream to the obligation.
  fixture.plan.plan_generation = 5U;
  fixture.plan.cuda_compute_streams = {cuda_compute_stream()};
  fixture.plan.profile_manifest.materialization_obligations = {tensor_device_stream_obligation()};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);

  // Key assertion: stream_id mismatch between obligation and resource causes kFatalFailure.
  fixture.plan.cuda_compute_streams.front().stream_id = "tensor-stream.mismatch";
  created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kFatalFailure);
  EXPECT_TRUE(state.failed());
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, CleansMultipleCompletedObligationsInReverseCanonicalOrder) {
  StartedLane fixture;
  fixture.plan.scheduler_plan = serial_scheduler(2U);
  fixture.plan.execution_lanes = {default_lane(), explicit_lane(1U, ExecutionPartitionId{2U})};
  auto lanes = ExecutionLaneSet::create(fixture.plan.scheduler_plan, fixture.plan.execution_lanes);
  ASSERT_TRUE(lanes) << runtime_error_message(lanes.error());
  fixture.lanes = std::move(lanes).value();
  ASSERT_TRUE(fixture.lanes->start());

  auto first = lane_obligation();
  auto second = lane_obligation();
  second.object_id = "lane:1";
  fixture.plan.profile_manifest.materialization_obligations = {first, second};
  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();

  // Key setup: obligations for lane:0 and lane:1 both complete at kStart in order.
  ASSERT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  ASSERT_EQ(state.transition(1U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kOk);
  state.cleanup_remaining();

  // Key assertion: cleanup proceeds in reverse canonical order — lane:1 before lane:0.
  ASSERT_EQ(state.cleanup_order().size(), 2U);
  EXPECT_EQ(state.cleanup_order()[0], 1U);
  EXPECT_EQ(state.cleanup_order()[1], 0U);
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(state.record(1U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest,
     LifecycleGraphFailsInCanonicalOrderAndCleansCompletedBarriersInReverseExactlyOnce) {
  LifecycleMaterializationProbeOperator::start_count.store(0U, std::memory_order_relaxed);
  LifecycleMaterializationProbeOperator::stop_count.store(0U, std::memory_order_relaxed);
  const ExecutionPlan baseline = three_lane_materialization_plan();
  ASSERT_TRUE(baseline.ok());
  const auto accepted = ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  ASSERT_EQ(accepted->scheduler_plan.execution_lane_count, 3U);

  auto plan = std::make_shared<ExecutionPlanState>(*accepted);
  // Key setup: obligations 0 and 1 use the valid lane strategy; obligation 2 uses an
  //   invalid strategy (sorted before the valid lane:2 entry); obligation 3 is unreachable.
  auto lane_zero = lane_obligation();
  auto lane_one = lane_obligation();
  lane_one.object_id = "lane:1";
  auto failing = lane_obligation();
  failing.object_id = "lane:2";
  failing.strategy_id = "aaa.invalid-before-valid-lane-strategy";
  auto never_reached = lane_obligation();
  never_reached.object_id = "lane:2";
  plan->profile_manifest.materialization_obligations = {
      lane_zero,
      lane_one,
      failing,
      never_reached,
  };

  auto reconstructed = reconstruct_operators(*plan);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  auto operators = std::move(reconstructed).value();
  auto lanes = ExecutionLaneSet::create(plan->scheduler_plan, plan->execution_lanes);
  ASSERT_TRUE(lanes) << runtime_error_message(lanes.error());
  auto lane_set = std::move(lanes).value();
  const auto lanes_started = lane_set->start();
  ASSERT_TRUE(lanes_started);

  const std::atomic_bool cancellation{};
  constexpr RunGenerationId kRunGeneration = 17U;
  auto package = matching_package();
  package.context.plan_generation = plan->plan_generation;
  package.context.run_generation = kRunGeneration;
  auto created = RuntimeLifecycleExecutor::create(
      *plan, operators, kRunGeneration, &cancellation, nullptr, lane_set.get(), package);
  ASSERT_TRUE(created) << created.error().message;
  auto lifecycle = std::move(created).value();

  EXPECT_FALSE(lifecycle->startup());
  EXPECT_EQ(lifecycle->failure_token(), "PROFILE_MATERIALIZATION_FAILED");
  const RuntimeMaterializationState* const materialization =
      RuntimeLifecycleExecutorTestAccess::materialization(*lifecycle);
  ASSERT_NE(materialization, nullptr);
  ASSERT_EQ(materialization->size(), 4U);
  // Key assertion: failure at index 2 stops forward progress; 0 and 1 are cleaned in
  //   reverse order; operator start/stop never fire since kStart was not reached.
  EXPECT_EQ(materialization->record(0U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(materialization->record(1U).result, RuntimeMaterializationResult::kCleaned);
  EXPECT_EQ(materialization->record(2U).result, RuntimeMaterializationResult::kFailed);
  EXPECT_EQ(materialization->record(3U).result, RuntimeMaterializationResult::kPending);
  EXPECT_EQ(materialization->record(0U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->record(1U).cleanup_actions, 1U);
  ASSERT_EQ(materialization->cleanup_order().size(), 2U);
  EXPECT_EQ(materialization->cleanup_order()[0], 1U);
  EXPECT_EQ(materialization->cleanup_order()[1], 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::start_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::stop_count.load(std::memory_order_relaxed), 0U);

  (void)lifecycle->shutdown();
  EXPECT_EQ(materialization->record(0U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->record(1U).cleanup_actions, 1U);
  EXPECT_EQ(materialization->cleanup_order().size(), 2U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::start_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(LifecycleMaterializationProbeOperator::stop_count.load(std::memory_order_relaxed), 0U);
  const auto lanes_joined = lane_set->join();
  EXPECT_TRUE(lanes_joined) << runtime_error_message(lanes_joined.error());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, FailsUnknownStrategyWithoutWideningThePlan) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  auto obligation = lane_obligation();
  obligation.strategy_id = "holoscan.runtime.unknown/v1";
  fixture.plan.profile_manifest.materialization_obligations = {obligation};

  auto created =
      RuntimeMaterializationState::create(fixture.plan, matching_package(), fixture.lanes.get());
  ASSERT_TRUE(created) << created.error().message;
  auto state = std::move(created).value();
  EXPECT_EQ(state.transition(0U, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
            LifecycleStatus::kFatalFailure);
  EXPECT_TRUE(state.failed());
  EXPECT_EQ(state.record(0U).result, RuntimeMaterializationResult::kFailed);
  EXPECT_EQ(fixture.plan.profile_manifest.materialization_obligations.front().strategy_id,
            "holoscan.runtime.unknown/v1");
  EXPECT_TRUE(fixture.lanes->join());
}

// REQ: RUNTIME-010
TEST(RuntimeMaterializationTest, RejectsNoncanonicalOrUnboundedObligationsBeforeCallbacks) {
  auto fixture = started_lane();
  ASSERT_NE(fixture.lanes, nullptr);
  const auto package = matching_package();

  // Key setup: four invalid obligation configurations exercised in sequence.
  // Key assertion: all four produce PROFILE_MATERIALIZATION_FAILED before any callback.
  auto incompatible_package = package;
  incompatible_package.result = RuntimePackageCapabilityMatchResult::kCapabilityUnsatisfied;
  fixture.plan.profile_manifest.materialization_obligations = {lane_obligation()};
  auto created =
      RuntimeMaterializationState::create(fixture.plan, incompatible_package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  auto first = lane_obligation();
  first.object_id = "z";
  auto second = lane_obligation();
  second.object_id = "a";
  fixture.plan.profile_manifest.materialization_obligations = {first, second};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  auto invalid = lane_obligation();
  invalid.cleanup_stage_mask = 0U;
  fixture.plan.profile_manifest.materialization_obligations = {invalid};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");

  invalid = lane_obligation();
  invalid.object_id.assign(kRuntimeMaterializationMaximumTokenBytes + 1U, 'x');
  fixture.plan.profile_manifest.materialization_obligations = {invalid};
  created = RuntimeMaterializationState::create(fixture.plan, package, fixture.lanes.get());
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().message, "PROFILE_MATERIALIZATION_FAILED");
  EXPECT_TRUE(fixture.lanes->join());
}

}  // namespace
}  // namespace holoscan::detail
