/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/process_placement_internal.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

// Captures per-session lane startup evidence and diagnostic sequences for post-run inspection.
// Stores each publish call as a separate sequence so tests can assert on publication count and
// per-lane record fields. Used by both RunSessionLaneEvidenceTest cases.
class LaneEvidenceDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic> diagnostics) override {
    const std::lock_guard lock(mutex_);
    diagnostic_sequences_.emplace_back(diagnostics.begin(), diagnostics.end());
  }

  void publish_lane_startup_evidence(
      std::span<const detail::ExecutionLaneStartupEvidence> evidence) override {
    const std::lock_guard lock(mutex_);
    lane_sequences_.emplace_back(evidence.begin(), evidence.end());
  }

  [[nodiscard]] std::vector<std::vector<Diagnostic>> diagnostics() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::ExecutionLaneStartupEvidence>> lane_evidence()
      const {
    const std::lock_guard lock(mutex_);
    return lane_sequences_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::vector<Diagnostic>> diagnostic_sequences_;
  std::vector<std::vector<detail::ExecutionLaneStartupEvidence>> lane_sequences_;
};

// Controllable OS placement adapter: succeeds with a fixed CPUAffinity{37}/kOther snapshot
// when fail=false, or returns kUnavailable when fail=true. Used to exercise both the happy
// path (ArchivesAcceptedPlacementReadback) and the startup-barrier failure path.
class SessionPlacementAdapter final : public detail::ExecutionLanePlacementAdapter {
 public:
  explicit SessionPlacementAdapter(bool fail) : fail_(fail) {}

  [[nodiscard]] expected<void, detail::ExecutionLanePlacementError> apply(
      const detail::ExecutionLaneOsPlacement&) override {
    apply_count_.fetch_add(1U, std::memory_order_relaxed);
    if (fail_) {
      return make_unexpected(detail::ExecutionLanePlacementError{
          .kind = detail::ExecutionLanePlacementErrorKind::kUnavailable,
          .error = Error{ErrorCode::kNotSupported, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE"},
      });
    }
    return {};
  }

  [[nodiscard]] expected<detail::ExecutionLanePlacementSnapshot,
                         detail::ExecutionLanePlacementError>
  snapshot() override {
    return detail::ExecutionLanePlacementSnapshot{
        .affinity = CPUAffinity{37U},
        .policy = SchedulingPolicy::kOther,
        .priority = std::nullopt,
        .deadline = std::nullopt,
        .reset_on_fork = false,
    };
  }

  [[nodiscard]] expected<void, detail::ExecutionLanePlacementError> restore(
      const detail::ExecutionLanePlacementSnapshot&) override {
    return {};
  }

  [[nodiscard]] std::size_t apply_count() const noexcept {
    return apply_count_.load(std::memory_order_relaxed);
  }

 private:
  bool fail_{};
  std::atomic<std::size_t> apply_count_{};
};

class WorkerLaneProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    consumer_started_ = false;
  }

  static void record_consumer_start() {
    {
      const std::lock_guard lock(mutex_);
      consumer_started_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_for_consumer_start() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [] { return consumer_started_; });
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline bool consumer_started_{};
};

class WorkerLaneProducer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output").max_emits_per_compute(1U);
    spec.notification_source(notification_, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output_;
  NotificationSource notification_;
};

class WorkerLaneConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input_, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override { WorkerLaneProbeState::record_consumer_start(); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input_;
};

class DescriptorSchemaWorkerProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    producer_started_ = false;
    consumer_started_ = false;
  }

  static void record_start(bool producer) {
    {
      const std::lock_guard lock(mutex_);
      (producer ? producer_started_ : consumer_started_) = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait_for_start(bool producer) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [producer] {
      return producer ? producer_started_ : consumer_started_;
    });
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline bool producer_started_{};
  static inline bool consumer_started_{};
};

class DescriptorSchemaWorkerProducer final : public Operator<> {
 public:
  explicit DescriptorSchemaWorkerProducer(MemoryKind memory_kind) : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = memory_kind_,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        })
        .payload_options(PayloadOptions{
            .schema = schema_identity<schema::ImageT>(),
            .max_tensor_byte_span = 8U,
            .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
        });
    spec.notification_source(notification_, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  void start() override { DescriptorSchemaWorkerProbeState::record_start(true); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output_;

 private:
  MemoryKind memory_kind_{MemoryKind::kUnknown};
  NotificationSource notification_;
};

class DescriptorSchemaWorkerConsumer final : public Operator<> {
 public:
  explicit DescriptorSchemaWorkerConsumer(MemoryKind memory_kind) : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = memory_kind_,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input_});
    return result;
  }

  void start() override { DescriptorSchemaWorkerProbeState::record_start(false); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input_;

 private:
  MemoryKind memory_kind_{MemoryKind::kUnknown};
};

[[nodiscard]] ExecutionPlan make_worker_local_lane_test_plan() {
  Graph graph{"runtime-worker-local-lane-test"};
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto producer = graph.op<WorkerLaneProducer>("producer");
  auto consumer = graph.op<WorkerLaneConsumer>("consumer");
  graph.partition("worker-lane-producer").add(producer);
  graph.partition("worker-lane-consumer").add(consumer);
  graph.add_flow(producer->output_, consumer->input_, ConnectionOptions{.queue_depth = 1U});

  constexpr std::array producer_selector{detail::GraphObjectSelector{.graph_path = "producer"}};
  constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
  CompileOptions options;
  options.deployment
      .bind_thread_placement("worker-lane-producer",
                             ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset})
      .bind_thread_placement("worker-lane-consumer",
                             ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset});
  detail::ProcessPlacementBinding process_placement;
  process_placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = producer_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = consumer_selector,
                             .launch_group = "consumer-process",
                         });
  return detail::compile_with_process_placement(
      graph, std::move(options), std::move(process_placement));
}

[[nodiscard]] ExecutionPlan make_descriptor_schema_worker_test_plan(MemoryKind memory_kind,
                                                                    int device_ordinal) {
  Graph graph{memory_kind == MemoryKind::kCudaDevice
                  ? "runtime-descriptor-schema-device-worker-test"
                  : "runtime-descriptor-schema-pinned-worker-test"};
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto producer = graph.op<DescriptorSchemaWorkerProducer>("producer", memory_kind);
  auto consumer = graph.op<DescriptorSchemaWorkerConsumer>("consumer", memory_kind);
  graph.partition("descriptor-schema-worker-producer").add(producer);
  graph.partition("descriptor-schema-worker-consumer").add(consumer);
  graph.add_flow(producer->output_, consumer->input_, ConnectionOptions{.queue_depth = 1U});

  constexpr std::array producer_selector{detail::GraphObjectSelector{.graph_path = "producer"}};
  constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
  CompileOptions options;
  options.deployment
      .bind_thread_placement("descriptor-schema-worker-producer",
                             ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset})
      .bind_thread_placement("descriptor-schema-worker-consumer",
                             ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "producer",
          .output_port = "output",
          .device = DeviceId{device_ordinal},
      });
  detail::ProcessPlacementBinding process_placement;
  process_placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = producer_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = consumer_selector,
                             .launch_group = "consumer-process",
                         });
  return detail::compile_with_process_placement(
      graph, std::move(options), std::move(process_placement));
}

[[nodiscard]] ProcessGroupId process_group_for_vertex(const ExecutionPlan& plan, VertexId vertex) {
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto placement = std::find_if(placements.begin(),
                                      placements.end(),
                                      [&](const detail::VertexProcessPlacementPlanRecord& record) {
                                        return record.vertex == vertex;
                                      });
  return placement == placements.end() ? ProcessGroupId{} : placement->process_group;
}

[[nodiscard]] std::uint32_t execution_lane_for_vertex(const ExecutionPlan& plan, VertexId vertex) {
  const auto decision =
      std::find_if(plan.partition_decisions().begin(),
                   plan.partition_decisions().end(),
                   [&](const PartitionDecisionRecord& record) { return record.vertex == vertex; });
  if (decision == plan.partition_decisions().end()) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  const auto partition = std::find_if(
      plan.execution_partitions().begin(),
      plan.execution_partitions().end(),
      [&](const ExecutionPartitionRecord& record) { return record.id == decision->partition; });
  return partition == plan.execution_partitions().end() ? std::numeric_limits<std::uint32_t>::max()
                                                        : partition->execution_lane;
}

// Layers a placement adapter and diagnostic sink onto a compiled plan for lane-evidence tests.
// Enables the direct runtime so execution proceeds without a full HoloMQ deployment.
[[nodiscard]] ExecutionPlan with_lane_test_services(
    const ExecutionPlan& source,
    const std::shared_ptr<detail::ExecutionLanePlacementAdapter>& placement,
    const std::shared_ptr<detail::RuntimeStructuralDiagnosticSink>& sink) {
  const ExecutionPlan direct = test::enable_direct_runtime(source);
  const ExecutionPlan placed =
      detail::RuntimeTestPlanAccess::with_lane_placement_adapter(direct, placement);
  return detail::RuntimeTestPlanAccess::with_diagnostic_sink(placed, sink);
}

// REQ: RUNTIME-031
TEST(RunSessionLaneEvidenceTest, ProcessGroupWorkerStartsOnlyItsOwnedExecutionLane) {
  WorkerLaneProbeState::reset();
  const ExecutionPlan compiled = make_worker_local_lane_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(compiled);
  ASSERT_EQ(roles.size(), 2U);
  const auto consumer_role =
      std::find_if(roles.begin(), roles.end(), [](const detail::ProcessRolePlanRecord& record) {
        return record.role == "consumer";
      });
  ASSERT_NE(consumer_role, roles.end());
  ASSERT_EQ(consumer_role->members.size(), 1U);
  const VertexId consumer = consumer_role->members.front();
  const ProcessGroupId consumer_group = process_group_for_vertex(compiled, consumer);
  ASSERT_TRUE(consumer_group);
  const std::uint32_t consumer_lane = execution_lane_for_vertex(compiled, consumer);
  ASSERT_LT(consumer_lane, compiled.scheduler_plan().execution_lane_count);

  const auto placement = std::make_shared<SessionPlacementAdapter>(false);
  const auto sink = std::make_shared<LaneEvidenceDiagnosticSink>();
  const ExecutionPlan placed =
      detail::RuntimeTestPlanAccess::with_lane_placement_adapter(compiled, placement);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(placed, sink);

  // Key setup: run only the consumer process group from a two-group, two-role plan.
  RunSession session = detail::run_async_worker(plan, consumer_group);
  ASSERT_TRUE(WorkerLaneProbeState::wait_for_consumer_start());
  session.request_stop();
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_TRUE(session.diagnostics().empty());
  const auto published = sink->lane_evidence();
  ASSERT_EQ(published.size(), 1U);
  // Key assertion: the worker materializes one local lane, not the plan's remote producer lane.
  ASSERT_EQ(published.front().size(), 1U);
  EXPECT_EQ(published.front().front().execution_lane, consumer_lane);
  EXPECT_EQ(placement->apply_count(), 1U);
}

// Descriptor-backed schema workers need the same frozen CUDA stream on the publisher and importer
// sides so same-kind pinned/device staging cannot silently fall back through pageable host.
// REQ: CHAN-106
TEST(RunSessionLaneEvidenceTest, DescriptorSchemaWorkersMaterializeSourceAndTargetDeviceResources) {
  if (cudaFree(nullptr) != cudaSuccess) {
    GTEST_SKIP() << "CUDA-backed Tensor providers are unavailable";
  }
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);

  for (const MemoryKind memory_kind : {MemoryKind::kPinnedHost, MemoryKind::kCudaDevice}) {
    SCOPED_TRACE(memory_kind == MemoryKind::kPinnedHost ? "pinned_host" : "device");
    DescriptorSchemaWorkerProbeState::reset();
    const ExecutionPlan plan = make_descriptor_schema_worker_test_plan(memory_kind, device_ordinal);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto process_roles = detail::ProcessPlacementPlanAccess::process_roles(plan);
    ASSERT_EQ(process_roles.size(), 2U);
    ASSERT_EQ(plan.tensor_device_resources().size(), 2U);

    const auto producer_role = std::find_if(
        process_roles.begin(),
        process_roles.end(),
        [](const detail::ProcessRolePlanRecord& record) { return record.role == "producer"; });
    const auto consumer_role = std::find_if(
        process_roles.begin(),
        process_roles.end(),
        [](const detail::ProcessRolePlanRecord& record) { return record.role == "consumer"; });
    ASSERT_NE(producer_role, process_roles.end());
    ASSERT_NE(consumer_role, process_roles.end());
    ASSERT_EQ(producer_role->members.size(), 1U);
    ASSERT_EQ(consumer_role->members.size(), 1U);
    const ProcessGroupId producer_group =
        process_group_for_vertex(plan, producer_role->members.front());
    const ProcessGroupId consumer_group =
        process_group_for_vertex(plan, consumer_role->members.front());
    ASSERT_TRUE(producer_group);
    ASSERT_TRUE(consumer_group);

    {
      // Key setup: reconstruct the importer without a live publisher; its exact target-group
      // resource must still reach lifecycle Start.
      RunSession session = detail::run_async_worker(plan, consumer_group);
      const bool started = DescriptorSchemaWorkerProbeState::wait_for_start(false);
      session.request_stop();
      session.wait();
      ASSERT_TRUE(started) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                             : session.diagnostics().front().token);
      ASSERT_EQ(session.termination(), RunTermination::kStopped);
      ASSERT_TRUE(session.diagnostics().empty());
    }
    {
      // Key assertion: the producer group independently materializes and binds the source stream
      // required by readiness-ordered schema staging.
      RunSession session = detail::run_async_worker(plan, producer_group);
      const bool started = DescriptorSchemaWorkerProbeState::wait_for_start(true);
      session.request_stop();
      session.wait();
      ASSERT_TRUE(started) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                             : session.diagnostics().front().token);
      ASSERT_EQ(session.termination(), RunTermination::kStopped);
      ASSERT_TRUE(session.diagnostics().empty());
    }
  }
}

// REQ: RUNTIME-011
// REQ: RUNTIME-031
// GoogleTest fatal assertions terminate these evidence tests before optional access, but
// clang-tidy does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(RunSessionLaneEvidenceTest, ArchivesAcceptedPlacementReadbackAfterTerminalCompletion) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_role_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto placement = std::make_shared<SessionPlacementAdapter>(false);
  const auto sink = std::make_shared<LaneEvidenceDiagnosticSink>();
  const ExecutionPlan plan = with_lane_test_services(compiled, placement, sink);
  ASSERT_EQ(plan.execution_lanes().size(), compiled.execution_lanes().size());
  ASSERT_EQ(plan.json(), compiled.json());
  ASSERT_EQ(plan.digest(), compiled.digest());
  test::RuntimeProbeState::reset();

  // Key setup: SessionPlacementAdapter(false) succeeds; snapshot returns affinity={37}/kOther.
  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  ASSERT_TRUE(session.diagnostics().empty());
  // Key assertion: only the explicit partition lane is materialized, with exact accepted evidence.
  const auto published = sink->lane_evidence();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const auto& record = published.front().front();
  const auto accepted = std::find_if(plan.execution_lanes().begin(),
                                     plan.execution_lanes().end(),
                                     [&](const ExecutionLanePlanRecord& lane) {
                                       return lane.execution_lane == record.execution_lane;
                                     });
  ASSERT_NE(accepted, plan.execution_lanes().end());
  EXPECT_EQ(record.partition, accepted->partition);
  EXPECT_EQ(record.wake_mode, accepted->accepted_placement.wake_mode);
  EXPECT_EQ(record.irq_isolation, accepted->accepted_placement.irq_isolation);
  EXPECT_EQ(record.disposition, detail::ExecutionLaneStartupDisposition::kReady);
  EXPECT_EQ(record.phase, detail::ExecutionLaneStartupPhase::kAllLanesBarrier);
  ASSERT_TRUE(record.applied.has_value());
  EXPECT_EQ(record.applied->affinity.cores, (std::vector<std::uint32_t>{37U}));
  EXPECT_EQ(record.applied->policy, SchedulingPolicy::kOther);
  EXPECT_FALSE(record.applied->priority.has_value());
  EXPECT_FALSE(record.applied->deadline.has_value());
  EXPECT_FALSE(record.applied->reset_on_fork);
  EXPECT_FALSE(record.error.has_value());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-011
// REQ: RUNTIME-031
TEST(RunSessionLaneEvidenceTest, StartupBarrierFailurePublishesStableEvidenceBeforeCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_role_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto placement = std::make_shared<SessionPlacementAdapter>(true);
  const auto sink = std::make_shared<LaneEvidenceDiagnosticSink>();
  const ExecutionPlan plan = with_lane_test_services(compiled, placement, sink);
  test::RuntimeProbeState::reset();

  // Key setup: SessionPlacementAdapter(true) injects kUnavailable during placement apply.
  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "PLATFORM_CAPABILITY_UNAVAILABLE");
  // Key assertion: compute was never entered; at least one lane carries kUnavailable disposition.
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));

  const auto diagnostics = sink->diagnostics();
  ASSERT_EQ(diagnostics.size(), 1U);
  ASSERT_EQ(diagnostics.front().size(), 1U);
  EXPECT_EQ(diagnostics.front().front().token, "PLATFORM_CAPABILITY_UNAVAILABLE");

  const auto published = sink->lane_evidence();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const auto failed =
      std::count_if(published.front().begin(), published.front().end(), [](const auto& record) {
        return record.disposition == detail::ExecutionLaneStartupDisposition::kUnavailable;
      });
  EXPECT_GE(failed, 1);
  for (std::size_t index = 0U; index < published.front().size(); ++index) {
    SCOPED_TRACE(index);
    const auto& record = published.front()[index];
    const auto accepted = std::find_if(plan.execution_lanes().begin(),
                                       plan.execution_lanes().end(),
                                       [&](const ExecutionLanePlanRecord& lane) {
                                         return lane.execution_lane == record.execution_lane;
                                       });
    ASSERT_NE(accepted, plan.execution_lanes().end());
    EXPECT_EQ(record.partition, accepted->partition);
    EXPECT_EQ(record.wake_mode, accepted->accepted_placement.wake_mode);
    EXPECT_EQ(record.irq_isolation, accepted->accepted_placement.irq_isolation);
    EXPECT_NE(record.disposition, detail::ExecutionLaneStartupDisposition::kPending);
    EXPECT_FALSE(record.applied.has_value());
    ASSERT_TRUE(record.error.has_value());
    if (record.disposition == detail::ExecutionLaneStartupDisposition::kUnavailable) {
      EXPECT_EQ(record.phase, detail::ExecutionLaneStartupPhase::kPlacementApply);
      EXPECT_EQ(record.error->code, ErrorCode::kNotSupported);
      EXPECT_EQ(record.error->message, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE");
    } else {
      EXPECT_EQ(record.disposition, detail::ExecutionLaneStartupDisposition::kStopped);
      EXPECT_EQ(record.phase, detail::ExecutionLaneStartupPhase::kAllLanesBarrier);
      EXPECT_EQ(record.error->code, ErrorCode::kInterrupted);
      EXPECT_EQ(record.error->message, "RUNTIME_LANE_STARTUP_CANCELLED");
    }
  }
}
// NOLINTEND(bugprone-unchecked-optional-access)

}  // namespace
}  // namespace holoscan
