/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/memory/block_memory_pool.hpp>

#include "common/allocation_counter.hpp"
#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "run_session_tensor_test_support.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::run_session_tensor;

// Captures private CUDA teardown evidence when a Green Context Tensor system test fails. Used by
// GreenContextTensorFlowUsesCrossContextReadinessAndRetirement to keep failure output actionable.
class RuntimeGreenContextDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic>) override {}

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    details_.assign(details.begin(), details.end());
  }

  [[nodiscard]] std::vector<detail::RuntimeDiagnosticDetail> details() const {
    const std::lock_guard lock(mutex_);
    return details_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<detail::RuntimeDiagnosticDetail> details_;
};

constexpr std::uint32_t kGreenContextTensorSampleCount = 64U;

[[nodiscard]] constexpr std::uint8_t green_context_tensor_sample_byte(
    std::uint32_t sample_index) noexcept {
  return static_cast<std::uint8_t>((sample_index % 0xfeU) + 1U);
}

[[nodiscard]] constexpr std::uint64_t green_context_tensor_sample_value(
    std::uint32_t sample_index) noexcept {
  return static_cast<std::uint64_t>(green_context_tensor_sample_byte(sample_index)) *
         0x0101010101010101ULL;
}

// Synchronizes a Green Context Tensor source/sink pair and records their streams and final value.
// Used by RunSessionTest.GreenContextTensorFlowUsesCrossContextReadinessAndRetirement.
struct RuntimeGreenContextTensorProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    source_stream = nullptr;
    sink_stream = nullptr;
    observed_value = 0U;
    completed = false;
    error.reset();
  }

  static void record_source(cudaStream_t stream) {
    const std::lock_guard lock(mutex);
    source_stream = stream;
  }

  static void record_sink(cudaStream_t stream, std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      sink_stream = stream;
      observed_value = value;
      completed = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return completed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline cudaStream_t source_stream{};
  static inline cudaStream_t sink_stream{};
  static inline std::uint64_t observed_value{};
  static inline bool completed{};
  static inline std::optional<ErrorCode> error;
};

// Publishes a deterministic sequence of device Tensor loans on its assigned Green Context stream.
// Used by RunSessionTest.GreenContextTensorFlowUsesCrossContextReadinessAndRetirement.
class RuntimeGreenContextTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 64U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_count_ == kGreenContextTensorSampleCount) {
      return {};
    }
    const std::uint32_t sample_index = emitted_count_;
    const auto stream = context.cuda_stream().get();
    RuntimeGreenContextTensorProbeState::record_source(stream);
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      // A slow consumer can temporarily retain every fixed pool slot. Both queue pressure and
      // backing exhaustion are retryable for this clock-driven source.
      if (loan.error().code == ErrorCode::kBackpressured ||
          loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      RuntimeGreenContextTensorProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      RuntimeGreenContextTensorProbeState::fail(writer.error().code);
      return make_unexpected(std::move(writer.error()));
    }
    if (stream == nullptr || cudaMemsetAsync(writer->data(),
                                             green_context_tensor_sample_byte(sample_index),
                                             writer->byte_size(),
                                             stream) != cudaSuccess) {
      RuntimeGreenContextTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      RuntimeGreenContextTensorProbeState::fail(committed.error().code);
      return make_unexpected(std::move(committed.error()));
    }
    auto published = output.emit(std::move(*loan));
    if (!published) {
      if (published.error().code == ErrorCode::kBackpressured) {
        return {};
      }
      RuntimeGreenContextTensorProbeState::fail(published.error().code);
    } else {
      ++emitted_count_;
    }
    return published;
  }

  Output<Tensor> output;

 private:
  std::uint32_t emitted_count_{};
};

// Receives the deterministic device Tensor sequence and validates it on the assigned compute
// stream. Used by RunSessionTest.GreenContextTensorFlowUsesCrossContextReadinessAndRetirement.
class RuntimeGreenContextTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 64U, 1U}},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      RuntimeGreenContextTensorProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto stream = context.cuda_stream().get();
    std::uint64_t value = 0U;
    auto copied = sample->data.copy_to_host(&value, sizeof(value), context.cuda_stream());
    if (stream == nullptr || !copied || cudaStreamSynchronize(stream) != cudaSuccess) {
      const Error error = copied ? Error{ErrorCode::kFailure} : std::move(copied.error());
      RuntimeGreenContextTensorProbeState::fail(error.code);
      return make_unexpected(error);
    }
    if (value != green_context_tensor_sample_value(received_count_)) {
      RuntimeGreenContextTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure, "Green Context Tensor value mismatch"});
    }
    ++received_count_;
    if (received_count_ == kGreenContextTensorSampleCount) {
      RuntimeGreenContextTensorProbeState::record_sink(stream, value);
    }
    return {};
  }

  Input<Tensor> input;

 private:
  std::uint32_t received_count_{};
};

// Gates foreign Tensor publication until a subscriber connects and records the producer result.
// Used by RunSessionTest.GreenContextExternalTensorCopyUsesPlannedServiceStream.
struct RuntimeGreenContextTransferProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    emit_allowed = false;
    completed = false;
    producer_stream = nullptr;
    error.reset();
  }

  static void allow_emit() { emit_allowed.store(true, std::memory_order_release); }

  static void finish(cudaStream_t stream, std::optional<ErrorCode> failure = std::nullopt) {
    {
      const std::lock_guard lock(mutex);
      producer_stream = stream;
      error = failure;
      completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return completed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic_bool emit_allowed{false};
  static inline bool completed{};
  static inline cudaStream_t producer_stream{};
  static inline std::optional<ErrorCode> error;
};

// Publishes one externally allocated device Tensor so Runtime must stage it through the planned
// transfer-service stream. Used by
// RunSessionTest.GreenContextExternalTensorCopyUsesPlannedServiceStream.
class RuntimeGreenContextForeignTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 64U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  void start() override {
    pool_ = std::make_shared<BlockMemoryPool>(MemoryStorageType::kDevice, 8U, 1U, 0);
    pool_ready_ = static_cast<bool>(pool_->initialize_pool());
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_ ||
        !RuntimeGreenContextTransferProbeState::emit_allowed.load(std::memory_order_acquire)) {
      return {};
    }
    emitted_ = true;
    const auto stream = context.cuda_stream().get();
    if (!pool_ready_ || stream == nullptr) {
      RuntimeGreenContextTransferProbeState::finish(stream, ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto tensor = make_tensor(TensorShapeView{},
                              DLDataType{kDLUInt, 64U, 1U},
                              std::nullopt,
                              MemoryStorageType::kDevice,
                              pool_,
                              context.cuda_stream());
    if (!tensor) {
      RuntimeGreenContextTransferProbeState::finish(stream, tensor.error().code);
      return make_unexpected(std::move(tensor).error());
    }
    auto data = tensor->mutable_data();
    if (!data || cudaMemsetAsync(*data, 0x4D, 8U, stream) != cudaSuccess ||
        !tensor->set_readiness_stream(stream)) {
      RuntimeGreenContextTransferProbeState::finish(stream, ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto emitted = output.emit(Sample<Tensor>{std::move(*tensor), {}});
    if (emitted) {
      RuntimeGreenContextTransferProbeState::finish(stream);
    } else {
      RuntimeGreenContextTransferProbeState::finish(stream, emitted.error().code);
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  std::shared_ptr<BlockMemoryPool> pool_;
  bool pool_ready_{};
  bool emitted_{};
};

// Graph topology (partition placement varies by test):
//   source --[queue_depth=1]--> sink
// The operators either share one dedicated partition, occupy two dedicated partitions, or place
// the sink on the residual Green Context.
[[nodiscard]] ExecutionPlan make_runtime_green_context_tensor_plan(
    const CudaGreenContextCapability& capability,
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    const bool shared_partition = false, const bool residual_sink = false) {
  Graph graph{shared_partition ? "runtime-green-context-tensor-shared"
              : residual_sink  ? "runtime-green-context-tensor-residual"
                               : "runtime-green-context-tensor-dedicated"};
  auto source = graph.op<RuntimeGreenContextTensorSourceOperator>("source");
  auto sink = graph.op<RuntimeGreenContextTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  if (shared_partition) {
    graph.partition("shared-partition").add(source).add(sink);
  } else {
    graph.partition("source-partition").add(source);
    if (residual_sink) {
      graph.auto_partition_remaining();
    } else {
      graph.partition("sink-partition").add(sink);
    }
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "Green Context Tensor snapshot unavailable"}};
  }
  const auto output_port =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.name == "output";
      });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "Green Context Tensor output unresolved"}};
  }
  const std::uint32_t quantum =
      std::lcm(capability.min_sm_partition_size, capability.sm_coscheduled_alignment);
  CompileOptions options;
  if (shared_partition) {
    options.deployment.bind_cuda_green_context("shared-partition",
                                               {.device = DeviceId{0}, .sm_count = quantum});
  } else {
    options.deployment.bind_cuda_green_context("source-partition",
                                               {.device = DeviceId{0}, .sm_count = quantum});
    if (!residual_sink) {
      options.deployment.bind_cuda_green_context("sink-partition",
                                                 {.device = DeviceId{0}, .sm_count = quantum});
    }
  }
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output_port->id.value},
                                        .device = DeviceId{0}},
  };
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      std::move(options),
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      });
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// Graph topology:
//   source(dedicated partition) --> exported HoloMQ topic(residual service stream)
[[nodiscard]] ExecutionPlan make_runtime_green_context_transfer_plan(
    const CudaGreenContextCapability& capability,
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    const std::string& endpoint_name) {
  Graph graph{"runtime-green-context-transfer"};
  auto source = graph.op<RuntimeGreenContextForeignTensorSourceOperator>("source");
  graph.partition("source-partition").add(source);
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 64U, 1U},
                         .rank = 0U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime transfer snapshot is unavailable"}};
  }
  const auto output_port =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.name == "output";
      });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime transfer output is unresolved"}};
  }
  const std::uint32_t quantum =
      std::lcm(capability.min_sm_partition_size, capability.sm_coscheduled_alignment);
  CompileOptions options;
  options.deployment.bind_cuda_green_context("source-partition",
                                             {.device = DeviceId{0}, .sm_count = quantum});
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output_port->id.value},
                                        .device = DeviceId{0}},
  };
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      std::move(options),
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      });
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextTensorFlowUsesCrossContextReadinessAndRetirement) {
  // Key setup: run a Tensor source and sink on two dedicated Green Contexts while Tensor-pool
  // retirement tracks completion across the two streams.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 3U) {
    GTEST_SKIP() << "device cannot provide two dedicated groups plus a residual";
  }

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeGreenContextTensorProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RuntimeGreenContextDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_runtime_green_context_tensor_plan(*capability, control), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_pools().size(), 1U);
  ASSERT_EQ(plan.cuda_green_context_resources().size(), 3U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  const auto& allocation = plan.tensor_allocations().front();
  const auto& device_resource = plan.tensor_device_resources().front();
  EXPECT_EQ(device_resource.pool_count, 1U);
  EXPECT_EQ(device_resource.slot_capacity, allocation.slot_count);
  EXPECT_EQ(device_resource.readiness_capacity, allocation.readiness_capacity);
  EXPECT_EQ(device_resource.release_capacity, allocation.release_capacity);
  EXPECT_EQ(device_resource.retirement_capacity, allocation.retirement_capacity);
  EXPECT_EQ(device_resource.quarantine_capacity, allocation.quarantine_capacity);

  std::optional<RunSession> session{run_async(plan)};
  const bool observed = RuntimeGreenContextTensorProbeState::wait();
  session->request_stop();
  session->wait();

  ASSERT_TRUE(observed) << (session->diagnostics().empty() ? "no runtime diagnostic"
                                                           : session->diagnostics().front().token);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  const auto diagnostic_details = diagnostic_sink->details();
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token) << " "
      << (diagnostic_details.empty() ? std::string{"no private diagnostic detail"}
                                     : runtime_error_message(diagnostic_details.front().error));
  // Key assertion: distinct streams exchange all samples through real readiness, wait, and
  // retirement events, and every CUDA resource is destroyed with the session.
  {
    const std::lock_guard lock(RuntimeGreenContextTensorProbeState::mutex);
    EXPECT_FALSE(RuntimeGreenContextTensorProbeState::error.has_value());
    EXPECT_NE(RuntimeGreenContextTensorProbeState::source_stream, nullptr);
    EXPECT_NE(RuntimeGreenContextTensorProbeState::sink_stream, nullptr);
    EXPECT_NE(RuntimeGreenContextTensorProbeState::source_stream,
              RuntimeGreenContextTensorProbeState::sink_stream);
    EXPECT_EQ(RuntimeGreenContextTensorProbeState::observed_value,
              green_context_tensor_sample_value(kGreenContextTensorSampleCount - 1U));
  }
  session.reset();
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_streams_created, 2U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
  EXPECT_GE(metrics.cuda_readiness_records, kGreenContextTensorSampleCount);
  EXPECT_GE(metrics.cuda_stream_waits, kGreenContextTensorSampleCount);
  EXPECT_EQ(metrics.cuda_readiness_record_elisions, 0U);
  EXPECT_GT(metrics.cuda_release_stream_assignments, 0U);
  EXPECT_GE(metrics.cuda_release_records, kGreenContextTensorSampleCount);
  EXPECT_GT(metrics.cuda_retirement_queries, 0U);
  EXPECT_EQ(metrics.cuda_events_created,
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 4U);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextEventMaterializationFailurePreventsCallbacks) {
  // Graph topology:
  //   source(dedicated) --> sink(dedicated), with retirement on the residual
  // Key setup: inject a readiness-event creation failure while materializing a valid two-partition
  // Green Context Tensor plan.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 3U) {
    GTEST_SKIP() << "device cannot provide two dedicated groups plus a residual";
  }

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  control->arm_fault(detail::RuntimeTensorOutputTestFault::kEventMaterializationFailure);
  RuntimeGreenContextTensorProbeState::reset();
  const ExecutionPlan plan = make_runtime_green_context_tensor_plan(*capability, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: kArm fails before either Operator callback and no partially created event or
  // active slot survives rollback.
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_FALSE(session.diagnostics().empty());
  {
    const std::lock_guard lock(RuntimeGreenContextTensorProbeState::mutex);
    EXPECT_EQ(RuntimeGreenContextTensorProbeState::source_stream, nullptr);
    EXPECT_EQ(RuntimeGreenContextTensorProbeState::sink_stream, nullptr);
  }
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_events_created, 0U);
  EXPECT_EQ(metrics.cuda_events_destroyed, 0U);
  EXPECT_EQ(metrics.active_slot_count, 0U);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextTensorFlowElidesSamePartitionReadiness) {
  // Key setup: place the Tensor source and sink together in one dedicated Green Context partition.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 1U) {
    GTEST_SKIP() << "device cannot provide one dedicated group";
  }

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeGreenContextTensorProbeState::reset();
  const ExecutionPlan plan =
      make_runtime_green_context_tensor_plan(*capability, control, true, false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);

  std::optional<RunSession> session{run_async(plan)};
  const bool observed = RuntimeGreenContextTensorProbeState::wait();
  session->request_stop();
  session->wait();
  ASSERT_TRUE(observed) << (session->diagnostics().empty() ? "no runtime diagnostic"
                                                           : session->diagnostics().front().token);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  // Key assertion: both operators use one stream and Runtime takes the same-stream elision path.
  {
    const std::lock_guard lock(RuntimeGreenContextTensorProbeState::mutex);
    EXPECT_FALSE(RuntimeGreenContextTensorProbeState::error.has_value());
    EXPECT_NE(RuntimeGreenContextTensorProbeState::source_stream, nullptr);
    EXPECT_EQ(RuntimeGreenContextTensorProbeState::source_stream,
              RuntimeGreenContextTensorProbeState::sink_stream);
  }
  session.reset();
  const auto metrics = control->snapshot(0U);
  EXPECT_GT(metrics.cuda_readiness_record_elisions, 0U);
  EXPECT_GT(metrics.cuda_stream_wait_elisions, 0U);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(metrics.cuda_events_created,
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 4U);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextTensorFlowCrossesDedicatedAndResidualStreams) {
  // Key setup: bind the source to a dedicated Green Context and leave the sink on the residual.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 2U) {
    GTEST_SKIP() << "device cannot provide one dedicated group plus a residual";
  }

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeGreenContextTensorProbeState::reset();
  const ExecutionPlan plan =
      make_runtime_green_context_tensor_plan(*capability, control, false, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 1U);
  const auto source_partition = plan.partitions().lookup_named("source-partition");
  const auto sink_partition = plan.partitions().lookup_for_path("sink");
  ASSERT_TRUE(source_partition);
  ASSERT_TRUE(sink_partition);
  EXPECT_EQ(source_partition->origin(), ExecutionPartitionOrigin::kAuthored);
  EXPECT_EQ(sink_partition->origin(), ExecutionPartitionOrigin::kAutomatic);
  EXPECT_NE(source_partition->id(), sink_partition->id());
  EXPECT_NE(source_partition->execution_lane(), sink_partition->execution_lane());
  ASSERT_EQ(plan.cuda_green_context_assignments().size(), 1U);
  EXPECT_EQ(plan.cuda_green_context_assignments().front().partition, source_partition->id());
  EXPECT_TRUE(std::ranges::none_of(plan.cuda_green_context_assignments(), [&](const auto& row) {
    return row.partition == sink_partition->id();
  }));
  const auto source_stream = std::ranges::find(
      plan.cuda_compute_streams(), source_partition->id(), &CudaComputeStreamPlanRecord::partition);
  const auto sink_stream = std::ranges::find(
      plan.cuda_compute_streams(), sink_partition->id(), &CudaComputeStreamPlanRecord::partition);
  ASSERT_NE(source_stream, plan.cuda_compute_streams().end());
  ASSERT_NE(sink_stream, plan.cuda_compute_streams().end());
  EXPECT_NE(source_stream->green_context_id, sink_stream->green_context_id);
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);

  std::optional<RunSession> session{run_async(plan)};
  const bool observed = RuntimeGreenContextTensorProbeState::wait();
  session->request_stop();
  session->wait();
  ASSERT_TRUE(observed) << (session->diagnostics().empty() ? "no runtime diagnostic"
                                                           : session->diagnostics().front().token);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  // Key assertion: distinct dedicated/residual streams perform readiness, wait, and retirement
  // operations for every sample without leaking events.
  {
    const std::lock_guard lock(RuntimeGreenContextTensorProbeState::mutex);
    EXPECT_FALSE(RuntimeGreenContextTensorProbeState::error.has_value());
    EXPECT_NE(RuntimeGreenContextTensorProbeState::source_stream, nullptr);
    EXPECT_NE(RuntimeGreenContextTensorProbeState::sink_stream, nullptr);
    EXPECT_NE(RuntimeGreenContextTensorProbeState::source_stream,
              RuntimeGreenContextTensorProbeState::sink_stream);
  }
  session.reset();
  const auto metrics = control->snapshot(0U);
  EXPECT_GE(metrics.cuda_readiness_records, kGreenContextTensorSampleCount);
  EXPECT_GE(metrics.cuda_stream_waits, kGreenContextTensorSampleCount);
  EXPECT_GE(metrics.cuda_release_records, kGreenContextTensorSampleCount);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(metrics.cuda_events_created,
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 4U);
}

// REQ: TENSOR-STORAGE-014
// REQ: CHAN-134
TEST(RunSessionTest, GreenContextExternalTensorCopyUsesPlannedServiceStream) {
  // Key setup: export one foreign device Tensor from a dedicated partition through a HoloMQ topic
  // whose staging copy is assigned to the residual service stream.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 2U) {
    GTEST_SKIP() << "device cannot provide one dedicated group plus a service residual";
  }

  const std::string endpoint_name = unique_runtime_graph_name("runtime-green-context-transfer");
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeGreenContextTransferProbeState::reset();
  const ExecutionPlan plan =
      make_runtime_green_context_transfer_plan(*capability, control, endpoint_name);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  const auto service_stream = std::find_if(
      plan.cuda_compute_streams().begin(),
      plan.cuda_compute_streams().end(),
      [](const auto& stream) {
        return !stream.partition && stream.owner_id.starts_with("service.tensor.transfer.topic.");
      });
  ASSERT_NE(service_stream, plan.cuda_compute_streams().end());
  ASSERT_TRUE(service_stream->green_context_id.has_value());
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  const auto copied_transfer =
      std::find_if(plan.tensor_transfer_plans().begin(),
                   plan.tensor_transfer_plans().end(),
                   [](const auto& transfer) {
                     return transfer.disposition == PlanDisposition::kAccepted &&
                            transfer.requires_exportable_copy_pool;
                   });
  ASSERT_NE(copied_transfer, plan.tensor_transfer_plans().end());

  // Key assertion: the raw Tensor copy row names the same process, device, and service stream as
  // its compiler-selected exportable staging pool.
  EXPECT_EQ(copied_transfer->interface_process_group,
            plan.tensor_interface_pools().front().allocator_process_group);
  EXPECT_EQ(copied_transfer->interface_device, plan.tensor_interface_pools().front().device);
  EXPECT_EQ(copied_transfer->service_owner_id, service_stream->owner_id);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  std::optional<RunSession> session{run_async(plan)};
  const auto connection_deadline = std::chrono::steady_clock::now() + 5s;
  while (!subscriber->has_publishers() && std::chrono::steady_clock::now() < connection_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(subscriber->has_publishers());
  RuntimeGreenContextTransferProbeState::allow_emit();
  ASSERT_TRUE(RuntimeGreenContextTransferProbeState::wait());
  {
    const std::lock_guard lock(RuntimeGreenContextTransferProbeState::mutex);
    const auto observed_copy = control->service_copy_snapshot();
    ASSERT_FALSE(RuntimeGreenContextTransferProbeState::error.has_value())
        << static_cast<std::uint32_t>(
               RuntimeGreenContextTransferProbeState::error.value_or(ErrorCode::kSuccess))
        << " service copies=" << observed_copy.count;
  }

  test::UniqueCudaStream verify_stream;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(verify_stream), cudaSuccess);
  expected<Sample<Tensor>, Error> received = make_unexpected(Error{ErrorCode::kNotReady});
  const auto receive_deadline = std::chrono::steady_clock::now() + 5s;
  while (!received && received.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < receive_deadline) {
    received = subscriber->receive_sample();
    if (!received) {
      std::this_thread::sleep_for(1ms);
    }
  }
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  std::uint64_t value{};
  ASSERT_TRUE(received->data.copy_to_host(&value, sizeof(value), verify_stream.get()));
  ASSERT_EQ(cudaStreamSynchronize(verify_stream.get()), cudaSuccess);
  // Key assertion: the subscriber observes the exact value and metrics identify the planned
  // service stream as distinct from the producer stream.
  EXPECT_EQ(value, 0x4D4D4D4D4D4D4D4DULL);

  session->request_stop();
  session->wait();
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  const auto copy = control->service_copy_snapshot();
  EXPECT_EQ(copy.count, 1U);
  EXPECT_NE(copy.producer_stream, nullptr);
  EXPECT_NE(copy.service_stream, nullptr);
  EXPECT_NE(copy.producer_stream, copy.service_stream);
  session.reset();
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
}

}  // namespace
}  // namespace holoscan
