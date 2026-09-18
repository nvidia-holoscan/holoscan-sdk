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
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
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
#include "compiler/process_placement_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_test_graph.hpp"
#include "support/host_unique_test_name.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

// Runtime-materializing tests publish into the host-wide HoloMQ namespace. Give each invocation
// its own graph identity so concurrent CTest runs and parallel SDK containers cannot derive the
// same Channel service names.
[[nodiscard]] std::string unique_runtime_graph_name(const std::string_view base) {
  return test::make_host_unique_test_name(base);
}

[[nodiscard]] bool wait_for_runtime_phase_count(
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const detail::RuntimePhaseCounter counter, const std::uint64_t minimum) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (counters->snapshot()[detail::RuntimePhaseCounters::index(counter)] >= minimum) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return counters->snapshot()[detail::RuntimePhaseCounters::index(counter)] >= minimum;
}

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

// A zero-limit producer must complete its ordinary lifecycle without admitting compute. Returning
// failure from compute makes any accidental callback admission fail closed and directly visible.
class RuntimeZeroComputeCatalogTensorSourceOperator final : public Operator<> {
 public:
  static void reset() {
    compute_calls_.store(0U, std::memory_order_release);
    const std::lock_guard lock(start_mutex_);
    start_entered_ = false;
    start_released_ = false;
  }

  [[nodiscard]] static std::uint32_t compute_calls() noexcept {
    return compute_calls_.load(std::memory_order_acquire);
  }

  [[nodiscard]] static bool wait_for_start() {
    std::unique_lock lock(start_mutex_);
    return start_cv_.wait_for(lock, 10s, [] { return start_entered_; });
  }

  static void release_start() {
    {
      const std::lock_guard lock(start_mutex_);
      start_released_ = true;
    }
    start_cv_.notify_all();
  }

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
            .bounds = tensor_bounds(16U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 0U});
    return result;
  }

  void start() override {
    std::unique_lock lock(start_mutex_);
    start_entered_ = true;
    start_cv_.notify_all();
    start_cv_.wait(lock, [] { return start_released_; });
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_calls_.fetch_add(1U, std::memory_order_acq_rel);
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  Output<Tensor> output;

 private:
  static inline std::atomic<std::uint32_t> compute_calls_{};
  static inline std::mutex start_mutex_;
  static inline std::condition_variable start_cv_;
  static inline bool start_entered_{};
  static inline bool start_released_{};
};

// Coordinates catalog-export terminal tests. The start gate gives the public required subscriber
// a live publisher to attach to before the sole producer callback retains or quarantines a slot.
struct RuntimeCatalogTensorTerminalProbeState {
  static void reset() {
    release_loan();
    const std::lock_guard lock(mutex);
    start_entered = false;
    start_released = false;
    loan_retained = false;
    fault_result_recorded = false;
    fault_commit_rejected = false;
    error.reset();
  }

  static void enter_start_gate() {
    std::unique_lock lock(mutex);
    start_entered = true;
    cv.notify_all();
    cv.wait(lock, [] { return start_released; });
  }

  [[nodiscard]] static bool wait_for_start() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 10s, [] { return start_entered; });
  }

  static void release_start() {
    {
      const std::lock_guard lock(mutex);
      start_released = true;
    }
    cv.notify_all();
  }

  static void retain(TensorOutputLoan loan) {
    {
      const std::lock_guard lock(mutex);
      retained_loan.emplace(std::move(loan));
      loan_retained = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_loan() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return loan_retained || error.has_value(); });
  }

  static void record_fault_result(const bool rejected) {
    {
      const std::lock_guard lock(mutex);
      fault_commit_rejected = rejected;
      fault_result_recorded = true;
      if (!rejected) {
        error = ErrorCode::kFailure;
      }
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_fault_result() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return fault_result_recorded || error.has_value(); });
  }

  static void fail(const ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  static void release_loan() {
    std::optional<TensorOutputLoan> released;
    {
      const std::lock_guard lock(mutex);
      if (retained_loan) {
        released.emplace(std::move(*retained_loan));
      }
      retained_loan.reset();
    }
    released.reset();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<TensorOutputLoan> retained_loan;
  static inline std::optional<ErrorCode> error;
  static inline bool start_entered{};
  static inline bool start_released{};
  static inline bool loan_retained{};
  static inline bool fault_result_recorded{};
  static inline bool fault_commit_rejected{};
};

class RuntimeCatalogRetainedTensorLoanSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  void start() override { RuntimeCatalogTensorTerminalProbeState::enter_start_gate(); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      RuntimeCatalogTensorTerminalProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    RuntimeCatalogTensorTerminalProbeState::retain(std::move(*loan));
    return {};
  }

  Output<Tensor> output;
};

class RuntimeCatalogQuarantinedTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  void start() override { RuntimeCatalogTensorTerminalProbeState::enter_start_gate(); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      RuntimeCatalogTensorTerminalProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      const ErrorCode code = writer ? ErrorCode::kFailure : writer.error().code;
      RuntimeCatalogTensorTerminalProbeState::fail(code);
      return make_unexpected(Error{code});
    }
    *writer->data_as<std::uint64_t>() = 1U;
    auto committed = std::move(*writer).commit();
    const bool rejected = !committed && committed.error().code == ErrorCode::kFailure;
    RuntimeCatalogTensorTerminalProbeState::record_fault_result(rejected);
    if (!rejected) {
      return make_unexpected(committed ? Error{ErrorCode::kFailure} : std::move(committed.error()));
    }
    return {};
  }

  Output<Tensor> output;
};

// ComputeCalls{0} closes the producer-local gate before its first callback, but required external
// Tensor outputs still receive the same explicit successor-sequence EOS as positive limits.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallsZeroPublishesRequiredCatalogTensorEndOfStreamWithoutCompute) {
  RuntimeZeroComputeCatalogTensorSourceOperator::reset();
  const std::string graph_name =
      unique_runtime_graph_name("runtime-zero-compute-required-catalog-tensor");
  const std::string endpoint_name = graph_name + "-topic";
  Graph graph{graph_name};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto source = graph.op<RuntimeZeroComputeCatalogTensorSourceOperator>("source");

  static constexpr std::array axes{TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  // The start gate keeps the already-materialized publisher live; this bound is only a finite
  // diagnostic limit for the public required-peer discovery path.
  topic_options.attach = TopicAttachPolicy{
      .peer = PeerRequirement::kRequired,
      .max_wait = 10s,
  };
  topic_options.bounds.subscriber = SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = axes},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
  graph.add_flow(source->output,
                 exported.input(),
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());

  // Runtime materializes the catalog publisher before lifecycle start. Hold that live publisher in
  // start() while the public required subscriber attaches, then release zero-limit EOS closure.
  RunSession session = run_async(plan);
  struct StartGateRelease {
    ~StartGateRelease() { RuntimeZeroComputeCatalogTensorSourceOperator::release_start(); }
  } const start_gate_release;
  const bool source_started = RuntimeZeroComputeCatalogTensorSourceOperator::wait_for_start();
  auto subscriber = [&]() -> expected<ChannelSubscriber<Tensor>, Error> {
    if (!source_started) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    return client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  }();
  if (!source_started || !subscriber) {
    session.request_stop();
  }
  RuntimeZeroComputeCatalogTensorSourceOperator::release_start();
  if (!source_started || !subscriber) {
    session.wait();
  }

  ASSERT_TRUE(source_started);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  expected<Sample<Tensor>, Error> terminal = make_unexpected(Error{ErrorCode::kNotReady});
  const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
  while (!terminal && terminal.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < terminal_deadline) {
    terminal = subscriber->receive_sample();
    if (!terminal) {
      std::this_thread::sleep_for(1ms);
    }
  }
  const bool observed_terminal = !terminal && terminal.error().code == ErrorCode::kClosed;
  if (!observed_terminal) {
    session.request_stop();
  }

  const auto completion_deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < completion_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool completed_without_timeout = !session.is_running();
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(observed_terminal) << (terminal ? "unexpected Tensor payload"
                                              : std::string("terminal receive failed: ") +
                                                    runtime_error_message(terminal.error()));
  EXPECT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(RuntimeZeroComputeCatalogTensorSourceOperator::compute_calls(), 0U);
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);

  auto repeated_terminal = subscriber->receive_sample();
  ASSERT_FALSE(repeated_terminal);
  EXPECT_EQ(repeated_terminal.error().code, ErrorCode::kClosed);
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 1U);
}

template <typename SourceOperator>
[[nodiscard]] ExecutionPlan make_runtime_catalog_terminal_tensor_plan(
    const std::string& graph_name, const std::span<const std::string> endpoint_names) {
  Graph graph{graph_name};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto source = graph.op<SourceOperator>("source");

  for (const std::string& endpoint_name : endpoint_names) {
    TopicOptions topic_options;
    topic_options.endpoint_name = endpoint_name;
    topic_options.attach = TopicAttachPolicy{
        .peer = PeerRequirement::kRequired,
        .max_wait = 10s,
    };
    topic_options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 1U,
        .max_borrowed_samples = 1U,
    };
    topic_options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 0U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    auto exported =
        graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
    graph.add_flow(source->output,
                   exported.input(),
                   ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
  }
  return compile(graph);
}

template <typename SourceOperator>
[[nodiscard]] ExecutionPlan make_runtime_catalog_terminal_tensor_plan(
    const std::string& graph_name, const std::string& endpoint_name) {
  return make_runtime_catalog_terminal_tensor_plan<SourceOperator>(
      graph_name, std::span<const std::string>{&endpoint_name, 1U});
}

// A terminal catalog close has no authority while an unpublished direct Tensor loan remains
// retained. The public subscriber must observe neither EOS nor completion before that loan drops.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RunSessionTest, CatalogTensorEndOfStreamWaitsForRetainedOutputLoanAcrossFanout) {
  RuntimeCatalogTensorTerminalProbeState::reset();
  const std::string graph_name =
      unique_runtime_graph_name("runtime-catalog-retained-unpublished-tensor-loan");
  const std::array endpoint_names{graph_name + "-first", graph_name + "-second"};
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan compiled =
      make_runtime_catalog_terminal_tensor_plan<RuntimeCatalogRetainedTensorLoanSourceOperator>(
          graph_name, endpoint_names);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());

  RunSession session = run_async(plan);
  struct SessionCleanup {
    RunSession* session{};
    ~SessionCleanup() {
      RuntimeCatalogTensorTerminalProbeState::release_start();
      RuntimeCatalogTensorTerminalProbeState::release_loan();
      if (session->is_running()) {
        session->request_stop();
      }
      session->wait();
    }
  } const cleanup{&session};

  const bool source_started = RuntimeCatalogTensorTerminalProbeState::wait_for_start();
  const auto create_subscriber =
      [&](const std::string& endpoint_name) -> expected<ChannelSubscriber<Tensor>, Error> {
    if (!source_started) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    return client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  };
  auto first_subscriber = create_subscriber(endpoint_names[0]);
  auto second_subscriber = create_subscriber(endpoint_names[1]);
  if (!source_started || !first_subscriber || !second_subscriber) {
    session.request_stop();
  }
  RuntimeCatalogTensorTerminalProbeState::release_start();
  if (!source_started || !first_subscriber || !second_subscriber) {
    session.wait();
  }

  ASSERT_TRUE(source_started);
  ASSERT_TRUE(first_subscriber) << runtime_error_message(first_subscriber.error());
  ASSERT_TRUE(second_subscriber) << runtime_error_message(second_subscriber.error());
  const std::array subscribers{&*first_subscriber, &*second_subscriber};
  ASSERT_TRUE(RuntimeCatalogTensorTerminalProbeState::wait_for_loan());
  ASSERT_TRUE(wait_for_runtime_phase_count(
      counters, detail::RuntimePhaseCounter::kActivationsCompleted, 1U));
  {
    const std::lock_guard lock(RuntimeCatalogTensorTerminalProbeState::mutex);
    ASSERT_FALSE(RuntimeCatalogTensorTerminalProbeState::error.has_value());
    ASSERT_TRUE(RuntimeCatalogTensorTerminalProbeState::retained_loan.has_value());
  }

  bool premature_terminal{};
  const auto no_eos_deadline = std::chrono::steady_clock::now() + 100ms;
  while (std::chrono::steady_clock::now() < no_eos_deadline) {
    for (ChannelSubscriber<Tensor>* const subscriber : subscribers) {
      auto observed = subscriber->receive_sample();
      if (!observed && observed.error().code == ErrorCode::kClosed) {
        premature_terminal = true;
        break;
      }
      EXPECT_FALSE(observed);
      if (!observed) {
        EXPECT_EQ(observed.error().code, ErrorCode::kNotReady);
      }
    }
    if (premature_terminal) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_FALSE(premature_terminal);
  for (ChannelSubscriber<Tensor>* const subscriber : subscribers) {
    EXPECT_EQ(
        detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber),
        0U);
  }
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);

  RuntimeCatalogTensorTerminalProbeState::release_loan();
  bool observed_every_terminal = true;
  for (ChannelSubscriber<Tensor>* const subscriber : subscribers) {
    expected<Sample<Tensor>, Error> terminal = make_unexpected(Error{ErrorCode::kNotReady});
    const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
    while (!terminal && terminal.error().code == ErrorCode::kNotReady &&
           std::chrono::steady_clock::now() < terminal_deadline) {
      terminal = subscriber->receive_sample();
      if (!terminal) {
        std::this_thread::sleep_for(1ms);
      }
    }
    observed_every_terminal =
        observed_every_terminal && !terminal && terminal.error().code == ErrorCode::kClosed;
  }
  if (!observed_every_terminal) {
    session.request_stop();
  }
  const auto completion_deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < completion_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool completed_without_timeout = !session.is_running();
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(observed_every_terminal);
  EXPECT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  for (ChannelSubscriber<Tensor>* const subscriber : subscribers) {
    EXPECT_EQ(
        detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber),
        1U);
  }
}

// A commit-readiness fault quarantines its unpublished slot. Completion must fail before the
// catalog binding consumes terminal authority, leaving the public subscriber with zero EOS.
// REQ: CHAN-136
// REQ: RUNTIME-050
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, CatalogTensorQuarantinedOutputFailsWithoutPublishingEndOfStream) {
  RuntimeCatalogTensorTerminalProbeState::reset();
  const std::string graph_name = unique_runtime_graph_name("runtime-catalog-quarantined-tensor");
  const std::string endpoint_name = graph_name + "-topic";
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  control->arm_fault(detail::RuntimeTensorOutputTestFault::kCommitReadinessFailure);
  const ExecutionPlan compiled =
      make_runtime_catalog_terminal_tensor_plan<RuntimeCatalogQuarantinedTensorSourceOperator>(
          graph_name, endpoint_name);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());

  RunSession session = run_async(plan);
  struct SessionCleanup {
    RunSession* session{};
    ~SessionCleanup() {
      RuntimeCatalogTensorTerminalProbeState::release_start();
      if (session->is_running()) {
        session->request_stop();
      }
      session->wait();
    }
  } const cleanup{&session};

  const bool source_started = RuntimeCatalogTensorTerminalProbeState::wait_for_start();
  auto subscriber = [&]() -> expected<ChannelSubscriber<Tensor>, Error> {
    if (!source_started) {
      return make_unexpected(Error{ErrorCode::kTimeout});
    }
    return client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  }();
  if (!source_started || !subscriber) {
    session.request_stop();
  }
  RuntimeCatalogTensorTerminalProbeState::release_start();
  if (!source_started || !subscriber) {
    session.wait();
  }

  ASSERT_TRUE(source_started);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(RuntimeCatalogTensorTerminalProbeState::wait_for_fault_result());
  {
    const std::lock_guard lock(RuntimeCatalogTensorTerminalProbeState::mutex);
    ASSERT_FALSE(RuntimeCatalogTensorTerminalProbeState::error.has_value());
    EXPECT_TRUE(RuntimeCatalogTensorTerminalProbeState::fault_commit_rejected);
  }
  const auto terminal_deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < terminal_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool failed_without_timeout = !session.is_running();
  if (!failed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(failed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  // This graph has a compile-concrete Tensor pool and no lifecycle rate candidate. The failed
  // readiness record therefore quarantines the output before completion can close its catalog
  // binding; QUEUE_REPLACEMENT_NOT_ADMITTED is reserved for lifecycle replacement admission.
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(
      detail::RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(*subscriber), 0U);
  const auto snapshot = control->snapshot(0U);
  constexpr std::uint32_t quarantine_claim =
      1U << static_cast<std::uint8_t>(TensorSlotClaimReason::kQuarantine);
  EXPECT_NE(snapshot.observed_claim_mask & quarantine_claim, 0U);
}

// Holds one already-admitted Tensor sink callback until the test explicitly releases it. The
// source callback can therefore reach its positive ComputeCalls limit while downstream work is
// still owned by Runtime.
struct RuntimeComputeCallTensorCallbackDrainProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    callback_entered = false;
    callback_returned = false;
    release_requested = false;
    observed_value = 0U;
    error.reset();
  }

  static void fail(const ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  [[nodiscard]] static expected<void, Error> enter_and_wait(const std::uint64_t value) {
    std::unique_lock lock(mutex);
    observed_value = value;
    callback_entered = true;
    cv.notify_all();
    if (!cv.wait_for(lock, 5s, [] { return release_requested; })) {
      error = ErrorCode::kFailure;
      callback_returned = true;
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    callback_returned = true;
    return {};
  }

  [[nodiscard]] static bool wait_for_callback_entry() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return callback_entered || error.has_value(); });
  }

  static void release_callback() {
    {
      const std::lock_guard lock(mutex);
      release_requested = true;
    }
    cv.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<ErrorCode> error;
  static inline std::uint64_t observed_value{};
  static inline bool callback_entered{};
  static inline bool callback_returned{};
  static inline bool release_requested{};
};

class RuntimeComputeCallTensorCallbackDrainSourceOperator final : public Operator<> {
 public:
  static constexpr std::uint64_t kValue = 39U;

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      RuntimeComputeCallTensorCallbackDrainProbeState::fail(emitted.error().code);
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static expected<void, Error> fail(const ErrorCode code) {
    RuntimeComputeCallTensorCallbackDrainProbeState::fail(code);
    return make_unexpected(Error{code});
  }
};

class RuntimeComputeCallTensorCallbackDrainSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{.representation = {
                                            .memory_kind = MemoryKind::kHost,
                                            .dtype = DLDataType{kDLUInt, 64U, 1U},
                                            .rank = 0U,
                                        }});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      const ErrorCode code = sample ? ErrorCode::kFailure : sample.error().code;
      RuntimeComputeCallTensorCallbackDrainProbeState::fail(code);
      return make_unexpected(Error{code});
    }
    return RuntimeComputeCallTensorCallbackDrainProbeState::enter_and_wait(
        *sample->data.data_as<std::uint64_t>());
  }

  Input<Tensor> input;
};

// Retains one unpublished direct Tensor loan after its limit-reaching source callback returns.
// Destroying the loan is the only test action that releases its producer claim.
struct RuntimeComputeCallRetainedTensorLoanDrainProbeState {
  static void reset() {
    release_loan();
    const std::lock_guard lock(mutex);
    loan_retained = false;
    sink_compute_calls.store(0U, std::memory_order_release);
    error.reset();
  }

  static void retain(TensorOutputLoan loan) {
    {
      const std::lock_guard lock(mutex);
      retained_loan.emplace(std::move(loan));
      loan_retained = true;
    }
    cv.notify_all();
  }

  static void fail(const ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_loan() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return loan_retained || error.has_value(); });
  }

  static void release_loan() {
    std::optional<TensorOutputLoan> released;
    {
      const std::lock_guard lock(mutex);
      if (retained_loan) {
        released.emplace(std::move(*retained_loan));
      }
      retained_loan.reset();
    }
    released.reset();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<TensorOutputLoan> retained_loan;
  static inline std::optional<ErrorCode> error;
  static inline std::atomic<std::uint32_t> sink_compute_calls{};
  static inline bool loan_retained{};
};

class RuntimeComputeCallRetainedTensorLoanDrainSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      RuntimeComputeCallRetainedTensorLoanDrainProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    RuntimeComputeCallRetainedTensorLoanDrainProbeState::retain(std::move(*loan));
    return {};
  }

  Output<Tensor> output;
};

class RuntimeComputeCallRetainedTensorLoanDrainSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{.representation = {
                                            .memory_kind = MemoryKind::kHost,
                                            .dtype = DLDataType{kDLUInt, 64U, 1U},
                                            .rank = 0U,
                                        }});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RuntimeComputeCallRetainedTensorLoanDrainProbeState::sink_compute_calls.fetch_add(
        1U, std::memory_order_acq_rel);
    RuntimeComputeCallRetainedTensorLoanDrainProbeState::fail(ErrorCode::kFailure);
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  Input<Tensor> input;
};

// A positive-limit producer has already returned successfully, but an admitted downstream
// callback still owns the final Tensor. Natural completion must remain unfrozen until it returns.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionWaitsForAdmittedTensorCallback) {
  RuntimeComputeCallTensorCallbackDrainProbeState::reset();
  Graph graph{unique_runtime_graph_name("runtime-compute-call-admitted-tensor-callback")};
  const auto source = graph.op<RuntimeComputeCallTensorCallbackDrainSourceOperator>("source");
  const auto sink = graph.op<RuntimeComputeCallTensorCallbackDrainSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  struct SessionCleanup {
    RunSession* session{};
    ~SessionCleanup() {
      RuntimeComputeCallTensorCallbackDrainProbeState::release_callback();
      if (session->is_running()) {
        session->request_stop();
      }
      session->wait();
    }
  } const cleanup{&session};

  ASSERT_TRUE(RuntimeComputeCallTensorCallbackDrainProbeState::wait_for_callback_entry());
  {
    const std::lock_guard lock(RuntimeComputeCallTensorCallbackDrainProbeState::mutex);
    ASSERT_FALSE(RuntimeComputeCallTensorCallbackDrainProbeState::error.has_value());
    EXPECT_TRUE(RuntimeComputeCallTensorCallbackDrainProbeState::callback_entered);
    EXPECT_FALSE(RuntimeComputeCallTensorCallbackDrainProbeState::callback_returned);
    EXPECT_EQ(RuntimeComputeCallTensorCallbackDrainProbeState::observed_value,
              RuntimeComputeCallTensorCallbackDrainSourceOperator::kValue);
  }
  ASSERT_TRUE(wait_for_runtime_phase_count(
      counters, detail::RuntimePhaseCounter::kActivationsCompleted, 1U));
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);
  EXPECT_TRUE(session.operator_completion_progress().empty());

  RuntimeComputeCallTensorCallbackDrainProbeState::release_callback();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.configured_limit, 1U);
  EXPECT_EQ(progress.successful_compute_calls, 1U);
  EXPECT_TRUE(progress.limit_reached);
  const std::lock_guard lock(RuntimeComputeCallTensorCallbackDrainProbeState::mutex);
  EXPECT_TRUE(RuntimeComputeCallTensorCallbackDrainProbeState::callback_returned);
  EXPECT_FALSE(RuntimeComputeCallTensorCallbackDrainProbeState::error.has_value());
}

// Output closure is backpressured by an unpublished Tensor loan even after the source's sole
// successful compute call is fully retired. Releasing that claim permits natural completion.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionWaitsForRetainedTensorOutputLoan) {
  RuntimeComputeCallRetainedTensorLoanDrainProbeState::reset();
  Graph graph{unique_runtime_graph_name("runtime-compute-call-retained-tensor-loan")};
  const auto source = graph.op<RuntimeComputeCallRetainedTensorLoanDrainSourceOperator>("source");
  const auto sink = graph.op<RuntimeComputeCallRetainedTensorLoanDrainSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  struct SessionCleanup {
    RunSession* session{};
    ~SessionCleanup() {
      RuntimeComputeCallRetainedTensorLoanDrainProbeState::release_loan();
      if (session->is_running()) {
        session->request_stop();
      }
      session->wait();
    }
  } const cleanup{&session};

  ASSERT_TRUE(RuntimeComputeCallRetainedTensorLoanDrainProbeState::wait_for_loan());
  {
    const std::lock_guard lock(RuntimeComputeCallRetainedTensorLoanDrainProbeState::mutex);
    ASSERT_FALSE(RuntimeComputeCallRetainedTensorLoanDrainProbeState::error.has_value());
    ASSERT_TRUE(RuntimeComputeCallRetainedTensorLoanDrainProbeState::retained_loan.has_value());
  }
  ASSERT_TRUE(wait_for_runtime_phase_count(
      counters, detail::RuntimePhaseCounter::kActivationsCompleted, 1U));
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);
  EXPECT_TRUE(session.operator_completion_progress().empty());

  RuntimeComputeCallRetainedTensorLoanDrainProbeState::release_loan();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.configured_limit, 1U);
  EXPECT_EQ(progress.successful_compute_calls, 1U);
  EXPECT_TRUE(progress.limit_reached);
  EXPECT_EQ(RuntimeComputeCallRetainedTensorLoanDrainProbeState::sink_compute_calls.load(
                std::memory_order_acquire),
            0U);
  const std::lock_guard lock(RuntimeComputeCallRetainedTensorLoanDrainProbeState::mutex);
  EXPECT_FALSE(RuntimeComputeCallRetainedTensorLoanDrainProbeState::retained_loan.has_value());
  EXPECT_FALSE(RuntimeComputeCallRetainedTensorLoanDrainProbeState::error.has_value());
}

// Managed storage retains its physical CUDA ordinal in the private plan, but DLPack requires the
// public device id to remain zero even when that backing was allocated from another GPU context.
// REQ: TENSOR-STORAGE-009
TEST(RuntimeTensorResources, ManagedDlpackMetadataDoesNotExposeThePhysicalCudaOrdinal) {
  constexpr DLDevice managed =
      detail::runtime_tensor_dl_device(MemoryKind::kCudaManaged, DeviceId{7});
  EXPECT_EQ(managed.device_type, kDLCUDAManaged);
  EXPECT_EQ(managed.device_id, 0);

  constexpr DLDevice device =
      detail::runtime_tensor_dl_device(MemoryKind::kCudaDevice, DeviceId{7});
  EXPECT_EQ(device.device_type, kDLCUDA);
  EXPECT_EQ(device.device_id, 7);
}

// Central probe singleton for direct-Tensor output tests. Captures pool
// alignment, exhaustion, retirement, and scalar-metadata flags set by
// RuntimeDirectTensorSourceOperator and records the published pointer and the
// sink's retained Tensor for cross-operator comparison.
// Used by: DirectTensorPoolPreservesAlignmentExhaustionRetirementAndScalarMetadata
struct RuntimeDirectTensorProbeState {
  static void reset(std::shared_ptr<detail::RuntimeTensorOutputTestControl> test_control = {},
                    MemoryKind selected_memory_kind = MemoryKind::kHost) {
    const std::lock_guard lock(mutex);
    control = std::move(test_control);
    memory_kind = selected_memory_kind;
    if (control) {
      control->set_metric_time_ns(0U);
    }
    source_completed = false;
    source_initial_completed = false;
    sink_completed = false;
    aligned_slots = false;
    exhausted = false;
    canceled_ready_slot = false;
    rejected_invalid_layout = false;
    scalar_metadata = false;
    host_copy_supported = false;
    active_writer_retained = false;
    published_pointer_preserved = false;
    retained_pointer_not_reused = false;
    final_claim_released_for_reuse = false;
    pre_submit_backpressured = false;
    validation_precedence = false;
    warm_path_allocations = std::numeric_limits<std::size_t>::max();
    observed_value = 0U;
    published_pointer = nullptr;
    retained_tensor.reset();
    source_error.reset();
  }

  static void record_initial_source(bool aligned, bool observed_exhaustion, bool canceled_ready,
                                    bool rejected_layout, bool active_writer_safe,
                                    bool backpressured, bool precedence, void* pointer) {
    {
      const std::lock_guard lock(mutex);
      aligned_slots = aligned;
      exhausted = observed_exhaustion;
      canceled_ready_slot = canceled_ready;
      rejected_invalid_layout = rejected_layout;
      active_writer_retained = active_writer_safe;
      pre_submit_backpressured = backpressured;
      validation_precedence = precedence;
      published_pointer = pointer;
      source_initial_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool sink_retains_published_tensor() {
    const std::lock_guard lock(mutex);
    return retained_tensor.has_value();
  }

  [[nodiscard]] static const void* published_data_pointer() {
    const std::lock_guard lock(mutex);
    return published_pointer;
  }

  // Returns the memory kind selected in the most recent reset() call.
  [[nodiscard]] static MemoryKind selected_memory_kind() {
    const std::lock_guard lock(mutex);
    return memory_kind;
  }

  static void release_retained_tensor() {
    const std::lock_guard lock(mutex);
    retained_tensor.reset();
  }

  static void complete_source_retirement_probe(bool unavailable_while_retained,
                                               bool reusable_after_release,
                                               std::size_t allocations) {
    {
      const std::lock_guard lock(mutex);
      retained_pointer_not_reused = unavailable_while_retained;
      final_claim_released_for_reuse = reusable_after_release;
      warm_path_allocations = allocations;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void fail_source(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      source_error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void record_sink(const Sample<Tensor>& sample, bool scalar, bool copied,
                          std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      scalar_metadata = scalar;
      host_copy_supported = copied;
      published_pointer_preserved = sample.data.data() == published_pointer;
      observed_value = value;
      retained_tensor = sample.data;
      sink_completed = true;
    }
    cv.notify_all();
  }

  static void set_metric_time_ns(std::uint64_t value) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected;
    {
      const std::lock_guard lock(mutex);
      selected = control;
    }
    if (selected) {
      selected->set_metric_time_ns(value);
    }
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return source_completed && sink_completed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::shared_ptr<detail::RuntimeTensorOutputTestControl> control;
  static inline MemoryKind memory_kind{MemoryKind::kHost};
  static inline bool source_completed{};
  static inline bool source_initial_completed{};
  static inline bool sink_completed{};
  static inline bool aligned_slots{};
  static inline bool exhausted{};
  static inline bool canceled_ready_slot{};
  static inline bool rejected_invalid_layout{};
  static inline bool scalar_metadata{};
  static inline bool host_copy_supported{};  // true if sink's copy_to_host() succeeded
  static inline bool active_writer_retained{};
  static inline bool published_pointer_preserved{};
  static inline bool retained_pointer_not_reused{};
  static inline bool final_claim_released_for_reuse{};
  static inline bool pre_submit_backpressured{};
  static inline bool validation_precedence{};
  static inline std::size_t warm_path_allocations{};
  static inline std::uint64_t observed_value{};
  static inline const void* published_pointer{};
  static inline std::optional<Tensor> retained_tensor;
  static inline std::optional<ErrorCode> source_error;
};

struct RuntimeTensorContractAdmissionProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    completed = false;
    error.reset();
  }

  static void finish(std::optional<ErrorCode> result = std::nullopt) {
    {
      const std::lock_guard lock(mutex);
      completed = true;
      error = result;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return completed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool completed{};
  static inline std::optional<ErrorCode> error;
};

// Captures the lifecycle ordering and published rate observed by the no-maximum
// Tensor candidate fixture. The private pool observer lets the user Allocate
// callback prove that Core completed physical materialization first.
struct RuntimeLifecycleTensorCandidateProbeState {
  static void reset(std::shared_ptr<detail::RuntimeTensorOutputTestControl> test_control,
                    bool skip_discover_binding = false) {
    const std::lock_guard lock(mutex);
    control = std::move(test_control);
    allocate_saw_pool = false;
    allocate_rate = {};
    compute_rate = {};
    received = false;
    observed_value = 0U;
    error.reset();
    omit_discover_binding = skip_discover_binding;
  }

  static void record_allocate(const PortRateView& rate) {
    const std::lock_guard lock(mutex);
    allocate_rate = rate;
    allocate_saw_pool = control != nullptr && control->snapshot(0U).maximum_pool_generation != 0U;
  }

  static void record_compute(const PortRateView& rate) {
    const std::lock_guard lock(mutex);
    compute_rate = rate;
  }

  static void record_sink(std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      observed_value = value;
      received = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return received || error.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::shared_ptr<detail::RuntimeTensorOutputTestControl> control;
  static inline bool allocate_saw_pool{};
  static inline PortRateView allocate_rate;
  static inline PortRateView compute_rate;
  static inline bool received{};
  static inline std::uint64_t observed_value{};
  static inline std::optional<ErrorCode> error;
  static inline bool omit_discover_binding{};
};

class RuntimeFixedRateTensorEvidenceSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(Rate::hz(60U))
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    auto rate = context.rate(output);
    if (!rate) {
      return fail(rate.error().code);
    }
    RuntimeLifecycleTensorCandidateProbeState::record_compute(*rate);
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto published = output.emit(std::move(*loan));
    if (!published) {
      return fail(published.error().code);
    }
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kValue = 0x4652455348524154ULL;

 private:
  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeLifecycleTensorCandidateProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  bool emitted_{};
};

struct RuntimeMixedStageTensorProbeState {
  static void reset(std::shared_ptr<detail::RuntimeTensorOutputTestControl> test_control) {
    const std::lock_guard lock(mutex);
    control = std::move(test_control);
    allocate_rate = {};
    allocate_saw_pool = false;
    allocated = false;
    error.reset();
  }

  static void record_allocate(const PortRateView& rate) {
    {
      const std::lock_guard lock(mutex);
      allocate_rate = rate;
      allocate_saw_pool = control != nullptr && control->snapshot(1U).maximum_pool_generation != 0U;
      allocated = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return allocated || error.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::shared_ptr<detail::RuntimeTensorOutputTestControl> control;
  static inline PortRateView allocate_rate;
  static inline bool allocate_saw_pool{};
  static inline bool allocated{};
  static inline std::optional<ErrorCode> error;
};

class RuntimeArmFiniteMaximumTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kArm)
        .max_nominal_rate(Rate::hz(120U))
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle()
        .stage(LifecycleStage::kArm, &RuntimeArmFiniteMaximumTensorSourceOperator::bind_rate)
        .stage(LifecycleStage::kAllocate, &RuntimeArmFiniteMaximumTensorSourceOperator::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus bind_rate(LifecycleContext& context) noexcept {
    auto bound = context.bind_nominal_rate(output, Rate::hz(60U));
    if (!bound) {
      RuntimeMixedStageTensorProbeState::fail(bound.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    auto rate = context.rate(output);
    if (!rate) {
      RuntimeMixedStageTensorProbeState::fail(rate.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    RuntimeMixedStageTensorProbeState::record_allocate(*rate);
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;
};

class RuntimeLifecycleTensorCandidateSourceOperator final : public Operator<> {
 public:
  explicit RuntimeLifecycleTensorCandidateSourceOperator(
      LifecycleStage binding_stage = LifecycleStage::kDiscover)
      : binding_stage_(binding_stage) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(binding_stage_)
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle()
        .stage(binding_stage_, &RuntimeLifecycleTensorCandidateSourceOperator::bind_rate)
        .stage(LifecycleStage::kAllocate, &RuntimeLifecycleTensorCandidateSourceOperator::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    auto rate = context.rate(output);
    if (!rate) {
      return fail(rate.error().code);
    }
    RuntimeLifecycleTensorCandidateProbeState::record_compute(*rate);
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  LifecycleStatus bind_rate(LifecycleContext& context) noexcept {
    {
      const std::lock_guard lock(RuntimeLifecycleTensorCandidateProbeState::mutex);
      if (RuntimeLifecycleTensorCandidateProbeState::omit_discover_binding) {
        return LifecycleStatus::kOk;
      }
    }
    auto bound = context.bind_nominal_rate(output, Rate::hz(60U));
    if (!bound) {
      RuntimeLifecycleTensorCandidateProbeState::fail(bound.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    auto rate = context.rate(output);
    if (!rate) {
      RuntimeLifecycleTensorCandidateProbeState::fail(rate.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    RuntimeLifecycleTensorCandidateProbeState::record_allocate(*rate);
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kValue = 0x4155544F52415445ULL;

 private:
  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeLifecycleTensorCandidateProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  LifecycleStage binding_stage_{LifecycleStage::kDiscover};
  bool emitted_{};
};

// A remote no-maximum provider must remain entirely inactive when its process group is not
// launched. These counters distinguish an omitted lifecycle binding/materialization path from an
// ordinary successful local Tensor run.
struct RuntimeRemoteLifecycleTensorWorkerProbe {
  static void reset() noexcept {
    bind_calls.store(0U, std::memory_order_release);
    allocate_calls.store(0U, std::memory_order_release);
    compute_calls.store(0U, std::memory_order_release);
  }

  static inline std::atomic<std::uint32_t> bind_calls{};
  static inline std::atomic<std::uint32_t> allocate_calls{};
  static inline std::atomic<std::uint32_t> compute_calls{};
};

class RuntimeRemoteNoMaximumLifecycleTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle()
        .stage(LifecycleStage::kDiscover,
               &RuntimeRemoteNoMaximumLifecycleTensorSourceOperator::bind_rate)
        .stage(LifecycleStage::kAllocate,
               &RuntimeRemoteNoMaximumLifecycleTensorSourceOperator::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    RuntimeRemoteLifecycleTensorWorkerProbe::compute_calls.fetch_add(1U, std::memory_order_acq_rel);
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  LifecycleStatus bind_rate(LifecycleContext&) noexcept {
    RuntimeRemoteLifecycleTensorWorkerProbe::bind_calls.fetch_add(1U, std::memory_order_acq_rel);
    return LifecycleStatus::kFatalFailure;
  }

  LifecycleStatus allocate(LifecycleContext&) noexcept {
    RuntimeRemoteLifecycleTensorWorkerProbe::allocate_calls.fetch_add(1U,
                                                                      std::memory_order_acq_rel);
    return LifecycleStatus::kFatalFailure;
  }

  Output<Tensor> output;
};

class RuntimeLifecycleTensorCandidateSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U}},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      const ErrorCode code = sample ? ErrorCode::kFailure : sample.error().code;
      RuntimeLifecycleTensorCandidateProbeState::fail(code);
      return make_unexpected(Error{code});
    }
    RuntimeLifecycleTensorCandidateProbeState::record_sink(*sample->data.data_as<std::uint64_t>());
    return {};
  }

  Input<Tensor> input;
};

struct RuntimeLifecycleTensorRateRestartProbeState {
  struct RestartTiming {
    std::int64_t request_to_discover_ns{};
    std::int64_t discover_to_allocate_ns{};
    std::int64_t allocate_to_successor_compute_ns{};
    std::int64_t request_to_successor_compute_ns{};
  };

  static void reset(Rate selected_restart_rate, Rate selected_initial_rate = Rate::hz(60U)) {
    const std::lock_guard lock(mutex);
    initial_rate = selected_initial_rate;
    restart_rate = selected_restart_rate;
    discover_count = 0U;
    sink_count = 0U;
    first_value = 0U;
    second_value = 0U;
    first_compute_rate = {};
    second_compute_rate = {};
    allocate_count = 0U;
    restart_requested_at = {};
    restart_discover_at = {};
    restart_allocate_at = {};
    successor_compute_at = {};
    restart_accepted = false;
    source_completed = false;
    error.reset();
  }

  [[nodiscard]] static Rate discover_rate() {
    const std::lock_guard lock(mutex);
    const auto callback_index = discover_count++;
    if (callback_index == 0U) {
      return initial_rate;
    }
    if (callback_index == 1U) {
      restart_discover_at = std::chrono::steady_clock::now();
    }
    return restart_rate;
  }

  static void record_compute(const PortRateView& rate, bool successor) {
    const std::lock_guard lock(mutex);
    (successor ? second_compute_rate : first_compute_rate) = rate;
    if (successor) {
      successor_compute_at = std::chrono::steady_clock::now();
    }
  }

  static void record_restart_requested() {
    const std::lock_guard lock(mutex);
    restart_requested_at = std::chrono::steady_clock::now();
  }

  static void record_allocate() {
    const std::lock_guard lock(mutex);
    const auto callback_index = allocate_count++;
    if (callback_index == 1U) {
      restart_allocate_at = std::chrono::steady_clock::now();
    }
  }

  static void record_restart(bool accepted) {
    const std::lock_guard lock(mutex);
    restart_accepted = accepted;
  }

  static void record_source_complete() {
    {
      const std::lock_guard lock(mutex);
      source_completed = true;
    }
    cv.notify_all();
  }

  static void record_sink(std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      if (sink_count++ == 0U) {
        first_value = value;
      } else {
        second_value = value;
      }
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool first_sample_observed() {
    const std::lock_guard lock(mutex);
    return sink_count != 0U;
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [] { return error.has_value() || (source_completed && sink_count >= 2U); });
  }

  [[nodiscard]] static std::optional<RestartTiming> restart_timing() {
    const std::lock_guard lock(mutex);
    if (restart_requested_at == std::chrono::steady_clock::time_point() ||
        restart_discover_at == std::chrono::steady_clock::time_point() ||
        restart_allocate_at == std::chrono::steady_clock::time_point() ||
        successor_compute_at == std::chrono::steady_clock::time_point()) {
      return std::nullopt;
    }
    const auto elapsed = [](const auto begin, const auto end) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    };
    return RestartTiming{
        .request_to_discover_ns = elapsed(restart_requested_at, restart_discover_at),
        .discover_to_allocate_ns = elapsed(restart_discover_at, restart_allocate_at),
        .allocate_to_successor_compute_ns = elapsed(restart_allocate_at, successor_compute_at),
        .request_to_successor_compute_ns = elapsed(restart_requested_at, successor_compute_at),
    };
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline Rate initial_rate{Rate::hz(60U)};
  static inline Rate restart_rate{Rate::hz(30U)};
  static inline std::uint32_t discover_count{};
  static inline std::size_t sink_count{};
  static inline std::uint64_t first_value{};
  static inline std::uint64_t second_value{};
  static inline PortRateView first_compute_rate;
  static inline PortRateView second_compute_rate;
  static inline std::uint32_t allocate_count{};
  static inline std::chrono::steady_clock::time_point restart_requested_at;
  static inline std::chrono::steady_clock::time_point restart_discover_at;
  static inline std::chrono::steady_clock::time_point restart_allocate_at;
  static inline std::chrono::steady_clock::time_point successor_compute_at;
  static inline bool restart_accepted{};
  static inline bool source_completed{};
  static inline std::optional<ErrorCode> error;
};

class RuntimeLifecycleTensorRateRestartSourceOperator final : public Operator<> {
 public:
  explicit RuntimeLifecycleTensorRateRestartSourceOperator(
      std::optional<Rate> maximum_rate = std::nullopt,
      LifecycleStage binding_stage = LifecycleStage::kDiscover,
      LifecycleStage restart_from = LifecycleStage::kDiscover,
      MemoryKind memory_kind = MemoryKind::kHost) noexcept
      : maximum_rate_(maximum_rate),
        binding_stage_(binding_stage),
        restart_from_(restart_from),
        memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output").lifecycle_nominal_rate(binding_stage_);
    if (maximum_rate_) {
      builder.max_nominal_rate(*maximum_rate_);
    }
    builder.max_emits_per_compute(1U).produces_tensor(TensorOutputSpec{
        .representation = {.memory_kind = memory_kind_,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 0U},
        .bounds = tensor_bounds(8U),
        .storage = TensorOutputStorage::kRuntimePool,
    });
    spec.lifecycle()
        .stage(binding_stage_, &RuntimeLifecycleTensorRateRestartSourceOperator::discover)
        .stage(LifecycleStage::kAllocate,
               &RuntimeLifecycleTensorRateRestartSourceOperator::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (phase_ == 0U) {
      auto emitted = emit_value(kFirstValue, context, false);
      if (emitted) {
        phase_ = 1U;
      }
      return emitted;
    }
    if (phase_ == 1U) {
      if (!RuntimeLifecycleTensorRateRestartProbeState::first_sample_observed()) {
        return {};
      }
      RuntimeLifecycleTensorRateRestartProbeState::record_restart_requested();
      auto receipt = context.request_restart(LifecycleRestartOptions{.from_stage = restart_from_});
      RuntimeLifecycleTensorRateRestartProbeState::record_restart(receipt.has_value());
      if (!receipt) {
        return fail(receipt.error().code);
      }
      phase_ = 2U;
      return {};
    }
    if (phase_ == 2U) {
      auto emitted = emit_value(kSecondValue, context, true);
      if (emitted) {
        phase_ = 3U;
        RuntimeLifecycleTensorRateRestartProbeState::record_source_complete();
      }
      return emitted;
    }
    return {};
  }

  LifecycleStatus discover(LifecycleContext& context) noexcept {
    auto bound = context.bind_nominal_rate(
        output, RuntimeLifecycleTensorRateRestartProbeState::discover_rate());
    if (!bound) {
      RuntimeLifecycleTensorRateRestartProbeState::fail(bound.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus allocate(LifecycleContext&) noexcept {
    RuntimeLifecycleTensorRateRestartProbeState::record_allocate();
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] expected<void, Error> emit_value(std::uint64_t value, ExecutionContext& context,
                                                 bool successor) {
    auto rate = context.rate(output);
    if (!rate) {
      return fail(rate.error().code);
    }
    RuntimeLifecycleTensorRateRestartProbeState::record_compute(*rate, successor);
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = value;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeLifecycleTensorRateRestartProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  static constexpr std::uint64_t kFirstValue = 11U;
  static constexpr std::uint64_t kSecondValue = 22U;
  std::uint32_t phase_{};
  std::optional<Rate> maximum_rate_;
  LifecycleStage binding_stage_{LifecycleStage::kDiscover};
  LifecycleStage restart_from_{LifecycleStage::kDiscover};
  MemoryKind memory_kind_{MemoryKind::kHost};
};

class RuntimeLifecycleTensorRateRestartSinkOperator final : public Operator<> {
 public:
  explicit RuntimeLifecycleTensorRateRestartSinkOperator(
      MemoryKind memory_kind = MemoryKind::kHost) noexcept
      : memory_kind_(memory_kind) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = memory_kind_, .dtype = DLDataType{kDLUInt, 64U, 1U}},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 500ms}).input(input, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto batch = input.selected_batch();
    if (!batch) {
      RuntimeLifecycleTensorRateRestartProbeState::fail(batch.error().code);
      return make_unexpected(std::move(batch.error()));
    }
    for (const auto& sample : *batch) {
      if (sample.data.data_as<std::uint64_t>() == nullptr) {
        RuntimeLifecycleTensorRateRestartProbeState::fail(ErrorCode::kFailure);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      RuntimeLifecycleTensorRateRestartProbeState::record_sink(
          *sample.data.data_as<std::uint64_t>());
    }
    return {};
  }

  Input<Tensor> input;

 private:
  MemoryKind memory_kind_{MemoryKind::kHost};
};

class RuntimeTensorContractAdmissionSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    constexpr std::array<std::int64_t, 2U> contiguous_maximum_shape{2, 3};
    constexpr std::array bounded_axes{
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 2U},
        TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 1U},
    };
    spec.output(contiguous, "contiguous")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 2U},
            .bounds = tensor_bounds(contiguous_maximum_shape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(bounded, "bounded")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 2U,
                               .strides = TensorStridePolicy::kBoundedStrided},
            .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 5U, .axes = bounded_axes},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (completed_) {
      return {};
    }
    completed_ = true;
    constexpr DLDataType kDtype{kDLUInt, 8U, 1U};
    constexpr std::array<std::int64_t, 2U> in_range_shape{2, 2};
    constexpr std::array<std::int64_t, 2U> maximum_shape{2, 3};
    constexpr std::array<std::int64_t, 2U> axis_overflow_shape{3, 2};
    constexpr std::array<std::int64_t, 2U> padded_strides{3, 1};
    constexpr std::array<std::int64_t, 2U> contiguous_strides{2, 1};

    const auto noncontiguous = contiguous.allocate_tensor(TensorLoanRequest{
        .shape = in_range_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{padded_strides},
    });
    const auto oversized_axis = contiguous.allocate_tensor(TensorLoanRequest{
        .shape = axis_overflow_shape,
        .dtype = kDtype,
    });
    const auto oversized_stride = bounded.allocate_tensor(TensorLoanRequest{
        .shape = in_range_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{padded_strides},
    });
    const bool rejected =
        !noncontiguous && noncontiguous.error().code == ErrorCode::kInvalidArgument &&
        !oversized_axis && oversized_axis.error().code == ErrorCode::kInvalidArgument &&
        !oversized_stride && oversized_stride.error().code == ErrorCode::kInvalidArgument;

    auto valid_contiguous = contiguous.allocate_tensor(maximum_shape);
    auto valid_bounded = bounded.allocate_tensor(TensorLoanRequest{
        .shape = in_range_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{contiguous_strides},
    });
    if (!rejected || !valid_contiguous || !valid_bounded) {
      RuntimeTensorContractAdmissionProbeState::finish(ErrorCode::kFailure);
      return {};
    }
    const auto finish_loan = [](TensorOutputLoan loan) -> expected<void, Error> {
      auto writer = loan.write_host();
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return committed;
      }
      return std::move(loan).cancel();
    };
    auto first_finished = finish_loan(std::move(*valid_contiguous));
    auto second_finished = finish_loan(std::move(*valid_bounded));
    if (!first_finished || !second_finished) {
      RuntimeTensorContractAdmissionProbeState::finish(
          !first_finished ? first_finished.error().code : second_finished.error().code);
      return {};
    }
    RuntimeTensorContractAdmissionProbeState::finish();
    return {};
  }

  Output<Tensor> contiguous;
  Output<Tensor> bounded;

 private:
  bool completed_{};
};

// Exercises every code path of the direct Tensor output API in a single compute
// call: invalid layouts, active-writer confinement, alignment verification, pool
// exhaustion, cancellation of ready slots, scalar publishing, and backpressure.
// On subsequent calls it probes slot-retirement ordering while the sink retains
// a reference to the published slot.
// Used by: DirectTensorPoolPreservesAlignmentExhaustionRetirementAndScalarMetadata
class RuntimeDirectTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(2U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = RuntimeDirectTensorProbeState::selected_memory_kind(),
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            // Deliberately use an odd span. Each physical slot must still preserve alignment.
            .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 17U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      if (!RuntimeDirectTensorProbeState::sink_retains_published_tensor()) {
        return {};
      }
      auto retained_probe = output.allocate_tensor(TensorShapeView{});
      if (!retained_probe) {
        return fail_source(retained_probe.error().code);
      }
      auto retained_writer = retained_probe->write_host();
      if (!retained_writer) {
        return fail_source(retained_writer.error().code);
      }
      const bool not_reused_while_retained =
          retained_writer->data() != RuntimeDirectTensorProbeState::published_data_pointer();
      auto retained_commit = std::move(*retained_writer).commit();
      if (!retained_commit) {
        return fail_source(retained_commit.error().code);
      }
      auto retained_cancel = std::move(*retained_probe).cancel();
      if (!retained_cancel) {
        return fail_source(retained_cancel.error().code);
      }

      RuntimeDirectTensorProbeState::release_retained_tensor();
      bool reused_after_release = false;
      std::size_t warm_path_allocations = std::numeric_limits<std::size_t>::max();
      {
        const test::ScopedAllocationCounter allocations;
        auto reusable = output.allocate_tensor(TensorShapeView{});
        if (!reusable) {
          return fail_source(reusable.error().code);
        }
        auto reusable_writer = reusable->write_host();
        if (!reusable_writer) {
          return fail_source(reusable_writer.error().code);
        }
        reused_after_release =
            reusable_writer->data() == RuntimeDirectTensorProbeState::published_data_pointer();
        auto reusable_commit = std::move(*reusable_writer).commit();
        if (!reusable_commit) {
          return fail_source(reusable_commit.error().code);
        }
        auto reusable_cancel = std::move(*reusable).cancel();
        if (!reusable_cancel) {
          return fail_source(reusable_cancel.error().code);
        }
        warm_path_allocations = allocations.count();
      }
      RuntimeDirectTensorProbeState::complete_source_retirement_probe(
          not_reused_while_retained, reused_after_release, warm_path_allocations);
      return {};
    }
    emitted_ = true;
    const auto fail = [](ErrorCode code) -> expected<void, Error> {
      RuntimeDirectTensorProbeState::fail_source(code);
      return make_unexpected(Error{code});
    };

    constexpr DLDataType kDtype{kDLUInt, 64U, 1U};
    const std::array<std::int64_t, 1U> vector_shape{2};
    const std::array<std::int64_t, 1U> valid_stride{8};
    const std::array<std::int64_t, 1U> invalid_stride{3};
    const std::array<std::int64_t, 1U> negative_axis{-1};
    const std::array<std::int64_t, 2U> zero_then_negative_axis{0, -1};
    const std::array<std::int64_t, 2U> overlapping_shape{2, 2};
    const std::array<std::int64_t, 2U> overlapping_strides{0, 8};
    const std::array<std::int64_t, 3U> excessive_rank{1, 1, 1};
    const auto invalid_code = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = DLDataType{0xFFU, 8U, 1U},
    });
    const auto wrong_dtype = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = DLDataType{kDLFloat, 64U, 1U},
    });
    const auto invalid_axis_stride = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{invalid_stride},
    });
    const auto invalid_rank = output.allocate_tensor(TensorLoanRequest{
        .shape = excessive_rank,
        .dtype = kDtype,
    });
    const auto invalid_axis = output.allocate_tensor(TensorLoanRequest{
        .shape = negative_axis,
        .dtype = kDtype,
    });
    const auto invalid_trailing_axis = output.allocate_tensor(TensorLoanRequest{
        .shape = zero_then_negative_axis,
        .dtype = kDtype,
    });
    const auto overlapping_layout = output.allocate_tensor(TensorLoanRequest{
        .shape = overlapping_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{overlapping_strides},
    });
    const bool rejected_layout =
        !invalid_code && invalid_code.error().code == ErrorCode::kInvalidArgument && !wrong_dtype &&
        wrong_dtype.error().code == ErrorCode::kInvalidArgument && !invalid_axis_stride &&
        invalid_axis_stride.error().code == ErrorCode::kInvalidArgument && !invalid_rank &&
        invalid_rank.error().code == ErrorCode::kInvalidArgument && !invalid_axis &&
        invalid_axis.error().code == ErrorCode::kInvalidArgument && !invalid_trailing_axis &&
        invalid_trailing_axis.error().code == ErrorCode::kInvalidArgument && !overlapping_layout &&
        overlapping_layout.error().code == ErrorCode::kInvalidArgument;

    auto active_loan = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    if (!active_loan) {
      return fail(active_loan.error().code);
    }
    auto active_writer = active_loan->write_host();
    if (!active_writer) {
      return fail(active_writer.error().code);
    }
    void* const active_pointer = active_writer->data();
    auto invalid_active_emit = output.emit(std::move(*active_loan));
    if (invalid_active_emit || invalid_active_emit.error().code != ErrorCode::kInvalidArgument) {
      return fail(ErrorCode::kFailure);
    }
    auto replacement = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    if (!replacement) {
      return fail(replacement.error().code);
    }
    auto replacement_writer = replacement->write_host();
    if (!replacement_writer) {
      return fail(replacement_writer.error().code);
    }
    const bool active_writer_safe = replacement_writer->data() != active_pointer;
    auto replacement_commit = std::move(*replacement_writer).commit();
    if (!replacement_commit) {
      return fail(replacement_commit.error().code);
    }
    auto replacement_cancel = std::move(*replacement).cancel();
    if (!replacement_cancel) {
      return fail(replacement_cancel.error().code);
    }
    auto invalid_active_commit = std::move(*active_writer).commit();
    if (invalid_active_commit ||
        invalid_active_commit.error().code != ErrorCode::kInvalidArgument) {
      return fail(ErrorCode::kFailure);
    }
    auto active_recycled = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    if (!active_recycled) {
      return fail(active_recycled.error().code);
    }
    auto active_recycled_writer = active_recycled->write_host();
    if (!active_recycled_writer || active_recycled_writer->data() != active_pointer) {
      return fail(ErrorCode::kFailure);
    }
    auto active_recycled_commit = std::move(*active_recycled_writer).commit();
    if (!active_recycled_commit) {
      return fail(active_recycled_commit.error().code);
    }
    auto active_recycled_cancel = std::move(*active_recycled).cancel();
    if (!active_recycled_cancel) {
      return fail(active_recycled_cancel.error().code);
    }

    auto first = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    auto second = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    if (!first || !second) {
      return fail(ErrorCode::kFailure);
    }
    auto first_writer = first->write_host();
    auto second_writer = second->write_host();
    if (!first_writer || !second_writer) {
      return fail(ErrorCode::kFailure);
    }
    const bool aligned =
        reinterpret_cast<std::uintptr_t>(first_writer->data()) % alignof(std::uint64_t) == 0U &&
        reinterpret_cast<std::uintptr_t>(second_writer->data()) % alignof(std::uint64_t) == 0U &&
        first->byte_strides() && first->byte_strides().size() == 1U &&
        first->byte_strides()[0] == valid_stride[0];
    auto first_commit = std::move(*first_writer).commit();
    auto second_commit = std::move(*second_writer).commit();
    if (!first_commit || !second_commit) {
      return fail(ErrorCode::kFailure);
    }

    RuntimeDirectTensorProbeState::set_metric_time_ns(50U);
    auto unavailable = output.allocate_tensor(TensorLoanRequest{
        .shape = vector_shape,
        .dtype = kDtype,
        .byte_strides = TensorByteStridesSpan{valid_stride},
    });
    const bool exhausted =
        !unavailable && unavailable.error().code == ErrorCode::kResourceExhausted;
    const auto invalid_while_exhausted = output.allocate_tensor(TensorLoanRequest{
        .shape = excessive_rank,
        .dtype = kDtype,
    });
    const bool metadata_precedes_exhaustion =
        !invalid_while_exhausted &&
        invalid_while_exhausted.error().code == ErrorCode::kInvalidArgument;
    RuntimeDirectTensorProbeState::set_metric_time_ns(60U);
    auto canceled_first = std::move(*first).cancel();
    auto canceled_second = std::move(*second).cancel();
    if (!canceled_first || !canceled_second) {
      return fail(ErrorCode::kFailure);
    }

    auto scalar = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = kDtype,
    });
    if (!scalar || !scalar->shape().empty()) {
      return fail(ErrorCode::kFailure);
    }
    auto writer = scalar->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(ErrorCode::kFailure);
    }
    *writer->data_as<std::uint64_t>() = kPublishedValue;
    void* const published_pointer = writer->data();
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto published = output.emit(std::move(*scalar));
    if (!published) {
      return fail(published.error().code);
    }

    auto active_when_full = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = kDtype,
    });
    if (!active_when_full) {
      return fail(active_when_full.error().code);
    }
    auto active_when_full_writer = active_when_full->write_host();
    if (!active_when_full_writer) {
      return fail(active_when_full_writer.error().code);
    }
    auto active_when_full_emit = output.emit(std::move(*active_when_full));
    const bool writer_precedes_backpressure =
        !active_when_full_emit && active_when_full_emit.error().code == ErrorCode::kInvalidArgument;
    auto active_when_full_commit = std::move(*active_when_full_writer).commit();
    if (active_when_full_commit ||
        active_when_full_commit.error().code != ErrorCode::kInvalidArgument) {
      return fail(ErrorCode::kFailure);
    }

    auto rejected = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = kDtype,
    });
    if (!rejected) {
      return fail(rejected.error().code);
    }
    auto rejected_writer = rejected->write_host();
    if (!rejected_writer) {
      return fail(rejected_writer.error().code);
    }
    auto rejected_commit = std::move(*rejected_writer).commit();
    if (!rejected_commit) {
      return fail(rejected_commit.error().code);
    }
    auto backpressured = output.emit(std::move(*rejected));
    const bool exact_backpressure =
        !backpressured && backpressured.error().code == ErrorCode::kBackpressured;
    if (!exact_backpressure) {
      return fail(backpressured ? ErrorCode::kFailure : backpressured.error().code);
    }
    RuntimeDirectTensorProbeState::record_initial_source(
        aligned,
        exhausted,
        true,
        rejected_layout,
        active_writer_safe,
        exact_backpressure,
        metadata_precedes_exhaustion && writer_precedes_backpressure,
        published_pointer);
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kPublishedValue = 0x0123456789ABCDEFULL;

 private:
  [[nodiscard]] static expected<void, Error> fail_source(ErrorCode code) {
    RuntimeDirectTensorProbeState::fail_source(code);
    return make_unexpected(Error{code});
  }

  bool emitted_{};
};

// Receives the published scalar Tensor, verifies scalar DLPack metadata (null
// shape/strides pointers, ndim==0), reads the expected sentinel value, and
// retains the Tensor to allow the source to probe slot-retirement ordering.
// Used by: DirectTensorPoolPreservesAlignmentExhaustionRetirementAndScalarMetadata
class RuntimeDirectTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = RuntimeDirectTensorProbeState::selected_memory_kind(),
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    const std::uint64_t* const value = sample->data.data_as<std::uint64_t>();
    if (value == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    const auto& context = detail::TensorAccess::context(sample->data);
    std::uint64_t copied_value{};
    const auto copied = sample->data.copy_to_host(&copied_value, sizeof(copied_value));
    RuntimeDirectTensorProbeState::record_sink(
        *sample,
        sample->data.ndim() == 0 && sample->data.shape_span().empty() &&
            sample->data.dlpack_strides_span().empty() && context != nullptr &&
            context->tensor.dl_tensor.shape == nullptr &&
            context->tensor.dl_tensor.strides == nullptr && sample->metadata.sequence == 1U,
        copied && copied_value == *value,
        *value);
    return {};
  }

  Input<Tensor> input;
};

// Minimal one-shot state used to prove transported Tensor contributor validation without the
// direct-output stress fixture's intentional backpressure and escaped-handle behavior.
struct RuntimeTransportTensorProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    received = false;
    value = 0U;
  }

  static void record(std::uint64_t observed) {
    {
      const std::lock_guard lock(mutex);
      value = observed;
      received = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return received; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool received{};
  static inline std::uint64_t value{};
};

class RuntimeTransportTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    auto published = output.emit(std::move(*loan));
    if (!published) {
      return make_unexpected(std::move(published.error()));
    }
    emitted_ = true;
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kValue = 0x13579BDF2468ACE0ULL;

 private:
  bool emitted_{};
};

class RuntimeTransportTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(sample ? Error{ErrorCode::kFailure} : std::move(sample.error()));
    }
    RuntimeTransportTensorProbeState::record(*sample->data.data_as<std::uint64_t>());
    return {};
  }

  Input<Tensor> input;
};

// Used by: DirectTensorPoolWaitsForEveryFanoutConsumerStreamBeforeReuse,
//          ComputeCallCompletionWaitsForCudaConsumerStreamRetirement
struct RuntimeDirectTensorFanoutStreamProbeState {
  static void reset(cudaStream_t blocked, cudaStream_t quick) {
    const std::lock_guard lock(mutex);
    blocked_stream = blocked;
    quick_stream = quick;
    first_pointer = nullptr;
    sink_count = 0U;
    probe_count = 0U;
    blocked_callback_started = false;
    premature_reuse = false;
    error.reset();
    release_blocked_stream.store(false, std::memory_order_release);
  }

  static void set_first_pointer(const void* pointer) {
    const std::lock_guard lock(mutex);
    first_pointer = pointer;
  }

  static void record_probe(const void* pointer) {
    {
      const std::lock_guard lock(mutex);
      ++probe_count;
      if (pointer == first_pointer && !release_blocked_stream.load(std::memory_order_acquire)) {
        premature_reuse = true;
      }
    }
    cv.notify_all();
  }

  static void record_sink() {
    {
      const std::lock_guard lock(mutex);
      ++sink_count;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  static void blocking_stream_callback(void*) {
    {
      const std::lock_guard lock(mutex);
      blocked_callback_started = true;
    }
    cv.notify_all();
    release_blocked_stream.wait(false, std::memory_order_acquire);
  }

  [[nodiscard]] static bool wait_for_blocked_stream() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] {
      return (blocked_callback_started && sink_count == 2U) || error.has_value();
    });
  }

  [[nodiscard]] static bool wait_for_probes(std::uint64_t minimum) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [minimum] { return probe_count >= minimum || error.has_value(); });
  }

  static void release_blocked() {
    release_blocked_stream.store(true, std::memory_order_release);
    release_blocked_stream.notify_all();
  }

  [[nodiscard]] static bool wait_for_retirement_queries(
      const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
      const std::uint64_t minimum) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (control->snapshot(0U).cuda_retirement_queries >= minimum) {
        return true;
      }
      {
        const std::lock_guard lock(mutex);
        if (error.has_value()) {
          return true;
        }
      }
      std::this_thread::sleep_for(1ms);
    }
    return false;
  }

  [[nodiscard]] static bool wait_for_first_slot_retirement(
      const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      // OutputPool initializes its LIFO free list so the first publication uses slot zero.
      if (control->snapshot(0U).minimum_retired_slot == 0U) {
        return true;
      }
      {
        const std::lock_guard lock(mutex);
        if (error.has_value()) {
          return true;
        }
      }
      std::this_thread::sleep_for(1ms);
    }
    return false;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic_bool release_blocked_stream{false};
  static inline cudaStream_t blocked_stream{};
  static inline cudaStream_t quick_stream{};
  static inline const void* first_pointer{};
  static inline std::uint64_t sink_count{};
  static inline std::uint64_t probe_count{};
  static inline bool blocked_callback_started{};
  static inline bool premature_reuse{};
  static inline std::optional<ErrorCode> error;
};

// Emits one CUDA-managed Tensor on the first activation and then on each
// subsequent call probes whether the slot is correctly withheld from reuse while
// consumer streams are still outstanding. Pool exhaustion is recorded as a probe.
// Used by: DirectTensorPoolWaitsForEveryFanoutConsumerStreamBeforeReuse,
//          ComputeCallCompletionWaitsForCudaConsumerStreamRetirement
class RuntimeDirectTensorFanoutStreamSourceOperator final : public Operator<> {
 public:
  explicit RuntimeDirectTensorFanoutStreamSourceOperator(bool complete_after_one = false)
      : complete_after_one_(complete_after_one) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    if (complete_after_one_) {
      result.complete_after(ComputeCalls{.limit = 1U});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        RuntimeDirectTensorFanoutStreamProbeState::record_probe(nullptr);
        return {};
      }
      RuntimeDirectTensorFanoutStreamProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      RuntimeDirectTensorFanoutStreamProbeState::fail(writer.error().code);
      return make_unexpected(std::move(writer.error()));
    }
    void* const pointer = writer->data();
    if (context.cuda_stream().get() == nullptr ||
        cudaMemsetAsync(pointer, 0x5a, writer->byte_size(), context.cuda_stream().get()) !=
            cudaSuccess) {
      RuntimeDirectTensorFanoutStreamProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (!emitted_) {
      RuntimeDirectTensorFanoutStreamProbeState::set_first_pointer(pointer);
    } else {
      RuntimeDirectTensorFanoutStreamProbeState::record_probe(pointer);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      RuntimeDirectTensorFanoutStreamProbeState::fail(committed.error().code);
      return make_unexpected(std::move(committed.error()));
    }
    if (!emitted_) {
      emitted_ = true;
      auto published = output.emit(std::move(*loan));
      if (!published) {
        RuntimeDirectTensorFanoutStreamProbeState::fail(published.error().code);
      }
      return published;
    }
    auto canceled = std::move(*loan).cancel();
    if (!canceled) {
      RuntimeDirectTensorFanoutStreamProbeState::fail(canceled.error().code);
    }
    return canceled;
  }

  Output<Tensor> output;

 private:
  bool complete_after_one_{};
  bool emitted_{};
};

// Fanout sink template. kBlockStream=true installs a host callback that holds
// the CUDA stream until the test releases it, forcing the source to observe pool
// exhaustion; kBlockStream=false completes immediately without blocking.
// Used by: DirectTensorPoolWaitsForEveryFanoutConsumerStreamBeforeReuse,
//          ComputeCallCompletionWaitsForCudaConsumerStreamRetirement
template <bool kBlockStream, bool kUseExecutionContextStream = false>
class RuntimeDirectTensorFanoutStreamSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
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
      RuntimeDirectTensorFanoutStreamProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    cudaStream_t stream =
        kUseExecutionContextStream
            ? context.cuda_stream().get()
            : (kBlockStream ? RuntimeDirectTensorFanoutStreamProbeState::blocked_stream
                            : RuntimeDirectTensorFanoutStreamProbeState::quick_stream);
    auto synchronized = sample->data.synchronize_for_consumer_stream(stream);
    if (!synchronized || !sample->data.set_deallocation_stream(stream)) {
      const Error error =
          synchronized ? Error{ErrorCode::kFailure} : std::move(synchronized.error());
      RuntimeDirectTensorFanoutStreamProbeState::fail(error.code);
      return make_unexpected(error);
    }
    if constexpr (kBlockStream) {
      if (cudaLaunchHostFunc(stream,
                             RuntimeDirectTensorFanoutStreamProbeState::blocking_stream_callback,
                             nullptr) != cudaSuccess) {
        RuntimeDirectTensorFanoutStreamProbeState::fail(ErrorCode::kFailure);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    RuntimeDirectTensorFanoutStreamProbeState::record_sink();
    return {};
  }

  Input<Tensor> input;
};

// Shared probe for lifecycle restart tests. Records the first and second sink
// values across a restart boundary and optionally retains the escaped Tensor to
// race session teardown against backing-claim release.
// Used by: DirectTensorPoolDeepRestart*, DirectTensorPoolShallowRestart*,
//          DirectTensorSameStreamElision*, DirectTensorMixedCatalogFanout*
struct RuntimeDirectTensorRestartProbeState {
  static void reset(LifecycleStage stage = LifecycleStage::kAllocate,
                    cudaStream_t consumer = nullptr) {
    const std::lock_guard lock(mutex);
    sink_count = 0U;
    first_value = 0U;
    second_value = 0U;
    source_completed = false;
    restart_accepted = false;
    stale_guard_rejected = false;
    restart_allocate_count = 0U;
    restart_stage = stage;
    consumer_stream = consumer;
    escaped_tensor.reset();
    error.reset();
  }

  static void record_sink(const Tensor& tensor, std::uint64_t value) {
    {
      const std::lock_guard lock(mutex);
      if (sink_count == 0U) {
        first_value = value;
      } else if (sink_count == 1U) {
        second_value = value;
      }
      escaped_tensor = tensor;
      ++sink_count;
    }
    cv.notify_all();
  }

  static void record_source_complete(bool accepted, bool stale_rejected) {
    {
      const std::lock_guard lock(mutex);
      restart_accepted = accepted;
      stale_guard_rejected = stale_rejected;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return source_completed && sink_count >= 2U; });
  }

  [[nodiscard]] static bool first_sample_observed() {
    const std::lock_guard lock(mutex);
    return sink_count != 0U;
  }

  [[nodiscard]] static LifecycleStage requested_restart_stage() {
    const std::lock_guard lock(mutex);
    return restart_stage;
  }

  [[nodiscard]] static cudaStream_t selected_consumer_stream() {
    const std::lock_guard lock(mutex);
    return consumer_stream;
  }

  static void release_escaped_tensor() {
    const std::lock_guard lock(mutex);
    escaped_tensor.reset();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::size_t sink_count{};
  static inline std::uint64_t first_value{};
  static inline std::uint64_t second_value{};
  static inline bool source_completed{};
  static inline bool restart_accepted{};
  static inline bool stale_guard_rejected{};
  static inline std::uint32_t restart_allocate_count{};
  static inline LifecycleStage restart_stage{LifecycleStage::kAllocate};
  static inline cudaStream_t consumer_stream{};
  static inline std::optional<Tensor> escaped_tensor;
  static inline std::optional<ErrorCode> error;
};

// Three-phase source: emits kFirstValue, then waits for sink confirmation,
// allocates a stale write guard and issues a self-restart request, then on
// re-entry emits kSecondValue and confirms the pre-restart guard is rejected.
// Used by: DirectTensorPoolDeepRestartPreservesCommittedOutputAndRejectsStaleGuard
class RuntimeDirectTensorRestartSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(2U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle().stage(LifecycleStage::kAllocate,
                           &RuntimeDirectTensorRestartSourceOperator::allocate_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (phase_ == 0U) {
      auto first = emit_value(kFirstValue, context.cuda_stream());
      if (!first) {
        return first;
      }
      phase_ = 1U;
      return {};
    }
    if (phase_ == 1U) {
      // Restart only after the sink has retained the committed predecessor
      // sample. Queued old-generation ingress is intentionally discarded by
      // the Channel restart fence, while an already delivered Tensor must keep
      // its backing claim alive across pool replacement.
      if (!RuntimeDirectTensorRestartProbeState::first_sample_observed()) {
        return {};
      }
      auto stale = output.allocate_tensor(request());
      if (!stale) {
        return fail(stale.error().code);
      }
      stale_loan_.emplace(std::move(*stale));
      auto stale_writer = stale_loan_->write(context.cuda_stream());
      if (!stale_writer) {
        return fail(stale_writer.error().code);
      }
      stale_writer_.emplace(std::move(*stale_writer));
      auto receipt = context.request_restart(LifecycleRestartOptions{
          .from_stage = RuntimeDirectTensorRestartProbeState::requested_restart_stage()});
      if (!receipt) {
        return fail(receipt.error().code);
      }
      restart_accepted_ = true;
      phase_ = 2U;
      return {};
    }
    if (source_completed_) {
      return {};
    }
    if (!stale_writer_) {
      return fail(ErrorCode::kFailure);
    }

    auto stale_commit = std::move(*stale_writer_).commit();
    const bool stale_rejected =
        !stale_commit && (stale_commit.error().code == ErrorCode::kClosed ||
                          stale_commit.error().code == ErrorCode::kInvalidArgument);
    stale_writer_.reset();
    stale_loan_.reset();

    auto replacement = output.allocate_tensor(request());
    if (!replacement) {
      return fail(replacement.error().code);
    }
    auto replacement_writer = replacement->write(context.cuda_stream());
    if (!replacement_writer) {
      return fail(replacement_writer.error().code);
    }
    *replacement_writer->data_as<std::uint64_t>() = kSecondValue;
    auto committed = std::move(*replacement_writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*replacement));
    if (!emitted) {
      return fail(emitted.error().code);
    }
    source_completed_ = true;
    RuntimeDirectTensorRestartProbeState::record_source_complete(restart_accepted_, stale_rejected);
    return {};
  }

  LifecycleStatus allocate_stage(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
      ++RuntimeDirectTensorRestartProbeState::restart_allocate_count;
    }
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static TensorLoanRequest request() noexcept {
    return TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    };
  }

  [[nodiscard]] expected<void, Error> emit_value(std::uint64_t value, ::cuda::stream_ref stream) {
    auto loan = output.allocate_tensor(request());
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write(stream);
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = value;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted.error()))};
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDirectTensorRestartProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  static constexpr std::uint64_t kFirstValue = 11U;
  static constexpr std::uint64_t kSecondValue = 22U;
  std::optional<TensorOutputLoan> stale_loan_;
  std::optional<TensorOutputWriteGuard> stale_writer_;
  std::uint32_t phase_{};
  bool restart_accepted_{};
  bool source_completed_{};
};

// Receives Tensor outputs across a restart boundary, records each value, and
// optionally synchronises and retains the most recent Tensor on an external
// stream to probe cross-session pointer safety on teardown.
// Used by: DirectTensorPoolDeepRestart* and related tests.
class RuntimeDirectTensorRestartSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(sample ? Error{ErrorCode::kFailure} : std::move(sample.error()));
    }
    const auto consumer_stream = RuntimeDirectTensorRestartProbeState::selected_consumer_stream();
    auto ready = consumer_stream != nullptr
                     ? sample->data.synchronize_for_consumer_stream(consumer_stream)
                     : sample->data.synchronize_for_host();
    if (!ready) {
      return make_unexpected(std::move(ready.error()));
    }
    if (consumer_stream != nullptr && (cudaStreamSynchronize(consumer_stream) != cudaSuccess ||
                                       !sample->data.set_deallocation_stream(consumer_stream))) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    RuntimeDirectTensorRestartProbeState::record_sink(sample->data,
                                                      *sample->data.data_as<std::uint64_t>());
    return {};
  }

  Input<Tensor> input;
};

// Probe singleton for injected-fault tests. Records whether pool/slot generation
// corruption and commit/post-submit faults each consumed their loan authority
// exactly once and whether the one valid publication reached the sink.
// Used by: DirectTensorTerminalFaultsConsumeAuthorityWithoutRetry
struct RuntimeDirectTensorFaultProbeState {
  static void reset(std::shared_ptr<detail::RuntimeTensorOutputTestControl> test_control,
                    bool escape_writer = false) {
    const std::lock_guard lock(mutex);
    control = std::move(test_control);
    source_completed = false;
    sink_count = 0U;
    pool_generation_failed_closed = false;
    slot_generation_failed_closed = false;
    commit_failed = false;
    failed_commit_consumed = false;
    publication_indeterminate = false;
    indeterminate_consumed = false;
    publication_partial = false;
    partial_consumed = false;
    error.reset();
  }

  static void arm_fault(detail::RuntimeTensorOutputTestFault fault) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected;
    {
      const std::lock_guard lock(mutex);
      selected = control;
    }
    if (selected) {
      selected->arm_fault(fault);
    }
  }

  static void arm_corruption(detail::RuntimeTensorOutputTestCorruption corruption) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected;
    {
      const std::lock_guard lock(mutex);
      selected = control;
    }
    if (selected) {
      selected->arm_corruption(corruption);
    }
  }

  static void record_source(bool pool_generation_failure, bool slot_generation_failure, bool commit,
                            bool commit_consumed, bool indeterminate,
                            bool indeterminate_consumed_once, bool partial,
                            bool partial_consumed_once) {
    {
      const std::lock_guard lock(mutex);
      pool_generation_failed_closed = pool_generation_failure;
      slot_generation_failed_closed = slot_generation_failure;
      commit_failed = commit;
      failed_commit_consumed = commit_consumed;
      publication_indeterminate = indeterminate;
      indeterminate_consumed = indeterminate_consumed_once;
      publication_partial = partial;
      partial_consumed = partial_consumed_once;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void record_sink() {
    {
      const std::lock_guard lock(mutex);
      ++sink_count;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return source_completed && sink_count != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::shared_ptr<detail::RuntimeTensorOutputTestControl> control;
  static inline bool source_completed{};
  static inline std::size_t sink_count{};
  static inline bool pool_generation_failed_closed{};
  static inline bool slot_generation_failed_closed{};
  static inline bool commit_failed{};
  static inline bool failed_commit_consumed{};
  static inline bool publication_indeterminate{};
  static inline bool indeterminate_consumed{};
  static inline bool publication_partial{};
  static inline bool partial_consumed{};
  static inline std::optional<ErrorCode> error;
};

// Exercises pool- and slot-generation corruption, commit-readiness failure, and both terminal
// post-submit publication classes. Verifies each loan is consumed and the bounded input observes a
// valid already-published sample.
// Used by: DirectTensorTerminalFaultsConsumeAuthorityWithoutRetry
class RuntimeDirectTensorFaultSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(7U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (completed_) {
      return {};
    }
    completed_ = true;

    // Exercise the ordinary cancellation-to-retirement path once so the private snapshot proves
    // that the complete admitted claim vocabulary was active at least once before terminal zero.
    auto retired_loan = output.allocate_tensor(request());
    if (!retired_loan) {
      return fail(retired_loan.error().code);
    }
    auto retired_writer = retired_loan->write_host();
    if (!retired_writer) {
      return fail(retired_writer.error().code);
    }
    auto retired_commit = std::move(*retired_writer).commit();
    if (!retired_commit) {
      return fail(retired_commit.error().code);
    }
    auto retired = std::move(*retired_loan).cancel();
    if (!retired) {
      return fail(retired.error().code);
    }

    auto pool_corrupt_loan = output.allocate_tensor(request());
    if (!pool_corrupt_loan) {
      return fail(pool_corrupt_loan.error().code);
    }
    RuntimeDirectTensorFaultProbeState::arm_corruption(
        detail::RuntimeTensorOutputTestCorruption::kPoolGeneration);
    auto pool_corrupt_writer = pool_corrupt_loan->write_host();
    const bool pool_generation_failure =
        !pool_corrupt_writer && pool_corrupt_writer.error().code == ErrorCode::kFailure;
    auto pool_corrupt_emit = output.emit(std::move(*pool_corrupt_loan));
    const bool pool_corrupt_consumed =
        !pool_corrupt_emit && pool_corrupt_emit.error().code == ErrorCode::kInvalidArgument;

    auto slot_corrupt_loan = output.allocate_tensor(request());
    if (!slot_corrupt_loan) {
      return fail(slot_corrupt_loan.error().code);
    }
    auto slot_corrupt_writer = slot_corrupt_loan->write_host();
    if (!slot_corrupt_writer) {
      return fail(slot_corrupt_writer.error().code);
    }
    RuntimeDirectTensorFaultProbeState::arm_corruption(
        detail::RuntimeTensorOutputTestCorruption::kSlotGeneration);
    auto slot_corrupt_commit = std::move(*slot_corrupt_writer).commit();
    const bool slot_generation_failure =
        !slot_corrupt_commit && slot_corrupt_commit.error().code == ErrorCode::kFailure;
    auto slot_corrupt_emit = output.emit(std::move(*slot_corrupt_loan));
    const bool slot_corrupt_consumed =
        !slot_corrupt_emit && slot_corrupt_emit.error().code == ErrorCode::kInvalidArgument;

    auto failed_loan = output.allocate_tensor(request());
    if (!failed_loan) {
      return fail(failed_loan.error().code);
    }
    auto failed_writer = failed_loan->write_host();
    if (!failed_writer) {
      return fail(failed_writer.error().code);
    }
    RuntimeDirectTensorFaultProbeState::arm_fault(
        detail::RuntimeTensorOutputTestFault::kCommitReadinessFailure);
    auto failed_commit = std::move(*failed_writer).commit();
    const bool exact_commit_failure =
        !failed_commit && failed_commit.error().code == ErrorCode::kFailure;
    auto failed_emit = output.emit(std::move(*failed_loan));
    const bool commit_authority_consumed =
        !failed_emit && failed_emit.error().code == ErrorCode::kInvalidArgument;

    auto uncertain_loan = output.allocate_tensor(request());
    if (!uncertain_loan) {
      return fail(uncertain_loan.error().code);
    }
    auto uncertain_writer = uncertain_loan->write_host();
    if (!uncertain_writer) {
      return fail(uncertain_writer.error().code);
    }
    *uncertain_writer->data_as<std::uint64_t>() = 42U;
    auto uncertain_commit = std::move(*uncertain_writer).commit();
    if (!uncertain_commit) {
      return fail(uncertain_commit.error().code);
    }
    RuntimeDirectTensorFaultProbeState::arm_fault(
        detail::RuntimeTensorOutputTestFault::kPostSubmitIndeterminate);
    auto uncertain = output.emit(std::move(*uncertain_loan));
    const bool exact_indeterminate =
        !uncertain && uncertain.error().code == ErrorCode::kPublicationIndeterminate;
    auto retry = output.emit(std::move(*uncertain_loan));
    const bool consumed_once = !retry && retry.error().code == ErrorCode::kInvalidArgument;

    auto partial_loan = output.allocate_tensor(request());
    if (!partial_loan) {
      return fail(partial_loan.error().code);
    }
    auto partial_writer = partial_loan->write_host();
    if (!partial_writer) {
      return fail(partial_writer.error().code);
    }
    *partial_writer->data_as<std::uint64_t>() = 43U;
    auto partial_commit = std::move(*partial_writer).commit();
    if (!partial_commit) {
      return fail(partial_commit.error().code);
    }
    RuntimeDirectTensorFaultProbeState::arm_fault(
        detail::RuntimeTensorOutputTestFault::kPostSubmitPartial);
    auto partial = output.emit(std::move(*partial_loan));
    const bool exact_partial =
        !partial && partial.error().code == ErrorCode::kPublicationPartial &&
        partial.error().message == "PUBLICATION_PARTIAL delivered=0 expected=1";
    auto partial_retry = output.emit(std::move(*partial_loan));
    const bool partial_consumed_once =
        !partial_retry && partial_retry.error().code == ErrorCode::kInvalidArgument;

    RuntimeDirectTensorFaultProbeState::record_source(
        pool_generation_failure && pool_corrupt_consumed,
        slot_generation_failure && slot_corrupt_consumed,
        exact_commit_failure,
        commit_authority_consumed,
        exact_indeterminate,
        consumed_once,
        exact_partial,
        partial_consumed_once);
    return {};
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static TensorLoanRequest request() noexcept {
    return TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}};
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDirectTensorFaultProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  bool completed_{};
};

// Accepts a Tensor value published before one of the two injected post-submit terminal results.
// Used by: DirectTensorTerminalFaultsConsumeAuthorityWithoutRetry
class RuntimeDirectTensorFaultSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(sample ? Error{ErrorCode::kFailure} : std::move(sample.error()));
    }
    const std::uint64_t value = *sample->data.data_as<std::uint64_t>();
    if (value != 42U && value != 43U) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    RuntimeDirectTensorFaultProbeState::record_sink();
    return {};
  }

  Input<Tensor> input;
};

// Probe for the emit-precedence test. Captures whether each error category was
// rejected in the documented order and whether the cross-callback loan attempt
// was correctly diagnosed with kLoanActivationGenerationMismatch.
// Used by: DirectTensorEmitConfinementAndCheckedFailurePrecedenceAreExact,
//          DirectTensorLoanOutlivingRunSessionFailsClosed
struct RuntimeDirectTensorPrecedenceProbeState {
  static void reset(std::shared_ptr<detail::RuntimeTensorOutputTestControl> test_control,
                    bool escape_writer = false) {
    const std::lock_guard lock(mutex);
    control = std::move(test_control);
    after_callback_loan.reset();
    after_callback_guard.reset();
    escape_writer_requested = escape_writer;
    after_callback_loan_ready = false;
    after_callback_probe_completed = false;
    after_callback_rejected = false;
    after_callback_reason_exact = false;
    source_completed = false;
    wrong_output_precedes_closed = false;
    plan_precedes_activation = false;
    activation_precedes_writer = false;
    invariant_precedes_metadata = false;
    error.reset();
  }

  static void publish_after_callback_loan(TensorOutputLoan loan) {
    {
      const std::lock_guard lock(mutex);
      after_callback_loan.emplace(std::move(loan));
      after_callback_loan_ready = true;
    }
    cv.notify_all();
  }

  static void publish_after_callback_guard(TensorOutputLoan loan, TensorOutputWriteGuard guard) {
    {
      const std::lock_guard lock(mutex);
      after_callback_loan.emplace(std::move(loan));
      after_callback_guard.emplace(std::move(guard));
      after_callback_loan_ready = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool should_escape_writer() {
    const std::lock_guard lock(mutex);
    return escape_writer_requested;
  }

  [[nodiscard]] static bool wait_for_after_callback_loan() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return after_callback_loan_ready; });
  }

  static void probe_after_callback_use() {
    std::optional<TensorOutputLoan> loan;
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected_control;
    {
      const std::lock_guard lock(mutex);
      if (after_callback_loan) {
        loan.emplace(std::move(*after_callback_loan));
      }
      after_callback_loan.reset();
      selected_control = control;
    }
    if (!loan || !selected_control) {
      {
        const std::lock_guard lock(mutex);
        error = ErrorCode::kFailure;
        after_callback_probe_completed = true;
      }
      cv.notify_all();
      return;
    }
    const auto before = selected_control->last_diagnostic();
    auto writer = loan->write_host();
    const auto after = selected_control->last_diagnostic();
    const bool rejected = !writer && writer.error().code == ErrorCode::kInvalidArgument;
    const bool exact = after.count == before.count + 1U &&
                       after.record.reason ==
                           detail::TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch;
    loan.reset();
    {
      const std::lock_guard lock(mutex);
      after_callback_rejected = rejected;
      after_callback_reason_exact = exact;
      after_callback_probe_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool after_callback_probe_done() {
    const std::lock_guard lock(mutex);
    return after_callback_probe_completed;
  }

  [[nodiscard]] static std::uint64_t diagnostic_count() {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected_control;
    {
      const std::lock_guard lock(mutex);
      selected_control = control;
    }
    return selected_control->last_diagnostic().count;
  }

  [[nodiscard]] static bool diagnostic_matches(std::uint64_t prior_count,
                                               detail::TensorOutputDiagnosticReason reason) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected_control;
    {
      const std::lock_guard lock(mutex);
      selected_control = control;
    }
    const auto snapshot = selected_control->last_diagnostic();
    return snapshot.count == prior_count + 1U && snapshot.record.reason == reason;
  }

  static void arm_fault(detail::RuntimeTensorOutputTestFault fault) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected_control;
    {
      const std::lock_guard lock(mutex);
      selected_control = control;
    }
    selected_control->arm_fault(fault);
  }

  static void arm_corruption(detail::RuntimeTensorOutputTestCorruption corruption) {
    std::shared_ptr<detail::RuntimeTensorOutputTestControl> selected_control;
    {
      const std::lock_guard lock(mutex);
      selected_control = control;
    }
    selected_control->arm_corruption(corruption);
  }

  static void record_source(bool wrong_output, bool plan, bool activation, bool invariant) {
    {
      const std::lock_guard lock(mutex);
      wrong_output_precedes_closed = wrong_output;
      plan_precedes_activation = plan;
      activation_precedes_writer = activation;
      invariant_precedes_metadata = invariant;
      source_completed = true;
    }
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      source_completed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return source_completed && after_callback_probe_completed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::shared_ptr<detail::RuntimeTensorOutputTestControl> control;
  static inline std::optional<TensorOutputLoan> after_callback_loan;
  static inline std::optional<TensorOutputWriteGuard> after_callback_guard;
  static inline bool escape_writer_requested{};
  static inline bool after_callback_loan_ready{};
  static inline bool after_callback_probe_completed{};
  static inline bool after_callback_rejected{};
  static inline bool after_callback_reason_exact{};
  static inline bool source_completed{};
  static inline bool wrong_output_precedes_closed{};
  static inline bool plan_precedes_activation{};
  static inline bool activation_precedes_writer{};
  static inline bool invariant_precedes_metadata{};
  static inline std::optional<ErrorCode> error;
};

// Two-output source that exercises the complete checked-emit precedence order:
// wrong-output identity before pool-closed, plan-generation before activation,
// activation before active-writer, and slot invariant before metadata. Also
// passes a loan to an external callback to probe cross-thread use rejection.
// Used by: DirectTensorEmitConfinementAndCheckedFailurePrecedenceAreExact
class RuntimeDirectTensorPrecedenceSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    configure_output(spec.output(primary, "primary"));
    configure_output(spec.output(secondary, "secondary"));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (phase_ == 0U) {
      auto wrong_output = allocate_ready(primary);
      if (!wrong_output) {
        return fail(wrong_output.error().code);
      }
      const std::uint64_t wrong_before =
          RuntimeDirectTensorPrecedenceProbeState::diagnostic_count();
      RuntimeDirectTensorPrecedenceProbeState::arm_fault(
          detail::RuntimeTensorOutputTestFault::kEmitClosedBinding);
      const auto wrong = secondary.emit(std::move(*wrong_output));
      wrong_output_precedes_closed_ =
          !wrong && wrong.error().code == ErrorCode::kInvalidArgument &&
          RuntimeDirectTensorPrecedenceProbeState::diagnostic_matches(
              wrong_before, detail::TensorOutputDiagnosticReason::kLoanOutputMismatch);

      auto plan_and_activation = allocate_ready(primary);
      if (!plan_and_activation) {
        return fail(plan_and_activation.error().code);
      }
      plan_and_activation_loan_.emplace(std::move(*plan_and_activation));

      auto activation_and_writer = primary.allocate_tensor(request());
      if (!activation_and_writer) {
        return fail(activation_and_writer.error().code);
      }
      activation_and_writer_loan_.emplace(std::move(*activation_and_writer));
      auto writer = activation_and_writer_loan_->write_host();
      if (!writer) {
        return fail(writer.error().code);
      }
      activation_and_writer_guard_.emplace(std::move(*writer));

      auto after_callback = primary.allocate_tensor(request());
      if (!after_callback) {
        return fail(after_callback.error().code);
      }
      if (RuntimeDirectTensorPrecedenceProbeState::should_escape_writer()) {
        auto escaped_writer = after_callback->write_host();
        if (!escaped_writer) {
          return fail(escaped_writer.error().code);
        }
        RuntimeDirectTensorPrecedenceProbeState::publish_after_callback_guard(
            std::move(*after_callback), std::move(*escaped_writer));
      } else {
        RuntimeDirectTensorPrecedenceProbeState::publish_after_callback_loan(
            std::move(*after_callback));
      }
      phase_ = 1U;
      return {};
    }
    if (phase_ != 1U || !RuntimeDirectTensorPrecedenceProbeState::after_callback_probe_done()) {
      return {};
    }
    if (!plan_and_activation_loan_ || !activation_and_writer_loan_ ||
        !activation_and_writer_guard_) {
      return fail(ErrorCode::kFailure);
    }

    const std::uint64_t plan_before = RuntimeDirectTensorPrecedenceProbeState::diagnostic_count();
    RuntimeDirectTensorPrecedenceProbeState::arm_corruption(
        detail::RuntimeTensorOutputTestCorruption::kPlanGeneration);
    const auto stale_plan = primary.emit(std::move(*plan_and_activation_loan_));
    plan_precedes_activation_ =
        !stale_plan && stale_plan.error().code == ErrorCode::kInvalidArgument &&
        RuntimeDirectTensorPrecedenceProbeState::diagnostic_matches(
            plan_before, detail::TensorOutputDiagnosticReason::kLoanPlanGenerationMismatch);
    plan_and_activation_loan_.reset();

    const std::uint64_t activation_before =
        RuntimeDirectTensorPrecedenceProbeState::diagnostic_count();
    const auto stale_activation = primary.emit(std::move(*activation_and_writer_loan_));
    activation_precedes_writer_ =
        !stale_activation && stale_activation.error().code == ErrorCode::kInvalidArgument &&
        RuntimeDirectTensorPrecedenceProbeState::diagnostic_matches(
            activation_before,
            detail::TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch);
    auto stale_commit = std::move(*activation_and_writer_guard_).commit();
    if (stale_commit || stale_commit.error().code != ErrorCode::kInvalidArgument) {
      return fail(stale_commit ? ErrorCode::kFailure : stale_commit.error().code);
    }
    activation_and_writer_guard_.reset();
    activation_and_writer_loan_.reset();

    auto invariant_and_metadata = allocate_ready(primary);
    if (!invariant_and_metadata) {
      return fail(invariant_and_metadata.error().code);
    }
    const std::uint64_t invariant_before =
        RuntimeDirectTensorPrecedenceProbeState::diagnostic_count();
    RuntimeDirectTensorPrecedenceProbeState::arm_corruption(
        detail::RuntimeTensorOutputTestCorruption::kSlotGeneration);
    const auto invariant = primary.emit(
        std::move(*invariant_and_metadata),
        EmitOptions{.capture_time = TimePoint{.timestamp_ns = 1, .clock_id = ClockId{999U}}});
    invariant_precedes_metadata_ =
        !invariant && invariant.error().code == ErrorCode::kFailure &&
        RuntimeDirectTensorPrecedenceProbeState::diagnostic_matches(
            invariant_before, detail::TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch);

    phase_ = 2U;
    RuntimeDirectTensorPrecedenceProbeState::record_source(wrong_output_precedes_closed_,
                                                           plan_precedes_activation_,
                                                           activation_precedes_writer_,
                                                           invariant_precedes_metadata_);
    return {};
  }

  Output<Tensor> primary;
  Output<Tensor> secondary;

 private:
  static void configure_output(TensorOutputSpecBuilder& output) {
    output.max_emits_per_compute(4U).produces_tensor(TensorOutputSpec{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = MemoryKind::kHost,
                .dtype = DLDataType{kDLUInt, 64U, 1U},
            },
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .storage = TensorOutputStorage::kRuntimePool,
    });
  }

  [[nodiscard]] static TensorLoanRequest request() noexcept {
    return TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}};
  }

  [[nodiscard]] static expected<TensorOutputLoan, Error> allocate_ready(Output<Tensor>& output) {
    auto loan = output.allocate_tensor(request());
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    return std::move(*loan);
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDirectTensorPrecedenceProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  std::optional<TensorOutputLoan> plan_and_activation_loan_;
  std::optional<TensorOutputLoan> activation_and_writer_loan_;
  std::optional<TensorOutputWriteGuard> activation_and_writer_guard_;
  std::uint32_t phase_{};
  bool wrong_output_precedes_closed_{};
  bool plan_precedes_activation_{};
  bool activation_precedes_writer_{};
  bool invariant_precedes_metadata_{};
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
    const cudaStream_t stream = context.cuda_stream().get();
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
    const cudaStream_t stream = context.cuda_stream().get();
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
    const cudaStream_t stream = context.cuda_stream().get();
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

// Captures the metadata one descriptor-backed Tensor arrived with, so a test can assert on the
// publication coordinate Runtime stamped rather than on payload bytes.
struct RuntimeTensorPublicationProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    observed.reset();
    error.reset();
  }

  static void fail(const ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  static void record(const SampleMetadata& metadata) {
    {
      const std::lock_guard lock(mutex);
      observed = metadata;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed.has_value() || error.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<SampleMetadata> observed;
  static inline std::optional<ErrorCode> error;
};

class RuntimeTensorPublicationSourceOperator final : public Operator<> {
 public:
  static constexpr std::uint64_t kValue = 0x5150515051505150U;

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      RuntimeTensorPublicationProbeState::fail(emitted.error().code);
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static expected<void, Error> fail(const ErrorCode code) {
    RuntimeTensorPublicationProbeState::fail(code);
    return make_unexpected(Error{code});
  }
};

// Publishes one host-written pinned Tensor through a device-placed external catalog endpoint.
// Used by: PinnedHostTensorHostWritesDoNotRecordCudaReadiness
class RuntimePinnedHostTensorPublicationSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kPinnedHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = tensor_bounds(sizeof(kValue)),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer).error());
    }
    *writer->data_as<std::uint64_t>() = kValue;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    return output.emit(std::move(*loan));
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kValue = 0x50494E4E4544484FULL;
};

class RuntimeTensorPublicationSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = {
                                            .memory_kind = MemoryKind::kHost,
                                            .dtype = DLDataType{kDLUInt, 64U, 1U},
                                            .rank = 0U,
                                        }});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      RuntimeTensorPublicationProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    RuntimeTensorPublicationProbeState::record(sample->metadata);
    return {};
  }

  Input<Tensor> input;
};

// Graph topology:
//   source --[queue_depth=1]--> sink
// Both endpoints are ordinary in-graph ports, so the sample never crosses an external boundary.
[[nodiscard]] ExecutionPlan make_runtime_tensor_publication_plan() {
  Graph graph{"runtime-tensor-publication"};
  auto source = graph.op<RuntimeTensorPublicationSourceOperator>("source");
  auto sink = graph.op<RuntimeTensorPublicationSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  return compile(graph);
}

// Graph topology:
//   source --[queue_depth=1]--> sink
//   max_byte_span=17 forces odd per-slot span; confirms dtype alignment is maintained.
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control = {}) {
  Graph graph{"runtime-direct-tensor"};
  auto source = graph.op<RuntimeDirectTensorSourceOperator>("source");
  auto sink = graph.op<RuntimeDirectTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  return control ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control)
                 : plan;
}

[[nodiscard]] ExecutionPlan make_runtime_fixed_rate_tensor_evidence_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"runtime-fixed-rate-tensor-evidence"};
  auto source = graph.op<RuntimeFixedRateTensorEvidenceSourceOperator>("source");
  auto sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-evidence-clock"));
  const ExecutionPlan plan = compile(graph);
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

[[nodiscard]] ExecutionPlan make_runtime_mixed_stage_tensor_evidence_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"runtime-mixed-stage-tensor-evidence"};
  auto fixed_source = graph.op<RuntimeFixedRateTensorEvidenceSourceOperator>("fixed-source");
  auto fixed_sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("fixed-sink");
  auto arm_source = graph.op<RuntimeArmFiniteMaximumTensorSourceOperator>("arm-source");
  auto arm_sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("arm-sink");
  graph.add_flow(fixed_source->output, fixed_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(arm_source->output, arm_sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-mixed-stage-clock"));
  const ExecutionPlan plan = compile(graph);
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// Graph topology:
//   Discover-selected Tensor source --[rate-sized direct queue]--> OnEach sink
// No admitted maximum exists, so both the queue and output pool are materialized
// from the session-local rate generation at the first Core kAllocate barrier.
[[nodiscard]] ExecutionPlan make_runtime_lifecycle_tensor_candidate_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    LifecycleStage binding_stage = LifecycleStage::kDiscover) {
  Graph graph{unique_runtime_graph_name("runtime-lifecycle-tensor-candidate")};
  auto source = graph.op<RuntimeLifecycleTensorCandidateSourceOperator>("source", binding_stage);
  auto sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-candidate-clock"));
  const ExecutionPlan plan = compile(graph);
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// The two disconnected components deliberately share one execution plan but not a process group.
// Starting only the local worker must not require the remote no-maximum lifecycle candidate to
// bind a rate or materialize its Tensor pool.
[[nodiscard]] ExecutionPlan make_runtime_worker_local_fixed_tensor_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{unique_runtime_graph_name("runtime-worker-local-fixed-tensor")};
  auto local_source = graph.op<RuntimeFixedRateTensorEvidenceSourceOperator>("local-source");
  auto local_sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("local-sink");
  auto remote_source =
      graph.op<RuntimeRemoteNoMaximumLifecycleTensorSourceOperator>("remote-source");
  auto remote_sink = graph.op<RuntimeLifecycleTensorCandidateSinkOperator>("remote-sink");
  graph.add_flow(local_source->output, local_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(remote_source->output, remote_sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-worker-local-tensor-clock"));

  constexpr std::array local_members{
      detail::GraphObjectSelector{.graph_path = "local-source"},
      detail::GraphObjectSelector{.graph_path = "local-sink"},
  };
  constexpr std::array remote_members{
      detail::GraphObjectSelector{.graph_path = "remote-source"},
      detail::GraphObjectSelector{.graph_path = "remote-sink"},
  };
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("local",
                         detail::ProcessPlacement{
                             .members = local_members,
                             .launch_group = "local-process",
                         })
      .bind_process_role("remote",
                         detail::ProcessPlacement{
                             .members = remote_members,
                             .launch_group = "remote-process",
                         });
  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

[[nodiscard]] ExecutionPlan make_runtime_lifecycle_tensor_rate_restart_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    std::optional<Rate> maximum_rate = std::nullopt,
    LifecycleStage binding_stage = LifecycleStage::kDiscover,
    LifecycleStage restart_from = LifecycleStage::kDiscover,
    MemoryKind memory_kind = MemoryKind::kHost) {
  Graph graph{unique_runtime_graph_name("runtime-lifecycle-tensor-rate-restart")};
  auto source = graph.op<RuntimeLifecycleTensorRateRestartSourceOperator>(
      "source", maximum_rate, binding_stage, restart_from, memory_kind);
  auto sink = graph.op<RuntimeLifecycleTensorRateRestartSinkOperator>("sink", memory_kind);
  graph.add_flow(source->output, sink->input);
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-restart-clock"));
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  const ExecutionPlan plan = compile(graph, std::move(options));
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

[[nodiscard]] ExecutionPlan make_runtime_tensor_contract_admission_plan() {
  Graph graph{unique_runtime_graph_name("runtime-tensor-contract-admission")};
  static_cast<void>(graph.op<RuntimeTensorContractAdmissionSourceOperator>("source"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

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

// Graph topology:
//   source --[queue_depth=1, forced HoloMQ]--> sink
[[nodiscard]] ExecutionPlan make_runtime_transport_tensor_plan() {
  Graph graph{"runtime-transport-tensor"};
  auto source = graph.op<RuntimeTransportTensorSourceOperator>("source");
  auto sink = graph.op<RuntimeTransportTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return detail::compile_with_process_local_holomq(graph, CompileOptions{});
}

// Graph topology:
//   source --[queue_depth=1]--> sink-a-blocked  (holds CUDA stream via host callback)
//          --[queue_depth=1]--> sink-b-quick    (completes without blocking)
// Device binding for the output port enables managed-memory fanout with two streams.
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_fanout_stream_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    const bool complete_after_one = false,
    const std::string_view graph_name = "runtime-direct-tensor-fanout-streams") {
  Graph graph{std::string{graph_name}};
  auto source =
      graph.op<RuntimeDirectTensorFanoutStreamSourceOperator>("source", complete_after_one);
  auto blocked = graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<true>>("sink-a-blocked");
  auto quick = graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<false>>("sink-b-quick");
  graph.add_flow(source->output, blocked->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, quick->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{
        Error{ErrorCode::kFailure, "runtime fanout Tensor graph snapshot is unavailable"}};
  }
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto output_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime fanout Tensor output is unresolved"}};
  }
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output_port->id.value},
                                        .device = DeviceId{0}},
  };
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      {},
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      });
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// Graph topology (each operator has a dedicated Green Context partition):
//   source --> sink-a-blocked
//          --> sink-b-quick
[[nodiscard]] ExecutionPlan make_runtime_green_context_tensor_fanout_plan(
    const CudaGreenContextCapability& capability,
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    std::string_view graph_name) {
  Graph graph{std::string{graph_name}};
  auto source = graph.op<RuntimeDirectTensorFanoutStreamSourceOperator>("source");
  auto blocked =
      graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<true, true>>("sink-a-blocked");
  auto quick = graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<false, true>>("sink-b-quick");
  graph.add_flow(source->output, blocked->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, quick->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("blocked-partition").add(blocked);
  graph.partition("quick-partition").add(quick);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime fanout snapshot is unavailable"}};
  }
  const auto output_port =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.name == "output";
      });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime fanout output is unresolved"}};
  }
  const std::uint32_t quantum =
      std::lcm(capability.min_sm_partition_size, capability.sm_coscheduled_alignment);
  CompileOptions options;
  options.deployment
      .bind_cuda_green_context("source-partition", {.device = DeviceId{0}, .sm_count = quantum})
      .bind_cuda_green_context("blocked-partition", {.device = DeviceId{0}, .sm_count = quantum})
      .bind_cuda_green_context("quick-partition", {.device = DeviceId{0}, .sm_count = quantum});
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
//   source --[queue_depth=2]--> sink
//          --[queue_depth=2]--> exported  (optional, when add_catalog_export=true)
// Self-restart is permitted on source; lifecycle transaction capacity is expanded for
// the restart fence. Device binding is injected via compile-with-runtime-evidence.
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_restart_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control = {},
    bool add_catalog_export = false,
    std::string_view graph_name = "runtime-direct-tensor-restart") {
  Graph graph{std::string{graph_name}};
  auto source = graph.op<RuntimeDirectTensorRestartSourceOperator>("source");
  auto sink = graph.op<RuntimeDirectTensorRestartSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  if (add_catalog_export) {
    static std::atomic<std::uint64_t> endpoint_ordinal{};
    const std::string endpoint_name =
        "runtime-direct-tensor-mixed-export-" +
        std::to_string(endpoint_ordinal.fetch_add(1U, std::memory_order_relaxed));
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber =
        SubscriberEndpointBounds{.queue_depth = 2U, .max_borrowed_samples = 2U};
    options.tensor = TensorBoundaryContract{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = MemoryKind::kHost,
                .dtype = DLDataType{kDLUInt, 64U, 1U},
                .rank = 0U,
                .strides = TensorStridePolicy::kContiguousOnly,
            },
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 2U});
  }
  graph.lifecycle().allow_self_restart(source);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto output_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime restart Tensor output is unresolved"}};
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
  return control ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control)
                 : plan;
}

// Graph topology:
//   source --[queue_depth=2]--> sink
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_fault_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"runtime-direct-tensor-fault"};
  auto source = graph.op<RuntimeDirectTensorFaultSourceOperator>("source");
  auto sink = graph.op<RuntimeDirectTensorFaultSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compile(graph), control);
}

// Graph topology:
//   source (primary + secondary outputs, no downstream sink)
// Isolation from a sink lets the test inject precedence faults freely.
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_precedence_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"runtime-direct-tensor-precedence"};
  static_cast<void>(graph.op<RuntimeDirectTensorPrecedenceSourceOperator>("source"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compile(graph), control);
}

// Preparing observation storage must not erase caller-owned injection configuration. Runtime
// materialization prepares the control after the test arms it, so faults, corruption, and the
// deterministic metric clock remain live until consumed or explicitly replaced.
// REQ: OPERATOR-025
TEST(RunSessionTest, RuntimeTensorOutputTestControlPreparePreservesPendingInjection) {
  detail::RuntimeTensorOutputTestControl control;
  control.prepare(/*pool_count=*/1U);
  control.arm_fault(detail::RuntimeTensorOutputTestFault::kCommitReadinessFailure);
  control.arm_corruption(detail::RuntimeTensorOutputTestCorruption::kPlanGeneration);
  control.set_metric_time_ns(17U);

  control.prepare(/*pool_count=*/1U);

  EXPECT_TRUE(control.consume_fault(detail::RuntimeTensorOutputTestFault::kCommitReadinessFailure));
  EXPECT_EQ(control.consume_corruption(),
            detail::RuntimeTensorOutputTestCorruption::kPlanGeneration);
  EXPECT_EQ(control.metric_time_ns(), 17U);
}

TEST(RunSessionTest, RuntimeTensorOutputTestControlTargetsOnePoolWithoutEarlyConsumption) {
  detail::RuntimeTensorOutputTestControl control;
  control.prepare(/*pool_count=*/3U);
  control.arm_fault_for_pool(detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure,
                             1U);
  control.prepare(/*pool_count=*/3U);

  EXPECT_FALSE(control.consume_fault(
      detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure, 0U));
  EXPECT_FALSE(
      control.consume_fault(detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure));
  EXPECT_TRUE(control.consume_fault(
      detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure, 1U));
  EXPECT_FALSE(control.consume_fault(
      detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure, 1U));
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
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 3U);
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
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 3U);
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
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 3U);
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
        << static_cast<std::uint32_t>(*RuntimeGreenContextTransferProbeState::error)
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

// The production RunSession materializes exact bounded output pools before arm. The odd per-slot
// span proves later slots remain dtype-aligned without hidden aggregate padding; cancellation of a
// ready slot returns its fixed capacity, and scalar DLPack metadata remains pointer-free at rank 0.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, NoMaximumTensorRateMaterializesBeforeUserAllocateWithoutPlanPlaceholder) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_candidate_plan(control);

  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_TRUE(plan.tensor_allocations().empty());
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  ASSERT_TRUE(plan.provider_allocation_plans().front().tensor_template.has_value());
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  EXPECT_EQ(plan.queue_sizing_recipes().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);

  RunSession session = run_async(plan);
  const bool observed = RuntimeLifecycleTensorCandidateProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(RuntimeLifecycleTensorCandidateProbeState::mutex);
  ASSERT_FALSE(RuntimeLifecycleTensorCandidateProbeState::error.has_value());
  EXPECT_TRUE(RuntimeLifecycleTensorCandidateProbeState::allocate_saw_pool);
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::allocate_rate.resolved_nominal_rate,
            Rate::hz(60U));
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::allocate_rate.state,
            PortRateViewState::kResolved);
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::compute_rate,
            RuntimeLifecycleTensorCandidateProbeState::allocate_rate);
  EXPECT_TRUE(RuntimeLifecycleTensorCandidateProbeState::received);
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::observed_value,
            RuntimeLifecycleTensorCandidateSourceOperator::kValue);
}

// kConfigure is an equally valid pre-allocation discovery point. Its globally closed value must
// drive the no-maximum direct-host candidate before the first user kAllocate callback.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, NoMaximumConfigureTensorRateMaterializesBeforeUserAllocate) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control);
  const ExecutionPlan plan =
      make_runtime_lifecycle_tensor_candidate_plan(control, LifecycleStage::kConfigure);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  EXPECT_TRUE(plan.tensor_allocations().empty());

  RunSession session = run_async(plan);
  const bool observed = RuntimeLifecycleTensorCandidateProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(RuntimeLifecycleTensorCandidateProbeState::mutex);
  ASSERT_FALSE(RuntimeLifecycleTensorCandidateProbeState::error.has_value());
  EXPECT_TRUE(RuntimeLifecycleTensorCandidateProbeState::allocate_saw_pool);
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::allocate_rate.resolved_nominal_rate,
            Rate::hz(60U));
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::allocate_rate.state,
            PortRateViewState::kResolved);
  EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::compute_rate,
            RuntimeLifecycleTensorCandidateProbeState::allocate_rate);
  EXPECT_TRUE(RuntimeLifecycleTensorCandidateProbeState::received);
}

// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, MissingCandidateRateFailsBeforeAllocateWithoutPartialReadinessCleanup) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control, true);
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_candidate_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "PORT_RATE_BINDING_MISSING");
  EXPECT_FALSE(RuntimeLifecycleTensorCandidateProbeState::allocate_saw_pool);
  EXPECT_EQ(control->snapshot(0U).maximum_pool_generation, 0U);
}

// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, DiscoverRateRestartReusesCompatibleTensorBackingAndPublishesNewBinding) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(30U));
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_rate_restart_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_TRUE(plan.tensor_allocations().empty());
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());

  RunSession session = run_async(plan);
  const bool observed = RuntimeLifecycleTensorRateRestartProbeState::wait();
  const auto retained_metrics = control->snapshot(0U);
  session.request_stop();
  session.wait();
  const auto released_metrics = control->snapshot(0U);

  std::size_t observed_sink_count{};
  std::uint32_t observed_discover_count{};
  std::uint32_t observed_allocate_count{};
  bool observed_source_completed{};
  bool observed_restart_accepted{};
  std::optional<ErrorCode> observed_error;
  PortRateView observed_initial_compute_rate;
  PortRateView observed_successor_compute_rate;
  {
    const std::lock_guard lock(RuntimeLifecycleTensorRateRestartProbeState::mutex);
    observed_sink_count = RuntimeLifecycleTensorRateRestartProbeState::sink_count;
    observed_discover_count = RuntimeLifecycleTensorRateRestartProbeState::discover_count;
    observed_allocate_count = RuntimeLifecycleTensorRateRestartProbeState::allocate_count;
    observed_source_completed = RuntimeLifecycleTensorRateRestartProbeState::source_completed;
    observed_restart_accepted = RuntimeLifecycleTensorRateRestartProbeState::restart_accepted;
    observed_error = RuntimeLifecycleTensorRateRestartProbeState::error;
    observed_initial_compute_rate = RuntimeLifecycleTensorRateRestartProbeState::first_compute_rate;
    observed_successor_compute_rate =
        RuntimeLifecycleTensorRateRestartProbeState::second_compute_rate;
  }

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token)
                        << " sink_count=" << observed_sink_count
                        << " discover_count=" << observed_discover_count
                        << " source_completed=" << observed_source_completed
                        << " restart_accepted=" << observed_restart_accepted << " error="
                        << (observed_error ? static_cast<std::uint32_t>(*observed_error) : 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeLifecycleTensorRateRestartProbeState::mutex);
    ASSERT_FALSE(RuntimeLifecycleTensorRateRestartProbeState::error.has_value());
    EXPECT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::restart_accepted);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::discover_count, 2U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::allocate_count, 2U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::sink_count, 2U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::first_value, 11U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::second_value, 22U);
    EXPECT_EQ(observed_initial_compute_rate.resolved_nominal_rate, Rate::hz(60U));
    EXPECT_EQ(observed_successor_compute_rate.resolved_nominal_rate, Rate::hz(30U));
    EXPECT_EQ(observed_initial_compute_rate.run_generation,
              observed_successor_compute_rate.run_generation);
    EXPECT_NE(observed_initial_compute_rate.rate_generation,
              observed_successor_compute_rate.rate_generation);
  }
  const auto timing = RuntimeLifecycleTensorRateRestartProbeState::restart_timing();
  ASSERT_TRUE(timing.has_value());
  EXPECT_GE(timing->request_to_discover_ns, 0);
  EXPECT_GE(timing->discover_to_allocate_ns, 0);
  EXPECT_GE(timing->allocate_to_successor_compute_ns, 0);
  EXPECT_GT(timing->request_to_successor_compute_ns, 0);
  EXPECT_EQ(timing->request_to_successor_compute_ns,
            timing->request_to_discover_ns + timing->discover_to_allocate_ns +
                timing->allocate_to_successor_compute_ns);
  RecordProperty("restart_request_to_discover_ns", timing->request_to_discover_ns);
  RecordProperty("restart_discover_to_user_allocate_ns", timing->discover_to_allocate_ns);
  RecordProperty("restart_user_allocate_to_successor_compute_ns",
                 timing->allocate_to_successor_compute_ns);
  RecordProperty("restart_request_to_successor_compute_ns",
                 timing->request_to_successor_compute_ns);
  const auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->published(), nullptr);
  const auto providers = generations->provider_allocations(*generations->published());
  ASSERT_EQ(providers.size(), 1U);
  const auto& provider = providers.front();
  EXPECT_EQ(generations->published()->run_generation,
            observed_successor_compute_rate.run_generation);
  EXPECT_EQ(generations->published()->rate_generation,
            observed_successor_compute_rate.rate_generation);
  EXPECT_EQ(provider.run_generation, observed_successor_compute_rate.run_generation);
  EXPECT_EQ(provider.rate_generation, observed_successor_compute_rate.rate_generation);
  EXPECT_EQ(provider.action, AllocationAction::kReusedLarger);
  ASSERT_TRUE(provider.allocation_generation);
  ASSERT_TRUE(provider.storage_binding_generation);
  ASSERT_TRUE(provider.previous_allocation_generation);
  EXPECT_EQ(provider.previous_allocation_generation, provider.allocation_generation);
  ASSERT_TRUE(provider.provider_materialized_entries);
  EXPECT_FALSE(provider.provider_reserved_bytes_total);
  EXPECT_EQ(retained_metrics.materialization_count, 1U);
  EXPECT_EQ(retained_metrics.live_materialization_count, 1U);
  EXPECT_EQ(retained_metrics.peak_live_materialization_count, 1U);
  EXPECT_EQ(retained_metrics.maximum_allocation_generation, 1U);
  EXPECT_NE(retained_metrics.physical_slot_count, 0U);
  EXPECT_NE(retained_metrics.active_entry_limit, 0U);
  EXPECT_LT(retained_metrics.active_entry_limit, retained_metrics.physical_slot_count);
  EXPECT_EQ(retained_metrics.live_physical_entry_count, retained_metrics.physical_slot_count);
  EXPECT_EQ(retained_metrics.peak_live_physical_entry_count, retained_metrics.physical_slot_count);
  EXPECT_EQ(retained_metrics.live_requested_payload_backing_bytes,
            static_cast<std::uint64_t>(retained_metrics.physical_slot_count) * 8U);
  EXPECT_EQ(retained_metrics.peak_live_requested_payload_backing_bytes,
            retained_metrics.live_requested_payload_backing_bytes);
  EXPECT_EQ(retained_metrics.allocation_action,
            detail::RuntimeTensorPoolAllocationAction::kReusedLarger);
  EXPECT_LT(retained_metrics.minimum_pool_generation, retained_metrics.maximum_pool_generation);
  EXPECT_GE(retained_metrics.maximum_pool_generation, 2U);
  EXPECT_EQ(retained_metrics.maximum_published_pool_generation,
            retained_metrics.maximum_pool_generation);
  EXPECT_EQ(provider.allocation_generation->value, retained_metrics.maximum_allocation_generation);
  EXPECT_EQ(provider.provider_materialized_entries, retained_metrics.physical_slot_count);
  EXPECT_EQ(released_metrics.live_materialization_count, 0U);
  EXPECT_EQ(released_metrics.live_physical_entry_count, 0U);
  EXPECT_EQ(released_metrics.live_requested_payload_backing_bytes, 0U);
  EXPECT_EQ(released_metrics.peak_live_materialization_count,
            retained_metrics.peak_live_materialization_count);
  EXPECT_EQ(released_metrics.peak_live_physical_entry_count,
            retained_metrics.peak_live_physical_entry_count);
  EXPECT_EQ(released_metrics.peak_live_requested_payload_backing_bytes,
            retained_metrics.peak_live_requested_payload_backing_bytes);

  RecordProperty("restart_probe_kind", "instrumented_conformance");
  RecordProperty("restart_discover_callback_count", observed_discover_count);
  RecordProperty("restart_allocate_callback_count", observed_allocate_count);
  RecordProperty("restart_initial_run_generation", observed_initial_compute_rate.run_generation);
  RecordProperty("restart_successor_run_generation",
                 observed_successor_compute_rate.run_generation);
  RecordProperty("restart_initial_rate_generation", observed_initial_compute_rate.rate_generation);
  RecordProperty("restart_successor_rate_generation",
                 observed_successor_compute_rate.rate_generation);
  RecordProperty("restart_published_run_generation", generations->published()->run_generation);
  RecordProperty("restart_published_rate_generation", generations->published()->rate_generation);
  RecordProperty("restart_retained_allocation_generation", provider.allocation_generation->value);
  RecordProperty("restart_previous_allocation_generation",
                 provider.previous_allocation_generation->value);
  RecordProperty("restart_maximum_allocation_generation",
                 retained_metrics.maximum_allocation_generation);
  RecordProperty("restart_storage_binding_generation", provider.storage_binding_generation->value);
  RecordProperty("restart_allocation_action", "reused_larger");
  RecordProperty("restart_materialization_count", retained_metrics.materialization_count);
  RecordProperty("restart_live_materialization_count", retained_metrics.live_materialization_count);
  RecordProperty("restart_peak_live_materialization_count",
                 retained_metrics.peak_live_materialization_count);
  RecordProperty("restart_physical_slot_count", retained_metrics.physical_slot_count);
  RecordProperty("restart_active_entry_limit", retained_metrics.active_entry_limit);
  RecordProperty("restart_live_physical_entry_count", retained_metrics.live_physical_entry_count);
  RecordProperty("restart_peak_live_physical_entry_count",
                 retained_metrics.peak_live_physical_entry_count);
  RecordProperty("restart_live_logical_requested_payload_bytes",
                 retained_metrics.live_requested_payload_backing_bytes);
  RecordProperty("restart_peak_live_logical_requested_payload_bytes",
                 retained_metrics.peak_live_requested_payload_backing_bytes);
  RecordProperty("restart_logical_requested_payload_byte_span_per_slot", 8U);
  RecordProperty("restart_provider_materialized_entries", *provider.provider_materialized_entries);
  RecordProperty("restart_post_stop_live_materialization_count",
                 released_metrics.live_materialization_count);
  RecordProperty("restart_post_stop_live_physical_entry_count",
                 released_metrics.live_physical_entry_count);
  RecordProperty("restart_post_stop_live_logical_requested_payload_bytes",
                 released_metrics.live_requested_payload_backing_bytes);
  RecordProperty("restart_provider_reserved_bytes_available", "false");
  RecordProperty("restart_provider_reserved_bytes_unavailable_reason",
                 "provider_does_not_publish_reservation_bytes");
  RecordProperty("restart_minimum_pool_generation", retained_metrics.minimum_pool_generation);
  RecordProperty("restart_maximum_pool_generation", retained_metrics.maximum_pool_generation);
  RecordProperty("restart_maximum_published_pool_generation",
                 retained_metrics.maximum_published_pool_generation);
}

// An unchanged lifecycle-selected rate must rebind the existing physical backing exactly. The
// private resource witness also proves that no successor materialization was created and that the
// test-observed requested payload bytes return to zero when the session releases its pool.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, SameDiscoverRateRestartReusesExactTensorBackingWithoutOverlap) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(60U));
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_rate_restart_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  detail::RuntimeTensorOutputTestControl::PoolSnapshot live;
  {
    RunSession session = run_async(plan);
    ASSERT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::wait())
        << (session.diagnostics().empty() ? "no runtime diagnostic"
                                          : session.diagnostics().front().token);
    live = control->snapshot(0U);
    session.request_stop();
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kStopped);
    ASSERT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

    const auto generations = session.capture_rate_generations();
    ASSERT_TRUE(generations) << generations.error().message;
    ASSERT_NE(generations->published(), nullptr);
    const auto providers = generations->provider_allocations(*generations->published());
    ASSERT_EQ(providers.size(), 1U);
    EXPECT_EQ(providers.front().action, AllocationAction::kReusedExact);
    ASSERT_TRUE(providers.front().allocation_generation);
    ASSERT_TRUE(providers.front().storage_binding_generation);
    EXPECT_EQ(providers.front().allocation_generation->value, 1U);
    EXPECT_GE(providers.front().storage_binding_generation->value, 2U);

    EXPECT_EQ(live.materialization_count, 1U);
    EXPECT_EQ(live.live_materialization_count, 1U);
    EXPECT_EQ(live.peak_live_materialization_count, 1U);
    EXPECT_EQ(live.live_physical_entry_count, live.physical_slot_count);
    EXPECT_EQ(live.peak_live_physical_entry_count, live.physical_slot_count);
    EXPECT_EQ(live.live_requested_payload_backing_bytes,
              static_cast<std::uint64_t>(live.physical_slot_count) * 8U);
    EXPECT_EQ(live.peak_live_requested_payload_backing_bytes,
              live.live_requested_payload_backing_bytes);
  }

  const auto released = control->snapshot(0U);
  EXPECT_EQ(released.live_materialization_count, 0U);
  EXPECT_EQ(released.live_physical_entry_count, 0U);
  EXPECT_EQ(released.live_requested_payload_backing_bytes, 0U);
  EXPECT_EQ(released.peak_live_materialization_count, 1U);
  EXPECT_EQ(released.peak_live_physical_entry_count, live.physical_slot_count);
  EXPECT_EQ(released.peak_live_requested_payload_backing_bytes,
            static_cast<std::uint64_t>(live.physical_slot_count) * 8U);
}

// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, MaximumBackedTensorRateGrowthWithinAdmittedCeilingReusesPhysicalBacking) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(60U), Rate::hz(30U));
  const ExecutionPlan plan =
      make_runtime_lifecycle_tensor_rate_restart_plan(control, Rate::hz(120U));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);

  RunSession session = run_async(plan);
  const bool observed = RuntimeLifecycleTensorRateRestartProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeLifecycleTensorRateRestartProbeState::mutex);
    ASSERT_FALSE(RuntimeLifecycleTensorRateRestartProbeState::error.has_value());
    EXPECT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::restart_accepted);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::sink_count, 2U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::first_compute_rate.resolved_nominal_rate,
              Rate::hz(30U));
    EXPECT_EQ(
        RuntimeLifecycleTensorRateRestartProbeState::second_compute_rate.resolved_nominal_rate,
        Rate::hz(60U));
  }
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.materialization_count, 1U);
  EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
  EXPECT_GT(metrics.physical_slot_count, metrics.active_entry_limit);
  EXPECT_EQ(metrics.allocation_action, detail::RuntimeTensorPoolAllocationAction::kReusedLarger);
  EXPECT_GE(metrics.maximum_pool_generation, 2U);
  EXPECT_GE(metrics.maximum_published_pool_generation, 2U);
}

// A finite admitted ceiling permits a forced-incompatible candidate to materialize a successor.
// Publication overlaps predecessor and candidate until the restart fence retires the old pool;
// rollback/rejection tests below retain the predecessor instead.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, FiniteMaximumRateRestartMaterializesAdmittedReplacementWithBoundedOverlap) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  control->arm_fault(detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure);
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(30U));
  const ExecutionPlan plan =
      make_runtime_lifecycle_tensor_rate_restart_plan(control, Rate::hz(120U));
  ASSERT_TRUE(plan.ok()) << plan.json();

  detail::RuntimeTensorOutputTestControl::PoolSnapshot live;
  {
    RunSession session = run_async(plan);
    ASSERT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::wait())
        << (session.diagnostics().empty() ? "no runtime diagnostic"
                                          : session.diagnostics().front().token);
    live = control->snapshot(0U);
    session.request_stop();
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kStopped);
    ASSERT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

    EXPECT_EQ(live.materialization_count, 2U);
    EXPECT_EQ(live.live_materialization_count, 2U);
    EXPECT_EQ(live.peak_live_materialization_count, 2U);
    EXPECT_EQ(live.live_physical_entry_count,
              static_cast<std::uint64_t>(live.physical_slot_count) * 2U);
    EXPECT_EQ(live.peak_live_physical_entry_count, live.live_physical_entry_count);
    EXPECT_EQ(live.live_requested_payload_backing_bytes,
              static_cast<std::uint64_t>(live.physical_slot_count) * 2U * 8U);
    EXPECT_EQ(live.peak_live_requested_payload_backing_bytes,
              live.live_requested_payload_backing_bytes);
    EXPECT_EQ(live.maximum_allocation_generation, 2U);
    EXPECT_EQ(live.allocation_action, detail::RuntimeTensorPoolAllocationAction::kReplaced);
    EXPECT_GE(live.maximum_published_pool_generation, 2U);
  }

  const auto released = control->snapshot(0U);
  EXPECT_EQ(released.live_materialization_count, 0U);
  EXPECT_EQ(released.live_physical_entry_count, 0U);
  EXPECT_EQ(released.live_requested_payload_backing_bytes, 0U);
  EXPECT_EQ(released.peak_live_materialization_count, 2U);
  EXPECT_EQ(released.peak_live_physical_entry_count,
            static_cast<std::uint64_t>(live.physical_slot_count) * 2U);
  EXPECT_EQ(released.peak_live_requested_payload_backing_bytes,
            static_cast<std::uint64_t>(live.physical_slot_count) * 2U * 8U);
}

// A kWarmUp/kArm binding may change only the logical active limit over backing admitted before
// kAllocate. If retained backing is incompatible with reuse, restart must fail closed instead of
// allocating a replacement after the allocation deadline.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, ArmRateRestartNeverAllocatesReplacementBacking) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  control->arm_fault(detail::RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure);
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(30U));
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_rate_restart_plan(
      control, Rate::hz(120U), LifecycleStage::kArm, LifecycleStage::kArm);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "QUEUE_REPLACEMENT_NOT_ADMITTED");
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.materialization_count, 1U);
  EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
  EXPECT_EQ(metrics.maximum_published_pool_generation, 1U);
  EXPECT_EQ(metrics.allocation_action, detail::RuntimeTensorPoolAllocationAction::kFresh);
}

// A Discover-selected rate above the declared maximum is rejected by the rate barrier before the
// restart can enter kAllocate. The rejected candidate remains inspectable while the previously
// published generation and its sole materialization are not replaced.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, AboveMaximumDiscoverRateRestartFailsBeforeAllocateAndRetainsPublishedBacking) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(121U));
  const ExecutionPlan plan =
      make_runtime_lifecycle_tensor_rate_restart_plan(control, Rate::hz(120U));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kCompileConcrete);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "PORT_RATE_BINDING_ABOVE_MAXIMUM");
  {
    const std::lock_guard lock(RuntimeLifecycleTensorRateRestartProbeState::mutex);
    EXPECT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::restart_accepted);
    // Initial + rejected restart-forward + restoration callbacks; kAllocate runs only initially.
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::discover_count, 3U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::allocate_count, 1U);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::sink_count, 1U);
    EXPECT_TRUE(RuntimeLifecycleTensorRateRestartProbeState::error.has_value());
  }

  const auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->published(), nullptr);
  ASSERT_NE(generations->last_rejected(), nullptr);
  EXPECT_NE(generations->published()->rate_generation,
            generations->last_rejected()->rate_generation);
  EXPECT_EQ(generations->last_rejected()->diagnostic.token, "PORT_RATE_BINDING_ABOVE_MAXIMUM");
  EXPECT_EQ(generations->last_rejected()->diagnostic.id,
            detail::diagnostic_id("PORT_RATE_BINDING_ABOVE_MAXIMUM"));

  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.materialization_count, 1U);
  EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
  EXPECT_EQ(metrics.maximum_published_pool_generation, 1U);
  EXPECT_EQ(metrics.allocation_action, detail::RuntimeTensorPoolAllocationAction::kFresh);
  EXPECT_EQ(metrics.live_materialization_count, 0U);
  EXPECT_EQ(metrics.peak_live_materialization_count, 1U);
  EXPECT_EQ(metrics.live_physical_entry_count, 0U);
  EXPECT_EQ(metrics.peak_live_physical_entry_count, metrics.physical_slot_count);
}

// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, LifecycleCandidateRateGrowthFailsClosedBeforePublishingUndersizedBacking) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorRateRestartProbeState::reset(Rate::hz(60U), Rate::hz(30U));
  const ExecutionPlan plan = make_runtime_lifecycle_tensor_rate_restart_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  EXPECT_EQ(plan.provider_allocation_plans().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);

  RunSession session = run_async(plan);
  auto waiter = std::async(std::launch::async, [&session] { session.wait(); });
  const auto completed = waiter.wait_for(5s);
  if (completed != std::future_status::ready) {
    session.request_stop();
    waiter.wait();
  }

  ASSERT_EQ(completed, std::future_status::ready)
      << (session.diagnostics().empty() ? "no runtime diagnostic"
                                        : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "QUEUE_REPLACEMENT_NOT_ADMITTED");
  const auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->last_rejected(), nullptr);
  EXPECT_EQ(generations->last_rejected()->diagnostic.token, "QUEUE_REPLACEMENT_NOT_ADMITTED");
  EXPECT_EQ(generations->last_rejected()->diagnostic.id,
            detail::diagnostic_id("QUEUE_REPLACEMENT_NOT_ADMITTED"));
  EXPECT_EQ(generations->last_rejected()->diagnostic.provided, "QUEUE_REPLACEMENT_NOT_ADMITTED");
  {
    const std::lock_guard lock(RuntimeLifecycleTensorRateRestartProbeState::mutex);
    EXPECT_EQ(RuntimeLifecycleTensorRateRestartProbeState::first_compute_rate.resolved_nominal_rate,
              Rate::hz(30U));
  }
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.materialization_count, 1U);
  EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
  EXPECT_EQ(metrics.allocation_action, detail::RuntimeTensorPoolAllocationAction::kFresh);
  EXPECT_EQ(metrics.maximum_published_pool_generation, 1U);
  EXPECT_EQ(metrics.live_materialization_count, 0U);
  EXPECT_EQ(metrics.peak_live_materialization_count, 1U);
  EXPECT_EQ(metrics.live_physical_entry_count, 0U);
  EXPECT_EQ(metrics.peak_live_physical_entry_count, metrics.physical_slot_count);
  EXPECT_EQ(metrics.live_requested_payload_backing_bytes, 0U);
  EXPECT_EQ(metrics.peak_live_requested_payload_backing_bytes,
            static_cast<std::uint64_t>(metrics.physical_slot_count) * 8U);
}

// REQ: OPERATOR-025
// REQ: OPERATOR-029
// REQ: CHAN-098
// REQ: TENSOR-STORAGE-011
// REQ: COMPILE-069
TEST(RunSessionTest, FixedRateTensorProviderPublishesFreshAllocationAndBindingEvidence) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_fixed_rate_tensor_evidence_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);

  RunSession session = run_async(plan);
  const bool observed = RuntimeLifecycleTensorCandidateProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->published(), nullptr);
  const auto providers = generations->provider_allocations(*generations->published());
  ASSERT_EQ(providers.size(), 1U);
  EXPECT_EQ(providers.front().recipe, plan.provider_allocation_plans().front().id);
  EXPECT_EQ(providers.front().action, AllocationAction::kFresh);
  ASSERT_TRUE(providers.front().allocation_generation);
  ASSERT_TRUE(providers.front().storage_binding_generation);
  EXPECT_EQ(providers.front().provider_materialized_entries,
            plan.tensor_allocations().front().slot_count);
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.materialization_count, 1U);
  EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
}

// A targeted worker starts only the local fixed-rate Tensor component. Its disconnected remote
// peer is a no-maximum lifecycle candidate whose callbacks fail if invoked, so the successful
// local run proves that no remote lifecycle binding or pool materialization is required.
// REQ: LIFECYCLE-012
// REQ: COMPILE-069
TEST(RunSessionTest, WorkerIgnoresDisconnectedRemoteNoMaximumTensorCandidate) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control);
  RuntimeRemoteLifecycleTensorWorkerProbe::reset();
  const ExecutionPlan plan = make_runtime_worker_local_fixed_tensor_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 2U);

  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto local_vertex = std::ranges::find_if(
      state->graph.vertices(), [](const auto& vertex) { return vertex.name == "local-source"; });
  const auto remote_vertex = std::ranges::find_if(
      state->graph.vertices(), [](const auto& vertex) { return vertex.name == "remote-source"; });
  ASSERT_NE(local_vertex, state->graph.vertices().end());
  ASSERT_NE(remote_vertex, state->graph.vertices().end());
  const auto local_output = std::ranges::find_if(state->ports, [&](const auto& port) {
    return port.owner == local_vertex->id && port.direction == PortDirection::kOutput;
  });
  const auto remote_output = std::ranges::find_if(state->ports, [&](const auto& port) {
    return port.owner == remote_vertex->id && port.direction == PortDirection::kOutput;
  });
  ASSERT_NE(local_output, state->ports.end());
  ASSERT_NE(remote_output, state->ports.end());
  const auto local_provider = std::ranges::find_if(
      plan.provider_allocation_plans(),
      [&](const auto& provider) { return provider.output == local_output->id; });
  const auto remote_provider = std::ranges::find_if(
      plan.provider_allocation_plans(),
      [&](const auto& provider) { return provider.output == remote_output->id; });
  ASSERT_NE(local_provider, plan.provider_allocation_plans().end());
  ASSERT_NE(remote_provider, plan.provider_allocation_plans().end());
  EXPECT_EQ(local_provider->geometry_mode, AllocationGeometryMode::kCompileConcrete);
  EXPECT_EQ(remote_provider->geometry_mode, AllocationGeometryMode::kLifecycleCandidate);

  const ProcessGroupId local_group = detail::process_group_for(*state, local_vertex->id);
  ASSERT_TRUE(local_group);
  RunSession session = detail::run_async_worker(plan, local_group);
  const bool observed = RuntimeLifecycleTensorCandidateProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(RuntimeRemoteLifecycleTensorWorkerProbe::bind_calls.load(std::memory_order_acquire),
            0U);
  EXPECT_EQ(RuntimeRemoteLifecycleTensorWorkerProbe::allocate_calls.load(std::memory_order_acquire),
            0U);
  EXPECT_EQ(RuntimeRemoteLifecycleTensorWorkerProbe::compute_calls.load(std::memory_order_acquire),
            0U);
  {
    const std::lock_guard lock(RuntimeLifecycleTensorCandidateProbeState::mutex);
    ASSERT_FALSE(RuntimeLifecycleTensorCandidateProbeState::error.has_value());
    EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::compute_rate.resolved_nominal_rate,
              Rate::hz(60U));
    EXPECT_EQ(RuntimeLifecycleTensorCandidateProbeState::observed_value,
              RuntimeFixedRateTensorEvidenceSourceOperator::kValue);
  }

  const auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->published(), nullptr);
  EXPECT_EQ(generations->queue_sizings(*generations->published()).size(), 1U);
  EXPECT_LT(generations->port_rates(*generations->published()).size(), plan.port_rates().size());
  const auto providers = generations->provider_allocations(*generations->published());
  ASSERT_EQ(providers.size(), 1U);
  EXPECT_EQ(providers.front().recipe, local_provider->id);
  EXPECT_EQ(std::ranges::count(
                providers, remote_provider->id, &ProviderAllocationGenerationRecord::recipe),
            0U);
  EXPECT_EQ(control->snapshot(0U).materialization_count, 1U);
  EXPECT_EQ(control->snapshot(1U).materialization_count, 0U);
}

// A fixed provider is ready before allocation while a finite-maximum lifecycle provider remains
// nominally unresolved until kArm. Core may allocate the finite physical ceiling early, but must
// not fabricate its selected nominal rate or publish duplicate provider evidence before kArm.
// REQ: OPERATOR-025
// REQ: OPERATOR-029
// REQ: CHAN-098
// REQ: TENSOR-STORAGE-011
// REQ: COMPILE-069
TEST(RunSessionTest, MixedStageTensorProvidersPublishOneFreshRecordAfterArm) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeLifecycleTensorCandidateProbeState::reset(control);
  RuntimeMixedStageTensorProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_mixed_stage_tensor_evidence_plan(control);

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.provider_allocation_plans().size(), 2U);
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(
      plan.provider_allocation_plans(), [](const ProviderAllocationPlanRecord& provider) {
        return provider.geometry_mode == AllocationGeometryMode::kCompileConcrete;
      }));

  RunSession session = run_async(plan);
  ASSERT_TRUE(RuntimeMixedStageTensorProbeState::wait())
      << (session.diagnostics().empty() ? "no runtime diagnostic"
                                        : session.diagnostics().front().token);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeMixedStageTensorProbeState::mutex);
    ASSERT_FALSE(RuntimeMixedStageTensorProbeState::error.has_value());
    EXPECT_TRUE(RuntimeMixedStageTensorProbeState::allocate_saw_pool);
    EXPECT_EQ(RuntimeMixedStageTensorProbeState::allocate_rate.state,
              PortRateViewState::kPendingBarrier);
    EXPECT_FALSE(RuntimeMixedStageTensorProbeState::allocate_rate.resolved_nominal_rate);
    EXPECT_EQ(RuntimeMixedStageTensorProbeState::allocate_rate.max_nominal_rate, Rate::hz(120U));
  }

  const auto generations = session.capture_rate_generations();
  ASSERT_TRUE(generations) << generations.error().message;
  ASSERT_NE(generations->published(), nullptr);
  const auto providers = generations->provider_allocations(*generations->published());
  ASSERT_EQ(providers.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(providers, [](const auto& provider) {
    return provider.action == AllocationAction::kFresh && provider.allocation_generation &&
           provider.storage_binding_generation && provider.provider_materialized_entries;
  }));
  for (std::size_t index = 0U; index < 2U; ++index) {
    const auto metrics = control->snapshot(index);
    EXPECT_EQ(metrics.materialization_count, 1U);
    EXPECT_EQ(metrics.maximum_allocation_generation, 1U);
    EXPECT_NE(metrics.physical_slot_count, 0U);
    EXPECT_NE(metrics.active_entry_limit, 0U);
    EXPECT_EQ(metrics.allocation_action, detail::RuntimeTensorPoolAllocationAction::kFresh);
  }
}

// REQ: OPERATOR-025
// REQ: OPERATOR-029
// REQ: CHAN-098
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorPoolPreservesAlignmentExhaustionRetirementAndScalarMetadata) {
  // Key setup: max_byte_span=17 (odd) with max_outstanding_loans=2 means the pool
  // has exactly two physical slots; the source exhausts them, cancels both, then
  // publishes a scalar and probes the retired slot for reuse.
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_direct_tensor_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 17U);
  EXPECT_EQ(plan.tensor_allocations().front().max_outstanding_loans, 2U);
  EXPECT_TRUE(plan.tensor_device_resources().empty());

  RunSession session = run_async(plan);
  const bool observed = RuntimeDirectTensorProbeState::wait();
  session.request_stop();
  session.wait();

  std::optional<ErrorCode> source_error;
  bool source_initial_completed{};
  bool source_completed{};
  bool sink_completed{};
  {
    const std::lock_guard lock(RuntimeDirectTensorProbeState::mutex);
    source_error = RuntimeDirectTensorProbeState::source_error;
    source_initial_completed = RuntimeDirectTensorProbeState::source_initial_completed;
    source_completed = RuntimeDirectTensorProbeState::source_completed;
    sink_completed = RuntimeDirectTensorProbeState::sink_completed;
  }
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token)
                        << " source_initial=" << source_initial_completed
                        << " source=" << source_completed << " sink=" << sink_completed
                        << " source_error="
                        << (source_error.has_value() ? static_cast<std::uint32_t>(*source_error)
                                                     : 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const detail::RuntimeTensorOutputTestControl::PoolSnapshot metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.metrics.minimum_publisher_loan_credit, 0U);
  EXPECT_EQ(metrics.metrics.minimum_effective_free_slot_depth, 0U);
  EXPECT_EQ(metrics.metrics.zero_depth_total_ns, 60U);
  EXPECT_EQ(metrics.metrics.zero_depth_longest_ns, 60U);
  EXPECT_GE(metrics.metrics.zero_depth_interval_count, 1U);
  EXPECT_EQ(metrics.metrics.failed_loan_count, 1U);
  EXPECT_GT(metrics.metrics.failed_loan_physical_free_slot_depth, 0U);
  EXPECT_EQ(metrics.metrics.failed_loan_publisher_credit, 0U);
  EXPECT_EQ(metrics.metrics.failed_loan_effective_free_slot_depth, 0U);
  EXPECT_FALSE(metrics.metrics.failed_loan_physical_limited);
  EXPECT_TRUE(metrics.metrics.failed_loan_credit_limited);
  EXPECT_NE(metrics.metrics.failed_loan_blocking_claim_mask &
                (1U << static_cast<std::uint8_t>(TensorSlotClaimReason::kProducerLoan)),
            0U);
  EXPECT_NE(metrics.metrics.failed_loan_blocking_claim_mask &
                (1U << static_cast<std::uint8_t>(TensorSlotClaimReason::kProducerReadiness)),
            0U);
  EXPECT_LT(metrics.metrics.failed_loan_oldest_retained_slot,
            plan.tensor_allocations().front().slot_count);
  EXPECT_EQ(metrics.metrics.failed_loan_oldest_retained_age_ns, 50U);
  EXPECT_NE(metrics.metrics.failed_loan_oldest_retained_claim_mask, 0U);
  EXPECT_EQ(metrics.metrics.retry_cancellation_total_started, 1U);
  EXPECT_EQ(metrics.metrics.peak_retry_cancellation_demand, 1U);
  const std::lock_guard lock(RuntimeDirectTensorProbeState::mutex);
  // Key assertion: every pool-level behavioral flag is true and warm_path_allocations
  // is zero (slot reused without malloc after retirement).
  EXPECT_FALSE(RuntimeDirectTensorProbeState::source_error.has_value());
  EXPECT_TRUE(RuntimeDirectTensorProbeState::aligned_slots);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::exhausted);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::canceled_ready_slot);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::rejected_invalid_layout);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::scalar_metadata);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::host_copy_supported);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::active_writer_retained);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::published_pointer_preserved);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::retained_pointer_not_reused);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::final_claim_released_for_reuse);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::pre_submit_backpressured);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::validation_precedence);
  EXPECT_EQ(RuntimeDirectTensorProbeState::warm_path_allocations, 0U);
  EXPECT_EQ(RuntimeDirectTensorProbeState::observed_value,
            RuntimeDirectTensorSourceOperator::kPublishedValue);
}

// The descriptor-backed Tensor route reaches a different publish site than the inline and schema
// routes, and its receive boundary validates the publication coordinate against the clock this
// session stamps from. Arrival is therefore itself an assertion: a coordinate the boundary refused
// would fail the sample instead of delivering it, so the sink would never observe anything.
//
// This is an in-graph flow, which is what makes it worth pinning separately. The catalog import
// flag that selects the absent-coordinate rule is only ever set for imported external topics, so
// an in-graph descriptor sample must carry the plan clock rather than canonical absence.
// REQ: CHAN-083
// REQ: CHAN-096
// REQ: RUNTIME-019
TEST(RunSessionTest, InGraphDescriptorTensorCarriesRuntimePublicationCoordinate) {
  RuntimeTensorPublicationProbeState::reset();
  const ExecutionPlan plan = make_runtime_tensor_publication_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = RuntimeTensorPublicationProbeState::wait();
  session.request_stop();
  session.wait();

  std::optional<SampleMetadata> metadata;
  std::optional<ErrorCode> error;
  {
    const std::lock_guard lock(RuntimeTensorPublicationProbeState::mutex);
    metadata = RuntimeTensorPublicationProbeState::observed;
    error = RuntimeTensorPublicationProbeState::error;
  }
  ASSERT_FALSE(error.has_value()) << "descriptor Tensor publication failed with error code "
                                  << static_cast<std::uint32_t>(*error);
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  ASSERT_TRUE(metadata.has_value());
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

  // Key assertion: the coordinate names the Graph default clock, which is what lets the consumer
  // place this sample on the same timeline Runtime published it against.
  EXPECT_EQ(metadata->payload_kind, PayloadKind::kDescriptorBacked);
  const ClockId publish_clock = plan.clock_catalog().root_default_clock;
  ASSERT_TRUE(publish_clock.valid());
  EXPECT_EQ(metadata->publish_clock_id, publish_clock);
  EXPECT_GT(metadata->publish_timestamp_ns, 0);
  EXPECT_TRUE(metadata->publish_time().valid());
}

// REQ: OPERATOR-042
// REQ: OPERATOR-044
TEST(RunSessionTest, TensorPoolsEnforceAxisAndStrideContractsBeforeAllocating) {
  RuntimeTensorContractAdmissionProbeState::reset();
  const ExecutionPlan plan = make_runtime_tensor_contract_admission_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);
  for (const auto& allocation : plan.tensor_allocations()) {
    EXPECT_EQ(allocation.axes.size(), 2U);
  }

  RunSession session = run_async(plan);
  const bool observed = RuntimeTensorContractAdmissionProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(RuntimeTensorContractAdmissionProbeState::mutex);
  EXPECT_FALSE(RuntimeTensorContractAdmissionProbeState::error.has_value());
}

// Host-written pinned storage has no asynchronous producer work, so the consumer does not need a
// Runtime CUDA stream. CUDA-written pinned storage remains covered by the readiness-fence tests.
// REQ: OPERATOR-030
TEST(RunSessionTest, DirectPinnedHostTensorSupportsHostConsumerWithoutCudaStream) {
  if (cudaFree(nullptr) != cudaSuccess) {
    GTEST_SKIP() << "CUDA pinned allocation is unavailable";
  }

  // Key setup: reset with kPinnedHost so the plan allocates a pinned pool with no device stream.
  RuntimeDirectTensorProbeState::reset({}, MemoryKind::kPinnedHost);
  const ExecutionPlan plan = make_runtime_direct_tensor_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kPinnedHost);
  // Key assertion: no TensorDeviceResource means no CUDA stream obligation on the consumer.
  EXPECT_TRUE(plan.tensor_device_resources().empty());

  RunSession session = run_async(plan);
  const bool observed = RuntimeDirectTensorProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(RuntimeDirectTensorProbeState::mutex);
  EXPECT_FALSE(RuntimeDirectTensorProbeState::source_error.has_value());
  EXPECT_TRUE(RuntimeDirectTensorProbeState::source_completed);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::sink_completed);
  EXPECT_TRUE(RuntimeDirectTensorProbeState::host_copy_supported);
  EXPECT_EQ(RuntimeDirectTensorProbeState::observed_value,
            RuntimeDirectTensorSourceOperator::kPublishedValue);
}

// A pinned-host pool may still carry a private CUDA device for external transport. Synchronous
// host writes do not acquire that stream and must not publish a CUDA readiness generation.
// REQ: OPERATOR-030
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, PinnedHostTensorHostWritesDoNotRecordCudaReadiness) {
  if (cudaFree(nullptr) != cudaSuccess) {
    GTEST_SKIP() << "CUDA pinned allocation is unavailable";
  }

  // Graph topology:
  //   source --[queue_depth=1]--> holomq://endpoint
  // Key setup: device-place an external pinned-host Tensor pool, then publish through write_host()
  // so the pool has CUDA readiness resources but the producer performs no asynchronous CUDA work.
  int device_ordinal = -1;
  ASSERT_EQ(cudaGetDevice(&device_ordinal), cudaSuccess);
  const std::string graph_name = unique_runtime_graph_name("runtime-pinned-host-tensor-export");
  const std::string endpoint_name = graph_name + "-topic";
  Graph graph{graph_name};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  auto source = graph.op<RuntimePinnedHostTensorPublicationSourceOperator>("source");
  TopicOptions topic_options;
  topic_options.endpoint_name = endpoint_name;
  topic_options.attach = TopicAttachPolicy{.peer = PeerRequirement::kRequired, .max_wait = 5s};
  topic_options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  topic_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 64U, 1U},
                         .rank = 0U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = tensor_bounds(sizeof(RuntimePinnedHostTensorPublicationSourceOperator::kValue)),
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});

  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "output",
      .device = DeviceId{device_ordinal},
  });
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_tensor_output_test_control(
      compile(graph, std::move(options)), control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(plan.tensor_allocations().front().device, DeviceId{device_ordinal});
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);

  auto client_runtime = ChannelRuntime::create();
  ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());
  RunSession session = run_async(plan);
  auto subscriber =
      client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  expected<Sample<Tensor>, Error> received = make_unexpected(Error{ErrorCode::kNotReady});
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!received && received.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < deadline) {
    received = subscriber->receive_sample();
    if (!received) {
      std::this_thread::sleep_for(1ms);
    }
  }
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received) << runtime_error_message(received.error());
  ASSERT_NE(received->data.data_as<std::uint64_t>(), nullptr);
  EXPECT_EQ(*received->data.data_as<std::uint64_t>(),
            RuntimePinnedHostTensorPublicationSourceOperator::kValue);
  // The one-shot source may naturally complete before the explicit cleanup stop reaches it.
  EXPECT_TRUE(session.termination() == RunTermination::kStopped ||
              session.termination() == RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = control->snapshot(0U);
  // Key assertion: neither a real readiness event nor the same-stream elision path runs for a
  // synchronous host writer. CUDA writers retain the positive readiness assertions above.
  EXPECT_EQ(snapshot.cuda_readiness_records, 0U);
  EXPECT_EQ(snapshot.cuda_readiness_record_elisions, 0U);
}

// Each fanout consumer registers a distinct application stream. Releasing the stream that was
// registered last must not permit slot reuse while earlier consumer work remains queued.
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, DirectTensorPoolWaitsForEveryFanoutConsumerStreamBeforeReuse) {
  // Key setup: blocked_stream is held via a host callback while quick_stream
  // completes immediately, forcing the source to observe slot exhaustion and
  // probe for premature pointer reuse on subsequent allocations.
  cudaStream_t blocked_stream = nullptr;
  if (cudaStreamCreateWithFlags(&blocked_stream, cudaStreamNonBlocking) != cudaSuccess) {
    GTEST_SKIP() << "CUDA stream creation is unavailable";
  }
  cudaStream_t quick_stream = nullptr;
  if (cudaStreamCreateWithFlags(&quick_stream, cudaStreamNonBlocking) != cudaSuccess) {
    static_cast<void>(cudaStreamDestroy(blocked_stream));
    GTEST_SKIP() << "second CUDA stream creation is unavailable";
  }
  struct StreamCleanup {
    cudaStream_t blocked{};
    cudaStream_t quick{};
    ~StreamCleanup() {
      static_cast<void>(cudaStreamDestroy(quick));
      static_cast<void>(cudaStreamDestroy(blocked));
    }
  };
  const StreamCleanup stream_cleanup{blocked_stream, quick_stream};

  RuntimeDirectTensorFanoutStreamProbeState::reset(blocked_stream, quick_stream);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_direct_tensor_fanout_stream_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);

  RunSession session = run_async(plan);
  struct BlockedStreamRelease {
    ~BlockedStreamRelease() { RuntimeDirectTensorFanoutStreamProbeState::release_blocked(); }
  };
  const BlockedStreamRelease release_on_exit;

  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_blocked_stream());
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_probes(16U));
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::premature_reuse);
  }
  EXPECT_NE(control->snapshot(0U).minimum_retired_slot, 0U);

  RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_first_slot_retirement(control));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    // Key assertion: premature_reuse remains false even after both stream callbacks
    // complete, proving the pool held the slot until every consumer deregistered.
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::premature_reuse);
  }
  const auto snapshot = control->snapshot(0U);
  // Fanout prevents compiler fusion, so producer readiness is recorded eagerly. The two
  // application-selected consumer streams then each take the real cross-stream wait path.
  EXPECT_GT(snapshot.cuda_readiness_records, 0U);
  EXPECT_EQ(snapshot.cuda_readiness_record_elisions, 0U);
  EXPECT_GT(snapshot.cuda_stream_waits, 0U);
  EXPECT_EQ(snapshot.cuda_host_waits, 0U);
  EXPECT_GE(snapshot.cuda_release_stream_assignments, 3U);
  EXPECT_GE(snapshot.cuda_release_records, 3U);
}

// A positive-limit producer and both downstream callbacks have retired, but one consumer's
// application-selected CUDA stream still owns the final Tensor. Natural completion must wait for
// the corresponding release event instead of treating callback return as device-work retirement.
// REQ: CHAN-136
// REQ: RUNTIME-050
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, ComputeCallCompletionWaitsForCudaConsumerStreamRetirement) {
  cudaStream_t blocked_stream = nullptr;
  if (cudaStreamCreateWithFlags(&blocked_stream, cudaStreamNonBlocking) != cudaSuccess) {
    GTEST_SKIP() << "CUDA stream creation is unavailable";
  }
  cudaStream_t quick_stream = nullptr;
  if (cudaStreamCreateWithFlags(&quick_stream, cudaStreamNonBlocking) != cudaSuccess) {
    static_cast<void>(cudaStreamDestroy(blocked_stream));
    GTEST_SKIP() << "second CUDA stream creation is unavailable";
  }
  struct StreamCleanup {
    cudaStream_t blocked{};
    cudaStream_t quick{};
    ~StreamCleanup() {
      static_cast<void>(cudaStreamDestroy(quick));
      static_cast<void>(cudaStreamDestroy(blocked));
    }
  };
  const StreamCleanup stream_cleanup{blocked_stream, quick_stream};

  RuntimeDirectTensorFanoutStreamProbeState::reset(blocked_stream, quick_stream);
  const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan compiled = make_runtime_direct_tensor_fanout_stream_plan(
      control, true, unique_runtime_graph_name("runtime-compute-call-cuda-consumer-drain"));
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);

  RunSession session = run_async(plan);
  struct SessionCleanup {
    RunSession* session{};
    ~SessionCleanup() {
      RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
      if (session->is_running()) {
        session->request_stop();
      }
      session->wait();
    }
  } const cleanup{&session};

  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_blocked_stream());
  ASSERT_TRUE(wait_for_runtime_phase_count(
      counters, detail::RuntimePhaseCounter::kActivationsCompleted, 3U));
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_retirement_queries(control, 2U));
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    ASSERT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_EQ(RuntimeDirectTensorFanoutStreamProbeState::sink_count, 2U);
    EXPECT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::blocked_callback_started);
  }
  const auto held = control->snapshot(0U);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kRunning);
  EXPECT_TRUE(session.operator_completion_progress().empty());
  EXPECT_GT(held.active_slot_count, 0U);
  EXPECT_NE(held.minimum_retired_slot, 0U);
  EXPECT_GT(held.cuda_stream_waits, 0U);
  EXPECT_GT(held.cuda_release_stream_assignments, 0U);
  EXPECT_GE(held.cuda_retirement_queries, 2U);
  EXPECT_EQ(held.cuda_retirement_completions, 0U);

  RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_first_slot_retirement(control));
  const auto completion_deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < completion_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool completed_without_timeout = !session.is_running();
  if (!completed_without_timeout) {
    session.request_stop();
  }
  session.wait();

  EXPECT_TRUE(completed_without_timeout);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    ASSERT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value())
        << static_cast<std::uint32_t>(
               RuntimeDirectTensorFanoutStreamProbeState::error.value_or(ErrorCode::kSuccess));
    EXPECT_EQ(RuntimeDirectTensorFanoutStreamProbeState::sink_count, 2U);
  }
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.configured_limit, 1U);
  EXPECT_EQ(progress.successful_compute_calls, 1U);
  EXPECT_TRUE(progress.limit_reached);
  const auto retired = control->snapshot(0U);
  EXPECT_EQ(retired.active_slot_count, 0U);
  EXPECT_EQ(retired.minimum_retired_slot, 0U);
  EXPECT_GT(retired.cuda_release_records, 0U);
  EXPECT_GT(retired.cuda_retirement_queries, 0U);
  EXPECT_GT(retired.cuda_retirement_completions, 0U);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextTensorFanoutWaitsForEveryConsumerBeforeReuse) {
  // Key setup: fan out one Tensor from a dedicated source to blocked and quick dedicated sinks.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 3U) {
    GTEST_SKIP() << "device cannot provide three dedicated Green Context groups";
  }

  RuntimeDirectTensorFanoutStreamProbeState::reset(nullptr, nullptr);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_green_context_tensor_fanout_plan(
      *capability,
      control,
      unique_runtime_graph_name("runtime-green-context-tensor-fanout-retirement"));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);

  std::optional<RunSession> session{run_async(plan)};
  struct BlockedStreamRelease {
    ~BlockedStreamRelease() { RuntimeDirectTensorFanoutStreamProbeState::release_blocked(); }
  };
  const BlockedStreamRelease release_on_exit;
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_blocked_stream());
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_probes(16U));
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::premature_reuse);
  }
  EXPECT_NE(control->snapshot(0U).minimum_retired_slot, 0U);

  RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_first_slot_retirement(control));
  session->request_stop();
  session->wait();
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  session.reset();

  const auto metrics = control->snapshot(0U);
  // Key assertion: no slot is reused before both consumers retire, and all streams/events clean up.
  EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::premature_reuse);
  EXPECT_GE(metrics.cuda_streams_created, 3U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
  EXPECT_GE(metrics.cuda_release_stream_assignments, 2U);
  EXPECT_GE(metrics.cuda_release_records, 2U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(metrics.cuda_events_created,
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 4U);
}

// REQ: TENSOR-STORAGE-014
TEST(RunSessionTest, GreenContextTensorCancellationQuiescesInflightConsumerEvents) {
  // Key setup: stop the three-partition fanout session while one consumer stream remains blocked.
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 3U) {
    GTEST_SKIP() << "device cannot provide three dedicated Green Context groups";
  }

  RuntimeDirectTensorFanoutStreamProbeState::reset(nullptr, nullptr);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_green_context_tensor_fanout_plan(
      *capability,
      control,
      unique_runtime_graph_name("runtime-green-context-tensor-fanout-cancellation"));
  ASSERT_TRUE(plan.ok()) << plan.json();

  std::optional<RunSession> session{run_async(plan)};
  struct BlockedStreamRelease {
    ~BlockedStreamRelease() { RuntimeDirectTensorFanoutStreamProbeState::release_blocked(); }
  };
  const BlockedStreamRelease release_on_exit;
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_blocked_stream());
  session->request_stop();
  RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
  session->wait();
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  session.reset();

  const auto metrics = control->snapshot(0U);
  // Key assertion: cancellation quiesces every slot claim and destroys all streams and events.
  EXPECT_TRUE(metrics.terminal);
  EXPECT_EQ(metrics.active_slot_count, 0U);
  EXPECT_EQ(metrics.active_claim_mask, 0U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
}

// The production binding validates the complete frozen order before publication. Each combined
// fault proves the earlier identity/state category wins, while a loan used from outside its
// callback thread or from a successor activation cannot regain output authority.
// REQ: OPERATOR-025
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorEmitConfinementAndCheckedFailurePrecedenceAreExact) {
  // Key setup: probe_after_callback_use() attempts to open a write guard on a loan
  // held from a previous callback activation — crossing the activation boundary must
  // be diagnosed as kLoanActivationGenerationMismatch with no partial state change.
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorPrecedenceProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_direct_tensor_precedence_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);

  RunSession session = run_async(plan);
  const bool loan_ready = RuntimeDirectTensorPrecedenceProbeState::wait_for_after_callback_loan();
  ASSERT_TRUE(loan_ready);
  RuntimeDirectTensorPrecedenceProbeState::probe_after_callback_use();
  const bool observed = RuntimeDirectTensorPrecedenceProbeState::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: all four precedence flags and after_callback_rejected are true,
  // and after_callback_reason_exact confirms the exact diagnostic code.
  const std::lock_guard lock(RuntimeDirectTensorPrecedenceProbeState::mutex);
  EXPECT_FALSE(RuntimeDirectTensorPrecedenceProbeState::error.has_value());
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::wrong_output_precedes_closed);
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::plan_precedes_activation);
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::activation_precedes_writer);
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::invariant_precedes_metadata);
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::after_callback_rejected);
  EXPECT_TRUE(RuntimeDirectTensorPrecedenceProbeState::after_callback_reason_exact);
}

// Escaped public authority keeps only its opaque slot state alive. Once RunSession destroys the
// owning pool, every callback observes the detached pool-control fence and fails closed instead of
// dereferencing the former OutputPool address.
// REQ: OPERATOR-025
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorLoanOutlivingRunSessionFailsClosed) {
  // Key setup: escape_writer=true causes the source to pass both a loan and its
  // open write guard to the probe state before the test destroys the session.
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorPrecedenceProbeState::reset(control, true);
  const ExecutionPlan plan = make_runtime_direct_tensor_precedence_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  std::optional<RunSession> session{run_async(plan)};
  ASSERT_TRUE(RuntimeDirectTensorPrecedenceProbeState::wait_for_after_callback_loan());
  session->request_stop();
  session->wait();
  session.reset();

  std::optional<TensorOutputLoan> escaped;
  std::optional<TensorOutputWriteGuard> escaped_guard;
  {
    const std::lock_guard lock(RuntimeDirectTensorPrecedenceProbeState::mutex);
    if (RuntimeDirectTensorPrecedenceProbeState::after_callback_loan) {
      escaped.emplace(std::move(*RuntimeDirectTensorPrecedenceProbeState::after_callback_loan));
    }
    if (RuntimeDirectTensorPrecedenceProbeState::after_callback_guard) {
      escaped_guard.emplace(
          std::move(*RuntimeDirectTensorPrecedenceProbeState::after_callback_guard));
    }
    RuntimeDirectTensorPrecedenceProbeState::after_callback_loan.reset();
    RuntimeDirectTensorPrecedenceProbeState::after_callback_guard.reset();
  }
  if (!escaped.has_value() || !escaped_guard.has_value()) {
    FAIL() << "expected the callback to retain both Tensor loan authorities";
    return;
  }
  TensorOutputLoan loan{std::move(*escaped)};
  TensorOutputWriteGuard guard{std::move(*escaped_guard)};
  EXPECT_EQ(loan.shape().size(), 0U);
  EXPECT_EQ(guard.data(), nullptr);
  // Key assertion: every operation on the escaped loan/guard returns kClosed,
  // confirming the pool-control fence is observed without dereferencing freed memory.
  auto committed = std::move(guard).commit();
  ASSERT_FALSE(committed);
  EXPECT_EQ(committed.error().code, ErrorCode::kClosed);
  escaped_guard.reset();
  auto write = loan.write_host();
  ASSERT_FALSE(write);
  EXPECT_EQ(write.error().code, ErrorCode::kClosed);
  auto canceled = std::move(loan).cancel();
  ASSERT_FALSE(canceled);
  EXPECT_EQ(canceled.error().code, ErrorCode::kClosed);
  escaped.reset();
}

// A deep self restart fences unpublished predecessor authority before callbacks, preserves a
// committed Tensor already owned by the Channel, and publishes a distinct replacement pool only
// after the kAllocate lifecycle suffix succeeds. The stale predecessor guard remains memory-safe
// and cannot commit into the successor epoch.
// REQ: COMPILE-049
// REQ: COMPILE-065
// REQ: CHAN-094
// REQ: OPERATOR-025
// REQ: RUNTIME-008
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorPoolDeepRestartPreservesCommittedOutputAndRejectsStaleGuard) {
  // Key setup: restart from kAllocate replaces the pool; the pre-restart write guard
  // is retained across the fence to confirm it cannot commit into the successor epoch.
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset();
  const ExecutionPlan plan =
      make_runtime_direct_tensor_restart_plan(control, false, "runtime-direct-tensor-restart-deep");
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());

  std::optional<RunSession> session{run_async(plan)};
  const bool observed = RuntimeDirectTensorRestartProbeState::wait();
  session->request_stop();
  session->wait();

  std::size_t observed_sink_count{};
  bool observed_source_completed{};
  bool observed_restart_accepted{};
  std::uint32_t observed_restart_allocate_count{};
  std::optional<ErrorCode> observed_source_error;
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    observed_sink_count = RuntimeDirectTensorRestartProbeState::sink_count;
    observed_source_completed = RuntimeDirectTensorRestartProbeState::source_completed;
    observed_restart_accepted = RuntimeDirectTensorRestartProbeState::restart_accepted;
    observed_restart_allocate_count = RuntimeDirectTensorRestartProbeState::restart_allocate_count;
    observed_source_error = RuntimeDirectTensorRestartProbeState::error;
  }
  ASSERT_TRUE(observed) << (session->diagnostics().empty() ? "no runtime diagnostic"
                                                           : session->diagnostics().front().token)
                        << " sink_count=" << observed_sink_count
                        << " source_completed=" << observed_source_completed
                        << " restart_accepted=" << observed_restart_accepted
                        << " restart_allocate_count=" << observed_restart_allocate_count
                        << " source_error="
                        << (observed_source_error.has_value()
                                ? static_cast<std::uint32_t>(*observed_source_error)
                                : 0U);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    // Key assertion: restart_allocate_count==1 confirms kAllocate ran once for the
    // new pool; stale_guard_rejected confirms the pre-restart guard was closed.
    EXPECT_FALSE(RuntimeDirectTensorRestartProbeState::error.has_value());
    EXPECT_TRUE(RuntimeDirectTensorRestartProbeState::restart_accepted);
    EXPECT_TRUE(RuntimeDirectTensorRestartProbeState::stale_guard_rejected);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::restart_allocate_count, 1U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::sink_count, 2U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::first_value, 11U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::second_value, 22U);
    ASSERT_TRUE(RuntimeDirectTensorRestartProbeState::escaped_tensor.has_value());
  }

  // Race the escaped Tensor's final backing claim against session-owned pool detachment. The
  // shared reference control either completes the release under a live pool pin or observes the
  // detached state; neither path dereferences a destroyed pool.
  std::atomic_bool release_started{false};
  std::thread releaser([&] {
    release_started.store(true, std::memory_order_release);
    RuntimeDirectTensorRestartProbeState::release_escaped_tensor();
  });
  while (!release_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  session.reset();
  releaser.join();

  const detail::RuntimeTensorOutputTestControl::PoolSnapshot snapshot = control->snapshot(0U);
  EXPECT_EQ(snapshot.minimum_pool_generation, 1U);
  EXPECT_EQ(snapshot.maximum_pool_generation, 2U);
  EXPECT_EQ(snapshot.maximum_published_pool_generation, 2U);
  EXPECT_TRUE(snapshot.terminal);
  EXPECT_EQ(snapshot.active_claim_mask, 0U);
  EXPECT_EQ(snapshot.active_slot_count, 0U);
  constexpr std::uint32_t quarantine_claim =
      1U << static_cast<std::uint8_t>(TensorSlotClaimReason::kQuarantine);
  const std::uint32_t normal_claims = snapshot.admitted_claim_mask & ~quarantine_claim;
  EXPECT_EQ(snapshot.observed_claim_mask & normal_claims, normal_claims);
  EXPECT_EQ(snapshot.observed_claim_mask & quarantine_claim, 0U);
  EXPECT_GT(snapshot.cuda_streams_created, 0U);
  EXPECT_EQ(snapshot.cuda_streams_destroyed, snapshot.cuda_streams_created);
  EXPECT_GT(snapshot.cuda_events_created, 0U);
  EXPECT_EQ(snapshot.cuda_events_destroyed, snapshot.cuda_events_created);
  // Source and sink use the same process-group/device stream. Their normal readiness handoff
  // therefore records no event and inserts no self-wait; the explicit host reads below lazily
  // record the readiness event they require.
  EXPECT_GE(snapshot.cuda_readiness_record_elisions, 3U);
  EXPECT_GE(snapshot.cuda_stream_wait_elisions, 2U);
  EXPECT_GE(snapshot.cuda_release_stream_elisions, 2U);
  EXPECT_EQ(snapshot.cuda_release_stream_assignments, 0U);
  EXPECT_EQ(snapshot.cuda_release_records, 0U);
  EXPECT_EQ(snapshot.cuda_host_waits, 2U);
  EXPECT_EQ(snapshot.cuda_readiness_records, snapshot.cuda_host_waits);
  EXPECT_GT(snapshot.cuda_retirement_queries, 0U);
  EXPECT_GT(snapshot.cuda_retirement_completions, 0U);
}

// A strict linear plan initially elides the readiness event, but application-selected work on a
// distinct stream must lazily record that retained event and insert a real cross-stream wait.
// REQ: COMPILE-049
// REQ: COMPILE-065
// REQ: CHAN-098
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, DirectTensorSameStreamElisionFallsBackToLazyCrossStreamFence) {
  // Key setup: consumer_stream is a non-blocking stream distinct from the source's
  // runtime stream; the compiler elides the readiness event for the linear path but
  // the sink's explicit cross-stream registration forces a lazy record + wait.
  cudaStream_t consumer_stream = nullptr;
  if (cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking) != cudaSuccess) {
    GTEST_SKIP() << "CUDA stream creation is unavailable";
  }
  struct StreamCleanup {
    cudaStream_t stream{};
    ~StreamCleanup() { static_cast<void>(cudaStreamDestroy(stream)); }
  };
  const StreamCleanup stream_cleanup{consumer_stream};

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset(LifecycleStage::kAllocate, consumer_stream);
  struct ProbeCleanup {
    ~ProbeCleanup() { RuntimeDirectTensorRestartProbeState::release_escaped_tensor(); }
  };
  const ProbeCleanup probe_cleanup;
  const ExecutionPlan plan =
      make_runtime_direct_tensor_restart_plan(control, false, "runtime-direct-tensor-lazy-fence");
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->process_local_channel_services.size(), 1U);
  // Key assertion: this stream-ordering evidence exercises the direct-retained Tensor handoff,
  // rather than proving only the older HoloMQ sidecar route.
  const auto& service = plan_state->process_local_channel_services.front();
  EXPECT_EQ(service.delivery_kind, detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  ASSERT_EQ(service.tensor_memory_kind, MemoryKind::kCudaManaged);
  ASSERT_TRUE(service.tensor_device.valid());
  EXPECT_FALSE(detail::tensor_service_storage_requires_stream(service, service.tensor_memory_kind));
  EXPECT_TRUE(detail::tensor_service_storage_requires_operator_cuda_stream(
      service, service.tensor_memory_kind));
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  ASSERT_FALSE(plan.cuda_compute_streams().front().fused_linear_edges.empty());

  RunSession session = run_async(plan);
  const bool observed = RuntimeDirectTensorRestartProbeState::wait();
  session.request_stop();
  session.wait();
  RuntimeDirectTensorRestartProbeState::release_escaped_tensor();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = control->snapshot(0U);
  // Key assertion: elisions > 0 confirms the same-stream fast path fired at least
  // once; readiness_records > 0 and stream_waits > 0 confirm the lazy fallback path.
  EXPECT_GT(snapshot.cuda_readiness_record_elisions, 0U);
  EXPECT_GT(snapshot.cuda_readiness_records, 0U);
  EXPECT_GT(snapshot.cuda_stream_waits, 0U);
  EXPECT_EQ(snapshot.cuda_host_waits, 0U);
  EXPECT_GT(snapshot.cuda_release_stream_assignments, 0U);
}

// One local successor plus one catalog export is valid mixed fanout, but it is not a strict
// linear chain. Compiler evidence must omit the edge so Runtime constructs with normal fences.
// REQ: COMPILE-049
// REQ: COMPILE-065
TEST(RunSessionTest, DirectTensorMixedCatalogFanoutConstructsWithFencesEnabled) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset();
  struct ProbeCleanup {
    ~ProbeCleanup() { RuntimeDirectTensorRestartProbeState::release_escaped_tensor(); }
  };
  const ProbeCleanup probe_cleanup;
  const ExecutionPlan plan =
      make_runtime_direct_tensor_restart_plan(control, true, "runtime-direct-tensor-mixed-export");
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_device_resources().size(), 1U);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  EXPECT_TRUE(plan.cuda_compute_streams().front().fused_linear_edges.empty());
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->execution_partitions.size(), 3U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 3U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 2U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 2U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  // Production automatic partitioning keeps the external-topic boundary separate and gives the
  // source and local Tensor consumer distinct callback owners. The synthetic external partition
  // has no reconstructed Operator owner.
  ASSERT_EQ(plan_state->partition_runners.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(plan_state->partition_runners, [](const auto& runner) {
    return runner.operators.size() == 1U;
  }));

  RunSession session = run_async(plan);
  const bool observed = RuntimeDirectTensorRestartProbeState::wait();
  session.request_stop();
  session.wait();
  RuntimeDirectTensorRestartProbeState::release_escaped_tensor();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = control->snapshot(0U);
  EXPECT_GT(snapshot.cuda_readiness_records, 0U);
  EXPECT_EQ(snapshot.cuda_readiness_record_elisions, 0U);
}

// Direct Tensor successor publication is preflighted before Channel, selector, and readiness
// generations commit. A failed preflight therefore leaves the predecessor pool generation as the
// only published identity and terminates the restart without a partially visible Tensor epoch.
// REQ: OPERATOR-025
// REQ: RUNTIME-008
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorRestartPreflightFailurePublishesNoSuccessorPool) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset();
  control->arm_fault(detail::RuntimeTensorOutputTestFault::kRestartPublicationPreflightFailure);
  const ExecutionPlan plan = make_runtime_direct_tensor_restart_plan(
      control, false, "runtime-direct-tensor-restart-preflight");
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  const auto snapshot = control->snapshot(0U);
  EXPECT_EQ(snapshot.minimum_pool_generation, 1U);
  // Teardown observes the rejected unbound candidate, but publication authority never advances.
  EXPECT_EQ(snapshot.maximum_pool_generation, 2U);
  EXPECT_EQ(snapshot.maximum_published_pool_generation, 1U);
}

// A restart beginning at kStart fences predecessor activation authority but retains the already
// materialized pool because kAllocate is outside the replayed suffix. The canceled predecessor
// slot is safely recycled in the same pool; the existing deep restart test proves kAllocate
// instead creates a distinct backing generation.
// REQ: CHAN-094
// REQ: OPERATOR-025
// REQ: RUNTIME-008
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorPoolShallowRestartRetainsMaterializedBacking) {
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset(LifecycleStage::kStart);
  const ExecutionPlan plan = make_runtime_direct_tensor_restart_plan(
      control, false, "runtime-direct-tensor-restart-shallow");
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);

  std::optional<RunSession> session{run_async(plan)};
  const bool observed = RuntimeDirectTensorRestartProbeState::wait();
  session->request_stop();
  session->wait();

  ASSERT_TRUE(observed) << (session->diagnostics().empty() ? "no runtime diagnostic"
                                                           : session->diagnostics().front().token);
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorRestartProbeState::error.has_value());
    EXPECT_TRUE(RuntimeDirectTensorRestartProbeState::restart_accepted);
    EXPECT_TRUE(RuntimeDirectTensorRestartProbeState::stale_guard_rejected);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::restart_allocate_count, 0U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::sink_count, 2U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::first_value, 11U);
    EXPECT_EQ(RuntimeDirectTensorRestartProbeState::second_value, 22U);
  }

  RuntimeDirectTensorRestartProbeState::release_escaped_tensor();
  session.reset();
  const detail::RuntimeTensorOutputTestControl::PoolSnapshot snapshot = control->snapshot(0U);
  EXPECT_EQ(snapshot.minimum_pool_generation, 1U);
  EXPECT_EQ(snapshot.maximum_pool_generation, 1U);
  EXPECT_EQ(snapshot.maximum_published_pool_generation, 1U);
  EXPECT_TRUE(snapshot.terminal);
  EXPECT_EQ(snapshot.active_claim_mask, 0U);
  EXPECT_EQ(snapshot.active_slot_count, 0U);
  // Each slot owns one readiness event, one Runtime-stream release event, and one bounded
  // application-selected release event for this one-subscriber graph.
  EXPECT_EQ(snapshot.cuda_events_created, 3U * plan.tensor_allocations().front().slot_count);
  EXPECT_EQ(snapshot.cuda_events_destroyed, snapshot.cuda_events_created);
  EXPECT_EQ(snapshot.cuda_streams_created, 1U);
  EXPECT_EQ(snapshot.cuda_streams_destroyed, 1U);
}

// Deterministic terminal injection proves a failed readiness commit quarantines and consumes its
// writer/loan authority, while a post-submit uncertain publication reaches the recipient exactly
// once, consumes the output authority, retains its backing claim, and cannot be retried.
// REQ: CHAN-076
// REQ: CHAN-111
// REQ: OPERATOR-025
// REQ: OPERATOR-032
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, DirectTensorTerminalFaultsConsumeAuthorityWithoutRetry) {
  // Key setup: inject distinct post-submit indeterminate and quantified-partial results after real
  // direct Tensor publications, then attempt to reuse both consumed loans.
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorFaultProbeState::reset(control);
  const ExecutionPlan plan = make_runtime_direct_tensor_fault_plan(control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  {
    RunSession session = run_async(plan);
    const bool observed = RuntimeDirectTensorFaultProbeState::wait();
    session.request_stop();
    session.wait();

    std::string observation;
    {
      const std::lock_guard lock(RuntimeDirectTensorFaultProbeState::mutex);
      observation = fmt::format("source_completed={} sink_count={} error={}",
                                RuntimeDirectTensorFaultProbeState::source_completed,
                                RuntimeDirectTensorFaultProbeState::sink_count,
                                RuntimeDirectTensorFaultProbeState::error.has_value()
                                    ? error_code_name(*RuntimeDirectTensorFaultProbeState::error)
                                    : "none");
    }
    ASSERT_TRUE(observed) << observation;
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    const std::lock_guard lock(RuntimeDirectTensorFaultProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFaultProbeState::error.has_value());
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::pool_generation_failed_closed);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::slot_generation_failed_closed);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::commit_failed);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::failed_commit_consumed);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::publication_indeterminate);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::indeterminate_consumed);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::publication_partial);
    EXPECT_TRUE(RuntimeDirectTensorFaultProbeState::partial_consumed);
    // Key assertion: both terminal results preserve their exact classification, prohibit retry,
    // and still let every retained recipient handle retire before teardown.
    EXPECT_GE(RuntimeDirectTensorFaultProbeState::sink_count, 1U);
    EXPECT_LE(RuntimeDirectTensorFaultProbeState::sink_count, 2U);
  }

  const detail::RuntimeTensorOutputTestControl::PoolSnapshot snapshot = control->snapshot(0U);
  EXPECT_TRUE(snapshot.terminal);
  EXPECT_EQ(snapshot.invariant_failure_count, 2U);
  EXPECT_EQ(snapshot.active_claim_mask, 0U);
  EXPECT_EQ(snapshot.active_slot_count, 0U);
  EXPECT_EQ(snapshot.observed_claim_mask & snapshot.admitted_claim_mask,
            snapshot.admitted_claim_mask);
}

// Runtime shall reject corrupted plan-owned publication/capacity facts before attaching the
// direct Tensor output. No callback may infer a retry or silently repair an understated pool.
// REQ: OPERATOR-025
TEST(RunSessionTest, RejectsMalformedDirectTensorPublicationAndCapacityContracts) {
  using Mutation = detail::RuntimeTestPlanAccess::TensorRuntimeMutation;
  constexpr std::array mutations{
      Mutation::kPublicationRetry,
      Mutation::kPublicationPreSubmitError,
      Mutation::kPublicationPostSubmitError,
      Mutation::kPublicationPostSubmitPartialError,
      Mutation::kPublicationAuthority,
      Mutation::kContributorCapacity,
      Mutation::kContributorReasonOrder,
      Mutation::kContributorPool,
      Mutation::kAllocationCapacity,
      Mutation::kAllocationAggregate,
  };
  const ExecutionPlan baseline = make_runtime_direct_tensor_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  for (const Mutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_tensor_runtime(baseline, mutation);
    ASSERT_TRUE(plan.ok());
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "PROFILE_MATERIALIZATION_FAILED"));
  }
}

// A transported Tensor pool separates queue occupancy from imported backing retention. Runtime
// reconstructs that topology rather than trusting contributor names alone.
// REQ: CHAN-093
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, RejectsMissingTransportRetentionAndAcceptsContributorReordering) {
  using Mutation = detail::RuntimeTestPlanAccess::TensorRuntimeMutation;
  RuntimeTransportTensorProbeState::reset();
  const ExecutionPlan baseline = make_runtime_transport_tensor_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  // Key setup: recreate the old three-slot shape by removing the transport dependency while
  // keeping all scalar pool facts coherent. The generic v4 artifact shape admits the omission;
  // Runtime owns the topology-specific requirement that transport retention remains separate.
  const ExecutionPlan malformed = detail::RuntimeTestPlanAccess::mutate_tensor_runtime(
      baseline, Mutation::kMissingTransportRetention);
  ASSERT_TRUE(malformed.ok());
  RunSession rejected = run_async(malformed);
  rejected.wait();

  // Key assertion: cold Runtime topology validation rejects the understated transported pool
  // before any Operator callback can use it.
  EXPECT_EQ(rejected.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(rejected, "PROFILE_MATERIALIZATION_FAILED"));

  // Key setup: reverse only the valid fixed contributor records.
  const ExecutionPlan reordered = detail::RuntimeTestPlanAccess::mutate_tensor_runtime(
      baseline, Mutation::kContributorRecordOrder);
  ASSERT_TRUE(reordered.ok());
  RuntimeTransportTensorProbeState::reset();
  RunSession accepted = run_async(reordered);
  const bool observed = RuntimeTransportTensorProbeState::wait();
  accepted.request_stop();
  accepted.wait();

  // Key assertion: contributor storage order is not part of the immutable pool contract.
  EXPECT_TRUE(observed);
  EXPECT_EQ(accepted.termination(), RunTermination::kStopped);
  EXPECT_TRUE(accepted.diagnostics().empty());
  EXPECT_EQ(RuntimeTransportTensorProbeState::value, RuntimeTransportTensorSourceOperator::kValue);
}

// The private production ledger has one entry for every canonical reason. Multiple logical
// reasons remain one physical-slot claim, duplicate terminal observation is idempotent, and the
// generation helper never wraps a stale physical identity back into service.
// REQ: TENSOR-STORAGE-011
TEST(RunSessionTest, TensorSlotClaimLedgerIsClosedBoundedAndNoWrap) {
  static_assert(detail::kTensorSlotClaimReasonCount == 17U);
  constexpr std::array<std::string_view, detail::kTensorSlotClaimReasonCount> expected_names{
      "producer_loan",
      "producer_writer",
      "producer_readiness",
      "accepted_publication",
      "holomq_transport_or_fanout",
      "subscriber_borrow_or_import",
      "consumer_cuda_release",
      "relay_or_reexport",
      "temporal_cache",
      "pending_or_selected_window",
      "reuse_last",
      "drain_all_activation",
      "dlpack_or_framework_borrow",
      "copy_or_recorder_materialization",
      "cancellation",
      "quarantine",
      "retirement_service",
  };
  detail::TensorSlotClaimLedger::Admission admitted;
  admitted.fill(true);
  detail::TensorSlotClaimLedger ledger;
  for (std::size_t index = 0U; index < admitted.size(); ++index) {
    const auto reason = static_cast<TensorSlotClaimReason>(index);
    EXPECT_EQ(tensor_slot_claim_reason_name(reason), expected_names[index]);
    EXPECT_TRUE(ledger.add(reason, admitted));
  }
  EXPECT_EQ(ledger.active_count(), 17U);
  EXPECT_TRUE(ledger.add(TensorSlotClaimReason::kRetirementService, admitted));
  EXPECT_EQ(ledger.active_count(), 17U);
  EXPECT_TRUE(ledger.contains(TensorSlotClaimReason::kProducerWriter));
  EXPECT_TRUE(ledger.contains(TensorSlotClaimReason::kSubscriberBorrowOrImport));
  EXPECT_TRUE(ledger.contains(TensorSlotClaimReason::kDlpackOrFrameworkBorrow));
  ledger.remove(TensorSlotClaimReason::kProducerWriter);
  EXPECT_EQ(ledger.active_count(), 16U);
  ledger.clear();
  EXPECT_FALSE(ledger.any());

  std::uint64_t generation = 1U;
  EXPECT_TRUE(detail::advance_tensor_slot_generation(generation));
  EXPECT_EQ(generation, 2U);
  generation = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(detail::advance_tensor_slot_generation(generation));
  EXPECT_EQ(generation, std::numeric_limits<std::uint64_t>::max());
  generation = 0U;
  EXPECT_FALSE(detail::advance_tensor_slot_generation(generation));
}

// Checked transition evidence is a bounded pointer-free record whose reason enumeration is an
// exact bijection with the maintained TENSOR_OUTPUT_* vocabulary. Overflow preserves the first
// admitted records and performs no growth or replacement.
// REQ: OPERATOR-025
TEST(RunSessionTest, TensorOutputCheckedDiagnosticsUseExactBoundedVocabulary) {
  static_assert(detail::kTensorOutputDiagnosticReasonCount == 26U);
  static_assert(std::is_trivially_copyable_v<detail::TensorOutputDiagnosticRecord>);
  constexpr std::array<std::string_view, detail::kTensorOutputDiagnosticReasonCount> expected{
      "TENSOR_OUTPUT_LOAN_OUTPUT_MISMATCH",
      "TENSOR_OUTPUT_LOAN_PLAN_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_ACTIVATION_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_POOL_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_SLOT_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_WRITER_ACTIVE",
      "TENSOR_OUTPUT_LOAN_READINESS_NOT_COMMITTED",
      "TENSOR_OUTPUT_ALLOCATION_CAPABILITY_MISSING",
      "TENSOR_OUTPUT_GUARD_DROPPED_WITHOUT_COMMIT",
      "TENSOR_OUTPUT_LOAN_DESTROYED_WITH_ACTIVE_WRITER",
      "TENSOR_OUTPUT_GUARD_OUTLIVED_LOAN",
      "TENSOR_OUTPUT_GUARD_DOUBLE_COMMIT",
      "TENSOR_OUTPUT_LOAN_DOUBLE_EMIT",
      "TENSOR_OUTPUT_LOAN_DOUBLE_CANCEL",
      "TENSOR_OUTPUT_LOAN_USE_AFTER_MOVE",
      "TENSOR_OUTPUT_GUARD_USE_AFTER_MOVE",
      "TENSOR_OUTPUT_LOAN_INVALID_STATE_TRANSITION",
      "TENSOR_OUTPUT_LOAN_REQUEST_LAYOUT_MISMATCH",
      "TENSOR_OUTPUT_LOAN_STREAM_DEVICE_MISMATCH",
      "TENSOR_OUTPUT_ALLOCATION_READINESS_RECORD_EXHAUSTED",
      "TENSOR_OUTPUT_LOAN_READINESS_RECORD_FAILED",
      "TENSOR_OUTPUT_LOAN_CANCEL_SAFETY_RECORD_FAILED",
      "TENSOR_OUTPUT_PUBLICATION_INDETERMINATE",
      "TENSOR_OUTPUT_LOAN_RETIREMENT_STALLED",
      "TENSOR_OUTPUT_LOAN_RELEASE_EVIDENCE_MISSING",
      "TENSOR_OUTPUT_PUBLICATION_PARTIAL",
  };
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    EXPECT_EQ(detail::tensor_output_diagnostic_reason_name(
                  static_cast<detail::TensorOutputDiagnosticReason>(index)),
              expected[index]);
  }

  std::array<detail::TensorOutputDiagnosticRecord, 2U> storage;
  detail::TensorOutputDiagnosticBuffer buffer;
  buffer.reset(storage);
  buffer.append({.slot = 1U, .reason = detail::TensorOutputDiagnosticReason::kLoanOutputMismatch});
  buffer.append({.slot = 2U, .reason = detail::TensorOutputDiagnosticReason::kLoanWriterActive});
  buffer.append({.slot = 3U, .reason = detail::TensorOutputDiagnosticReason::kLoanDoubleEmit});
  ASSERT_EQ(buffer.records().size(), 2U);
  EXPECT_EQ(buffer.records()[0].slot, 1U);
  EXPECT_EQ(buffer.records()[1].slot, 2U);
}

}  // namespace
}  // namespace holoscan
