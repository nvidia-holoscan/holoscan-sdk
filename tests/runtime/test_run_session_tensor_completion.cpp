/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>

#include "common/cuda_test_utils.hpp"
#include "compiler/execution_plan_state.hpp"
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

}  // namespace
}  // namespace holoscan
