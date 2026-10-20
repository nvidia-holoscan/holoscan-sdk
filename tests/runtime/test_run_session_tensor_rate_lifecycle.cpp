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
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/memory/block_memory_pool.hpp>

#include "common/allocation_counter.hpp"
#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "run_session_tensor_test_support.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::run_session_tensor;

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
  const auto* timing_value = optional_value(timing);
  ASSERT_NE(timing_value, nullptr);
  EXPECT_GE(timing_value->request_to_discover_ns, 0);
  EXPECT_GE(timing_value->discover_to_allocate_ns, 0);
  EXPECT_GE(timing_value->allocate_to_successor_compute_ns, 0);
  EXPECT_GT(timing_value->request_to_successor_compute_ns, 0);
  EXPECT_EQ(timing_value->request_to_successor_compute_ns,
            timing_value->request_to_discover_ns + timing_value->discover_to_allocate_ns +
                timing_value->allocate_to_successor_compute_ns);
  RecordProperty("restart_request_to_discover_ns", timing_value->request_to_discover_ns);
  RecordProperty("restart_discover_to_user_allocate_ns", timing_value->discover_to_allocate_ns);
  RecordProperty("restart_user_allocate_to_successor_compute_ns",
                 timing_value->allocate_to_successor_compute_ns);
  RecordProperty("restart_request_to_successor_compute_ns",
                 timing_value->request_to_successor_compute_ns);
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
  const auto* allocation_generation = optional_value(provider.allocation_generation);
  const auto* storage_binding_generation = optional_value(provider.storage_binding_generation);
  const auto* previous_allocation_generation =
      optional_value(provider.previous_allocation_generation);
  const auto* provider_materialized_entries =
      optional_value(provider.provider_materialized_entries);
  ASSERT_NE(allocation_generation, nullptr);
  ASSERT_NE(storage_binding_generation, nullptr);
  ASSERT_NE(previous_allocation_generation, nullptr);
  EXPECT_EQ(provider.previous_allocation_generation, provider.allocation_generation);
  ASSERT_NE(provider_materialized_entries, nullptr);
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
  EXPECT_EQ(allocation_generation->value, retained_metrics.maximum_allocation_generation);
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
  RecordProperty("restart_retained_allocation_generation", allocation_generation->value);
  RecordProperty("restart_previous_allocation_generation", previous_allocation_generation->value);
  RecordProperty("restart_maximum_allocation_generation",
                 retained_metrics.maximum_allocation_generation);
  RecordProperty("restart_storage_binding_generation", storage_binding_generation->value);
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
  RecordProperty("restart_provider_materialized_entries", *provider_materialized_entries);
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
    const auto& provider = providers.front();
    EXPECT_EQ(provider.action, AllocationAction::kReusedExact);
    const auto* allocation_generation = optional_value(provider.allocation_generation);
    const auto* storage_binding_generation = optional_value(provider.storage_binding_generation);
    ASSERT_NE(allocation_generation, nullptr);
    ASSERT_NE(storage_binding_generation, nullptr);
    EXPECT_EQ(allocation_generation->value, 1U);
    EXPECT_GE(storage_binding_generation->value, 2U);

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

// A deep self restart fences unpublished predecessor authority before callbacks, preserves a
// committed Tensor already owned by the Channel, and publishes a distinct replacement pool only
// after the kAllocate lifecycle suffix succeeds. The stale predecessor guard remains memory-safe
// and cannot commit into the successor epoch.
// REQ: COMPILE-049
// REQ: COMPILE-065
// REQ: CHAN-094
// REQ: OPERATOR-025
// REQ: RUNTIME-008
// REQ: RUNTIME-056
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
  cudaStream_t observed_predecessor_source_stream{};
  cudaStream_t observed_successor_source_stream{};
  std::optional<ErrorCode> observed_source_error;
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    observed_sink_count = RuntimeDirectTensorRestartProbeState::sink_count;
    observed_source_completed = RuntimeDirectTensorRestartProbeState::source_completed;
    observed_restart_accepted = RuntimeDirectTensorRestartProbeState::restart_accepted;
    observed_restart_allocate_count = RuntimeDirectTensorRestartProbeState::restart_allocate_count;
    observed_predecessor_source_stream =
        RuntimeDirectTensorRestartProbeState::predecessor_source_stream;
    observed_successor_source_stream =
        RuntimeDirectTensorRestartProbeState::successor_source_stream;
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
  ASSERT_NE(observed_predecessor_source_stream, nullptr);
  EXPECT_EQ(observed_successor_source_stream, observed_predecessor_source_stream);

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
  EXPECT_EQ(snapshot.cuda_streams_created, plan.cuda_compute_streams().size());
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
  // Each slot owns one readiness event, one producer release event, one Runtime-stream release
  // event, and one bounded application-selected release event for this one-subscriber graph.
  EXPECT_EQ(snapshot.cuda_events_created, 4U * plan.tensor_allocations().front().slot_count);
  EXPECT_EQ(snapshot.cuda_events_destroyed, snapshot.cuda_events_created);
  EXPECT_EQ(snapshot.cuda_streams_created, 1U);
  EXPECT_EQ(snapshot.cuda_streams_destroyed, 1U);
}

}  // namespace
}  // namespace holoscan
