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
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>

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
#include "support/optional_assertions.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::run_session_tensor;

struct RuntimeForwardedTensorProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    source_pointer = nullptr;
    sink_pointer = nullptr;
    probe_count = 0U;
    probe_result.reset();
    premature_reuse = false;
    sink_callback_started = false;
    release_sink_callback.store(false, std::memory_order_release);
    error.reset();
  }

  static void record_source(const void* pointer) {
    const std::lock_guard lock(mutex);
    source_pointer = pointer;
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    cv.notify_all();
  }

  static void CUDART_CB blocking_stream_callback(void*) noexcept {
    {
      // Notify under the lock so the waiter's reacquisition orders this call before the static
      // condition variable is destroyed at exit.
      const std::lock_guard lock(mutex);
      sink_callback_started = true;
      cv.notify_all();
    }
    while (!release_sink_callback.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  static void CUDART_CB stream_work_callback(void*) noexcept {}

  static void record_sink(const void* pointer) {
    const std::lock_guard lock(mutex);
    sink_pointer = pointer;
  }

  static void record_probe(const void* pointer) {
    {
      const std::lock_guard lock(mutex);
      ++probe_count;
      if (pointer == source_pointer && !release_sink_callback.load(std::memory_order_acquire)) {
        premature_reuse = true;
      }
    }
    cv.notify_all();
  }

  static void record_probe_exhaustion() {
    {
      const std::lock_guard lock(mutex);
      probe_result = ErrorCode::kResourceExhausted;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink_callback() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return sink_callback_started || error.has_value(); }) &&
           sink_callback_started && !error.has_value();
  }

  [[nodiscard]] static bool wait_for_probe_exhaustion() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return probe_result.has_value() || error.has_value(); }) &&
           probe_result == ErrorCode::kResourceExhausted && !error.has_value();
  }

  static void release_sink() noexcept {
    release_sink_callback.store(true, std::memory_order_release);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic_bool release_sink_callback{false};
  static inline const void* source_pointer{};
  static inline const void* sink_pointer{};
  static inline std::uint64_t probe_count{};
  static inline std::optional<ErrorCode> probe_result;
  static inline bool premature_reuse{};
  static inline bool sink_callback_started{};
  static inline std::optional<ErrorCode> error;
};

class RuntimeForwardedTensorSourceOperator final : public Operator<> {
 public:
  explicit RuntimeForwardedTensorSourceOperator(bool complete_after_one = true)
      : complete_after_one_(complete_after_one) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaManaged,
                               .dtype = DLDataType{kDLUInt, 64U, 1U}},
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
    if (probes_exhausted_) {
      return {};
    }
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      if (emitted_ && loan.error().code == ErrorCode::kResourceExhausted) {
        for (auto& retained : retained_probe_loans_) {
          auto canceled = std::move(retained).cancel();
          if (!canceled) {
            RuntimeForwardedTensorProbeState::fail(canceled.error().code);
            return canceled;
          }
        }
        retained_probe_loans_.clear();
        probes_exhausted_ = true;
        RuntimeForwardedTensorProbeState::record_probe_exhaustion();
        return {};
      }
      RuntimeForwardedTensorProbeState::fail(loan.error().code);
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      RuntimeForwardedTensorProbeState::fail(writer.error().code);
      return make_unexpected(std::move(writer.error()));
    }
    const auto stream = context.cuda_stream().get();
    if (stream == nullptr ||
        cudaMemsetAsync(writer->data(), 0x5A, writer->byte_size(), stream) != cudaSuccess) {
      RuntimeForwardedTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    void* const pointer = writer->data();
    if (emitted_) {
      RuntimeForwardedTensorProbeState::record_probe(pointer);
    } else {
      RuntimeForwardedTensorProbeState::record_source(pointer);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      RuntimeForwardedTensorProbeState::fail(committed.error().code);
      return make_unexpected(std::move(committed.error()));
    }
    if (emitted_) {
      retained_probe_loans_.emplace_back(std::move(*loan));
      return {};
    }
    emitted_ = true;
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      RuntimeForwardedTensorProbeState::fail(emitted.error().code);
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  bool complete_after_one_{};
  bool emitted_{};
  bool probes_exhausted_{};
  std::vector<TensorOutputLoan> retained_probe_loans_;
};

class RuntimeForwardedTensorRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation representation{
        .memory_kind = MemoryKind::kCudaManaged,
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    };
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = representation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kPassThrough,
        });
    spec.output(cuda_obligation, "cuda_obligation")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
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
      RuntimeForwardedTensorProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto stream = context.cuda_stream().get();
    if (stream == nullptr ||
        cudaLaunchHostFunc(stream,
                           RuntimeForwardedTensorProbeState::stream_work_callback,
                           nullptr) != cudaSuccess) {
      RuntimeForwardedTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      RuntimeForwardedTensorProbeState::fail(emitted.error().code);
    }
    return emitted;
  }

  Input<Tensor> input;
  Output<Tensor> output;
  Output<Tensor> cuda_obligation;
};

class RuntimeForwardedTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation representation{
        .memory_kind = MemoryKind::kCudaManaged,
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    };
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = representation});
    spec.output(cuda_obligation, "cuda_obligation")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
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
      RuntimeForwardedTensorProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto stream = context.cuda_stream().get();
    RuntimeForwardedTensorProbeState::record_sink(sample->data.data());
    if (stream == nullptr ||
        cudaLaunchHostFunc(stream,
                           RuntimeForwardedTensorProbeState::blocking_stream_callback,
                           nullptr) != cudaSuccess) {
      RuntimeForwardedTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return {};
  }

  Input<Tensor> input;
  Output<Tensor> cuda_obligation;
};

// Explicit output placement supplies each relay's device without a dummy allocation. Green
// Context placement selects the final compute stream for these forwarding operators.
class RuntimeGreenOnlyForwardedTensorRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation representation{
        .memory_kind = MemoryKind::kCudaManaged,
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    };
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = representation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kPassThrough,
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
      RuntimeForwardedTensorProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto stream = context.cuda_stream().get();
    if (stream == nullptr ||
        cudaLaunchHostFunc(stream,
                           RuntimeForwardedTensorProbeState::stream_work_callback,
                           nullptr) != cudaSuccess) {
      RuntimeForwardedTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      RuntimeForwardedTensorProbeState::fail(emitted.error().code);
    }
    return emitted;
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

class RuntimeGreenOnlyForwardedTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaManaged,
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
      RuntimeForwardedTensorProbeState::fail(sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    const auto stream = context.cuda_stream().get();
    RuntimeForwardedTensorProbeState::record_sink(sample->data.data());
    if (stream == nullptr ||
        cudaLaunchHostFunc(stream,
                           RuntimeForwardedTensorProbeState::blocking_stream_callback,
                           nullptr) != cudaSuccess) {
      RuntimeForwardedTensorProbeState::fail(ErrorCode::kFailure);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return {};
  }

  Input<Tensor> input;
};

// Used by: DirectTensorPoolWaitsForEveryFanoutConsumerStreamBeforeReuse,
//          ComputeCallCompletionWaitsForCudaConsumerStreamRetirement
struct RuntimeDirectTensorFanoutStreamProbeState {
  static void reset(cudaStream_t blocked, cudaStream_t quick, cudaStream_t overflow = nullptr) {
    const std::lock_guard lock(mutex);
    blocked_stream = blocked;
    quick_stream = quick;
    overflow_stream = overflow;
    source_context_stream = nullptr;
    blocked_context_stream = nullptr;
    quick_context_stream = nullptr;
    first_pointer = nullptr;
    sink_count = 0U;
    probe_count = 0U;
    blocked_callback_started = false;
    capacity_prefix_registered = false;
    producer_stream_registered = false;
    capacity_overflow_rejected = false;
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

  static void record_source_stream(cudaStream_t stream) {
    const std::lock_guard lock(mutex);
    source_context_stream = stream;
  }

  [[nodiscard]] static cudaStream_t source_stream() {
    const std::lock_guard lock(mutex);
    return source_context_stream;
  }

  static void record_sink(cudaStream_t stream, bool blocked) {
    {
      const std::lock_guard lock(mutex);
      (blocked ? blocked_context_stream : quick_context_stream) = stream;
      ++sink_count;
    }
    cv.notify_all();
  }

  static void record_capacity_prefix() {
    {
      const std::lock_guard lock(mutex);
      capacity_prefix_registered = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_capacity_prefix() {
    std::unique_lock lock(mutex);
    const bool observed =
        cv.wait_for(lock, 5s, [] { return capacity_prefix_registered || error.has_value(); });
    return observed && capacity_prefix_registered;
  }

  static void record_capacity_result(bool producer_registered, bool overflow_rejected) {
    {
      const std::lock_guard lock(mutex);
      producer_stream_registered = producer_registered;
      capacity_overflow_rejected = overflow_rejected;
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
      // Notify under the lock so the waiter's reacquisition orders this call before the static
      // condition variable is destroyed at exit.
      const std::lock_guard lock(mutex);
      blocked_callback_started = true;
      cv.notify_all();
    }
    release_blocked_stream.wait(false, std::memory_order_acquire);
  }

  [[nodiscard]] static bool wait_for_blocked_stream() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] {
      return (blocked_callback_started && sink_count == 2U) || error.has_value();
    });
  }

  [[nodiscard]] static bool wait_for_sinks() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return sink_count == 2U || error.has_value(); });
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
  static inline cudaStream_t overflow_stream{};
  static inline cudaStream_t source_context_stream{};
  static inline cudaStream_t blocked_context_stream{};
  static inline cudaStream_t quick_context_stream{};
  static inline const void* first_pointer{};
  static inline std::uint64_t sink_count{};
  static inline std::uint64_t probe_count{};
  static inline bool blocked_callback_started{};
  static inline bool capacity_prefix_registered{};
  static inline bool producer_stream_registered{};
  static inline bool capacity_overflow_rejected{};
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
    RuntimeDirectTensorFanoutStreamProbeState::record_source_stream(context.cuda_stream().get());
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
template <bool kBlockStream, bool kUseExecutionContextStream = false,
          bool kHoldStream = kBlockStream, bool kExerciseReleaseCapacity = false>
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
    if constexpr (kExerciseReleaseCapacity && !kBlockStream) {
      if (!RuntimeDirectTensorFanoutStreamProbeState::wait_for_capacity_prefix()) {
        RuntimeDirectTensorFanoutStreamProbeState::fail(ErrorCode::kFailure);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
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
    if constexpr (kExerciseReleaseCapacity && kBlockStream) {
      if (!sample->data.set_deallocation_stream(
              RuntimeDirectTensorFanoutStreamProbeState::blocked_stream) ||
          !sample->data.set_deallocation_stream(
              RuntimeDirectTensorFanoutStreamProbeState::quick_stream)) {
        RuntimeDirectTensorFanoutStreamProbeState::fail(ErrorCode::kFailure);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      RuntimeDirectTensorFanoutStreamProbeState::record_capacity_prefix();
    } else if constexpr (kExerciseReleaseCapacity) {
      const bool producer_registered = sample->data.set_deallocation_stream(
          RuntimeDirectTensorFanoutStreamProbeState::source_stream());
      RuntimeDirectTensorFanoutStreamProbeState::record_capacity_result(
          producer_registered,
          !sample->data.set_deallocation_stream(
              RuntimeDirectTensorFanoutStreamProbeState::overflow_stream));
    }
    if constexpr (kHoldStream) {
      if (cudaLaunchHostFunc(stream,
                             RuntimeDirectTensorFanoutStreamProbeState::blocking_stream_callback,
                             nullptr) != cudaSuccess) {
        RuntimeDirectTensorFanoutStreamProbeState::fail(ErrorCode::kFailure);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    RuntimeDirectTensorFanoutStreamProbeState::record_sink(context.cuda_stream().get(),
                                                           kBlockStream);
    return {};
  }

  Input<Tensor> input;
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

// Graph topology (one Operator per explicit execution partition):
//   source --> relay-a --> relay-b --> relay-c --> sink
[[nodiscard]] ExecutionPlan make_runtime_forwarded_tensor_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    CudaComputeStreamPolicy policy) {
  Graph graph{unique_runtime_graph_name("runtime-forwarded-tensor")};
  const auto source = graph.op<RuntimeForwardedTensorSourceOperator>("source");
  const auto relay_a = graph.op<RuntimeForwardedTensorRelayOperator>("relay-a");
  const auto relay_b = graph.op<RuntimeForwardedTensorRelayOperator>("relay-b");
  const auto relay_c = graph.op<RuntimeForwardedTensorRelayOperator>("relay-c");
  const auto sink = graph.op<RuntimeForwardedTensorSinkOperator>("sink");
  graph.add_flow(source->output, relay_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_a->output, relay_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_b->output, relay_c->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_c->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("relay-a-partition").add(relay_a);
  graph.partition("relay-b-partition").add(relay_b);
  graph.partition("relay-c-partition").add(relay_c);
  graph.partition("sink-partition").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-forwarded-tensor-clock"));
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = policy;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  for (const std::string_view operator_path : {"relay-a", "relay-b", "relay-c", "sink"}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = std::string{operator_path},
        .output_port = "cuda_obligation",
        .device = DeviceId{0},
    });
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// Graph topology (one explicitly assigned Green Context partition per Operator):
//   source --> relay-a --> relay-b --> relay-c --> sink
// Only the source declares an allocation. Green placement supplies each relay's device
// and final stream while all relays preserve the original backing.
[[nodiscard]] ExecutionPlan make_runtime_green_forwarded_tensor_plan(
    const CudaGreenContextCapability& capability,
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    CudaComputeStreamPolicy policy) {
  Graph graph{unique_runtime_graph_name("runtime-green-forwarded-tensor")};
  const auto source = graph.op<RuntimeForwardedTensorSourceOperator>("source", false);
  const auto relay_a = graph.op<RuntimeGreenOnlyForwardedTensorRelayOperator>("relay-a");
  const auto relay_b = graph.op<RuntimeGreenOnlyForwardedTensorRelayOperator>("relay-b");
  const auto relay_c = graph.op<RuntimeGreenOnlyForwardedTensorRelayOperator>("relay-c");
  const auto sink = graph.op<RuntimeGreenOnlyForwardedTensorSinkOperator>("sink");
  graph.add_flow(source->output, relay_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_a->output, relay_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_b->output, relay_c->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay_c->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("relay-a-partition").add(relay_a);
  graph.partition("relay-b-partition").add(relay_b);
  graph.partition("relay-c-partition").add(relay_c);
  graph.partition("sink-partition").add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-green-forwarded-tensor-clock"));

  const std::uint32_t quantum =
      std::lcm(capability.min_sm_partition_size, capability.sm_coscheduled_alignment);
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = policy;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  for (const std::string_view partition : {"source-partition",
                                           "relay-a-partition",
                                           "relay-b-partition",
                                           "relay-c-partition",
                                           "sink-partition"}) {
    options.deployment.bind_cuda_green_context(
        partition, CudaGreenContextPlacement{.device = DeviceId{0}, .sm_count = quantum});
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  return detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
}

// Graph topology:
//   source --[queue_depth=1]--> sink-a-blocked  (holds CUDA stream via host callback)
//          --[queue_depth=1]--> sink-b-quick    (completes without blocking)
// Device binding for the output port enables managed-memory fanout with two streams.
template <bool kUseExecutionContextStream = false, bool kHoldBlockedStream = true,
          bool kExerciseReleaseCapacity = false>
[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_fanout_stream_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    const bool complete_after_one = false,
    const std::string_view graph_name = "runtime-direct-tensor-fanout-streams",
    CudaComputeStreamPolicy policy = CudaComputeStreamPolicy::kPerExecutionPartition) {
  Graph graph{std::string{graph_name}};
  auto source =
      graph.op<RuntimeDirectTensorFanoutStreamSourceOperator>("source", complete_after_one);
  auto blocked = graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<true,
                                                                      kUseExecutionContextStream,
                                                                      kHoldBlockedStream,
                                                                      kExerciseReleaseCapacity>>(
      "sink-a-blocked");
  auto quick = graph.op<RuntimeDirectTensorFanoutStreamSinkOperator<false,
                                                                    kUseExecutionContextStream,
                                                                    false,
                                                                    kExerciseReleaseCapacity>>(
      "sink-b-quick");
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
  CompileOptions options;
  options.cuda_execution.compute_stream_policy = policy;
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      std::move(options),
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

// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
// REQ: RUNTIME-056
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, ForwardedTensorBackingReachesFinalPartitionUnderBothStreamPolicies) {
  if (cudaFree(nullptr) != cudaSuccess) {
    GTEST_SKIP() << "CUDA managed allocation is unavailable";
  }

  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    RuntimeForwardedTensorProbeState::reset();
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    const ExecutionPlan plan = make_runtime_forwarded_tensor_plan(control, policy);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.tensor_allocations().size(), 5U);
    EXPECT_TRUE(std::ranges::all_of(plan.tensor_allocations(), [](const auto& allocation) {
      return allocation.release_stream_capacity == 9U;
    }));
    ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 5U);
    EXPECT_EQ(plan.cuda_compute_streams().size(),
              policy == CudaComputeStreamPolicy::kPerExecutionPartition ? 5U : 1U);

    RunSession session = run_async(plan);
    const bool reached_final_stream = RuntimeForwardedTensorProbeState::wait_for_sink_callback();
    EXPECT_TRUE(session.is_running());
    RuntimeForwardedTensorProbeState::release_sink();
    session.wait();

    ASSERT_TRUE(reached_final_stream)
        << (session.diagnostics().empty() ? "no runtime diagnostic"
                                          : session.diagnostics().front().token);
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    {
      const std::lock_guard lock(RuntimeForwardedTensorProbeState::mutex);
      EXPECT_FALSE(RuntimeForwardedTensorProbeState::error.has_value());
      EXPECT_NE(RuntimeForwardedTensorProbeState::source_pointer, nullptr);
      EXPECT_EQ(RuntimeForwardedTensorProbeState::sink_pointer,
                RuntimeForwardedTensorProbeState::source_pointer);
    }
    const auto metrics = control->snapshot(0U);
    EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
    EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  }
}

// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
// REQ: RUNTIME-056
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, GreenOnlyForwardingBindingsRetainOriginalBackingUnderBothPolicies) {
  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability) {
    GTEST_SKIP() << runtime_error_message(capability.error());
  }
  const std::uint32_t quantum =
      std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
  if (quantum == 0U || capability->max_structured_context_count < 5U ||
      capability->total_sm_count < 5U * static_cast<std::uint64_t>(quantum)) {
    GTEST_SKIP() << "device cannot provide five dedicated Green Context groups";
  }

  for (const CudaComputeStreamPolicy policy : {CudaComputeStreamPolicy::kSharedProcessDevice,
                                               CudaComputeStreamPolicy::kPerExecutionPartition}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    RuntimeForwardedTensorProbeState::reset();
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    const ExecutionPlan plan =
        make_runtime_green_forwarded_tensor_plan(*capability, control, policy);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.tensor_allocations().size(), 1U);
    EXPECT_EQ(plan.tensor_allocations().front().release_stream_capacity, 9U);
    ASSERT_EQ(plan.cuda_green_context_assignments().size(), 5U);
    ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 5U);
    ASSERT_EQ(plan.cuda_compute_streams().size(), 5U);

    RunSession session = run_async(plan);
    struct SinkRelease {
      ~SinkRelease() { RuntimeForwardedTensorProbeState::release_sink(); }
    };
    const SinkRelease release_on_exit;
    ASSERT_TRUE(RuntimeForwardedTensorProbeState::wait_for_sink_callback());
    ASSERT_TRUE(RuntimeForwardedTensorProbeState::wait_for_probe_exhaustion());
    EXPECT_TRUE(session.is_running());
    {
      const std::lock_guard lock(RuntimeForwardedTensorProbeState::mutex);
      EXPECT_FALSE(RuntimeForwardedTensorProbeState::error.has_value());
      EXPECT_NE(RuntimeForwardedTensorProbeState::source_pointer, nullptr);
      EXPECT_EQ(RuntimeForwardedTensorProbeState::sink_pointer,
                RuntimeForwardedTensorProbeState::source_pointer);
      EXPECT_EQ(RuntimeForwardedTensorProbeState::probe_result, ErrorCode::kResourceExhausted);
      EXPECT_FALSE(RuntimeForwardedTensorProbeState::premature_reuse);
    }

    RuntimeForwardedTensorProbeState::release_sink();
    session.request_stop();
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    const auto metrics = control->snapshot(0U);
    EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
    EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  }
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

// REQ: MEM-STREAM-016
// REQ: RUNTIME-056
// REQ: TENSOR-STORAGE-012
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, PerPartitionTensorFanoutPublishesDistinctStreamsAndRetiresEveryConsumer) {
  // Key setup: let each automatically selected execution partition consume its compiler-owned
  // stream, while one sink holds its stream so premature output-slot reuse remains observable.
  RuntimeDirectTensorFanoutStreamProbeState::reset(nullptr, nullptr);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_direct_tensor_fanout_stream_plan<true>(
      control,
      false,
      unique_runtime_graph_name("runtime-per-partition-tensor-fanout"),
      CudaComputeStreamPolicy::kPerExecutionPartition);
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.cuda_compute_stream_policy(), CudaComputeStreamPolicy::kPerExecutionPartition);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 3U);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 3U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().release_stream_capacity, 5U);
  EXPECT_TRUE(std::ranges::all_of(plan.cuda_compute_streams(), [](const auto& stream) {
    return stream.partition.has_value() && stream.fused_linear_edges.empty();
  }));

  std::optional<RunSession> session{run_async(plan)};
  struct BlockedStreamRelease {
    ~BlockedStreamRelease() { RuntimeDirectTensorFanoutStreamProbeState::release_blocked(); }
  } const release_on_exit;
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_blocked_stream());
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_probes(8U));
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::premature_reuse);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream, nullptr);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::blocked_context_stream, nullptr);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::quick_context_stream, nullptr);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream,
              RuntimeDirectTensorFanoutStreamProbeState::blocked_context_stream);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream,
              RuntimeDirectTensorFanoutStreamProbeState::quick_context_stream);
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::blocked_context_stream,
              RuntimeDirectTensorFanoutStreamProbeState::quick_context_stream);
  }

  RuntimeDirectTensorFanoutStreamProbeState::release_blocked();
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_first_slot_retirement(control));
  session->request_stop();
  session->wait();
  EXPECT_EQ(session->termination(), RunTermination::kStopped);
  EXPECT_TRUE(session->diagnostics().empty())
      << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  session.reset();

  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_streams_created, 3U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
  EXPECT_EQ(metrics.cuda_events_destroyed, metrics.cuda_events_created);
  EXPECT_GE(metrics.cuda_release_stream_assignments, 2U);
  EXPECT_GE(metrics.cuda_release_records, 2U);
}

// REQ: MEM-STREAM-016
// REQ: RUNTIME-056
TEST(RunSessionTest, SharedPolicyPublishesOneNativeStreamToEveryFanoutPartition) {
  RuntimeDirectTensorFanoutStreamProbeState::reset(nullptr, nullptr);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_direct_tensor_fanout_stream_plan<true, false>(
      control,
      true,
      unique_runtime_graph_name("runtime-shared-tensor-fanout"),
      CudaComputeStreamPolicy::kSharedProcessDevice);
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.cuda_compute_stream_policy(), CudaComputeStreamPolicy::kSharedProcessDevice);
  ASSERT_EQ(plan.cuda_compute_streams().size(), 1U);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 3U);
  EXPECT_TRUE(std::ranges::all_of(plan.operator_cuda_stream_bindings(), [&](const auto& binding) {
    return binding.stream_id == plan.cuda_compute_streams().front().stream_id;
  }));

  RunSession session = run_async(plan);
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_sinks());
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_NE(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream, nullptr);
    EXPECT_EQ(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream,
              RuntimeDirectTensorFanoutStreamProbeState::blocked_context_stream);
    EXPECT_EQ(RuntimeDirectTensorFanoutStreamProbeState::source_context_stream,
              RuntimeDirectTensorFanoutStreamProbeState::quick_context_stream);
  }

  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_streams_created, 1U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
}

// REQ: MEM-STREAM-016
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, ReleaseCapacityAdmitsPartitionAndExternalStreamsThenRejectsOverflow) {
  cudaStream_t first_external = nullptr;
  if (cudaStreamCreateWithFlags(&first_external, cudaStreamNonBlocking) != cudaSuccess) {
    GTEST_SKIP() << "CUDA stream creation is unavailable";
  }
  cudaStream_t second_external = nullptr;
  if (cudaStreamCreateWithFlags(&second_external, cudaStreamNonBlocking) != cudaSuccess) {
    static_cast<void>(cudaStreamDestroy(first_external));
    GTEST_SKIP() << "second CUDA stream creation is unavailable";
  }
  cudaStream_t overflow = nullptr;
  if (cudaStreamCreateWithFlags(&overflow, cudaStreamNonBlocking) != cudaSuccess) {
    static_cast<void>(cudaStreamDestroy(second_external));
    static_cast<void>(cudaStreamDestroy(first_external));
    GTEST_SKIP() << "overflow CUDA stream creation is unavailable";
  }
  struct StreamCleanup {
    cudaStream_t first{};
    cudaStream_t second{};
    cudaStream_t overflow{};
    ~StreamCleanup() {
      static_cast<void>(cudaStreamDestroy(overflow));
      static_cast<void>(cudaStreamDestroy(second));
      static_cast<void>(cudaStreamDestroy(first));
    }
  } const stream_cleanup{first_external, second_external, overflow};

  RuntimeDirectTensorFanoutStreamProbeState::reset(first_external, second_external, overflow);
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = make_runtime_direct_tensor_fanout_stream_plan<true, false, true>(
      control,
      true,
      unique_runtime_graph_name("runtime-release-stream-capacity"),
      CudaComputeStreamPolicy::kPerExecutionPartition);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().release_stream_capacity, 5U);

  RunSession session = run_async(plan);
  ASSERT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::wait_for_sinks());
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock(RuntimeDirectTensorFanoutStreamProbeState::mutex);
    EXPECT_FALSE(RuntimeDirectTensorFanoutStreamProbeState::error.has_value());
    EXPECT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::capacity_prefix_registered);
    EXPECT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::producer_stream_registered);
    EXPECT_TRUE(RuntimeDirectTensorFanoutStreamProbeState::capacity_overflow_rejected);
  }
  const auto metrics = control->snapshot(0U);
  EXPECT_EQ(metrics.cuda_release_stream_assignments, 5U);
  EXPECT_EQ(metrics.cuda_streams_destroyed, metrics.cuda_streams_created);
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
            static_cast<std::uint64_t>(plan.tensor_allocations().front().slot_count) * 6U);
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

struct PoolReadinessTransitionProbe {
  cudaStream_t caller{};
  test::BlockingHostState gate;
  Tensor tensor;
  std::atomic_bool committed{};
  cudaStream_t stream{};
  std::shared_ptr<const holomq::cuda::StreamLifetime> lifetime;
  bool publish = true;
};

class PoolReadinessTransitionSource final : public Operator<> {
 public:
  explicit PoolReadinessTransitionSource(std::shared_ptr<PoolReadinessTransitionProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = TensorRepresentation{.memory_kind = MemoryKind::kCudaManaged,
                                                   .dtype = DLDataType{kDLUInt, 8U, 1U}},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::array<std::int64_t, 1U> shape{8};
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    auto view = writer->tensor();
    if (cudaMemsetAsync(writer->data(), 0x11, 8U, probe_->caller) != cudaSuccess ||
        !view.set_readiness_stream(probe_->caller) ||
        cudaStreamSynchronize(probe_->caller) != cudaSuccess ||
        cudaLaunchHostFunc(context.cuda_stream().get(),
                           test::blocking_host_callback,
                           &probe_->gate) != cudaSuccess ||
        cudaMemsetAsync(writer->data(), 0x7b, 8U, context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    probe_->tensor = std::move(view);
    probe_->stream = context.cuda_stream().get();
    probe_->lifetime = detail::TensorConsumerScope::lifetime(probe_->stream);
    if (!probe_->publish) {
      auto canceled = std::move(*loan).cancel();
      if (!canceled) {
        return canceled;
      }
      probe_->committed.store(true, std::memory_order_release);
      return {};
    }
    probe_->committed.store(true, std::memory_order_release);
    return output.emit(std::move(*loan));
  }
  Output<Tensor> output;

 private:
  std::shared_ptr<PoolReadinessTransitionProbe> probe_;
};

class PoolReadinessTransitionSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    return sample ? expected<void, Error>{} : make_unexpected(std::move(sample.error()));
  }
  Input<Tensor> input;
};

// An eager caller snapshot must become stale when commit selects elided pool readiness.
// Both host and different-stream reads must wait for the new, gated pool-stream write.
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, CallerToPoolReadinessInvalidatesTheEarlierEvent) {
  if (!test::has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  test::UniqueCudaStream caller;
  test::UniqueCudaStream consumer;
  test::UniqueCudaEvent consumed;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(caller), cudaSuccess);
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer), cudaSuccess);
  ASSERT_EQ(test::create_cuda_event(consumed), cudaSuccess);
  auto probe = std::make_shared<PoolReadinessTransitionProbe>();
  probe->caller = caller.get();
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  Graph graph{unique_runtime_graph_name("pool-readiness-transition")};
  auto source = graph.op<PoolReadinessTransitionSource>("source", probe);
  auto sink = graph.op<PoolReadinessTransitionSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  const auto plan = detail::RuntimeTestPlanAccess::with_tensor_output_test_control(
      compile(graph, options), control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  // Destruction always opens the GPU gate before the session is joined.
  const test::BlockingHostReleaseGuard release{probe->gate};
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!probe->committed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(probe->committed.load(std::memory_order_acquire));
  ASSERT_TRUE(test::wait_until_started(probe->gate));
  ASSERT_GT(control->snapshot(0U).cuda_readiness_record_elisions, 0U);
  ASSERT_TRUE(probe->tensor.synchronize_for_consumer_stream(consumer.get()));
  ASSERT_EQ(cudaEventRecord(consumed.get(), consumer.get()), cudaSuccess);
  std::atomic_bool host_entered{};
  auto host = std::async(std::launch::async, [&] {
    host_entered.store(true, std::memory_order_release);
    return probe->tensor.synchronize_for_host();
  });
  // An async future joins before release_on_exit; explicitly open the gate on every exit path.
  struct HostWaitCleanup {
    test::BlockingHostState& gate;
    ~HostWaitCleanup() { test::release_blocking_host(gate); }
  } const host_cleanup{probe->gate};
  while (!host_entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_EQ(host.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(cudaEventQuery(consumed.get()), cudaErrorNotReady);
  test::release_blocking_host(probe->gate);
  ASSERT_EQ(host.wait_for(5s), std::future_status::ready);
  EXPECT_TRUE(host.get());
  EXPECT_EQ(cudaEventSynchronize(consumed.get()), cudaSuccess);
  for (const auto value : std::span(probe->tensor.data_as<std::uint8_t>(), 8U)) {
    EXPECT_EQ(value, 0x7bU);
  }
  probe->tensor = Tensor{};
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// Exercise the real pool and native stream barrier, including foreign-thread final releases.
// REQ: TENSOR-STORAGE-011
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, PoolTensorReleaseDuringClosingWaitsForRealCudaQuiescence) {
  if (!test::has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  test::UniqueCudaStream caller;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(caller), cudaSuccess);
  auto probe = std::make_shared<PoolReadinessTransitionProbe>();
  probe->caller = caller.get();
  // Keep the sole public Tensor handle; no Channel sample can mask its final release.
  probe->publish = false;
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  Graph graph{unique_runtime_graph_name("pool-closing-release")};
  auto source = graph.op<PoolReadinessTransitionSource>("source", probe);
  auto sink = graph.op<PoolReadinessTransitionSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "output", .device = DeviceId{0}});
  const auto plan = detail::RuntimeTestPlanAccess::with_tensor_output_test_control(
      compile(graph, options), control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::optional<RunSession> session{run_async(plan)};
  const test::BlockingHostReleaseGuard release{probe->gate};
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!probe->committed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(probe->committed.load(std::memory_order_acquire));
  ASSERT_TRUE(test::wait_until_started(probe->gate));
  ASSERT_NE(probe->lifetime, nullptr);
  session->request_stop();
  const auto closing_deadline = std::chrono::steady_clock::now() + 5s;
  while (probe->lifetime->state() == holomq::cuda::StreamLifetime::State::kOpen &&
         std::chrono::steady_clock::now() < closing_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_EQ(probe->lifetime->state(), holomq::cuda::StreamLifetime::State::kClosing);
  const auto before = control->snapshot(0U);
  ASSERT_GT(before.active_slot_count, 0U);
  auto host_read = probe->tensor.synchronize_for_host();
  ASSERT_FALSE(host_read);
  EXPECT_EQ(host_read.error().code, ErrorCode::kClosed);
  EXPECT_NE(runtime_error_message(host_read.error()).find("read it after the session stops"),
            std::string_view::npos);
  const auto stream_read = probe->tensor.synchronize_for_consumer_stream(caller.get());
  ASSERT_FALSE(stream_read);
  EXPECT_EQ(stream_read.error().code, ErrorCode::kClosed);
  EXPECT_NE(runtime_error_message(stream_read.error()).find("read it after the session stops"),
            std::string_view::npos);
  std::jthread releaser([&] { probe->tensor = Tensor{}; });
  releaser.join();
  const auto closing = control->snapshot(0U);
  EXPECT_EQ(closing.active_slot_count, before.active_slot_count);
  EXPECT_EQ(closing.cuda_release_records, before.cuda_release_records);
  EXPECT_EQ(closing.cuda_events_destroyed, 0U);
  EXPECT_FALSE(session->termination() == RunTermination::kCompleted);
  test::release_blocking_host(probe->gate);
  session->wait();
  EXPECT_EQ(probe->lifetime->state(), holomq::cuda::StreamLifetime::State::kRetired);
  EXPECT_TRUE(session->diagnostics().empty());
  session.reset();
  const auto after = control->snapshot(0U);
  EXPECT_EQ(after.active_slot_count, 0U);
  EXPECT_EQ(after.invariant_failure_count, 0U);
  EXPECT_EQ(after.cuda_events_destroyed, after.cuda_events_created);
  EXPECT_EQ(after.cuda_streams_destroyed, after.cuda_streams_created);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Eager caller readiness keeps the producer independent of the Closing SDK consumer.
// REQ: TENSOR-STORAGE-015
TEST(RunSessionTest, PoolTensorRejectsClosingConsumerAndSkipsRetiredReleaseRecords) {
  if (!test::has_cuda_device()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset(LifecycleStage::kAllocate, nullptr, true);
  struct ProbeCleanup {
    ~ProbeCleanup() { RuntimeDirectTensorRestartProbeState::release_escaped_tensor(); }
  } const cleanup;
  const auto plan = make_runtime_direct_tensor_restart_plan(
      control, false, unique_runtime_graph_name("pool-closing-consumer"));
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  ASSERT_TRUE(RuntimeDirectTensorRestartProbeState::wait());
  Tensor escaped;
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    const auto& escaped_tensor = RuntimeDirectTensorRestartProbeState::escaped_tensor;
    HOLOSCAN_ASSERT_HAS_VALUE(escaped_tensor);
    escaped = *escaped_tensor;
  }
  test::UniqueCudaStream caller;
  test::UniqueCudaStream consumer;
  ASSERT_EQ(test::create_non_blocking_cuda_stream(caller), cudaSuccess);
  ASSERT_EQ(test::create_non_blocking_cuda_stream(consumer), cudaSuccess);
  ASSERT_TRUE(escaped.set_readiness_stream(caller.get()));
  auto lifetime = std::make_shared<holomq::cuda::StreamLifetime>(consumer.get(), 0);
  const std::shared_ptr<const holomq::cuda::StreamLifetime> identity = lifetime;
  ASSERT_TRUE(detail::TensorAccess::register_consumer_stream(escaped, consumer.get(), identity));
  test::BlockingHostState gate;
  ASSERT_EQ(cudaLaunchHostFunc(consumer.get(), test::blocking_host_callback, &gate), cudaSuccess);
  const test::BlockingHostReleaseGuard release{gate};
  ASSERT_TRUE(test::wait_until_started(gate));
  lifetime->close();
  {
    const detail::TensorConsumerScope scope{consumer.get(), &identity};
    auto read = escaped.synchronize_for_consumer_stream(consumer.get());
    ASSERT_FALSE(read);
    EXPECT_EQ(read.error().code, ErrorCode::kClosed);
    EXPECT_NE(runtime_error_message(read.error()).find("SDK CUDA consumer stream is closing"),
              std::string_view::npos);
    EXPECT_NE(runtime_error_message(read.error()).find("only supported during compute()"),
              std::string_view::npos);
  }
  test::release_blocking_host(gate);
  ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
  ASSERT_TRUE(lifetime->publish_retired());
  const auto before = control->snapshot(0U);
  RuntimeDirectTensorRestartProbeState::release_escaped_tensor();
  escaped = Tensor{};
  session.request_stop();
  session.wait();
  const auto after = control->snapshot(0U);
  EXPECT_EQ(after.cuda_release_records, before.cuda_release_records);
  EXPECT_EQ(after.invariant_failure_count, 0U);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// A retained pool Tensor can outlive the session that owned its elided readiness stream.
// Its backing is already quiescent; reads must not lazily record on the destroyed stream.
// REQ: TENSOR-STORAGE-011
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, EscapedDirectTensorReadinessAfterSessionTeardown) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset(LifecycleStage::kAllocate, nullptr, true);
  struct ProbeCleanup {
    ~ProbeCleanup() { RuntimeDirectTensorRestartProbeState::release_escaped_tensor(); }
  };
  const ProbeCleanup probe_cleanup;
  const ExecutionPlan plan = make_runtime_direct_tensor_restart_plan(
      control, false, "runtime-direct-tensor-escaped-readiness");
  ASSERT_TRUE(plan.ok()) << plan.json();
  {
    RunSession session = run_async(plan);
    ASSERT_TRUE(RuntimeDirectTensorRestartProbeState::wait());
    session.request_stop();
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kStopped);
    ASSERT_TRUE(session.diagnostics().empty());
  }
  Tensor escaped;
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    const auto& escaped_tensor = RuntimeDirectTensorRestartProbeState::escaped_tensor;
    HOLOSCAN_ASSERT_HAS_VALUE(escaped_tensor);
    escaped = *escaped_tensor;
  }
  const auto before = control->snapshot(0U);
  ASSERT_EQ(before.cuda_streams_destroyed, before.cuda_streams_created);
  ASSERT_GT(before.cuda_streams_created, 0U);
  ASSERT_GT(before.cuda_readiness_record_elisions, 0U);
  ASSERT_EQ(before.cuda_readiness_records, 0U);
  EXPECT_TRUE(escaped.synchronize_for_host());
  cudaStream_t consumer{};
  ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
  EXPECT_TRUE(escaped.synchronize_for_consumer_stream(consumer));
  EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
  const auto after = control->snapshot(0U);
  EXPECT_EQ(after.cuda_readiness_records, before.cuda_readiness_records);
  EXPECT_EQ(after.cuda_stream_waits, before.cuda_stream_waits);
  EXPECT_EQ(after.cuda_host_waits, before.cuda_host_waits);
}

// Session quiescence covers only the pool's own stream. A caller-supplied readiness stream is
// recorded eagerly even when the pool elides same-stream readiness, so reads after teardown wait
// on that event and never touch the caller's stream, which may already be destroyed.
// REQ: TENSOR-STORAGE-011
// REQ: TENSOR-STORAGE-012
TEST(RunSessionTest, EscapedTensorCallerReadinessStreamAfterSessionTeardown) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  RuntimeDirectTensorRestartProbeState::reset(LifecycleStage::kAllocate, nullptr, true);
  struct ProbeCleanup {
    ~ProbeCleanup() { RuntimeDirectTensorRestartProbeState::release_escaped_tensor(); }
  };
  const ProbeCleanup probe_cleanup;
  const ExecutionPlan plan = make_runtime_direct_tensor_restart_plan(
      control, false, "runtime-direct-tensor-caller-readiness");
  ASSERT_TRUE(plan.ok()) << plan.json();
  {
    RunSession session = run_async(plan);
    ASSERT_TRUE(RuntimeDirectTensorRestartProbeState::wait());
    session.request_stop();
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kStopped);
    ASSERT_TRUE(session.diagnostics().empty());
  }
  Tensor escaped;
  {
    const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
    const auto& escaped_tensor = RuntimeDirectTensorRestartProbeState::escaped_tensor;
    HOLOSCAN_ASSERT_HAS_VALUE(escaped_tensor);
    escaped = *escaped_tensor;
  }
  const auto before = control->snapshot(0U);
  ASSERT_GT(before.cuda_readiness_record_elisions, 0U);

  // Key setup: a caller stream replaces the elided pool-stream readiness, then is destroyed.
  cudaStream_t caller{};
  ASSERT_EQ(cudaStreamCreateWithFlags(&caller, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_TRUE(escaped.set_readiness_stream(caller));
  const auto recorded = control->snapshot(0U);
  EXPECT_EQ(recorded.cuda_readiness_records, before.cuda_readiness_records + 1U);
  EXPECT_EQ(recorded.cuda_readiness_record_elisions, before.cuda_readiness_record_elisions);
  ASSERT_EQ(cudaStreamDestroy(caller), cudaSuccess);

  // Key assertion: both reads wait on the recorded event without recording on the caller stream.
  EXPECT_TRUE(escaped.synchronize_for_host());
  cudaStream_t consumer{};
  ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
  EXPECT_TRUE(escaped.synchronize_for_consumer_stream(consumer));
  EXPECT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
  const auto after = control->snapshot(0U);
  EXPECT_EQ(after.cuda_readiness_records, recorded.cuda_readiness_records);
  EXPECT_EQ(after.cuda_host_waits, recorded.cuda_host_waits + 1U);
  EXPECT_EQ(after.cuda_stream_waits, recorded.cuda_stream_waits + 1U);
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
// REQ: COMPILE-086
// REQ: COMPILE-090
// REQ: COMPILE-097
// REQ: RUNTIME-056
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
  ASSERT_EQ(plan.cuda_compute_streams().size(), 2U);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 2U);
  EXPECT_NE(plan.operator_cuda_stream_bindings()[0].stream_id,
            plan.operator_cuda_stream_bindings()[1].stream_id);
  EXPECT_TRUE(std::ranges::all_of(plan.cuda_compute_streams(), [](const auto& stream) {
    return stream.fused_linear_edges.empty();
  }));
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

}  // namespace
}  // namespace holoscan
