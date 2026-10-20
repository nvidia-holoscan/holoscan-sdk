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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "cuda_execution/cuda_green_context_driver.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_fixture_observation.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

constexpr DLDataType kUint32{.code = kDLUInt, .bits = 32U, .lanes = 1U};
constexpr std::array<TensorAxisBounds, 1U> kSingleElementAxis{TensorAxisBounds{.max_extent = 1U}};
constexpr std::array<TensorAxisBounds, 1U> kFourElementAxis{TensorAxisBounds{.max_extent = 4U}};
constexpr std::uint64_t kInternalCaptureFrameId = 101U;
constexpr std::uint64_t kRegionEgressFrameId = 202U;

std::atomic<std::uint32_t>& reader_compute_count() {
  static std::atomic<std::uint32_t> count{};
  return count;
}

struct CUDAGraphIngressRestartProbe {
  void reset() noexcept {
    for (auto& address : addresses) {
      address.store(0U, std::memory_order_relaxed);
    }
    restart_accepted.store(false, std::memory_order_relaxed);
  }

  std::array<std::atomic<std::uintptr_t>, 2U> addresses{};
  std::atomic_bool restart_accepted{};
};

class CUDAGraphFixedTensorAllocationFailureControl final
    : public detail::RuntimeChannelSessionBindTestControl {
 public:
  explicit CUDAGraphFixedTensorAllocationFailureControl(
      const std::size_t successful_allocations) noexcept
      : successful_allocations_(successful_allocations) {}

  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
    bound_.store(true, std::memory_order_release);
    return detail::RuntimeChannelSessionTestAccess::
        fail_fixed_tensor_shape_allocation_after_successes(session, successful_allocations_);
  }

  [[nodiscard]] bool bound() const noexcept { return bound_.load(std::memory_order_acquire); }

 private:
  std::size_t successful_allocations_{};
  std::atomic_bool bound_{};
};

CUDAGraphIngressRestartProbe& cuda_graph_ingress_restart_probe() {
  static CUDAGraphIngressRestartProbe probe;
  return probe;
}

[[nodiscard]] TensorOutputSpec device_output_spec() {
  return TensorOutputSpec{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kCudaDevice,
              .dtype = kUint32,
              .rank = 1U,
              .strides = TensorStridePolicy::kContiguousOnly,
          },
      .bounds =
          TensorBounds{
              .max_rank = 1U,
              .max_byte_span = sizeof(std::uint32_t),
              .axes = kSingleElementAxis,
          },
      .storage = TensorOutputStorage::kRuntimePool,
  };
}

[[nodiscard]] TensorInputSpec device_input_spec() {
  return TensorInputSpec{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kCudaDevice,
              .dtype = kUint32,
              .rank = 1U,
              .strides = TensorStridePolicy::kContiguousOnly,
          },
  };
}

[[nodiscard]] TensorOutputSpec wide_device_output_spec() {
  TensorOutputSpec spec = device_output_spec();
  spec.bounds = TensorBounds{
      .max_rank = 1U, .max_byte_span = 4U * sizeof(std::uint32_t), .axes = kFourElementAxis};
  return spec;
}

[[nodiscard]] TensorOutputSpec pinned_host_output_spec() {
  TensorOutputSpec spec = wide_device_output_spec();
  spec.representation.memory_kind = MemoryKind::kPinnedHost;
  return spec;
}

[[nodiscard]] TensorOutputSpec fixed_pinned_host_output_spec() {
  TensorOutputSpec spec = device_output_spec();
  spec.representation.memory_kind = MemoryKind::kPinnedHost;
  return spec;
}

[[nodiscard]] TensorInputSpec pinned_host_input_spec() {
  TensorInputSpec spec = device_input_spec();
  spec.representation.memory_kind = MemoryKind::kPinnedHost;
  return spec;
}

class CUDAGraphRuntimeReader final : public Operator<> {
 public:
  explicit CUDAGraphRuntimeReader(std::uint32_t expected_emits = 2U)
      : expected_emits_(expected_emits) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    reader_compute_count().fetch_add(1U, std::memory_order_relaxed);
    if (emitted_count_ == expected_emits_) {
      return {};
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      // Retry on the next clock activation while the one-slot region ingress is still owned.
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemsetAsync(writer->data(), 0x2A, sizeof(std::uint32_t), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto published = output.emit(std::move(*loan));
    if (published) {
      ++emitted_count_;
    }
    return published;
  }

  Output<Tensor> output;

 private:
  std::uint32_t emitted_count_{};
  std::uint32_t expected_emits_{2U};
};

class CUDAGraphContendedIngressReader final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (!emit_requested.exchange(false, std::memory_order_acq_rel)) {
      return {};
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        emit_requested.store(true, std::memory_order_release);
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemsetAsync(writer->data(), 0x2A, sizeof(std::uint32_t), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return output.emit(std::move(*loan));
  }

  static void reset() noexcept { emit_requested.store(false, std::memory_order_relaxed); }

  static void request_emit() noexcept { emit_requested.store(true, std::memory_order_release); }

  static inline std::atomic_bool emit_requested{};
  Output<Tensor> output;
};

class CUDAGraphVariableShapeReader final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(wide_device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_count_ >= 2U) {
      return {};
    }
    const std::array<std::int64_t, 1U> shape{emitted_count_ == 0U ? 4 : 1};
    auto loan = output.allocate_tensor(shape);
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemsetAsync(writer->data(),
                        0x2A,
                        static_cast<std::size_t>(shape.front()) * sizeof(std::uint32_t),
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto emitted = output.emit(std::move(*loan));
    if (emitted) {
      ++emitted_count_;
    }
    return emitted;
  }

  Output<Tensor> output;

 private:
  std::uint32_t emitted_count_{};
};

class CUDAGraphRestartingReader final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    reader_compute_count().fetch_add(1U, std::memory_order_relaxed);
    if (phase_ >= 4U || (phase_ == 1U && cuda_graph_runtime_probe_writes() == 0U)) {
      return {};
    }

    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    auto& restart_probe = cuda_graph_ingress_restart_probe();
    if (phase_ == 0U || phase_ == 3U) {
      restart_probe.addresses[phase_ == 0U ? 0U : 1U].store(
          reinterpret_cast<std::uintptr_t>(writer->data()), std::memory_order_relaxed);
    }
    int value = 0x22;
    if (phase_ == 0U) {
      value = 0x11;
    } else if (phase_ == 1U) {
      value = 0x33;
    }
    if (cudaMemsetAsync(
            writer->data(), value, sizeof(std::uint32_t), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto published = output.emit(std::move(*loan));
    if (published) {
      ++phase_;
      if (phase_ == 2U) {
        // The second input is now waiting behind the occupied region output. Request restart
        // before yielding so generation cleanup encounters that gated readiness event.
        auto receipt = context.request_restart(
            LifecycleRestartOptions{.from_stage = LifecycleStage::kAllocate});
        cuda_graph_ingress_restart_probe().restart_accepted.store(receipt.has_value(),
                                                                  std::memory_order_relaxed);
        if (!receipt) {
          return make_unexpected(std::move(receipt.error()));
        }
        phase_ = 3U;
      }
    }
    return published;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static std::uint32_t cuda_graph_runtime_probe_writes() noexcept;

  std::uint32_t phase_{};
};

class CUDAGraphRuntimeDualReader final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(first, "first").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.output(second, "second").max_emits_per_compute(1U).produces_tensor(device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    reader_compute_count().fetch_add(1U, std::memory_order_relaxed);
    if (emitted_count_ == 2U) {
      return {};
    }

    if (first_emitted_count_ == emitted_count_) {
      auto first_result = try_emit(first, 0x11, context);
      if (!first_result) {
        return make_unexpected(std::move(first_result.error()));
      }
      if (!*first_result) {
        return {};
      }
      ++first_emitted_count_;
    }

    if (second_emitted_count_ == emitted_count_) {
      auto second_result = try_emit(second, 0x22, context);
      if (!second_result) {
        return make_unexpected(std::move(second_result.error()));
      }
      if (!*second_result) {
        return {};
      }
      ++second_emitted_count_;
    }

    ++emitted_count_;
    return {};
  }

  Output<Tensor> first;
  Output<Tensor> second;

 private:
  [[nodiscard]] static expected<bool, Error> try_emit(Output<Tensor>& output, int value,
                                                      ExecutionContext& context) {
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      // Keep both branch sequences aligned while either one-slot ingress remains owned.
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return false;
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemsetAsync(
            writer->data(), value, sizeof(std::uint32_t), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    auto published = output.emit(std::move(*loan));
    if (!published) {
      return make_unexpected(std::move(published.error()));
    }
    return true;
  }

  std::uint32_t emitted_count_{};
  std::uint32_t first_emitted_count_{};
  std::uint32_t second_emitted_count_{};
};

class CUDAGraphRuntimeHostReader final : public Operator<> {
 public:
  explicit CUDAGraphRuntimeHostReader(std::size_t expected_emits = 4U)
      : expected_emits_(expected_emits) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(pinned_host_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    constexpr std::array<std::uint32_t, 4U> kValues{
        0x11111111U, 0x22222222U, 0x33333333U, 0x44444444U};
    constexpr std::array<std::array<std::int64_t, 1U>, 4U> kShapes{{{4}, {4}, {4}, {4}}};
    if (emitted_count_ == expected_emits_) {
      return {};
    }
    auto loan = output.allocate_tensor(kShapes[emitted_count_]);
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    std::memcpy(writer->data(), &kValues[emitted_count_], sizeof(kValues[emitted_count_]));
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto published = output.emit(std::move(*loan));
    if (published) {
      ++emitted_count_;
    }
    return published;
  }

  Output<Tensor> output;

 private:
  std::size_t emitted_count_{};
  std::size_t expected_emits_{4U};
};

// Models a CPU-owned ingress buffer that the ingress producer copies into device boundary
// storage before the CUDA Graph region consumes it.
class CUDAGraphCPUStagedDeviceReader final : public Operator<> {
 public:
  explicit CUDAGraphCPUStagedDeviceReader(std::uint32_t expected_emits = 2U)
      : expected_emits_(expected_emits) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::array<std::uint32_t, 2U> kValues{0x11111111U, 0x22222222U};
    if (emitted_count_ == expected_emits_) {
      return {};
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemcpyAsync(writer->data(),
                        &kValues[emitted_count_],
                        sizeof(kValues[emitted_count_]),
                        cudaMemcpyHostToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    auto published = output.emit(std::move(*loan));
    if (published) {
      ++emitted_count_;
    }
    return published;
  }

  Output<Tensor> output;

 private:
  std::size_t emitted_count_{};
  std::size_t expected_emits_{};
};

struct CUDAGraphRuntimeProbe {
  void reset() {
    const std::lock_guard lock(mutex);
    ready = false;
    value = 0U;
    for (auto& count : starts) {
      count.store(0U, std::memory_order_relaxed);
    }
    for (auto& count : computes) {
      count.store(0U, std::memory_order_relaxed);
    }
    reader_compute_count().store(0U, std::memory_order_relaxed);
    ordering_errors.store(0U, std::memory_order_relaxed);
    graph_callback_started.store(false, std::memory_order_relaxed);
    graph_callback_finished.store(false, std::memory_order_relaxed);
    stop_before_graph_completion.store(0U, std::memory_order_relaxed);
    no_egress_launches.store(0U, std::memory_order_relaxed);
    writes.store(0U, std::memory_order_relaxed);
    for (auto& address : writer_addresses) {
      address.store(0U, std::memory_order_relaxed);
    }
    for (auto& observed : writer_values) {
      observed.store(0U, std::memory_order_relaxed);
    }
    for (auto& frame_id : writer_frame_ids) {
      frame_id.store(0U, std::memory_order_relaxed);
    }
    for (auto& sequence : writer_sequences) {
      sequence.store(0U, std::memory_order_relaxed);
    }
    for (auto& flags : writer_flags) {
      flags.store(0U, std::memory_order_relaxed);
    }
    for (auto& frame_id : capture_input_frame_ids) {
      frame_id.store(0U, std::memory_order_relaxed);
    }
    for (auto& flags : capture_input_flags) {
      flags.store(0U, std::memory_order_relaxed);
    }
    for (auto& count : branch_writes) {
      count.store(0U, std::memory_order_relaxed);
    }
    for (auto& branch_value : branch_values) {
      branch_value.store(0U, std::memory_order_relaxed);
    }
    for (auto& context : capture_green_contexts) {
      context.store(0U, std::memory_order_relaxed);
    }
    branch_threads = {};
  }

  void publish(std::uint32_t observed) {
    {
      const std::lock_guard lock(mutex);
      value = observed;
      ready = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return ready; });
  }

  [[nodiscard]] bool wait_for_graph_callback() {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [&] { return graph_callback_started.load(std::memory_order_acquire); });
  }

  [[nodiscard]] bool wait_for_no_egress_launches(std::uint32_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [&] { return no_egress_launches.load(std::memory_order_acquire) >= count; });
  }

  [[nodiscard]] bool wait_for_writes(std::uint32_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return writes.load(std::memory_order_acquire) >= count; });
  }

  void record_branch_thread(std::size_t branch) {
    const std::lock_guard lock(mutex);
    branch_threads[branch] = std::this_thread::get_id();
  }

  std::array<std::atomic<std::uint32_t>, 4U> starts{};
  std::array<std::atomic<std::uint32_t>, 4U> computes{};
  std::atomic<std::uint32_t> ordering_errors{};
  std::atomic_bool graph_callback_started{};
  std::atomic_bool graph_callback_finished{};
  std::atomic<std::uint32_t> stop_before_graph_completion{};
  std::atomic<std::uint32_t> no_egress_launches{};
  std::atomic<std::uint32_t> writes{};
  std::array<std::atomic<std::uintptr_t>, 4U> writer_addresses{};
  std::array<std::atomic<std::uint32_t>, 4U> writer_values{};
  std::array<std::atomic<std::uint64_t>, 4U> writer_frame_ids{};
  std::array<std::atomic<std::uint64_t>, 4U> writer_sequences{};
  std::array<std::atomic<std::uint32_t>, 4U> writer_flags{};
  std::array<std::atomic<std::uint64_t>, 4U> capture_input_frame_ids{};
  std::array<std::atomic<std::uint32_t>, 4U> capture_input_flags{};
  std::array<std::atomic<std::uint32_t>, 2U> branch_writes{};
  std::array<std::atomic<std::uint32_t>, 2U> branch_values{};
  std::array<std::atomic<std::uintptr_t>, 4U> capture_green_contexts{};
  std::array<std::thread::id, 2U> branch_threads{};
  std::mutex mutex;
  std::condition_variable cv;
  bool ready{};
  std::uint32_t value{};
};

CUDAGraphRuntimeProbe& cuda_graph_runtime_probe() {
  // Never destroyed: CUDA host functions notify it after the last wait, ordered only by stream
  // synchronization that ThreadSanitizer cannot see.
  static auto* const probe = new CUDAGraphRuntimeProbe{};
  return *probe;
}

std::uint32_t CUDAGraphRestartingReader::cuda_graph_runtime_probe_writes() noexcept {
  return cuda_graph_runtime_probe().writes.load(std::memory_order_acquire);
}

class CUDAGraphRuntimeTransform final : public Operator<> {
 public:
  explicit CUDAGraphRuntimeTransform(std::size_t index, bool delay_graph = false,
                                     bool record_green_context = false,
                                     std::uint32_t queue_depth = 1U, bool fail_capture = false,
                                     bool use_default_contract = false,
                                     EmitOptions capture_options = {})
      : index_(index),
        delay_graph_(delay_graph),
        record_green_context_(record_green_context),
        queue_depth_(queue_depth),
        fail_capture_(fail_capture),
        use_default_contract_(use_default_contract),
        capture_options_(capture_options) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(queue_depth_).expects_tensor(device_input_spec());
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (!use_default_contract_) {
      result.trigger(OnEach{input});
    }
    return result;
  }

  void start() override {
    cuda_graph_runtime_probe().starts[index_].fetch_add(1U, std::memory_order_relaxed);
  }

  void stop() override {
    auto& probe = cuda_graph_runtime_probe();
    if (probe.graph_callback_started.load(std::memory_order_acquire) &&
        !probe.graph_callback_finished.load(std::memory_order_acquire)) {
      probe.stop_before_graph_completion.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    // Production capture exposes its stream for recording graph work, but never as a live
    // consumer retirement stream. Recording last-release fences on it would pollute capture.
    if (!detail::TensorConsumerScope::capturing() || detail::TensorConsumerScope::stream() ||
        detail::TensorConsumerScope::lifetime(context.cuda_stream().get()) != nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_BINDING_INVALID"});
    }
    auto& probe = cuda_graph_runtime_probe();
    probe.computes[index_].fetch_add(1U, std::memory_order_relaxed);
    if (probe.starts[index_].load(std::memory_order_relaxed) != 1U ||
        reader_compute_count().load(std::memory_order_relaxed) != 0U) {
      probe.ordering_errors.fetch_add(1U, std::memory_order_relaxed);
    }
    if (fail_capture_) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_CALLBACK_FAILED"});
    }
    if (record_green_context_) {
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
      auto driver = cuda_execution_detail::cuda_green_context_driver();
      CUgreenCtx green_context{};
      if (!driver || (*driver)->stream_get_green_context == nullptr ||
          (*driver)->stream_get_green_context(
              reinterpret_cast<CUstream>(context.cuda_stream().get()), &green_context) !=
              CUDA_SUCCESS ||
          green_context == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      probe.capture_green_contexts[index_].store(reinterpret_cast<std::uintptr_t>(green_context),
                                                 std::memory_order_release);
#else
      return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
    }
    if (delay_graph_ &&
        cudaLaunchHostFunc(context.cuda_stream().get(), &delay_graph_callback, nullptr) !=
            cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    probe.capture_input_frame_ids[index_].store(sample->metadata.frame_id,
                                                std::memory_order_relaxed);
    probe.capture_input_flags[index_].store(static_cast<std::uint32_t>(sample->metadata.flags),
                                            std::memory_order_relaxed);
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemcpyAsync(writer->data(),
                        sample->data.data(),
                        sizeof(std::uint32_t),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return output.emit(std::move(*loan), capture_options_);
  }

  Input<Tensor> input;
  Output<Tensor> output;

 private:
  static void CUDART_CB delay_graph_callback(void*) {
    auto& probe = cuda_graph_runtime_probe();
    probe.graph_callback_started.store(true, std::memory_order_release);
    probe.cv.notify_all();
    std::this_thread::sleep_for(150ms);
    probe.graph_callback_finished.store(true, std::memory_order_release);
  }

  std::size_t index_{};
  bool delay_graph_{};
  bool record_green_context_{};
  std::uint32_t queue_depth_{1U};
  bool fail_capture_{};
  bool use_default_contract_{};
  EmitOptions capture_options_{};
};

class CUDAGraphRuntimeWriter final : public Operator<> {
 public:
  explicit CUDAGraphRuntimeWriter(std::uint32_t queue_depth = 1U,
                                  std::uint32_t expected_writes = 2U,
                                  bool hold_output_after_read = false)
      : queue_depth_(queue_depth),
        expected_writes_(expected_writes),
        hold_output_after_read_(hold_output_after_read) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(queue_depth_).expects_tensor(device_input_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    std::uint32_t observed{};
    if (cudaMemcpyAsync(&observed,
                        sample->data.data(),
                        sizeof(observed),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess ||
        cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (hold_output_after_read_ &&
        cudaLaunchHostFunc(context.cuda_stream().get(), &delay_output_release, nullptr) !=
            cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto& probe = cuda_graph_runtime_probe();
    std::uint32_t write_index{};
    {
      const std::lock_guard lock(probe.mutex);
      write_index = probe.writes.load(std::memory_order_relaxed);
      if (write_index < probe.writer_values.size()) {
        probe.writer_addresses[write_index].store(
            reinterpret_cast<std::uintptr_t>(sample->data.data()), std::memory_order_relaxed);
        probe.writer_values[write_index].store(observed, std::memory_order_relaxed);
        probe.writer_frame_ids[write_index].store(sample->metadata.frame_id,
                                                  std::memory_order_relaxed);
        probe.writer_sequences[write_index].store(sample->metadata.sequence,
                                                  std::memory_order_relaxed);
        probe.writer_flags[write_index].store(static_cast<std::uint32_t>(sample->metadata.flags),
                                              std::memory_order_relaxed);
      }
      // Publish the count only after the corresponding metadata is complete.
      probe.writes.store(write_index + 1U, std::memory_order_release);
    }
    probe.cv.notify_all();
    if (write_index + 1U == expected_writes_) {
      probe.publish(observed);
      context.request_completion();
    }
    return {};
  }

  Input<Tensor> input;

 private:
  static void CUDART_CB delay_output_release(void*) { std::this_thread::sleep_for(150ms); }

  std::uint32_t queue_depth_{1U};
  std::uint32_t expected_writes_{2U};
  bool hold_output_after_read_{};
};

class CUDAGraphAsyncSlowWriter final : public Operator<> {
 public:
  CUDAGraphAsyncSlowWriter(std::uint32_t expected_writes, std::uint32_t queue_depth)
      : expected_writes_(expected_writes), queue_depth_(queue_depth) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(queue_depth_).expects_tensor(device_input_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    // Keep this Tensor's release ordered behind asynchronous consumer work. The next fused-region
    // activation must remain pending until the ordinary one-slot RuntimePool retires this sample.
    if (cudaLaunchHostFunc(context.cuda_stream().get(), &delay_release, nullptr) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto& probe = cuda_graph_runtime_probe();
    const std::uint32_t count = probe.writes.fetch_add(1U, std::memory_order_release) + 1U;
    probe.cv.notify_all();
    if (count == expected_writes_) {
      probe.publish(count);
      context.request_completion();
    }
    return {};
  }

  Input<Tensor> input;

 private:
  static void CUDART_CB delay_release(void*) { std::this_thread::sleep_for(40ms); }

  std::uint32_t expected_writes_{};
  std::uint32_t queue_depth_{};
};

class CUDAGraphPinnedIngressTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(pinned_host_input_spec());
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (cudaLaunchHostFunc(context.cuda_stream().get(), &delay_read, nullptr) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemcpyAsync(writer->data(),
                        sample->data.data(),
                        sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return output.emit(std::move(*loan));
  }

  Input<Tensor> input;
  Output<Tensor> output;

 private:
  static void CUDART_CB delay_read(void*) { std::this_thread::sleep_for(150ms); }
};

class CUDAGraphRuntimeBranchWriter final : public Operator<> {
 public:
  explicit CUDAGraphRuntimeBranchWriter(std::size_t branch) : branch_(branch) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    std::uint32_t observed{};
    if (cudaMemcpyAsync(&observed,
                        sample->data.data(),
                        sizeof(observed),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess ||
        cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    auto& probe = cuda_graph_runtime_probe();
    probe.record_branch_thread(branch_);
    probe.branch_values[branch_].store(observed, std::memory_order_relaxed);
    probe.branch_writes[branch_].fetch_add(1U, std::memory_order_relaxed);
    if (probe.branch_writes[0U].load(std::memory_order_relaxed) >= 2U &&
        probe.branch_writes[1U].load(std::memory_order_relaxed) >= 2U) {
      probe.publish(observed);
      context.request_completion();
    }
    return {};
  }

  Input<Tensor> input;

 private:
  std::size_t branch_{};
};

class CUDAGraphRuntimeInputSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    const cudaError_t status =
        cudaLaunchHostFunc(context.cuda_stream().get(), &record_launch, nullptr);
    return status == cudaSuccess
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

  Input<Tensor> input;

 private:
  static void CUDART_CB record_launch(void*) {
    auto& probe = cuda_graph_runtime_probe();
    probe.no_egress_launches.fetch_add(1U, std::memory_order_release);
    probe.cv.notify_all();
  }
};

class CUDAGraphDirectTensorEmitTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    return sample ? output.emit(std::move(*sample))
                  : expected<void, Error>{make_unexpected(std::move(sample.error()))};
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Checks that capture preserves the logical Tensor span when the fixed slot has extra capacity.
class CUDAGraphLogicalSpanTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(wide_device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (writer->byte_size() != sizeof(std::uint32_t)) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_LOGICAL_SPAN_INCORRECT"});
    }
    if (cudaMemcpyAsync(writer->data(),
                        sample->data.data(),
                        sizeof(std::uint32_t),
                        cudaMemcpyDeviceToDevice,
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    return committed ? output.emit(std::move(*loan)) : committed;
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Enqueues capturable CUDA work but deliberately omits its declared Tensor output.
class CUDAGraphMissingOutputSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const cudaError_t status =
        cudaLaunchHostFunc(context.cuda_stream().get(), &empty_callback, nullptr);
    return status == cudaSuccess
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

  Output<Tensor> output;

 private:
  static void CUDART_CB empty_callback(void*) {}
};

struct CUDAGraphCaptureCancellationProbe {
  static inline std::atomic_bool started{};
};

enum class CUDAGraphCaptureStopResult : std::uint8_t {
  kInterrupted,
  kFailure,
  kSuccess,
};

class CUDAGraphCancellableCaptureSource final : public Operator<> {
 public:
  explicit CUDAGraphCancellableCaptureSource(
      CUDAGraphCaptureStopResult result = CUDAGraphCaptureStopResult::kInterrupted)
      : result_(result) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    CUDAGraphCaptureCancellationProbe::started.store(true, std::memory_order_release);
    while (!context.cancellation().stop_requested()) {
      std::this_thread::yield();
    }
    if (result_ == CUDAGraphCaptureStopResult::kInterrupted) {
      return make_unexpected(Error{ErrorCode::kInterrupted, "CUDA_GRAPH_CAPTURE_STOPPED"});
    }
    if (result_ == CUDAGraphCaptureStopResult::kFailure) {
      return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GRAPH_CAPTURE_CALLBACK_FAILED"});
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemsetAsync(writer->data(), 0x2A, sizeof(std::uint32_t), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return output.emit(std::move(*loan));
  }

  Output<Tensor> output;

 private:
  CUDAGraphCaptureStopResult result_;
};

enum class CUDAGraphIndependentRoot : std::uint8_t {
  kClock,
  kNotification,
  kReadable,
  kDoorbell,
  kSpin,
};

struct CUDAGraphIndependentNotificationState {
  static void reset() {
    const std::lock_guard lock(mutex);
    sender.reset();
  }

  static void publish(NotificationSender value) {
    {
      const std::lock_guard lock(mutex);
      sender = std::move(value);
    }
    cv.notify_all();
  }

  [[nodiscard]] static std::optional<NotificationSender> wait_for_sender() {
    std::unique_lock lock(mutex);
    if (!cv.wait_for(lock, 5s, [] { return sender.has_value(); })) {
      return std::nullopt;
    }
    return sender;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<NotificationSender> sender;
};

// Inputless captured source used to prove that each supported independent contract activates the
// fused operator while its original compute() remains capture-only.
class CUDAGraphIndependentSource final : public Operator<> {
 public:
  explicit CUDAGraphIndependentSource(CUDAGraphIndependentRoot root,
                                      std::optional<std::uint64_t> completion_limit = std::nullopt,
                                      std::uint32_t capture_emit_count = 1U,
                                      EmitOptions capture_options = {})
      : root_(root),
        completion_limit_(completion_limit),
        capture_emit_count_(capture_emit_count),
        capture_options_(capture_options) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(device_output_spec());
    spec.cuda_graph_capture_eligible();
    switch (root_) {
      case CUDAGraphIndependentRoot::kClock:
        break;
      case CUDAGraphIndependentRoot::kNotification:
        spec.notification_source(notification_, "activation")
            .capacity(2U)
            .sender_reference_capacity(1U);
        spec.lifecycle().stage(LifecycleStage::kArm, &CUDAGraphIndependentSource::arm);
        break;
      case CUDAGraphIndependentRoot::kReadable:
        spec.readable_source(readable_, "activation");
        break;
      case CUDAGraphIndependentRoot::kDoorbell:
        spec.doorbell_source(doorbell_, "activation").capacity(2U);
        break;
      case CUDAGraphIndependentRoot::kSpin:
        spec.spin_source(spin_, "activation").capacity(1U);
        break;
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    switch (root_) {
      case CUDAGraphIndependentRoot::kClock:
        result.trigger(OnClock{.period = 10ms});
        break;
      case CUDAGraphIndependentRoot::kNotification:
        result.trigger(OnNotified{notification_});
        break;
      case CUDAGraphIndependentRoot::kReadable:
        result.trigger(OnReadable{readable_});
        break;
      case CUDAGraphIndependentRoot::kDoorbell:
        result.trigger(OnDoorbell{doorbell_});
        break;
      case CUDAGraphIndependentRoot::kSpin:
        result.trigger(OnSpin{spin_});
        break;
    }
    if (completion_limit_) {
      result.complete_after(ComputeCalls{.limit = *completion_limit_});
    }
    return result;
  }

  void start() override {
    cuda_graph_runtime_probe().starts[0U].fetch_add(1U, std::memory_order_relaxed);
    constexpr std::uint32_t kValue = 0x51525354U;
    if (cudaMalloc(&device_value_, sizeof(kValue)) != cudaSuccess ||
        cudaMemcpy(device_value_, &kValue, sizeof(kValue), cudaMemcpyHostToDevice) != cudaSuccess) {
      if (device_value_ != nullptr) {
        static_cast<void>(cudaFree(device_value_));
      }
      device_value_ = nullptr;
    }
  }

  void stop() override {
    if (device_value_ != nullptr) {
      static_cast<void>(cudaFree(device_value_));
      device_value_ = nullptr;
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    cuda_graph_runtime_probe().computes[0U].fetch_add(1U, std::memory_order_relaxed);
    if (device_value_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    for (std::uint32_t emit_index = 0U; emit_index < capture_emit_count_; ++emit_index) {
      constexpr std::array<std::int64_t, 1U> kShape{1};
      auto loan = output.allocate_tensor(kShape);
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      if (cudaMemcpyAsync(writer->data(),
                          device_value_,
                          sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return committed;
      }
      auto emitted = output.emit(std::move(*loan), capture_options_);
      if (!emitted) {
        return emitted;
      }
    }
    return {};
  }

  Output<Tensor> output;

 private:
  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    CUDAGraphIndependentNotificationState::publish(std::move(*sender));
    return LifecycleStatus::kOk;
  }

  CUDAGraphIndependentRoot root_;
  std::optional<std::uint64_t> completion_limit_;
  std::uint32_t capture_emit_count_{1U};
  EmitOptions capture_options_{};
  NotificationSource notification_;
  ReadableSource readable_;
  DoorbellSource doorbell_;
  SpinSource spin_;
  void* device_value_{};
};

// Supplies physical Readable, Doorbell, and Spin witnesses through Runtime's normal binding API.
class CUDAGraphIndependentPhysicalFactory final : public detail::RuntimePortBindingFactory {
 public:
  struct BoundControl {
    detail::PhysicalReadinessControl control;
    ClockId reference_clock{};
  };

  explicit CUDAGraphIndependentPhysicalFactory(CUDAGraphIndependentRoot root) : root_(root) {}

  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    detail::RuntimeEventKind expected_kind{};
    if (root_ == CUDAGraphIndependentRoot::kReadable) {
      expected_kind = detail::RuntimeEventKind::kReadable;
    } else if (root_ == CUDAGraphIndependentRoot::kDoorbell) {
      expected_kind = detail::RuntimeEventKind::kDoorbell;
    } else if (root_ == CUDAGraphIndependentRoot::kSpin) {
      expected_kind = detail::RuntimeEventKind::kSpin;
    } else {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (request.kind != expected_kind) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool spin = expected_kind == detail::RuntimeEventKind::kSpin;
    const detail::PhysicalReadinessBindingDescriptor descriptor{
        .request = request,
        .wake_requirement = spin ? ReadinessWakeRequirement::kBusyPollRequired
                                 : ReadinessWakeRequirement::kDeploymentSelected,
        .wake_mode = spin ? WakeMode::kBusyPoll : WakeMode::kSleepingWaitset,
        .execution_lane = request.execution_lane,
        .poll_state = spin ? this : nullptr,
        .poll_asserted = spin ? &poll_false : nullptr,
    };
    auto physical = detail::PhysicalReadinessBindingFactory::create(
        std::span<const detail::PhysicalReadinessBindingDescriptor>{&descriptor, 1U});
    if (!physical) {
      return make_unexpected(std::move(physical.error()));
    }
    auto binding = (*physical)->bind(request, start_coordinate);
    if (!binding) {
      return make_unexpected(std::move(binding.error()));
    }
    auto control = (*physical)->control(
        request.source, request.plan_generation, request.run_generation, request.source_generation);
    if (!control) {
      return make_unexpected(std::move(control.error()));
    }
    {
      const std::lock_guard lock(mutex_);
      physical_ = std::move(*physical);
      bound_ =
          BoundControl{.control = std::move(*control), .reference_clock = request.reference_clock};
    }
    cv_.notify_all();
    return std::move(*binding);
  }

  [[nodiscard]] std::optional<BoundControl> wait_for_control() const {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, 5s, [&] { return bound_.has_value(); })) {
      return std::nullopt;
    }
    return bound_;
  }

 private:
  static expected<bool, Error> poll_false(void*) noexcept { return false; }

  CUDAGraphIndependentRoot root_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::shared_ptr<detail::PhysicalReadinessBindingFactory> physical_;
  std::optional<BoundControl> bound_;
};

class CUDAGraphPinnedEgressTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(fixed_pinned_host_output_spec());
    spec.cuda_graph_capture_eligible();
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    if (cudaMemcpyAsync(writer->data(),
                        sample->data.data(),
                        sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost,
                        context.cuda_stream().get()) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    return committed ? output.emit(std::move(*loan)) : committed;
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

class CUDAGraphHostWriteTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(device_input_spec());
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(fixed_pinned_host_output_spec());
    spec.cuda_graph_capture_eligible();
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
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(kShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer.error()));
    }
    std::memset(writer->data(), 0, sizeof(std::uint32_t));
    auto committed = std::move(*writer).commit();
    return committed ? output.emit(std::move(*loan)) : committed;
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

class CUDAGraphPinnedEgressWriter final : public Operator<> {
 public:
  explicit CUDAGraphPinnedEgressWriter(std::uint32_t expected_writes = 2U)
      : expected_writes_(expected_writes) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U).expects_tensor(pinned_host_input_spec());
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    auto ready = sample->data.synchronize_for_host();
    if (!ready) {
      return make_unexpected(std::move(ready.error()));
    }
    std::uint32_t observed{};
    std::memcpy(&observed, sample->data.data(), sizeof(observed));
    auto& probe = cuda_graph_runtime_probe();
    const auto count = probe.writes.fetch_add(1U, std::memory_order_release) + 1U;
    if (count <= probe.writer_values.size()) {
      probe.writer_values[count - 1U].store(observed, std::memory_order_relaxed);
    }
    probe.cv.notify_all();
    if (count == expected_writes_) {
      probe.publish(observed);
      context.request_completion();
    }
    return {};
  }

  Input<Tensor> input;

 private:
  std::uint32_t expected_writes_{2U};
};

class CUDAGraphClosedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.cuda_graph_capture_eligible(); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const cudaError_t status =
        cudaLaunchHostFunc(context.cuda_stream().get(), &record_launch, nullptr);
    return status == cudaSuccess
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

 private:
  static void CUDART_CB record_launch(void*) {
    auto& probe = cuda_graph_runtime_probe();
    probe.no_egress_launches.fetch_add(1U, std::memory_order_release);
    probe.cv.notify_all();
  }
};

[[nodiscard]] ExecutionPlan compile_cuda_graph_runtime(
    Graph& graph, CompileOptions options = {},
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& tensor_control = {}) {
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  if (!snapshot) {
    return {};
  }
  std::vector<detail::TensorDeviceBindingRecord> devices;
  for (const auto& port : snapshot->ports) {
    if (port.owner_kind == PortOwnerKind::kOperator && port.direction == PortDirection::kOutput &&
        port.setup.tensor_layout &&
        port.setup.tensor_layout->memory_kind == MemoryKind::kCudaDevice) {
      devices.push_back(
          detail::TensorDeviceBindingRecord{.port = PortId{port.id.value}, .device = DeviceId{0}});
    }
  }
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.tensor_device_bindings = devices;
  const ExecutionPlan plan =
      detail::compile_with_runtime_evidence(graph, std::move(options), evidence);
  return tensor_control
             ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, tensor_control)
             : plan;
}

[[nodiscard]] ExecutionPlan cuda_graph_runtime_plan(
    bool delay_graph = false, std::uint32_t queue_depth = 1U, bool fail_capture = false,
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& tensor_control = {}) {
  Graph graph{"cuda-graph-runtime-execution"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto first = graph.op<CUDAGraphRuntimeTransform>(
      "first", std::size_t{0U}, delay_graph, false, queue_depth, fail_capture);
  auto second =
      graph.op<CUDAGraphRuntimeTransform>("second", std::size_t{1U}, false, false, queue_depth);
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer", queue_depth);
  graph.add_flow(reader->output, first->input, ConnectionOptions{.queue_depth = queue_depth});
  graph.add_flow(first->output, second->input, ConnectionOptions{.queue_depth = queue_depth});
  graph.add_flow(second->output, writer->input, ConnectionOptions{.queue_depth = queue_depth});
  graph.region("captured").add(first).add(second).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph, {}, tensor_control);
}

[[nodiscard]] ExecutionPlan slow_egress_cuda_graph_plan(std::uint32_t frame_count) {
  constexpr std::uint32_t kFixedQueueDepth = 1U;
  Graph graph{"cuda-graph-output-backpressure"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader", frame_count);
  auto transform = graph.op<CUDAGraphRuntimeTransform>(
      "captured-transform", std::size_t{0U}, false, false, kFixedQueueDepth);
  auto writer = graph.op<CUDAGraphAsyncSlowWriter>("slow-writer", frame_count, kFixedQueueDepth);
  graph.add_flow(
      reader->output, transform->input, ConnectionOptions{.queue_depth = kFixedQueueDepth});
  graph.add_flow(
      transform->output, writer->input, ConnectionOptions{.queue_depth = kFixedQueueDepth});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan contended_ingress_cuda_graph_plan() {
  Graph graph{"cuda-graph-contended-ingress"};
  auto reader = graph.op<CUDAGraphContendedIngressReader>("reader");
  auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{0U});
  // Keep the writer active after the two observations under test. The test's explicit stop is then
  // the only terminal request, so it cannot race the writer's completion threshold.
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer", 1U, 3U);
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan lifecycle_restart_cuda_graph_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& tensor_control) {
  Graph graph{"cuda-graph-ingress-lifecycle-restart"};
  auto reader = graph.op<CUDAGraphRestartingReader>("reader");
  auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{0U});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer", 1U, 2U, true);
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.lifecycle().allow_self_restart(reader);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));

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
  return compile_cuda_graph_runtime(graph, std::move(options), tensor_control);
}

struct CUDAGraphIndependentSourcePlan {
  ExecutionPlan plan;
  std::shared_ptr<CUDAGraphIndependentPhysicalFactory> physical_factory;
};

[[nodiscard]] CUDAGraphIndependentSourcePlan independent_source_cuda_graph_plan(
    CUDAGraphIndependentRoot root) {
  Graph graph{"cuda-graph-independent-source"};
  auto source = graph.op<CUDAGraphIndependentSource>("captured-source", root);
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(source->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));

  CompileOptions options;
  if (root == CUDAGraphIndependentRoot::kSpin) {
    options.deployment.bind_thread_placement("runtime",
                                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  }
  ExecutionPlan plan = compile_cuda_graph_runtime(graph, std::move(options));
  std::shared_ptr<CUDAGraphIndependentPhysicalFactory> physical_factory;
  if (root == CUDAGraphIndependentRoot::kReadable || root == CUDAGraphIndependentRoot::kDoorbell ||
      root == CUDAGraphIndependentRoot::kSpin) {
    physical_factory = std::make_shared<CUDAGraphIndependentPhysicalFactory>(root);
    plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(plan, physical_factory);
  }
  return {.plan = std::move(plan), .physical_factory = std::move(physical_factory)};
}

[[nodiscard]] ExecutionPlan completion_limited_cuda_graph_plan(std::uint64_t limit) {
  Graph graph{"cuda-graph-compute-call-completion"};
  auto source = graph.op<CUDAGraphIndependentSource>(
      "captured-source", CUDAGraphIndependentRoot::kClock, std::optional{limit});
  auto writer =
      graph.op<CUDAGraphRuntimeWriter>("writer", 1U, static_cast<std::uint32_t>(limit + 1U));
  graph.add_flow(source->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan reverse_authored_inputless_region_plan() {
  Graph graph{"cuda-graph-reverse-authored-inputless-region"};
  auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{1U});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  auto source =
      graph.op<CUDAGraphIndependentSource>("captured-source", CUDAGraphIndependentRoot::kClock);
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan captured_emit_options_cuda_graph_plan() {
  Graph graph{"cuda-graph-captured-emit-options"};
  auto source = graph.op<CUDAGraphIndependentSource>(
      "captured-source",
      CUDAGraphIndependentRoot::kClock,
      std::nullopt,
      1U,
      EmitOptions{.frame_id = kInternalCaptureFrameId, .flags = SampleFlags::kInvalid});
  auto transform = graph.op<CUDAGraphRuntimeTransform>(
      "captured-transform",
      std::size_t{1U},
      false,
      false,
      1U,
      false,
      false,
      EmitOptions{.frame_id = kRegionEgressFrameId, .flags = SampleFlags::kDegraded});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan variable_ingress_shape_cuda_graph_plan() {
  Graph graph{"cuda-graph-variable-ingress-shape"};
  auto reader = graph.op<CUDAGraphVariableShapeReader>("reader");
  auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{0U});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer", 1U, 3U);
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan repeated_capture_egress_cuda_graph_plan() {
  Graph graph{"cuda-graph-repeated-capture-egress"};
  auto source = graph.op<CUDAGraphIndependentSource>(
      "captured-source", CUDAGraphIndependentRoot::kClock, std::nullopt, 2U);
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(source->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan notification_source_with_slow_egress_plan(
    const std::shared_ptr<detail::RuntimeLateReadinessDrainTestControl>& drain_control) {
  Graph graph{"cuda-graph-notification-output-backpressure"};
  auto source = graph.op<CUDAGraphIndependentSource>("captured-source",
                                                     CUDAGraphIndependentRoot::kNotification);
  auto writer = graph.op<CUDAGraphAsyncSlowWriter>("slow-writer", 2U, 1U);
  graph.add_flow(source->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return detail::RuntimeTestPlanAccess::with_late_readiness_drain_test_control(
      compile_cuda_graph_runtime(graph), drain_control);
}

[[nodiscard]] ExecutionPlan default_any_input_cuda_graph_plan() {
  Graph graph{"cuda-graph-default-any-input"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto transform = graph.op<CUDAGraphRuntimeTransform>(
      "captured-transform", std::size_t{0U}, false, false, 1U, false, true);
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan direct_tensor_emit_cuda_graph_plan() {
  Graph graph{"cuda-graph-direct-tensor-emit"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto transform = graph.op<CUDAGraphDirectTensorEmitTransform>("captured-transform");
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan logical_span_cuda_graph_plan() {
  Graph graph{"cuda-graph-logical-span"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto transform = graph.op<CUDAGraphLogicalSpanTransform>("captured-transform");
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan stale_geometry_cuda_graph_plan() {
  Graph graph{"cuda-graph-capture-output-evidence"};
  auto source = graph.op<CUDAGraphMissingOutputSource>("missing-output-source");
  auto downstream = graph.op<CUDAGraphRuntimeTransform>("downstream", std::size_t{1U});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(source->output, downstream->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(downstream->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  // Capture the downstream region first so its representative ingress initializes the shared
  // edge. The upstream region must still prove that its own compute() emitted that output.
  graph.region("downstream-region").add(downstream).constrain(RegionConstraint::cuda_graph());
  graph.region("upstream-region").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan cancellable_capture_cuda_graph_plan(
    CUDAGraphCaptureStopResult result = CUDAGraphCaptureStopResult::kInterrupted,
    std::shared_ptr<detail::RuntimeCudaExecutionTestControl> control = {}) {
  Graph graph{"cuda-graph-capture-cancellation"};
  auto source = graph.op<CUDAGraphCancellableCaptureSource>("captured-source", result);
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(source->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  const ExecutionPlan plan = compile_cuda_graph_runtime(graph);
  return control != nullptr ? detail::RuntimeTestPlanAccess::with_cuda_execution_test_control(
                                  plan, std::move(control))
                            : plan;
}

[[nodiscard]] ExecutionPlan pinned_host_egress_cuda_graph_plan() {
  Graph graph{"cuda-graph-pinned-host-egress"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto transform = graph.op<CUDAGraphPinnedEgressTransform>("captured-transform");
  auto writer = graph.op<CUDAGraphPinnedEgressWriter>("writer");
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan captured_host_write_cuda_graph_plan() {
  Graph graph{"cuda-graph-host-output-write"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto transform = graph.op<CUDAGraphHostWriteTransform>("captured-transform");
  auto writer = graph.op<CUDAGraphPinnedEgressWriter>("writer");
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
[[nodiscard]] ExecutionPlan closed_cuda_graph_region_plan(std::uint32_t sm_count) {
  Graph graph{"closed-cuda-graph-runtime-region"};
  auto source = graph.op<CUDAGraphClosedSource>("captured-source");
  graph.region("captured").add(source).constrain(RegionConstraint::cuda_graph());
  graph.partition("captured").add(source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  CompileOptions options;
  options.deployment.bind_cuda_green_context("captured",
                                             {.device = DeviceId{0}, .sm_count = sm_count});
  return compile_cuda_graph_runtime(graph, std::move(options));
}
#endif

[[nodiscard]] ExecutionPlan pinned_host_ingress_cuda_graph_plan() {
  Graph graph{"cuda-graph-pinned-host-ingress"};
  auto reader = graph.op<CUDAGraphRuntimeHostReader>("host-reader");
  auto transform = graph.op<CUDAGraphPinnedIngressTransform>("captured-transform");
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer", 1U, 4U);
  graph.add_flow(reader->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

enum class CUDAGraphBoundaryIngress : std::uint8_t {
  kCPUToPinnedHost,
  kCPUToDevice,
  kDevice,
};

enum class CUDAGraphBoundaryEgress : std::uint8_t {
  kPinnedHostToCPU,
  kDevice,
};

struct CUDAGraphBoundaryPlacement {
  CUDAGraphBoundaryIngress ingress;
  CUDAGraphBoundaryEgress egress;
  bool ingress_in_region_partition;
  bool egress_in_region_partition;
};

[[nodiscard]] ExecutionPlan cuda_graph_boundary_matrix_plan(
    const CUDAGraphBoundaryPlacement& placement) {
  constexpr std::uint32_t kFrameCount = 2U;
  Graph graph{"cuda-graph-boundary-matrix"};

  auto place = [&]<typename Source, typename RegionMembers, typename Sink>(
                   Source& source, RegionMembers add_region, Sink& sink) {
    auto region_partition = graph.partition("region-partition");
    if (placement.ingress_in_region_partition) {
      region_partition.add(source);
    } else {
      graph.partition("ingress-partition").add(source);
    }
    add_region(region_partition);
    if (placement.egress_in_region_partition) {
      region_partition.add(sink);
    } else {
      graph.partition("egress-partition").add(sink);
    }
  };  // NOLINT(readability/braces)

  const auto add_pinned_egress = [&]<typename Source, typename Transform>(Source& source,
                                                                          Transform& transform) {
    auto egress = graph.op<CUDAGraphPinnedEgressTransform>("pinned-egress");
    auto sink = graph.op<CUDAGraphPinnedEgressWriter>("cpu-egress", kFrameCount);
    graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(transform->output, egress->input, ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(egress->output, sink->input, ConnectionOptions{.queue_depth = 1U});
    graph.region("captured").add(transform).add(egress).constrain(RegionConstraint::cuda_graph());
    place(source, [&](auto& partition) { partition.add(transform).add(egress); }, sink);
  };  // NOLINT(readability/braces)

  const auto add_device_egress = [&]<typename Source, typename Transform>(Source& source,
                                                                          Transform& transform) {
    auto sink = graph.op<CUDAGraphRuntimeWriter>("device-egress", 1U, kFrameCount);
    graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1U});
    graph.region("captured").add(transform).constrain(RegionConstraint::cuda_graph());
    place(source, [&](auto& partition) { partition.add(transform); }, sink);
  };  // NOLINT(readability/braces)

  switch (placement.ingress) {
    case CUDAGraphBoundaryIngress::kCPUToPinnedHost: {
      auto source = graph.op<CUDAGraphRuntimeHostReader>("cpu-pinned-ingress", kFrameCount);
      auto transform = graph.op<CUDAGraphPinnedIngressTransform>("captured-transform");
      if (placement.egress == CUDAGraphBoundaryEgress::kPinnedHostToCPU) {
        add_pinned_egress(source, transform);
      } else {
        add_device_egress(source, transform);
      }
      break;
    }
    case CUDAGraphBoundaryIngress::kCPUToDevice: {
      auto source = graph.op<CUDAGraphCPUStagedDeviceReader>("cpu-device-ingress", kFrameCount);
      auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{0U});
      if (placement.egress == CUDAGraphBoundaryEgress::kPinnedHostToCPU) {
        add_pinned_egress(source, transform);
      } else {
        add_device_egress(source, transform);
      }
      break;
    }
    case CUDAGraphBoundaryIngress::kDevice: {
      auto source = graph.op<CUDAGraphRuntimeReader>("device-ingress", kFrameCount);
      auto transform = graph.op<CUDAGraphRuntimeTransform>("captured-transform", std::size_t{0U});
      if (placement.egress == CUDAGraphBoundaryEgress::kPinnedHostToCPU) {
        add_pinned_egress(source, transform);
      } else {
        add_device_egress(source, transform);
      }
      break;
    }
  }

  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan two_cuda_graph_regions_plan() {
  Graph graph{"two-cuda-graph-runtime-regions"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto first = graph.op<CUDAGraphRuntimeTransform>("first", std::size_t{0U});
  auto second = graph.op<CUDAGraphRuntimeTransform>("second", std::size_t{1U});
  auto third = graph.op<CUDAGraphRuntimeTransform>("third", std::size_t{2U});
  auto fourth = graph.op<CUDAGraphRuntimeTransform>("fourth", std::size_t{3U});
  auto writer = graph.op<CUDAGraphRuntimeWriter>("writer");
  graph.add_flow(reader->output, first->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(first->output, second->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second->output, third->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(third->output, fourth->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(fourth->output, writer->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured-a").add(first).add(second).constrain(RegionConstraint::cuda_graph());
  graph.region("captured-b").add(third).add(fourth).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] ExecutionPlan independent_cuda_graph_regions_plan() {
  Graph graph{"independent-cuda-graph-runtime-regions"};
  auto reader = graph.op<CUDAGraphRuntimeDualReader>("reader");
  auto first_a = graph.op<CUDAGraphRuntimeTransform>("first-a", std::size_t{0U});
  auto second_a = graph.op<CUDAGraphRuntimeTransform>("second-a", std::size_t{1U});
  auto first_b = graph.op<CUDAGraphRuntimeTransform>("first-b", std::size_t{2U});
  auto second_b = graph.op<CUDAGraphRuntimeTransform>("second-b", std::size_t{3U});
  auto writer_a = graph.op<CUDAGraphRuntimeBranchWriter>("writer-a", std::size_t{0U});
  auto writer_b = graph.op<CUDAGraphRuntimeBranchWriter>("writer-b", std::size_t{1U});
  graph.add_flow(reader->first, first_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(first_a->output, second_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_a->output, writer_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(reader->second, first_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(first_b->output, second_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_b->output, writer_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured-a").add(first_a).add(second_a).constrain(RegionConstraint::cuda_graph());
  graph.region("captured-b").add(first_b).add(second_b).constrain(RegionConstraint::cuda_graph());
  graph.partition("runtime").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
[[nodiscard]] ExecutionPlan two_partition_green_context_cuda_graph_regions_plan(
    std::uint32_t sm_count) {
  Graph graph{"two-partition-green-context-cuda-graph-regions"};
  auto reader_a = graph.op<CUDAGraphRuntimeReader>("reader-a");
  auto first_a = graph.op<CUDAGraphRuntimeTransform>("first-a", std::size_t{0U}, false, true);
  auto second_a = graph.op<CUDAGraphRuntimeTransform>("second-a", std::size_t{1U}, false, true);
  auto writer_a = graph.op<CUDAGraphRuntimeBranchWriter>("writer-a", std::size_t{0U});
  auto reader_b = graph.op<CUDAGraphRuntimeReader>("reader-b");
  auto first_b = graph.op<CUDAGraphRuntimeTransform>("first-b", std::size_t{2U}, false, true);
  auto second_b = graph.op<CUDAGraphRuntimeTransform>("second-b", std::size_t{3U}, false, true);
  auto writer_b = graph.op<CUDAGraphRuntimeBranchWriter>("writer-b", std::size_t{1U});

  graph.add_flow(reader_a->output, first_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(first_a->output, second_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_a->output, writer_a->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(reader_b->output, first_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(first_b->output, second_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_b->output, writer_b->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured-a").add(first_a).add(second_a).constrain(RegionConstraint::cuda_graph());
  graph.region("captured-b").add(first_b).add(second_b).constrain(RegionConstraint::cuda_graph());
  graph.partition("green-a").add(reader_a).add(first_a).add(second_a).add(writer_a);
  graph.partition("green-b").add(reader_b).add(first_b).add(second_b).add(writer_b);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));

  CompileOptions options;
  options.deployment
      .bind_cuda_green_context("green-a", {.device = DeviceId{0}, .sm_count = sm_count})
      .bind_cuda_green_context("green-b", {.device = DeviceId{0}, .sm_count = sm_count});
  return compile_cuda_graph_runtime(graph, std::move(options));
}
#endif

[[nodiscard]] ExecutionPlan no_egress_cuda_graph_plan() {
  Graph graph{"no-egress-cuda-graph-runtime-region"};
  auto reader = graph.op<CUDAGraphRuntimeReader>("reader");
  auto sink = graph.op<CUDAGraphRuntimeInputSink>("captured-sink");
  graph.add_flow(reader->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.region("captured").add(sink).constrain(RegionConstraint::cuda_graph());
  graph.set_default_clock(graph.add_clock<RealtimeClock>("default"));
  return compile_cuda_graph_runtime(graph);
}

[[nodiscard]] int run_cuda_graph_terminal_failure_probe(
    detail::RuntimeCudaExecutionFailurePoint failure_point, std::string_view expected_token,
    bool require_region_identity = false,
    std::optional<detail::RuntimeCudaExecutionObserverPoint> allocation_failure_point =
        std::nullopt) {
  if (cudaSetDevice(0) != cudaSuccess) {
    return 1;
  }
  auto control = std::make_shared<detail::RuntimeCudaExecutionTestControl>(failure_point);
  if (allocation_failure_point) {
    control->set_failure_observer(*allocation_failure_point, nullptr, [](void*) noexcept {
      test::fail_allocation_on_current_thread(0U);
    });
  }
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_cuda_execution_test_control(
      completion_limited_cuda_graph_plan(2U), control);
  if (!plan.ok()) {
    return 2;
  }

  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  if (plan_state == nullptr || plan_state->cuda_graph_execution_regions.size() != 1U) {
    return 3;
  }
  const auto& region = plan_state->cuda_graph_execution_regions.front();
  RunSession failed = run_async(plan);
  failed.wait();
  const bool original_failure =
      failed.termination() == RunTermination::kFailed &&
      std::ranges::any_of(failed.diagnostics(), [&](const Diagnostic& diagnostic) {
        return diagnostic.token == expected_token &&
               (!require_region_identity || (diagnostic.source.graph == region.graph &&
                                             diagnostic.source.vertex == region.fused_operator));
      });
  if (!original_failure) {
    return 4;
  }

  const ExecutionPlan successor_plan = completion_limited_cuda_graph_plan(2U);
  if (!successor_plan.ok()) {
    return 5;
  }
  RunSession successor = run_async(successor_plan);
  successor.wait();
  const bool successor_rejected =
      successor.termination() == RunTermination::kFailed &&
      std::ranges::any_of(successor.diagnostics(), [](const Diagnostic& diagnostic) {
        return diagnostic.token == "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT";
      });
  return successor_rejected ? 0 : 6;
}

[[nodiscard]] int run_cuda_graph_capture_abort_cleanup_failure_probe() {
  if (cudaSetDevice(0) != cudaSuccess) {
    return 1;
  }
  CUDAGraphCaptureCancellationProbe::started.store(false, std::memory_order_relaxed);
  auto control = std::make_shared<detail::RuntimeCudaExecutionTestControl>(
      detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphDestroy);
  const ExecutionPlan plan =
      cancellable_capture_cuda_graph_plan(CUDAGraphCaptureStopResult::kInterrupted, control);
  if (!plan.ok()) {
    return 2;
  }

  RunSession failed = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (!CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire)) {
    failed.request_stop();
    failed.wait();
    return 3;
  }
  failed.request_stop();
  failed.wait();
  const bool cleanup_failure =
      failed.termination() == RunTermination::kFailed && control->visits() == 1U &&
      std::ranges::any_of(failed.diagnostics(), [](const Diagnostic& diagnostic) {
        return diagnostic.token == "CUDA_GRAPH_RUNTIME_SHUTDOWN_FAILED";
      });
  if (!cleanup_failure) {
    return 4;
  }

  const ExecutionPlan successor_plan = completion_limited_cuda_graph_plan(2U);
  if (!successor_plan.ok()) {
    return 5;
  }
  RunSession successor = run_async(successor_plan);
  successor.wait();
  const bool successor_rejected =
      successor.termination() == RunTermination::kFailed &&
      std::ranges::any_of(successor.diagnostics(), [](const Diagnostic& diagnostic) {
        return diagnostic.token == "CUDA_GREEN_CONTEXT_ACTIVE_OWNER_CONFLICT";
      });
  return successor_rejected ? 0 : 6;
}

[[nodiscard]] bool wait_for_physical_retirement(const detail::PhysicalReadinessControl& control) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (control.outstanding() != 0U && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return control.outstanding() == 0U;
}

[[nodiscard]] bool post_when_notification_ready(const NotificationSender& sender) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  auto posted = sender.post();
  while (!posted &&
         (posted.error().code == ErrorCode::kNotReady ||
          posted.error().code == ErrorCode::kBackpressured) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
    posted = sender.post();
  }
  return posted.has_value();
}

class CUDAGraphIndependentSourceTest : public testing::TestWithParam<CUDAGraphIndependentRoot> {};

// REQ: RUNTIME-059
TEST_P(CUDAGraphIndependentSourceTest, ReplaysFromTheOriginalIndependentSourceContract) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  CUDAGraphIndependentNotificationState::reset();
  auto configured = independent_source_cuda_graph_plan(GetParam());
  ASSERT_TRUE(configured.plan.ok()) << configured.plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(configured.plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 1U);
  EXPECT_FALSE(plan_state->cuda_graph_execution_regions.front().ingress.has_value());
  EXPECT_TRUE(plan_state->cuda_graph_execution_regions.front().egress.has_value());

  RunSession session = run_async(configured.plan);
  if (GetParam() == CUDAGraphIndependentRoot::kNotification) {
    const auto sender = CUDAGraphIndependentNotificationState::wait_for_sender();
    HOLOSCAN_ASSERT_HAS_VALUE(sender) << "notification sender unavailable";
    ASSERT_TRUE(post_when_notification_ready(*sender));
    ASSERT_TRUE(probe.wait_for_writes(1U));
    ASSERT_TRUE(post_when_notification_ready(*sender));
  } else if (configured.physical_factory != nullptr) {
    const auto bound = configured.physical_factory->wait_for_control();
    ASSERT_TRUE(bound.has_value());
    const auto coordinate = [&](std::int64_t timestamp_ns) {
      return TimePoint{.timestamp_ns = timestamp_ns, .clock_id = bound->reference_clock};
    };
    if (GetParam() == CUDAGraphIndependentRoot::kReadable) {
      ASSERT_TRUE(bound->control.observe_readable(true, coordinate(1)));
      ASSERT_TRUE(probe.wait_for_writes(1U));
      ASSERT_TRUE(wait_for_physical_retirement(bound->control));
      ASSERT_TRUE(bound->control.observe_readable(false, coordinate(2)));
      ASSERT_TRUE(bound->control.observe_readable(true, coordinate(3)));
    } else if (GetParam() == CUDAGraphIndependentRoot::kDoorbell) {
      auto first = bound->control.ring_doorbell(1U, coordinate(1));
      ASSERT_TRUE(first);
      ASSERT_TRUE(first->witness_accepted);
      ASSERT_TRUE(probe.wait_for_writes(1U));
      ASSERT_TRUE(wait_for_physical_retirement(bound->control));
      auto second = bound->control.ring_doorbell(2U, coordinate(2));
      ASSERT_TRUE(second);
      ASSERT_TRUE(second->witness_accepted);
    } else {
      ASSERT_EQ(GetParam(), CUDAGraphIndependentRoot::kSpin);
      ASSERT_TRUE(bound->control.observe_spin(true, coordinate(1)));
      ASSERT_TRUE(probe.wait_for_writes(1U));
      ASSERT_TRUE(wait_for_physical_retirement(bound->control));
      ASSERT_TRUE(bound->control.observe_spin(false, coordinate(2)));
      ASSERT_TRUE(bound->control.observe_spin(true, coordinate(3)));
    }
  }

  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.starts[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.value, 0x51525354U);
}

INSTANTIATE_TEST_SUITE_P(SupportedIndependentContracts, CUDAGraphIndependentSourceTest,
                         testing::Values(CUDAGraphIndependentRoot::kClock,
                                         CUDAGraphIndependentRoot::kNotification,
                                         CUDAGraphIndependentRoot::kReadable,
                                         CUDAGraphIndependentRoot::kDoorbell,
                                         CUDAGraphIndependentRoot::kSpin));

// REQ: RUNTIME-062
TEST(RunSessionCUDAGraphTest, FixtureObservationRejectsFusedExecutionBeforeStartup) {
  const ExecutionPlan plan = completion_limited_cuda_graph_plan(2U);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto observation = std::make_shared<detail::RuntimeFixtureObservation>(8U, 8U, 8U);
  EXPECT_THROW(static_cast<void>(detail::start_operator_test_run(plan, observation)),
               std::invalid_argument);
  EXPECT_TRUE(observation->records().empty());
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, CountsOnlyFusedLaunchesForComputeCallCompletion) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = completion_limited_cuda_graph_plan(2U);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, BindsReverseAuthoredInputlessRegionToItsFusedSource) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = reverse_authored_inputless_region_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cuda_graph_capture_plans.size(), 1U);
  ASSERT_FALSE(state->cuda_graph_capture_plans.front().members.empty());
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U);
  EXPECT_EQ(state->cuda_graph_capture_plans.front().members.front().operator_ref,
            state->cuda_graph_capture_plans.front().source);
  EXPECT_EQ(state->cuda_graph_execution_regions.front().fused_operator,
            state->cuda_graph_capture_plans.front().members.front().vertex);

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.computes[1U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesOnlyCapturedRegionEgressEmitOptions) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = captured_emit_options_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.capture_input_frame_ids[1U].load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(probe.capture_input_flags[1U].load(std::memory_order_relaxed), 0U);
  ASSERT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  for (std::size_t index = 0U; index < 2U; ++index) {
    EXPECT_EQ(probe.writer_frame_ids[index].load(std::memory_order_relaxed), kRegionEgressFrameId);
    EXPECT_EQ(probe.writer_flags[index].load(std::memory_order_relaxed),
              static_cast<std::uint32_t>(SampleFlags::kDegraded));
  }
  EXPECT_LT(probe.writer_sequences[0U].load(std::memory_order_relaxed),
            probe.writer_sequences[1U].load(std::memory_order_relaxed));
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RejectsRuntimeIngressShapeThatDiffersFromCapturedShape) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = variable_ingress_shape_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_RUNTIME_INGRESS_MISMATCH");
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 1U);
}

// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, FixedIngressShapeAllocationFailureFailsStartupWithoutTermination) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  const ExecutionPlan baseline = cuda_graph_runtime_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto control = std::make_shared<CUDAGraphFixedTensorAllocationFailureControl>(0U);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(baseline, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_TRUE(control->bound());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_RUNTIME_INITIALIZATION_FAILED");
  EXPECT_NE(session.diagnostics().front().provided.find("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"),
            std::string::npos);
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, LaterFixedIngressShapeAllocationFailureRollsBackStartup) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  const ExecutionPlan baseline = independent_cuda_graph_regions_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto control = std::make_shared<CUDAGraphFixedTensorAllocationFailureControl>(1U);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(baseline, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_TRUE(control->bound());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_RUNTIME_INITIALIZATION_FAILED");
  EXPECT_NE(session.diagnostics().front().provided.find("CUDA_GRAPH_FIXED_TENSOR_BINDING_INVALID"),
            std::string::npos);
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RejectsSecondCapturedEgressEmission) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = repeated_capture_egress_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_CAPTURE_OUTPUT_REPEATED");
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RetainsOneInputlessTriggerUntilItsRegionActivationRetires) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  CUDAGraphIndependentNotificationState::reset();
  const auto drain_control = std::make_shared<detail::RuntimeLateReadinessDrainTestControl>();
  const ExecutionPlan plan = notification_source_with_slow_egress_plan(drain_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  drain_control->arm();
  RunSession session = run_async(plan);
  const auto sender = CUDAGraphIndependentNotificationState::wait_for_sender();
  const bool paused = drain_control->wait_until_paused();
  if (!sender || !paused) {
    drain_control->release();
    session.request_stop();
    session.wait();
    FAIL() << "notification source did not reach the controlled drain";
  }
  const auto first = sender->post();
  const auto second = sender->post();
  drain_control->release();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RejectsDirectTensorPublicationDuringCapture) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cuda_graph_runtime_probe().reset();
  const ExecutionPlan plan = direct_tensor_emit_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token,
            "CUDA_GRAPH_CAPTURE_DIRECT_TENSOR_EMIT_UNSUPPORTED");
  EXPECT_EQ(session.diagnostics().front().provided,
            "region_id=1,region_name=captured,error=kFailure: "
            "CUDA_GRAPH_CAPTURE_DIRECT_TENSOR_EMIT_UNSUPPORTED");
  EXPECT_EQ(cuda_graph_runtime_probe().writes.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RejectsCapturedOutputShapeSmallerThanTheFixedShape) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = logical_span_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_CAPTURE_FIXED_SHAPE_MISMATCH");
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RequiresEachRegionToProduceItsOwnCapturedOutputs) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cuda_graph_runtime_probe().reset();
  const ExecutionPlan plan = stale_geometry_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 2U);
  const auto* first_capture = detail::find_unique_cuda_graph_capture_plan_by_region(
      *state,
      state->cuda_graph_execution_regions.front().graph,
      state->cuda_graph_execution_regions.front().region);
  ASSERT_NE(first_capture, nullptr);
  ASSERT_EQ(first_capture->region_name, "downstream-region");

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_CAPTURE_OUTPUT_MISSING");
  EXPECT_EQ(session.diagnostics().front().provided,
            "region_id=2,region_name=upstream-region,error=kFailure: "
            "CUDA_GRAPH_CAPTURE_OUTPUT_MISSING");
}

// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, TreatsAStopDuringCaptureAsRequestedTermination) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  CUDAGraphCaptureCancellationProbe::started.store(false, std::memory_order_relaxed);
  const ExecutionPlan plan = cancellable_capture_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesMemberCaptureFailureThatRacesStop) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  CUDAGraphCaptureCancellationProbe::started.store(false, std::memory_order_relaxed);
  const ExecutionPlan plan =
      cancellable_capture_cuda_graph_plan(CUDAGraphCaptureStopResult::kFailure);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_CAPTURE_CALLBACK_FAILED");
}

// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesNativeCaptureFailureThatRacesStop) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  CUDAGraphCaptureCancellationProbe::started.store(false, std::memory_order_relaxed);
  const auto control = std::make_shared<detail::RuntimeCudaExecutionTestControl>(
      detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphCaptureEnd);
  const ExecutionPlan plan =
      cancellable_capture_cuda_graph_plan(CUDAGraphCaptureStopResult::kSuccess, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(CUDAGraphCaptureCancellationProbe::started.load(std::memory_order_acquire));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "CUDA_GRAPH_RUNTIME_INITIALIZATION_FAILED");
  EXPECT_NE(session.diagnostics().front().provided.find("CUDA Graph capture end failed"),
            std::string::npos);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, ReplaysNormalizedDefaultAnyInputContract) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = default_any_input_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PublishesPinnedHostEgressWithOrdinaryTensorReadiness) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = pinned_host_egress_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.value, 0x2A2A2A2AU);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RejectsHostOutputWriteDuringCapture) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  cuda_graph_runtime_probe().reset();
  const ExecutionPlan plan = captured_host_write_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token,
            "CUDA_GRAPH_CAPTURE_HOST_OUTPUT_WRITE_UNSUPPORTED");
  EXPECT_EQ(session.diagnostics().front().provided,
            "region_id=1,region_name=captured,error=kFailure: "
            "CUDA_GRAPH_CAPTURE_HOST_OUTPUT_WRITE_UNSUPPORTED");
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, ReconstructsCaptureStateForEachRunSession) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  const ExecutionPlan plan = cuda_graph_runtime_plan(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  for (std::size_t run = 0U; run < 2U; ++run) {
    auto& probe = cuda_graph_runtime_probe();
    probe.reset();
    RunSession session = run_async(plan);
    const bool received = probe.wait();
    if (!received) {
      session.request_stop();
    }
    session.wait();

    ASSERT_TRUE(received);
    EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(probe.computes[1U].load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  }
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, ReplaysClosedSingleOperatorRegionWithoutTensorBoundaries) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability || capability->structured_sm_quantum == 0U) {
    GTEST_SKIP() << "Device cannot create a minimum-size CUDA Green Context";
  }

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = closed_cuda_graph_region_plan(capability->structured_sm_quantum);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cuda_graph_execution_regions.size(), 1U);
  EXPECT_FALSE(state->cuda_graph_execution_regions.front().ingress.has_value());
  EXPECT_FALSE(state->cuda_graph_execution_regions.front().egress.has_value());

  RunSession session = run_async(plan);
  const bool launched = probe.wait_for_no_egress_launches(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(launched) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}
#endif

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, QuiescesAnInFlightGraphBeforeOperatorStop) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = cuda_graph_runtime_plan(true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool launched = probe.wait_for_graph_callback();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(launched);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(probe.graph_callback_finished.load(std::memory_order_acquire));
  EXPECT_EQ(probe.stop_before_graph_completion.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, CapturesMembersOnceAndPublishesAfterAsynchronousLaunch) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = cuda_graph_runtime_plan(false, 1U, false, tensor_control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 1U);

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token)
                        << ", fused vertex="
                        << plan_state->cuda_graph_execution_regions.front().fused_operator.value;
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(probe.starts[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.starts[1U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.computes[0U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.computes[1U].load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(probe.ordering_errors.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.value, 0x2A2A2A2AU);
  const std::uintptr_t first_output_address =
      probe.writer_addresses[0U].load(std::memory_order_relaxed);
  EXPECT_NE(first_output_address, 0U);
  EXPECT_EQ(probe.writer_addresses[1U].load(std::memory_order_relaxed), first_output_address);
  ASSERT_EQ(plan_state->tensor_allocations.size(), 3U);
  for (std::size_t index = 0U; index < plan_state->tensor_allocations.size(); ++index) {
    const auto pool = tensor_control->snapshot(index);
    EXPECT_EQ(pool.physical_slot_count, 1U);
    EXPECT_EQ(pool.active_entry_limit, 1U);
    EXPECT_EQ(pool.peak_live_physical_entry_count, 1U);
  }
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesFixedIngressAddressAcrossProducerLifecycleRestart) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& runtime_probe = cuda_graph_runtime_probe();
  runtime_probe.reset();
  auto& restart_probe = cuda_graph_ingress_restart_probe();
  restart_probe.reset();
  const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const ExecutionPlan plan = lifecycle_restart_cuda_graph_plan(tensor_control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 1U);
  const auto& ingress = plan_state->cuda_graph_execution_regions.front().ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(ingress);
  const PortId ingress_output = ingress->source_port;
  const auto ingress_allocation = std::ranges::find(
      plan_state->tensor_allocations, ingress_output, &TensorAllocationPlanRecord::output);
  ASSERT_NE(ingress_allocation, plan_state->tensor_allocations.end());
  const std::size_t ingress_pool_index = static_cast<std::size_t>(
      std::distance(plan_state->tensor_allocations.begin(), ingress_allocation));

  RunSession session = run_async(plan);
  const bool received = runtime_probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_TRUE(restart_probe.restart_accepted.load(std::memory_order_relaxed));
  const std::uintptr_t first_address = restart_probe.addresses[0U].load(std::memory_order_relaxed);
  EXPECT_NE(first_address, 0U);
  EXPECT_EQ(restart_probe.addresses[1U].load(std::memory_order_relaxed), first_address);
  EXPECT_EQ(runtime_probe.writes.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(runtime_probe.writer_values[0U].load(std::memory_order_relaxed), 0x11111111U);
  EXPECT_EQ(runtime_probe.writer_values[1U].load(std::memory_order_relaxed), 0x22222222U);
  const auto ingress_pool = tensor_control->snapshot(ingress_pool_index);
  EXPECT_EQ(ingress_pool.materialization_count, 1U);
  EXPECT_EQ(ingress_pool.maximum_allocation_generation, 1U);
  EXPECT_EQ(ingress_pool.peak_live_materialization_count, 1U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesCaptureFailureContextInStructuralDiagnostic) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = cuda_graph_runtime_plan(false, 1U, true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "CUDA_GRAPH_CAPTURE_CALLBACK_FAILED");
  EXPECT_EQ(diagnostic.service, "cuda_graph_capture");
  EXPECT_EQ(diagnostic.provided,
            "region_id=1,region_name=captured,error=kFailure: "
            "CUDA_GRAPH_CAPTURE_CALLBACK_FAILED");
}

// REQ: RUNTIME-054
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, AttributesReplayLaunchFailureToTheFusedRegion) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const auto control = std::make_shared<detail::RuntimeCudaExecutionTestControl>(
      detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphLaunch);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_cuda_execution_test_control(
      completion_limited_cuda_graph_plan(2U), control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 1U);
  const auto& region = plan_state->cuda_graph_execution_regions.front();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  const auto diagnostic = std::ranges::find(session.diagnostics(),
                                            std::string_view{"CUDA_GRAPH_RUNTIME_EXECUTION_FAILED"},
                                            &Diagnostic::token);
  ASSERT_NE(diagnostic, session.diagnostics().end());
  EXPECT_EQ(diagnostic->service, "cuda_graph_execution");
  EXPECT_EQ(diagnostic->source.graph, region.graph);
  EXPECT_EQ(diagnostic->source.vertex, region.fused_operator);
  EXPECT_EQ(diagnostic->provided,
            "region_id=1,region_name=captured,error=kFailure: "
            "CUDA Graph launch failed: cudaErrorUnknown");
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PublishesDelayedGraphResultsWithOneFixedSlot) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  // Delay device completion while two input samples traverse one fixed slot. Backpressure must
  // preserve ordinary Tensor stream ordering and deliver both outputs.
  const ExecutionPlan plan = cuda_graph_runtime_plan(true, 1U);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RetainsSourceActivationsWhileFixedEgressIsOwned) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  constexpr std::uint32_t kFrameCount = 4U;
  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = slow_egress_cuda_graph_plan(kFrameCount);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait_for_writes(kFrameCount);
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), kFrameCount);
}

// REQ: RUNTIME-025
// REQ: RUNTIME-026
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, ContendedIngressPollUsesTheExistingBoundedRetry) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  CUDAGraphContendedIngressReader::reset();
  const ExecutionPlan baseline = contended_ingress_cuda_graph_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto baseline_state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(baseline_state, nullptr);
  ASSERT_EQ(baseline_state->cuda_graph_execution_regions.size(), 1U);
  const auto& region = baseline_state->cuda_graph_execution_regions.front();
  ASSERT_TRUE(region.ingress.has_value());
  ASSERT_TRUE(region.egress.has_value());

  const auto control = std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(
      0U, detail::RuntimeChannelIncompletePollKind::kNonblocking, region.fused_operator);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(baseline);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(
          detail::RuntimeTestPlanAccess::with_phase_counters(ordinary, counters), control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  CUDAGraphContendedIngressReader::request_emit();
  ASSERT_TRUE(probe.wait_for_writes(1U)) << "warm-up CUDA Graph result was not delivered";

  control->arm(1U);
  CUDAGraphContendedIngressReader::request_emit();
  const bool delivered = probe.wait_for_writes(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(delivered) << "contended CUDA Graph ingress was not delivered";
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(control->forced(), 1U);
  const auto snapshot = counters->snapshot();
  EXPECT_GE(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kReadinessPollContentionRetries)],
            1U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, PreservesFixedShapeCPUWrittenPinnedIngressAcrossGraphIterations) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = pinned_host_ingress_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 4U);
  constexpr std::array<std::uint32_t, 4U> kExpected{
      0x11111111U, 0x22222222U, 0x33333333U, 0x44444444U};
  for (std::size_t index = 0U; index < kExpected.size(); ++index) {
    EXPECT_EQ(probe.writer_values[index].load(std::memory_order_relaxed), kExpected[index]);
  }
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, ExercisesUncoveredIngressEgressAndPartitionBoundaries) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  constexpr std::array kIngresses{CUDAGraphBoundaryIngress::kCPUToPinnedHost,
                                  CUDAGraphBoundaryIngress::kCPUToDevice,
                                  CUDAGraphBoundaryIngress::kDevice};
  constexpr std::array kEgresses{CUDAGraphBoundaryEgress::kPinnedHostToCPU,
                                 CUDAGraphBoundaryEgress::kDevice};
  constexpr std::array kPlacements{false, true};
  std::size_t exercised{};

  for (const auto ingress : kIngresses) {
    for (const auto egress : kEgresses) {
      for (const bool ingress_in_region_partition : kPlacements) {
        for (const bool egress_in_region_partition : kPlacements) {
          // These baseline paths are already covered by dedicated tests above. The matrix adds
          // only the remaining boundary-memory and partition-placement combinations.
          const bool existing_baseline = ingress_in_region_partition &&
                                         egress_in_region_partition &&
                                         ((ingress == CUDAGraphBoundaryIngress::kDevice &&
                                           egress == CUDAGraphBoundaryEgress::kDevice) ||
                                          (ingress == CUDAGraphBoundaryIngress::kCPUToPinnedHost &&
                                           egress == CUDAGraphBoundaryEgress::kDevice) ||
                                          (ingress == CUDAGraphBoundaryIngress::kDevice &&
                                           egress == CUDAGraphBoundaryEgress::kPinnedHostToCPU));
          if (existing_baseline) {
            continue;
          }

          const CUDAGraphBoundaryPlacement placement{
              .ingress = ingress,
              .egress = egress,
              .ingress_in_region_partition = ingress_in_region_partition,
              .egress_in_region_partition = egress_in_region_partition,
          };
          SCOPED_TRACE("ingress=" + std::to_string(static_cast<unsigned>(ingress)) +
                       " egress=" + std::to_string(static_cast<unsigned>(egress)) +
                       " ingress_in_region=" + std::to_string(ingress_in_region_partition) +
                       " egress_in_region=" + std::to_string(egress_in_region_partition));

          auto& probe = cuda_graph_runtime_probe();
          probe.reset();
          const ExecutionPlan plan = cuda_graph_boundary_matrix_plan(placement);
          ASSERT_TRUE(plan.ok()) << plan.json();

          RunSession session = run_async(plan);
          const bool received = probe.wait();
          if (!received) {
            session.request_stop();
          }
          session.wait();

          ASSERT_TRUE(received) << (session.diagnostics().empty()
                                        ? "no runtime diagnostic"
                                        : session.diagnostics().front().token);
          EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
          EXPECT_TRUE(session.diagnostics().empty());
          EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);

          constexpr std::array<std::uint32_t, 2U> kCPUValues{0x11111111U, 0x22222222U};
          constexpr std::array<std::uint32_t, 2U> kDeviceValues{0x2A2A2A2AU, 0x2A2A2A2AU};
          const auto& expected =
              ingress == CUDAGraphBoundaryIngress::kDevice ? kDeviceValues : kCPUValues;
          EXPECT_EQ(probe.writer_values[0U].load(std::memory_order_relaxed), expected[0U]);
          EXPECT_EQ(probe.writer_values[1U].load(std::memory_order_relaxed), expected[1U]);
          ++exercised;
        }
      }
    }
  }
  EXPECT_EQ(exercised, 21U);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RunsTwoChainedRegionsWithOneSharedFixedAllocation) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = two_cuda_graph_regions_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 2U);
  const auto& first_region = plan_state->cuda_graph_execution_regions[0U];
  const auto& second_region = plan_state->cuda_graph_execution_regions[1U];
  const auto& first_region_egress = first_region.egress;
  HOLOSCAN_ASSERT_HAS_VALUE(first_region_egress);
  const auto& second_region_ingress = second_region.ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(second_region_ingress);
  EXPECT_EQ(first_region.partition, second_region.partition);
  EXPECT_EQ(first_region_egress->edge, second_region_ingress->edge);
  const auto first_shared = std::ranges::find_if(first_region.buffers, [&](const auto& buffer) {
    return buffer.kind == detail::CUDAGraphCaptureBufferKind::kEgress &&
           buffer.edge == first_region_egress->edge;
  });
  const auto second_shared = std::ranges::find_if(second_region.buffers, [&](const auto& buffer) {
    return buffer.kind == detail::CUDAGraphCaptureBufferKind::kIngress &&
           buffer.edge == second_region_ingress->edge;
  });
  ASSERT_NE(first_shared, first_region.buffers.end());
  ASSERT_NE(second_shared, second_region.buffers.end());
  ASSERT_FALSE(first_shared->fixed_allocation_id.empty());
  EXPECT_EQ(first_shared->fixed_allocation_id, second_shared->fixed_allocation_id);

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  for (std::size_t index = 0U; index < probe.starts.size(); ++index) {
    EXPECT_EQ(probe.starts[index].load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(probe.computes[index].load(std::memory_order_relaxed), 1U);
  }
  EXPECT_EQ(probe.ordering_errors.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(probe.writes.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.value, 0x2A2A2A2AU);
}

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RunsIndependentRegionsWithDistinctBoundariesInOnePartition) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = independent_cuda_graph_regions_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 2U);
  const auto& first_region = plan_state->cuda_graph_execution_regions[0U];
  const auto& second_region = plan_state->cuda_graph_execution_regions[1U];
  const auto& first_region_ingress = first_region.ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(first_region_ingress);
  const auto& first_region_egress = first_region.egress;
  HOLOSCAN_ASSERT_HAS_VALUE(first_region_egress);
  const auto& second_region_ingress = second_region.ingress;
  HOLOSCAN_ASSERT_HAS_VALUE(second_region_ingress);
  const auto& second_region_egress = second_region.egress;
  HOLOSCAN_ASSERT_HAS_VALUE(second_region_egress);
  EXPECT_EQ(first_region.partition, second_region.partition);
  EXPECT_NE(first_region_ingress->edge, second_region_ingress->edge);
  EXPECT_NE(first_region_egress->edge, second_region_egress->edge);

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  for (std::size_t index = 0U; index < probe.starts.size(); ++index) {
    EXPECT_EQ(probe.starts[index].load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(probe.computes[index].load(std::memory_order_relaxed), 1U);
  }
  EXPECT_EQ(probe.branch_writes[0U].load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.branch_writes[1U].load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(probe.branch_values[0U].load(std::memory_order_relaxed), 0x11111111U);
  EXPECT_EQ(probe.branch_values[1U].load(std::memory_order_relaxed), 0x22222222U);
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RunsTwoPartitionsInDistinctGreenContexts) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto capability = query_cuda_green_context_capability(DeviceId{0});
  if (!capability || capability->structured_sm_quantum == 0U ||
      capability->max_structured_context_count < 2U) {
    GTEST_SKIP() << "Device cannot create two minimum-size CUDA Green Contexts";
  }

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan =
      two_partition_green_context_cuda_graph_regions_plan(capability->structured_sm_quantum);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->cuda_graph_execution_regions.size(), 2U);
  const auto& first_region = plan_state->cuda_graph_execution_regions[0U];
  const auto& second_region = plan_state->cuda_graph_execution_regions[1U];
  const auto& first_region_green_context_id = first_region.green_context_id;
  HOLOSCAN_ASSERT_HAS_VALUE(first_region_green_context_id);
  const auto& second_region_green_context_id = second_region.green_context_id;
  HOLOSCAN_ASSERT_HAS_VALUE(second_region_green_context_id);
  EXPECT_NE(first_region.partition, second_region.partition);
  EXPECT_NE(first_region.execution_lane, second_region.execution_lane);
  EXPECT_NE(first_region_green_context_id, second_region_green_context_id);
  std::size_t dedicated_contexts{};
  bool first_resource_found{};
  bool second_resource_found{};
  for (const auto& resource : plan_state->cuda_green_context_resources) {
    if (resource.role != CudaGreenContextResourceRole::kDedicatedPartition) {
      continue;
    }
    ++dedicated_contexts;
    EXPECT_EQ(resource.accepted_sm_count, capability->structured_sm_quantum);
    first_resource_found =
        first_resource_found || resource.green_context_id == *first_region_green_context_id;
    second_resource_found =
        second_resource_found || resource.green_context_id == *second_region_green_context_id;
  }
  EXPECT_EQ(dedicated_contexts, 2U);
  EXPECT_TRUE(first_resource_found);
  EXPECT_TRUE(second_resource_found);

  RunSession session = run_async(plan);
  const bool received = probe.wait();
  if (!received) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(received) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  for (std::size_t index = 0U; index < probe.starts.size(); ++index) {
    EXPECT_EQ(probe.starts[index].load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(probe.computes[index].load(std::memory_order_relaxed), 1U);
  }
  const std::uintptr_t first_context =
      probe.capture_green_contexts[0U].load(std::memory_order_acquire);
  const std::uintptr_t second_context =
      probe.capture_green_contexts[2U].load(std::memory_order_acquire);
  EXPECT_NE(first_context, 0U);
  EXPECT_NE(second_context, 0U);
  EXPECT_EQ(first_context, probe.capture_green_contexts[1U].load(std::memory_order_acquire));
  EXPECT_EQ(second_context, probe.capture_green_contexts[3U].load(std::memory_order_acquire));
  EXPECT_NE(first_context, second_context);
  EXPECT_NE(probe.branch_threads[0U], std::thread::id{});
  EXPECT_NE(probe.branch_threads[1U], std::thread::id{});
  EXPECT_NE(probe.branch_threads[0U], probe.branch_threads[1U]);
}
#endif

// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphTest, RetiresAnIngressActivationWithoutRegionEgress) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto& probe = cuda_graph_runtime_probe();
  probe.reset();
  const ExecutionPlan plan = no_egress_cuda_graph_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool completed = probe.wait_for_no_egress_launches(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(completed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                           : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(probe.no_egress_launches.load(std::memory_order_relaxed), 2U);
}

// These regressions use a re-executed death-test child because the required failure deliberately
// quarantines device 0 for the rest of its process. This proves successor rejection without
// contaminating later CUDA tests in the parent process.
// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphDeathTest, QuiesceFailureRejectsSameDeviceSuccessor) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce,
                  "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED")),
              testing::ExitedWithCode(0),
              "");
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-054
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphDeathTest,
     QuiesceDiagnosticAllocationFailureRetainsIdentityAndRejectsSameDeviceSuccessor) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce,
                  "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED",
                  true,
                  detail::RuntimeCudaExecutionObserverPoint::kQuiesceFailureRetention)),
              testing::ExitedWithCode(0),
              "");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce,
                  "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED",
                  true,
                  detail::RuntimeCudaExecutionObserverPoint::kFailureDiagnostic)),
              testing::ExitedWithCode(0),
              "");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce,
                  "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED",
                  true,
                  detail::RuntimeCudaExecutionObserverPoint::kQuiesceFailureRelease)),
              testing::ExitedWithCode(0),
              "");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce,
                  "CUDA_GRAPH_RUNTIME_QUIESCE_FAILED",
                  true,
                  detail::RuntimeCudaExecutionObserverPoint::kQuiesceFailureReturn)),
              testing::ExitedWithCode(0),
              "");
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphDeathTest, GraphDestroyFailureRejectsSameDeviceSuccessor) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_terminal_failure_probe(
                  detail::RuntimeCudaExecutionFailurePoint::kCUDAGraphDestroy,
                  "CUDA_GRAPH_RUNTIME_SHUTDOWN_FAILED")),
              testing::ExitedWithCode(0),
              "");
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-046
// REQ: RUNTIME-059
TEST(RunSessionCUDAGraphDeathTest,
     CaptureAbortGraphDestroyFailureOverridesStopAndRejectsSameDeviceSuccessor) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(run_cuda_graph_capture_abort_cleanup_failure_probe()),
              testing::ExitedWithCode(0),
              "");
}

}  // namespace
}  // namespace holoscan
