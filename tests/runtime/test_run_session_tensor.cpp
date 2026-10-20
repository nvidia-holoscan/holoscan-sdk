/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/memory/block_memory_pool.hpp>

#include "common/allocation_counter.hpp"
#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_contracts.hpp"
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
      Tensor tensor = writer->tensor();
      auto writable = tensor.mutable_data();
      const bool same_shape =
          tensor.shape_span().size() == loan.shape().size() &&
          std::equal(tensor.shape_span().begin(), tensor.shape_span().end(), loan.shape().begin());
      if (!writable || *writable != writer->data() || !same_shape ||
          tensor.dtype().code != loan.dtype().code || tensor.dtype().bits != loan.dtype().bits) {
        return make_unexpected(Error{ErrorCode::kFailure});
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

[[nodiscard]] ExecutionPlan make_runtime_tensor_contract_admission_plan() {
  Graph graph{unique_runtime_graph_name("runtime-tensor-contract-admission")};
  static_cast<void>(graph.op<RuntimeTensorContractAdmissionSourceOperator>("source"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
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
                                  << static_cast<std::uint32_t>(
                                         error.value_or(ErrorCode::kSuccess));
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  const auto* metadata_value = optional_value(metadata);
  ASSERT_NE(metadata_value, nullptr);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

  // Key assertion: the coordinate names the Graph default clock, which is what lets the consumer
  // place this sample on the same timeline Runtime published it against.
  EXPECT_EQ(metadata_value->payload_kind, PayloadKind::kDescriptorBacked);
  const ClockId publish_clock = plan.clock_catalog().root_default_clock;
  ASSERT_TRUE(publish_clock.valid());
  EXPECT_EQ(metadata_value->publish_clock_id, publish_clock);
  EXPECT_GT(metadata_value->publish_timestamp_ns, 0);
  EXPECT_TRUE(metadata_value->publish_time().valid());
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
