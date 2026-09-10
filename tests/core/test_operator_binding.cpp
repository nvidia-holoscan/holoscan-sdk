/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/time/manual_clock.hpp>
#include "core/domain/tensor_access.hpp"

#include <time/clock_registry_internal.hpp>

#include <core/operator_spec_compiler_access.hpp>

#include "operator_test_fixture.hpp"

namespace holoscan {

// Payload with no default constructor; exercises schema emission via explicit construction.
// Used by: ConstructorOnlySchemaPayloadUsesDirectEmissionStorage
struct ConstructorOnlySchemaPayload {
  explicit ConstructorOnlySchemaPayload(std::uint32_t value) : value(value) {}

  std::uint32_t value;
};

// Payload whose move constructor conditionally throws; probes move-exception safety in emit().
// Used by: InternalPostEntryValueMoveExceptionMapsToFailure,
//          PreEntryValueAndSampleMoveExceptionsRemainCallerVisible
struct ThrowingMoveSchemaPayload {
  std::uint32_t value{};
  bool throw_on_move{};

  ThrowingMoveSchemaPayload() = default;
  ThrowingMoveSchemaPayload(std::uint32_t value, bool throw_on_move)
      : value(value), throw_on_move(throw_on_move) {}
  ThrowingMoveSchemaPayload(const ThrowingMoveSchemaPayload&) = default;
  ThrowingMoveSchemaPayload& operator=(const ThrowingMoveSchemaPayload&) = default;
  // Intentionally throwing move construction is the behavior under test.
  // NOLINTNEXTLINE(performance-noexcept-move-constructor)
  ThrowingMoveSchemaPayload(ThrowingMoveSchemaPayload&& other) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    if (other.throw_on_move) {
      throw std::runtime_error{"synthetic payload move failure"};
    }
#endif
    value = other.value;
    throw_on_move = other.throw_on_move;
  }
  ThrowingMoveSchemaPayload& operator=(ThrowingMoveSchemaPayload&&) = default;
};

// Payload with a move-throw countdown; distinguishes an exception inside the capture
// operation from one thrown earlier at the emit() call boundary.
// Used by: BoundCaptureMoveFailureDoesNotConsumeSequence
struct LateThrowingMoveSchemaPayload {
  std::uint32_t value{};
  std::uint8_t successful_moves_before_throw{};

  LateThrowingMoveSchemaPayload() = default;
  LateThrowingMoveSchemaPayload(std::uint32_t value, std::uint8_t successful_moves_before_throw)
      : value(value), successful_moves_before_throw(successful_moves_before_throw) {}
  LateThrowingMoveSchemaPayload(const LateThrowingMoveSchemaPayload&) = default;
  LateThrowingMoveSchemaPayload& operator=(const LateThrowingMoveSchemaPayload&) = default;
  // The countdown distinguishes an exception inside the bound capture operation from earlier
  // moves that enter Output::emit and its private function table.
  // NOLINTNEXTLINE(performance-noexcept-move-constructor)
  LateThrowingMoveSchemaPayload(LateThrowingMoveSchemaPayload&& other) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    if (other.successful_moves_before_throw == 0) {
      throw std::runtime_error{"synthetic late payload move failure"};
    }
#endif
    value = other.value;
    successful_moves_before_throw =
        other.successful_moves_before_throw == 0
            ? 0
            : static_cast<std::uint8_t>(other.successful_moves_before_throw - 1U);
  }
  LateThrowingMoveSchemaPayload& operator=(LateThrowingMoveSchemaPayload&&) = default;
};

// Payload whose copy constructor conditionally throws; probes copy-exception safety in
// emit() and receive_data(). Used by: PreEntryValueAndSampleCopyExceptionsRemainCallerVisible
struct ThrowingCopySchemaPayload {
  std::uint32_t value{};
  bool throw_on_copy{};

  ThrowingCopySchemaPayload() = default;
  ThrowingCopySchemaPayload(std::uint32_t value, bool throw_on_copy)
      : value(value), throw_on_copy(throw_on_copy) {}
  ThrowingCopySchemaPayload(const ThrowingCopySchemaPayload& other)
      : value(other.value), throw_on_copy(other.throw_on_copy) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    if (other.throw_on_copy) {
      throw std::runtime_error{"synthetic payload copy failure"};
    }
#endif
  }
  ThrowingCopySchemaPayload& operator=(const ThrowingCopySchemaPayload&) = default;
  ThrowingCopySchemaPayload(ThrowingCopySchemaPayload&&) noexcept = default;
  ThrowingCopySchemaPayload& operator=(ThrowingCopySchemaPayload&&) noexcept = default;
};

// Minimal two-field POD with a runtime ChannelTypeIdentity and inline channel layout;
// exercises the inline_pod fast path in RuntimeErasedInputsAdaptScalarAndRawPodSelections.
struct RuntimeBindingPod {
  std::uint32_t first{};
  std::uint32_t second{};

  friend bool operator==(const RuntimeBindingPod&, const RuntimeBindingPod&) = default;
};

// Schema payload small enough to satisfy ChannelInlinePayload but routed through schema
// emission, not the inline_pod fast path. Used by: RuntimeTypeOpsRemainProcessLocalAndPodOnly
struct InlineLikeSchemaPayload {
  std::uint32_t value{};
};

template <>
consteval SchemaIdentity schema_identity<ConstructorOnlySchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.ConstructorOnlySchemaPayload",
      .version = 1,
  };
}

template <>
consteval SchemaIdentity schema_identity<ThrowingMoveSchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.ThrowingMoveSchemaPayload",
      .version = 1,
  };
}

template <>
consteval SchemaIdentity schema_identity<LateThrowingMoveSchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.LateThrowingMoveSchemaPayload",
      .version = 1,
  };
}

template <>
consteval SchemaIdentity schema_identity<ThrowingCopySchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.ThrowingCopySchemaPayload",
      .version = 1,
  };
}

template <>
consteval SchemaIdentity schema_identity<InlineLikeSchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.InlineLikeSchemaPayload",
      .version = 1,
  };
}

template <>
struct ChannelTypeIdentity<RuntimeBindingPod> {
  static constexpr std::string_view name = "holoscan.test.RuntimeBindingPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<RuntimeBindingPod> {
  static constexpr auto fields = pod_fields(
      pod_field("first", &RuntimeBindingPod::first, offsetof(RuntimeBindingPod, first)),
      pod_field("second", &RuntimeBindingPod::second, offsetof(RuntimeBindingPod, second)));
};

namespace {

static_assert(PublishedPayload<ConstructorOnlySchemaPayload>);
static_assert(!std::default_initializable<ConstructorOnlySchemaPayload>);
static_assert(PublishedPayload<ThrowingMoveSchemaPayload>);
static_assert(PublishedPayload<ThrowingCopySchemaPayload>);
static_assert(PublishedPayload<RuntimeBindingPod>);
static_assert(PublishedPayload<InlineLikeSchemaPayload>);
static_assert(ChannelInlinePayload<RuntimeBindingPod>);
static_assert(ChannelInlinePayload<InlineLikeSchemaPayload>);

void expect_metadata_equal(const SampleMetadata& lhs, const SampleMetadata& rhs) {
  EXPECT_EQ(lhs.version.major, rhs.version.major);
  EXPECT_EQ(lhs.version.minor, rhs.version.minor);
  EXPECT_EQ(lhs.sequence, rhs.sequence);
  EXPECT_EQ(lhs.capture_timestamp_ns, rhs.capture_timestamp_ns);
  EXPECT_EQ(lhs.source_clock_id, rhs.source_clock_id);
  EXPECT_EQ(lhs.publish_timestamp_ns, rhs.publish_timestamp_ns);
  EXPECT_EQ(lhs.publish_clock_id, rhs.publish_clock_id);
  EXPECT_EQ(lhs.trace_id, rhs.trace_id);
  EXPECT_EQ(lhs.frame_id, rhs.frame_id);
  EXPECT_EQ(lhs.byte_size, rhs.byte_size);
  EXPECT_EQ(lhs.payload_kind, rhs.payload_kind);
  EXPECT_EQ(lhs.flags, rhs.flags);
}

// Counts retain/release calls injected into a RuntimeInputBinding to verify read-claim
// lifetime bookkeeping. Used by: exercise_erased_input_adapter
struct RuntimeReadClaimProbe {
  std::uint32_t retain_count{};
  std::uint32_t release_count{};

  static void retain(void* state) noexcept {
    ++static_cast<RuntimeReadClaimProbe*>(state)->retain_count;
  }

  static void release(void* state) noexcept {
    ++static_cast<RuntimeReadClaimProbe*>(state)->release_count;
  }
};

// Synthetic type-erased input binding: holds two generation epochs (old and selected) with
// scalar and batch samples so that bind/unbind ordering can be tested without a real backend.
// Used by: RuntimeErasedInputsAdaptScalarAndRawPodSelections
template <PublishedPayload T>
struct ErasedInputProbe {
  ErasedInputProbe(Sample<T> scalar, std::array<Sample<T>, 2> batch, std::uint64_t generation)
      : scalar(std::move(scalar)), batch(std::move(batch)), generation(generation) {}

  static InputSelection selection(const void* opaque, std::uint64_t generation) noexcept {
    const auto& state = *static_cast<const ErasedInputProbe*>(opaque);
    return generation == state.generation
               ? InputSelection{InputState::kSelected, SelectionReason::kNone}
               : InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen};
  }

  static expected<detail::RuntimeScalarReadBinding, Error> scalar_read(
      const void* opaque, std::uint64_t generation, bool acquire_claim) noexcept {
    auto& state = *static_cast<ErasedInputProbe*>(const_cast<void*>(opaque));
    if (generation != state.generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    return detail::RuntimeScalarReadBinding{
        .sample = &state.scalar,
        .claim = acquire_claim
                     ? detail::PublishedReadClaimBinding{
                           .state = &state.claim,
                           .retain = &RuntimeReadClaimProbe::retain,
                           .release = &RuntimeReadClaimProbe::release,
                       }
                     : detail::PublishedReadClaimBinding{},
    };
  }

  static expected<detail::RuntimeBatchReadBinding, Error> batch_read(
      const void* opaque, std::uint64_t generation) noexcept {
    const auto& state = *static_cast<const ErasedInputProbe*>(opaque);
    if (generation != state.generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    return detail::RuntimeBatchReadBinding{
        .state = &state,
        .size = state.batch.size(),
        .active_generation = &state.generation,
        .expected_generation = generation,
        .sample_at = &sample_at,
    };
  }

  static const void* sample_at(const void* opaque, std::size_t index) noexcept {
    const auto& state = *static_cast<const ErasedInputProbe*>(opaque);
    return index < state.batch.size() ? &state.batch[index] : nullptr;
  }

  Sample<T> scalar;
  std::array<Sample<T>, 2> batch;
  std::uint64_t generation;
  RuntimeReadClaimProbe claim;

  static inline const detail::RuntimeInputBindingOps kOps{
      .selection = &selection,
      .scalar = &scalar_read,
      .batch = &batch_read,
  };
};

// Synthetic type-erased output binding: captures emitted samples and simulates allocation,
// loan cancellation, backpressure, and indeterminate-submission error injection.
// Used by: RuntimeErasedOutputsAdaptScalarAndRawPodEmissionAndLoans
template <PublishedPayload T>
struct ErasedOutputProbe {
  ErasedOutputProbe() : loan{.state = this, .cancel = &cancel_loan} {}

  static expected<void, Error> emit_sample(void* opaque, const void* sample) noexcept {
    auto& state = *static_cast<ErasedOutputProbe*>(opaque);
    if (sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    state.emitted = *static_cast<const Sample<T>*>(sample);
    return {};
  }

  static expected<detail::RuntimeLoanAllocation, Error> allocate(void* opaque) noexcept {
    auto& state = *static_cast<ErasedOutputProbe*>(opaque);
    state.loan_cancelled = false;
    if (state.return_malformed_allocation) {
      return detail::RuntimeLoanAllocation{.loan = &state.loan};
    }
    return detail::RuntimeLoanAllocation{.value = &state.loan_value, .loan = &state.loan};
  }

  static expected<void, Error> emit_loan(void* opaque, detail::RuntimeLoanBinding* loan,
                                         EmitOptions options) noexcept {
    auto& state = *static_cast<ErasedOutputProbe*>(opaque);
    if (loan != &state.loan) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (state.next_loan_error.has_value()) {
      const ErrorCode error = *state.next_loan_error;
      state.next_loan_error.reset();
      if (error == ErrorCode::kPublicationIndeterminate) {
        // The backend owns retirement after crossing its submission boundary. The public loan
        // wrapper must disarm without invoking cancellation or permitting caller retry.
        state.loan_submitted = true;
      }
      return make_unexpected(Error{error});
    }
    SampleMetadata metadata{};
    if (options.capture_time.has_value()) {
      metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
      metadata.source_clock_id = options.capture_time->clock_id;
    }
    metadata.frame_id = options.frame_id.value_or(0U);
    metadata.trace_id = options.trace.has_value() ? options.trace->trace_id : 0U;
    metadata.flags = options.flags;
    state.emitted = Sample<T>{state.loan_value, metadata};
    state.loan_submitted = true;
    return {};
  }

  static void cancel_loan(void* opaque) noexcept {
    static_cast<ErasedOutputProbe*>(opaque)->loan_cancelled = true;
  }

  T loan_value{};
  detail::RuntimeLoanBinding loan;
  std::optional<Sample<T>> emitted;
  bool loan_cancelled{};
  bool loan_submitted{};
  bool return_malformed_allocation{};
  std::optional<ErrorCode> next_loan_error;

  static inline const detail::RuntimeOutputBindingOps kOps{
      .emit_sample = &emit_sample,
      .allocate = &allocate,
      .emit_loan = &emit_loan,
  };
};

// Declares one scalar input and one POD output; used to confirm that runtime_port_type_ops
// returns process-local, inline-pod-aware function tables for each port type.
// Used by: RuntimeTypeOpsRemainProcessLocalAndPodOnly
class RuntimeTypeOpsCaptureOperator final : public Operator<> {
 public:
  Input<std::uint32_t> scalar_input{"scalar_input"};
  Output<RuntimeBindingPod> pod_output{"pod_output"};

  void setup(OperatorSpec& spec) override {
    spec.input(scalar_input);
    spec.output(pod_output);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Shared driver: binds two synthetic input generations (old, then selected) and verifies that
// scalar/batch data from the selected generation is visible while the old is ignored.
template <PublishedPayload T>
void exercise_erased_input_adapter(T old_value, T selected_value, std::array<T, 2> batch_values) {
  constexpr std::uint64_t kOldGeneration = 4U;
  constexpr std::uint64_t kSelectedGeneration = 9U;
  ErasedInputProbe<T> old_state{Sample<T>{old_value, SampleMetadata{.sequence = 1U}},
                                std::array{Sample<T>{old_value, SampleMetadata{.sequence = 2U}},
                                           Sample<T>{old_value, SampleMetadata{.sequence = 3U}}},
                                kOldGeneration};
  ErasedInputProbe<T> selected_state{
      Sample<T>{selected_value, SampleMetadata{.sequence = 11U}},
      std::array{Sample<T>{batch_values[0], SampleMetadata{.sequence = 12U}},
                 Sample<T>{batch_values[1], SampleMetadata{.sequence = 13U}}},
      kSelectedGeneration};
  detail::RuntimeInputBinding old_binding{
      .state = &old_state,
      .ops = &ErasedInputProbe<T>::kOps,
      .generation = kOldGeneration,
  };
  detail::RuntimeInputBinding selected_binding{
      .state = &selected_state,
      .ops = &ErasedInputProbe<T>::kOps,
      .generation = kSelectedGeneration,
  };

  Input<T> input{"input"};
  const auto* type_ops = detail::runtime_port_type_ops<T>();
  ASSERT_NE(type_ops->bind_input, nullptr);
  ASSERT_NE(type_ops->unbind_input, nullptr);
  type_ops->bind_input(&input, &old_binding);
  type_ops->bind_input(&input, &selected_binding);
  type_ops->unbind_input(&input, &old_binding);

  auto scalar_data = input.receive_data();
  ASSERT_TRUE(scalar_data);
  EXPECT_EQ(*scalar_data, selected_value);

  {
    auto scalar = input.receive();
    ASSERT_TRUE(scalar);
    EXPECT_EQ(scalar->data, selected_value);
    EXPECT_EQ(scalar->metadata.sequence, 11U);
    const Sample<T> retained_copy{*scalar};
    EXPECT_EQ(retained_copy.data, selected_value);
    EXPECT_EQ(selected_state.claim.retain_count, 1U);
  }
  EXPECT_EQ(selected_state.claim.release_count, 2U);

  auto batch = input.selected_batch();
  ASSERT_TRUE(batch);
  ASSERT_EQ(batch->size(), batch_values.size());
  EXPECT_EQ((*batch)[0].data, batch_values[0]);
  EXPECT_EQ((*batch)[0].metadata.sequence, 12U);
  EXPECT_EQ((*batch)[1].data, batch_values[1]);
  EXPECT_EQ((*batch)[1].metadata.sequence, 13U);

  ++selected_state.generation;
  auto stale = input.receive();
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);

  type_ops->unbind_input(&input, &selected_binding);
  auto unbound = input.receive();
  ASSERT_FALSE(unbound);
  EXPECT_EQ(unbound.error().code, ErrorCode::kInputUndeclared);
}

// Shared driver: exercises emit-by-value, zero-copy loan, abandon-loan, backpressure retry,
// indeterminate submission, and malformed-allocation paths for erased output adapters.
template <PublishedPayload T>
void exercise_erased_output_adapter(T first_value, T loan_value) {
  ErasedOutputProbe<T> old_state;
  ErasedOutputProbe<T> selected_state;
  detail::RuntimeOutputBinding old_binding{
      .state = &old_state,
      .ops = &ErasedOutputProbe<T>::kOps,
  };
  detail::RuntimeOutputBinding selected_binding{
      .state = &selected_state,
      .ops = &ErasedOutputProbe<T>::kOps,
  };

  Output<T> output{"output"};
  const auto* type_ops = detail::runtime_port_type_ops<T>();
  ASSERT_NE(type_ops->bind_output, nullptr);
  ASSERT_NE(type_ops->unbind_output, nullptr);
  type_ops->bind_output(&output, &old_binding);
  type_ops->bind_output(&output, &selected_binding);
  type_ops->unbind_output(&output, &old_binding);

  const SampleMetadata forwarded_metadata{
      .capture_timestamp_ns = 101,
      .source_clock_id = ClockId{.value = 3U},
      .frame_id = 17U,
      .flags = SampleFlags::kDegraded,
  };
  ASSERT_TRUE(output.emit(Sample<T>{first_value, forwarded_metadata}));
  ASSERT_TRUE(selected_state.emitted.has_value());
  // ASSERT_TRUE above is a fatal GTest guard, but clang-tidy cannot model its early return.
  const auto& first_emitted =
      selected_state.emitted.value();  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(first_emitted.data, first_value);
  expect_metadata_equal(first_emitted.metadata, forwarded_metadata);
  EXPECT_FALSE(old_state.emitted.has_value());

  {
    auto abandoned = output.allocate();
    ASSERT_TRUE(abandoned);
    abandoned->value() = loan_value;
  }
  EXPECT_TRUE(selected_state.loan_cancelled);
  EXPECT_FALSE(selected_state.loan_submitted);

  selected_state.loan_cancelled = false;
  auto loan = output.allocate();
  ASSERT_TRUE(loan);
  loan->value() = loan_value;
  const EmitOptions options{
      .capture_time = TimePoint{.timestamp_ns = 211, .clock_id = ClockId{.value = 5U}},
      .frame_id = 23U,
      .flags = SampleFlags::kInvalid,
  };
  ASSERT_TRUE(output.emit(std::move(*loan), options));
  EXPECT_TRUE(selected_state.loan_submitted);
  EXPECT_FALSE(selected_state.loan_cancelled);
  ASSERT_TRUE(selected_state.emitted.has_value());
  // ASSERT_TRUE above is a fatal GTest guard, but clang-tidy cannot model its early return.
  const auto& loan_emitted =
      selected_state.emitted.value();  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(loan_emitted.data, loan_value);
  EXPECT_EQ(loan_emitted.metadata.capture_time(), *options.capture_time);
  EXPECT_EQ(loan_emitted.metadata.frame_id, *options.frame_id);
  EXPECT_EQ(loan_emitted.metadata.flags, options.flags);

  selected_state.loan_cancelled = false;
  selected_state.loan_submitted = false;
  selected_state.next_loan_error = ErrorCode::kBackpressured;
  auto retryable = output.allocate();
  ASSERT_TRUE(retryable);
  retryable->value() = loan_value;
  auto rejected_before_submit = output.emit(std::move(*retryable));
  ASSERT_FALSE(rejected_before_submit);
  EXPECT_EQ(rejected_before_submit.error().code, ErrorCode::kBackpressured);
  EXPECT_FALSE(selected_state.loan_cancelled);
  EXPECT_FALSE(selected_state.loan_submitted);

  // A proven pre-submit rejection preserves the exact caller-owned loan for correction and
  // explicit resubmission during the same activation.
  retryable->value() = first_value;
  ASSERT_TRUE(output.emit(std::move(*retryable)));
  EXPECT_TRUE(selected_state.loan_submitted);
  EXPECT_FALSE(selected_state.loan_cancelled);
  ASSERT_TRUE(selected_state.emitted.has_value());
  // ASSERT_TRUE above is a fatal GTest guard, but clang-tidy cannot model its early return.
  const auto& retried_emitted =
      selected_state.emitted.value();  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(retried_emitted.data, first_value);

  selected_state.loan_cancelled = false;
  selected_state.loan_submitted = false;
  selected_state.next_loan_error = ErrorCode::kPublicationIndeterminate;
  {
    auto indeterminate = output.allocate();
    ASSERT_TRUE(indeterminate);
    indeterminate->value() = loan_value;
    auto uncertain = output.emit(std::move(*indeterminate));
    ASSERT_FALSE(uncertain);
    EXPECT_EQ(uncertain.error().code, ErrorCode::kPublicationIndeterminate);
    EXPECT_TRUE(selected_state.loan_submitted);
    EXPECT_FALSE(selected_state.loan_cancelled);
  }
  EXPECT_FALSE(selected_state.loan_cancelled);

  selected_state.return_malformed_allocation = true;
  selected_state.loan_cancelled = false;
  auto malformed = output.allocate();
  ASSERT_FALSE(malformed);
  EXPECT_EQ(malformed.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(selected_state.loan_cancelled);

  type_ops->unbind_output(&output, &selected_binding);
  auto unbound = output.emit(first_value);
  ASSERT_FALSE(unbound);
  EXPECT_EQ(unbound.error().code, ErrorCode::kNotReady);
}

// Reads the same frozen scalar input twice and emits 2× the first value; records sequence
// numbers to verify that both receive() calls see the same slot.
// Used by: TypedValueInvokesExactlyOneRealCallback
class DoublingOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};
  std::uint32_t callback_count{};
  std::uint64_t first_sequence{};
  std::uint64_t second_sequence{};

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    ++callback_count;
    HOLOSCAN_TRY(auto first, input_.receive());
    HOLOSCAN_TRY(auto second, input_.receive());
    first_sequence = first.metadata.sequence;
    second_sequence = second.metadata.sequence;
    return output_.emit(first.data * 2);
  }
};

// Passes a Sample<uint32_t> through unchanged; used to verify which metadata fields the
// emit path preserves, resets, or overwrites. Used by: ExplicitSamplePreservesSemanticMetadata
class ForwardingOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, input_.receive());
    return output_.emit(std::move(sample));
  }
};

// Consumes a depth-3 batch from a single input and emits the element sum; records per-element
// sequence numbers to verify non-contiguous batch ordering.
// Used by: NoncontiguousBatchPreservesDeclaredOrder
class BatchSumOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};
  std::array<std::uint64_t, 3> observed_sequences{};

  void setup(OperatorSpec& spec) override {
    spec.input(input_).queue_depth(3);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto batch, input_.selected_batch());
    if (batch.size() != observed_sequences.size()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    std::uint32_t sum{};
    for (std::size_t index = 0; index < batch.size(); ++index) {
      observed_sequences[index] = batch[index].metadata.sequence;
      sum += batch[index].data;
    }
    return output_.emit(sum);
  }
};

// Captures every ExecutionContext field during compute() for post-invocation comparison.
// Used by: ActivationContextIsOneShotAndBorrowed
class ContextProbeOperator final : public Operator<> {
 public:
  Tag observed_tag{};
  TagTime observed_tag_time{};
  TimePoint observed_activation_time{};
  ClockId observed_clock_id{};
  cudaStream_t observed_stream{};
  TraceContext observed_trace{};
  ActivationStatus observed_activation{ActivationReason::kNominal, OnViolation::kEmitWitness};
  bool observed_stop{};
  ParamGeneration observed_params_generation{};
  std::uint32_t callback_count{};

  expected<void, Error> compute(ExecutionContext& context) override {
    ++callback_count;
    observed_tag = context.tag();
    observed_tag_time = context.tag_time();
    observed_activation_time = context.activation_time();
    observed_clock_id = context.clock().id();
    observed_stream = context.cuda_stream().get();
    observed_trace = context.trace();
    observed_activation = context.activation();
    observed_stop = context.cancellation().stop_requested();
    observed_params_generation = context.params_generation();
    return {};
  }
};

// Minimal typed parameter struct backing ParameterProbeOperator.
struct ProbeParams {
  std::uint32_t threshold{};
};

// Captures params_generation and the params pointer from ExecutionContext each activation.
// Used by: ConfiguredParamsUsePositiveDefaultAndPermitExplicitInitialZero
class ParameterProbeOperator final : public Operator<ProbeParams> {
 public:
  ParamGeneration observed_generation{};
  const ProbeParams* observed_params{};

  expected<void, Error> compute(ExecutionContext& context) override {
    observed_generation = context.params_generation();
    observed_params = &context.params<ProbeParams>();
    return {};
  }
};

// Propagates any receive() error directly out of compute(); drives input-guard and
// error-precedence tests. Used by: UnboundInputFailurePropagatesWithoutCallbackRecovery
class ReceiveFailureOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto sample, input_.receive());
    (void)sample;
    return {};
  }
};

// Propagates any selected_batch() error out of compute(); drives batch-guard and
// shape-mismatch tests. Used by: MissingBatchDiffersFromSuccessfulEmptyBatch
class BatchFailureOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto batch, input_.selected_batch());
    (void)batch;
    return {};
  }
};

// Calls receive_data() twice and stores results in public fields; verifies that the same
// frozen scalar value is returned on both calls.
// Used by: ReceiveDataCopiesTheSameFrozenScalarSelection
class DataReadOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  std::uint32_t first{};
  std::uint32_t second{};

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(first, input_.receive_data());
    HOLOSCAN_TRY(second, input_.receive_data());
    return {};
  }
};

// Configurable operator (receive vs. receive_data) that makes three receive attempts and
// records each ErrorCode to verify malformed read-claim rejection.
// Used by: MalformedPrivateReadClaimsAreRejectedAndReleased
class MalformedClaimProbeOperator final : public Operator<> {
 public:
  enum class Accessor : std::uint8_t { kReceive, kReceiveData };

  explicit MalformedClaimProbeOperator(Accessor accessor) : accessor_(accessor) {}

  Input<std::uint32_t> input_{"input"};
  std::array<ErrorCode, 3> observed{};

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    for (ErrorCode& code : observed) {
      if (accessor_ == Accessor::kReceive) {
        auto result = input_.receive();
        code = result ? ErrorCode::kSuccess : result.error().code;
      } else {
        auto result = input_.receive_data();
        code = result ? ErrorCode::kSuccess : result.error().code;
      }
    }
    return {};
  }

 private:
  Accessor accessor_;
};

// Exercises the scalar-claim slot lifecycle: exhausts available slots, releases them by
// overwriting locals, and retains two live samples past compute() completion.
// Used by: ScalarReceiveClaimsUseFiniteIndependentColdSlots
class RetainedScalarReadOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  std::optional<Sample<std::uint32_t>> retained_first_;
  std::optional<Sample<std::uint32_t>> retained_second_;
  std::optional<Sample<std::uint32_t>> retained_copy_;
  std::uint32_t data_without_claim_{};

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto first, input_.receive());
    HOLOSCAN_TRY(auto second, input_.receive());
    std::optional<Sample<std::uint32_t>> shared_copy{first};

    HOLOSCAN_TRY(verify_exhausted());
    first = Sample<std::uint32_t>{0, {}};
    HOLOSCAN_TRY(verify_exhausted());

    second = Sample<std::uint32_t>{0, {}};
    HOLOSCAN_TRY(auto recovered, input_.receive());
    HOLOSCAN_TRY(verify_exhausted());

    shared_copy.reset();
    HOLOSCAN_TRY(auto recovered_after_final_copy, input_.receive());
    HOLOSCAN_TRY(verify_exhausted());

    HOLOSCAN_TRY(data_without_claim_, input_.receive_data());
    recovered = Sample<std::uint32_t>{0, {}};
    HOLOSCAN_TRY(auto second_data_copy, input_.receive_data());
    if (second_data_copy != data_without_claim_) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    HOLOSCAN_TRY(auto receive_after_data_copy, input_.receive());

    retained_first_.emplace(std::move(recovered_after_final_copy));
    retained_second_.emplace(std::move(receive_after_data_copy));
    retained_copy_.emplace(*retained_first_);
    return {};
  }

 private:
  [[nodiscard]] expected<void, Error> verify_exhausted() {
    auto result = input_.receive();
    if (result || result.error().code != ErrorCode::kResourceExhausted) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return {};
  }
};

// Retains two Sample<Tensor> values past fixture retirement to verify that the underlying
// DLManagedTensorContext stays alive as long as any Sample holds a reference.
// Used by: TensorReceivesShareImmutableBackingAfterFixtureRetirement
class TensorReadProbeOperator final : public Operator<> {
 public:
  Input<Tensor> input_{"input"};
  std::optional<Sample<Tensor>> first_;
  std::optional<Sample<Tensor>> second_;

  void setup(OperatorSpec& spec) override { spec.input(input_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto first, input_.receive());
    HOLOSCAN_TRY(auto second, input_.receive());
    first_.emplace(std::move(first));
    second_.emplace(std::move(second));
    return {};
  }
};

// Allocates a zero-copy loan each activation, writes 42, and emits it.
// Used by: GenericLoanPublishesOnePreconstructedBoundedValue
class LoanOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto loan, output_.allocate());
    loan.value() = 42;
    return output_.emit(std::move(loan));
  }
};

// Allocates a loan and retains it across activations; exposes publish_retained() so tests
// can verify stale-publish detection, wrong-output rejection, and post-fixture safety.
// Used by: RetainedLoanCanRetireSafelyAfterItsActivation
class RetainingLoanOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};
  std::optional<LoanedSample<std::uint32_t>> retained_;

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto loan, output_.allocate());
    loan.value() = 42;
    retained_.emplace(std::move(loan));
    return {};
  }

  [[nodiscard]] expected<void, Error> publish_retained() {
    if (!retained_.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return output_.emit(std::move(*retained_));
  }
};

// Emits a ConstructorOnlySchemaPayload to verify that the emit path supports schema
// payloads that require explicit construction and cannot be default-initialized.
// Used by: ConstructorOnlySchemaPayloadUsesDirectEmissionStorage
class ConstructorOnlyEmitOperator final : public Operator<> {
 public:
  Output<ConstructorOnlySchemaPayload> output_{"output"};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    return output_.emit(ConstructorOnlySchemaPayload{77});
  }
};

// Configurable operator whose Mode controls whether the ThrowingMoveSchemaPayload exception
// occurs before the emit() function boundary (caller-visible) or inside it (kFailure).
// Used by: InternalPostEntryValueMoveExceptionMapsToFailure
class ThrowingMoveEmitOperator final : public Operator<> {
 public:
  enum class Mode : std::uint8_t {
    kValue,
    kSample,
    kPreEntryValueMove,
    kPreEntrySampleMove,
    kSuccess,
  };

  Output<ThrowingMoveSchemaPayload> output_{"output"};
  Mode mode{Mode::kValue};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    if (mode == Mode::kSuccess) {
      return output_.emit(ThrowingMoveSchemaPayload{70, false});
    }
    if (mode == Mode::kValue) {
      const ThrowingMoveSchemaPayload value{71, true};
      return output_.emit(value);
    }

    if (mode == Mode::kPreEntryValueMove) {
      ThrowingMoveSchemaPayload value{72, true};
      return output_.emit(std::move(value));
    }

    Sample<ThrowingMoveSchemaPayload> sample{ThrowingMoveSchemaPayload{73, false},
                                             SampleMetadata{.frame_id = 5}};
    sample.data.throw_on_move = true;
    return mode == Mode::kPreEntrySampleMove ? output_.emit(std::move(sample))
                                             : output_.emit(sample);
  }
};

// Emits a LateThrowingMoveSchemaPayload with a countdown chosen to throw either inside the
// capture step (fail_capture=true) or well past the emit() boundary.
// Used by: BoundCaptureMoveFailureDoesNotConsumeSequence
class LateThrowingMoveEmitOperator final : public Operator<> {
 public:
  Output<LateThrowingMoveSchemaPayload> output_{"output"};
  bool fail_capture{true};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    const std::uint8_t successful_moves = fail_capture ? 2U : 8U;
    return output_.emit(LateThrowingMoveSchemaPayload{74, successful_moves});
  }
};

// Configurable operator whose Mode selects bare-value, Sample-wrapped, or non-throwing copy
// emission to cover all copy-exception paths.
// Used by: PreEntryValueAndSampleCopyExceptionsRemainCallerVisible
class ThrowingCopyEmitOperator final : public Operator<> {
 public:
  enum class Mode : std::uint8_t { kValue, kSample, kSuccess };

  Output<ThrowingCopySchemaPayload> output_{"output"};
  Mode mode{Mode::kValue};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    if (mode == Mode::kSuccess) {
      return output_.emit(ThrowingCopySchemaPayload{80, false});
    }
    if (mode == Mode::kValue) {
      const ThrowingCopySchemaPayload value{81, true};
      return output_.emit(value);
    }

    Sample<ThrowingCopySchemaPayload> sample{ThrowingCopySchemaPayload{82, false},
                                             SampleMetadata{.frame_id = 6}};
    sample.data.throw_on_copy = true;
    return output_.emit(sample);
  }
};

// Calls selected_batch() twice and retains the first view plus its begin() iterator across
// compute() to verify stable pointer identity and post-activation access enforcement.
// Used by: BatchBorrowAfterActivationRetirementTerminatesInEveryBuild
class BatchViewProbeOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  std::optional<SelectedBatchView<std::uint32_t>> retained_batch_;
  std::optional<SelectedBatchView<std::uint32_t>::const_iterator> retained_iterator_;
  std::array<const Sample<std::uint32_t>*, 3> first_addresses_{};
  std::array<const Sample<std::uint32_t>*, 3> second_addresses_{};
  std::array<std::uint64_t, 3> first_sequences_{};
  std::array<std::uint64_t, 3> second_sequences_{};

  void setup(OperatorSpec& spec) override { spec.input(input_).queue_depth(3); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto first, input_.selected_batch());
    HOLOSCAN_TRY(auto second, input_.selected_batch());
    if (first.size() != second.size() || first.size() > first_addresses_.size()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
      first_addresses_[index] = &first[index];
      second_addresses_[index] = &second[index];
      first_sequences_[index] = first[index].metadata.sequence;
      second_sequences_[index] = second[index].metadata.sequence;
    }
    retained_batch_.emplace(first);
    if (!first.empty()) {
      retained_iterator_.emplace(retained_batch_->begin());
    }
    return {};
  }
};

// Receives a ThrowingCopySchemaPayload via receive_data() (forcing a copy) and re-emits it;
// verifies that a receive-time copy exception retires both input and output ports.
// Used by: ReceiveDataSemanticCopyExceptionPropagatesAndRetiresActivation
class ThrowingCopyReceiveOperator final : public Operator<> {
 public:
  Input<ThrowingCopySchemaPayload> input_{"input"};
  Output<ThrowingCopySchemaPayload> output_{"output"};

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override {
    auto value = input_.receive_data();
    if (!value) {
      return make_unexpected(std::move(value).error());
    }
    return output_.emit(std::move(*value));
  }
};

// Emits one value successfully then returns a hard failure; confirms that the fixture
// retains already-staged output for inspection even when compute() fails.
// Used by: CallbackFailureRetainsAlreadyStagedOutputForInspection
class EmitThenFailOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(output_.emit(7));
    return make_unexpected(Error{ErrorCode::kFailure});
  }
};

// Declares max_emits_per_compute(2) but attempts three emissions; verifies that the burst
// limit is enforced and accepted emissions are retained as a prefix.
// Used by: BurstOverflowIsBoundedAndPreservesAcceptedPrefix
class BurstOverflowOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override { spec.output(output_).max_emits_per_compute(2); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(output_.emit(1));
    HOLOSCAN_TRY(output_.emit(2));
    return output_.emit(3);
  }
};

// Uses the default single-emit burst limit but attempts two emissions; confirms that the
// implicit limit of one publication per compute() is enforced.
// Used by: DefaultOutputBurstAdmitsExactlyOnePublication
class DefaultBurstOverflowOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(output_.emit(1));
    return output_.emit(2);
  }
};

// Declares an output but never emits; confirms that zero-emission activations succeed
// repeatedly without state leakage. Used by: EmptyOutputIsAValidRepeatableActivationResult
class EmptyOutputOperator final : public Operator<> {
 public:
  Output<std::uint32_t> output_{"output"};
  std::uint32_t callback_count{};

  void setup(OperatorSpec& spec) override { spec.output(output_); }

  expected<void, Error> compute(ExecutionContext&) override {
    ++callback_count;
    return {};
  }
};

// Declares an input and output sharing the name "value"; used to verify that setup()
// returns kDuplicateName. Used by: FreshSetupPassesRejectOnlyTheirOwnDuplicateNames
class DuplicateNameOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"value"};
  Output<std::uint32_t> output_{"value"};

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Dispatches to one invalid OperatorSpec builder call per Kind (zero queue depth, zero emits,
// etc.) to verify each constraint is caught at declaration time.
// Used by: SyntheticDeclarationRejectsInvalidFiniteBuilderBounds
class InvalidBuilderOperator final : public Operator<> {
 public:
  enum class Kind : std::uint8_t {
    kInputDepth,
    kOutputEmits,
    kTensorEmits,
    kTensorRank,
    kTensorBytes,
    kReadableCapacity,
    kSpinCapacity,
    kNotificationCapacity,
    kNotificationSenders,
    kDoorbellCapacity,
  };

  explicit InvalidBuilderOperator(Kind kind) : kind_(kind) {}

  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};
  Output<Tensor> tensor_{"tensor"};
  ReadableSource readable_;
  SpinSource spin_;
  NotificationSource notification_;
  DoorbellSource doorbell_;

  void setup(OperatorSpec& spec) override {
    switch (kind_) {
      case Kind::kInputDepth:
        spec.input(input_).queue_depth(0);
        break;
      case Kind::kOutputEmits:
        spec.output(output_).max_emits_per_compute(0);
        break;
      case Kind::kTensorEmits:
        spec.output(tensor_).max_emits_per_compute(0);
        break;
      case Kind::kTensorRank:
        spec.output(tensor_).produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost},
            .bounds = TensorBounds{.max_rank = 0U, .max_byte_span = 1024U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
        break;
      case Kind::kTensorBytes:
        spec.output(tensor_).produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost},
            .bounds = TensorBounds{.max_rank = 4U, .max_byte_span = 0U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
        break;
      case Kind::kReadableCapacity:
        spec.readable_source(readable_, "readable").capacity(2);
        break;
      case Kind::kSpinCapacity:
        spec.spin_source(spin_, "spin").capacity(0);
        break;
      case Kind::kNotificationCapacity:
        spec.notification_source(notification_, "notification").capacity(0);
        break;
      case Kind::kNotificationSenders:
        spec.notification_source(notification_, "notification").sender_reference_capacity(0);
        break;
      case Kind::kDoorbellCapacity:
        spec.doorbell_source(doorbell_, "doorbell").capacity(0);
        break;
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  Kind kind_;
};

template <typename Fixture, typename Port>
concept CanBindSyntheticOutput =
    requires(Fixture& fixture, Port& port) { fixture.bind_output(port); };

using InvalidBuilderFixture = test::SyntheticOperatorFixture<InvalidBuilderOperator>;
// REQ: OPERATOR-019
static_assert(CanBindSyntheticOutput<InvalidBuilderFixture, Output<std::uint32_t>>);
static_assert(!CanBindSyntheticOutput<InvalidBuilderFixture, Output<Tensor>>);
static_assert(!noexcept(std::declval<Output<ThrowingMoveSchemaPayload>&>().emit(
    std::declval<ThrowingMoveSchemaPayload>())));
static_assert(!noexcept(std::declval<Output<ThrowingMoveSchemaPayload>&>().emit(
    std::declval<Sample<ThrowingMoveSchemaPayload>>())));

TEST(OperatorBindingConformance, RuntimeTypeOpsRemainProcessLocalAndPodOnly) {
  RuntimeTypeOpsCaptureOperator op;
  OperatorSpec spec;
  op.setup(spec);

  ASSERT_EQ(detail::OperatorSpecCompilerAccess::port_count(spec), 2U);
  const auto scalar = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  const auto pod = detail::OperatorSpecCompilerAccess::port(spec, 1U);
  EXPECT_EQ(scalar.runtime_type_ops, detail::runtime_port_type_ops<std::uint32_t>());
  EXPECT_EQ(pod.runtime_type_ops, detail::runtime_port_type_ops<RuntimeBindingPod>());
  EXPECT_TRUE(scalar.runtime_type_ops->inline_pod);
  EXPECT_TRUE(pod.runtime_type_ops->inline_pod);

  const auto* schema_ops = detail::runtime_port_type_ops<InlineLikeSchemaPayload>();
  EXPECT_FALSE(schema_ops->inline_pod);
  EXPECT_EQ(schema_ops->construct_sample, nullptr);
  EXPECT_EQ(schema_ops->bind_input, nullptr);
  EXPECT_EQ(schema_ops->bind_output, nullptr);
}

TEST(OperatorBindingConformance, RuntimeTypeOpsMaterializeExactRawPodSamples) {
  const RuntimeBindingPod payload{.first = 17U, .second = 29U};
  const SampleMetadata metadata{
      .sequence = 31U,
      .capture_timestamp_ns = 37,
      .source_clock_id = ClockId{.value = 2U},
      .frame_id = 41U,
  };
  const auto* type_ops = detail::runtime_port_type_ops<RuntimeBindingPod>();
  ASSERT_TRUE(type_ops->inline_pod);
  ASSERT_EQ(type_ops->payload_size, sizeof(payload));
  ASSERT_EQ(type_ops->payload_alignment, alignof(RuntimeBindingPod));
  ASSERT_EQ(type_ops->sample_size, sizeof(Sample<RuntimeBindingPod>));
  ASSERT_EQ(type_ops->sample_alignment, alignof(Sample<RuntimeBindingPod>));
  ASSERT_NE(type_ops->construct_sample, nullptr);
  ASSERT_NE(type_ops->destroy_sample, nullptr);

  alignas(Sample<RuntimeBindingPod>) std::array<std::byte, sizeof(Sample<RuntimeBindingPod>)>
      storage{};
  ASSERT_TRUE(type_ops->construct_sample(
      storage.data(), std::as_bytes(std::span<const RuntimeBindingPod>{&payload, 1U}), metadata));
  const auto* sample =
      std::launder(reinterpret_cast<const Sample<RuntimeBindingPod>*>(storage.data()));  // NOLINT
  EXPECT_EQ(sample->data, payload);
  expect_metadata_equal(sample->metadata, metadata);
  EXPECT_EQ(type_ops->sample_payload(sample), &sample->data);
  EXPECT_EQ(type_ops->sample_metadata(sample), &sample->metadata);
  type_ops->destroy_sample(storage.data());

  const std::array<std::byte, sizeof(RuntimeBindingPod) - 1U> short_payload{};
  const auto rejected = type_ops->construct_sample(storage.data(), short_payload, metadata);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kReceiveError);
}

TEST(OperatorBindingConformance, RuntimeErasedInputsAdaptScalarAndRawPodSelections) {
  exercise_erased_input_adapter<std::uint32_t>(3U, 7U, std::array<std::uint32_t, 2>{11U, 13U});
  exercise_erased_input_adapter<RuntimeBindingPod>(
      RuntimeBindingPod{.first = 1U, .second = 2U},
      RuntimeBindingPod{.first = 3U, .second = 5U},
      std::array{RuntimeBindingPod{.first = 7U, .second = 11U},
                 RuntimeBindingPod{.first = 13U, .second = 17U}});
}

// REQ: RUNTIME-019
TEST(OperatorBindingConformance, RuntimeErasedOutputsAdaptScalarAndRawPodEmissionAndLoans) {
  exercise_erased_output_adapter<std::uint32_t>(19U, 23U);
  exercise_erased_output_adapter<RuntimeBindingPod>(RuntimeBindingPod{.first = 29U, .second = 31U},
                                                    RuntimeBindingPod{.first = 37U, .second = 41U});
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, TypedValueInvokesExactlyOneRealCallback) {
  //   fixture --[uint32=21]--> DoublingOperator --[uint32]--> fixture:captured
  DoublingOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_value(op.input_, std::uint32_t{21}));
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_TRUE(result);
  EXPECT_EQ(op.callback_count, 1);
  EXPECT_EQ(op.first_sequence, op.second_sequence);

  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 42);
  EXPECT_FALSE(captured[0].metadata.capture_time().valid());
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, ReceiveDataCopiesTheSameFrozenScalarSelection) {
  //   fixture --[uint32=33]--> DataReadOperator  (no bound output)
  DataReadOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_value(op.input_, std::uint32_t{33}));
  ASSERT_TRUE(fixture.invoke_once());

  EXPECT_EQ(op.first, 33);
  EXPECT_EQ(op.second, 33);
}

// REQ: OPERATOR-015
// REQ: OPERATOR-020
TEST(OperatorBindingConformance, MalformedPrivateReadClaimsAreRejectedAndReleased) {
  // Key setup: three Cases exercise omit-claim, partial-claim, and unexpected-claim states.
  struct Case {
    MalformedClaimProbeOperator::Accessor accessor;
    detail::SyntheticInputState state;
  };
  const auto make_state = [](bool omit, bool partial, bool unexpected) {
    return detail::SyntheticInputState{
        .selector_shape = detail::SyntheticSelectorShape::kScalar,
        .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
        .advertises_value = true,
        .omit_scalar_read_claim = omit,
        .return_partial_scalar_read_claim = partial,
        .return_unexpected_scalar_read_claim = unexpected,
    };
  };
  const std::array cases{
      Case{
          .accessor = MalformedClaimProbeOperator::Accessor::kReceive,
          .state = make_state(true, false, false),
      },
      Case{
          .accessor = MalformedClaimProbeOperator::Accessor::kReceive,
          .state = make_state(false, true, false),
      },
      Case{
          .accessor = MalformedClaimProbeOperator::Accessor::kReceiveData,
          .state = make_state(false, false, true),
      },
  };

  for (const Case& test_case : cases) {
    MalformedClaimProbeOperator op{test_case.accessor};
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(
        fixture.bind_sample_state(op.input_, Sample<std::uint32_t>{55, {}}, test_case.state));
    ASSERT_TRUE(fixture.invoke_once());
    // Key assertion: all three receive attempts yield kReceiveError regardless of accessor.
    EXPECT_EQ(
        op.observed,
        (std::array{ErrorCode::kReceiveError, ErrorCode::kReceiveError, ErrorCode::kReceiveError}));
  }
}

// REQ: OPERATOR-015
// REQ: OPERATOR-019
TEST(OperatorBindingConformance, ScalarReceiveClaimsUseFiniteIndependentColdSlots) {
  // Key setup: RetainedScalarReadOperator exhausts slots, releases via local overwrites,
  //            and retains two distinct Sample handles plus a copy across compute().
  RetainedScalarReadOperator op;
  const SampleMetadata metadata{
      .version = SampleMetadataVersion{.major = 3, .minor = 4},
      .sequence = 29,
      .capture_timestamp_ns = 31,
      .source_clock_id = ClockId{.value = 5},
      .publish_timestamp_ns = 37,
      .publish_clock_id = ClockId{.value = 7},
      .trace_id = 41,
      .frame_id = 43,
      .byte_size = 47,
      .payload_kind = PayloadKind::kSerializedBytes,
      .flags = SampleFlags::kInvalid | SampleFlags::kDegraded,
  };
  {
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_sample(op.input_, Sample<std::uint32_t>{33, metadata}));
    ASSERT_TRUE(fixture.invoke_once());
  }

  ASSERT_TRUE(op.retained_first_.has_value());
  ASSERT_TRUE(op.retained_second_.has_value());
  ASSERT_TRUE(op.retained_copy_.has_value());
  const Sample<std::uint32_t> retained_first =
      op.retained_first_.value_or(Sample<std::uint32_t>{0, {}});
  const Sample<std::uint32_t> retained_second =
      op.retained_second_.value_or(Sample<std::uint32_t>{0, {}});
  const Sample<std::uint32_t> retained_copy =
      op.retained_copy_.value_or(Sample<std::uint32_t>{0, {}});
  // Key assertion: all retained handles carry the same value and full metadata as the original.
  EXPECT_EQ(retained_first.data, 33);
  EXPECT_EQ(retained_second.data, 33);
  EXPECT_EQ(retained_copy.data, 33);
  EXPECT_EQ(op.data_without_claim_, 33);
  expect_metadata_equal(retained_first.metadata, metadata);
  expect_metadata_equal(retained_second.metadata, metadata);
  expect_metadata_equal(retained_copy.metadata, metadata);
  expect_metadata_equal(retained_first.metadata, retained_second.metadata);
  expect_metadata_equal(retained_first.metadata, retained_copy.metadata);
}

// REQ: OPERATOR-001
// REQ: OPERATOR-015
// REQ: OPERATOR-019
TEST(OperatorBindingConformance, TensorReceivesShareImmutableBackingAfterFixtureRetirement) {
  // Key setup: a 2×3 uint32 host tensor is bound and the fixture is fully retired before
  //            the retained samples are inspected; allocator also goes out of scope.
  TensorReadProbeOperator op;
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  const void* expected_data = nullptr;
  const std::int64_t* expected_shape = nullptr;
  const DLManagedTensorContext* expected_context = nullptr;
  std::weak_ptr<DLManagedTensorContext> context_lifetime;
  {
    const auto allocator = std::make_shared<UnboundedAllocator>();
    auto tensor = make_tensor(
        shape, DLDataType{kDLUInt, 32, 1}, std::nullopt, MemoryStorageType::kHost, allocator);
    ASSERT_TRUE(tensor);
    expected_data = tensor->data();
    expected_shape = tensor->shape_span().data();
    expected_context = detail::TensorAccess::context(*tensor).get();
    context_lifetime = detail::TensorAccess::context(*tensor);
    auto mutable_values = tensor->mutable_data_as<std::uint32_t>();
    ASSERT_TRUE(mutable_values) << mutable_values.error().message;
    auto* const values = mutable_values.value();
    for (std::uint32_t index = 0; index < 6; ++index) {
      values[index] = index + 1U;
    }
    {
      test::SyntheticOperatorFixture fixture{op};
      ASSERT_TRUE(fixture.declare_ports());
      ASSERT_TRUE(fixture.bind_sample(op.input_, Sample<Tensor>{*tensor, {}}));
      ASSERT_TRUE(fixture.invoke_once());
    }
  }

  ASSERT_FALSE(context_lifetime.expired());
  ASSERT_TRUE(op.first_.has_value());
  ASSERT_TRUE(op.second_.has_value());
  {
    const Sample<Tensor> first_sample = op.first_.value_or(Sample<Tensor>{Tensor{}, {}});
    const Sample<Tensor> second_sample = op.second_.value_or(Sample<Tensor>{Tensor{}, {}});
    const Tensor& first = first_sample.data;
    const Tensor& second = second_sample.data;
    EXPECT_EQ(first.data(), expected_data);
    EXPECT_EQ(second.data(), expected_data);
    EXPECT_EQ(detail::TensorAccess::context(first).get(), expected_context);
    EXPECT_EQ(detail::TensorAccess::context(second).get(), expected_context);
    ASSERT_EQ(first.shape_span().size(), shape.size());
    ASSERT_EQ(second.shape_span().size(), shape.size());
    // Key assertion: both samples point to the same data/shape/context as the original tensor.
    EXPECT_EQ(first.shape_span().data(), expected_shape);
    EXPECT_EQ(second.shape_span().data(), expected_shape);
    EXPECT_TRUE(std::equal(shape.begin(), shape.end(), first.shape_span().begin()));
    EXPECT_TRUE(std::equal(shape.begin(), shape.end(), second.shape_span().begin()));
    EXPECT_EQ(static_cast<const std::uint32_t*>(first.data())[5], 6);
    EXPECT_EQ(static_cast<const std::uint32_t*>(second.data())[0], 1);
    op.first_.reset();
    op.second_.reset();
    EXPECT_FALSE(context_lifetime.expired());
  }
  EXPECT_TRUE(context_lifetime.expired());
}

// REQ: OPERATOR-009
TEST(OperatorBindingConformance, GenericLoanPublishesOnePreconstructedBoundedValue) {
  LoanOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.invoke_once());

  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 42);

  auto first = fixture.take_captured(op.output_);
  ASSERT_TRUE(first);
  ASSERT_TRUE(fixture.invoke_once());
  const auto second = fixture.captured(op.output_);
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(second[0].data, 42);
}

// REQ: OPERATOR-009
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, RetainedLoanCanRetireSafelyAfterItsActivation) {
  // Key setup: loan is retained across activations; publish_retained() is called after
  //            the activation that produced the loan has already ended.
  RetainingLoanOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.invoke_once());
  ASSERT_TRUE(op.retained_.has_value());

  // Key assertion: publishing a stale loan returns kStaleEndpoint and leaves the loan intact.
  const auto stale_publish = op.publish_retained();
  ASSERT_FALSE(stale_publish);
  EXPECT_EQ(stale_publish.error().code, ErrorCode::kStaleEndpoint);
  if (!op.retained_.has_value()) {
    FAIL() << "stale publication unexpectedly consumed the retained loan";
    return;
  }
  EXPECT_EQ(op.retained_->value(), 42);

  op.retained_.reset();
  ASSERT_TRUE(fixture.invoke_once());
  if (!op.retained_.has_value()) {
    FAIL() << "second activation did not produce a retained loan";
    return;
  }
  EXPECT_EQ(op.retained_->value(), 42);
  op.retained_.reset();
}

// REQ: OPERATOR-009
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, RetainedLoanUsesTombstoneAcrossOutputRebind) {
  // Key setup: the output is rebound to a new fixture; the retained loan from the first
  //            fixture is then submitted via publish_retained().
  RetainingLoanOperator op;
  test::SyntheticOperatorFixture first_fixture{op};
  ASSERT_TRUE(first_fixture.declare_ports());
  ASSERT_TRUE(first_fixture.bind_output(op.output_));
  ASSERT_TRUE(first_fixture.invoke_once());

  test::SyntheticOperatorFixture rebound_fixture{op};
  ASSERT_TRUE(rebound_fixture.declare_ports());
  ASSERT_TRUE(rebound_fixture.bind_output(op.output_));
  // Key assertion: submitting a loan to the wrong output binding returns kInvalidArgument.
  const auto wrong_binding = op.publish_retained();
  ASSERT_FALSE(wrong_binding);
  EXPECT_EQ(wrong_binding.error().code, ErrorCode::kInvalidArgument);
  if (!op.retained_.has_value()) {
    FAIL() << "wrong-output rejection unexpectedly consumed the retained loan";
    return;
  }
  EXPECT_EQ(op.retained_->value(), 42);
  op.retained_.reset();
}

// REQ: OPERATOR-009
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, RetainedLoanOutlivesFixtureWithoutDanglingCancellation) {
  RetainingLoanOperator op;
  {
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_output(op.output_));
    ASSERT_TRUE(fixture.invoke_once());
  }

  if (!op.retained_.has_value()) {
    FAIL() << "fixture teardown unexpectedly consumed the retained loan";
    return;
  }
  EXPECT_EQ(op.retained_->value(), 42);
  op.retained_.reset();
}

// REQ: OPERATOR-009
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, IndeterminateLoanSubmissionConsumesFiniteAuthority) {
  // Key setup: the fixture is configured to return kPublicationIndeterminate on the first loan.
  LoanOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.fail_next_loan_publication_indeterminate(op.output_));

  const auto indeterminate = fixture.invoke_once();
  ASSERT_FALSE(indeterminate);
  EXPECT_EQ(indeterminate.error().code, ErrorCode::kPublicationIndeterminate);
  EXPECT_TRUE(fixture.captured(op.output_).empty());

  // Key assertion: the second invocation succeeds and the sequence counter is NOT incremented
  //               by the indeterminate activation (sequence remains at 1→2, not 1→3).
  ASSERT_TRUE(fixture.invoke_once());
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 42);
  EXPECT_EQ(captured[0].metadata.sequence, 2);
}

// REQ: OPERATOR-001
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, ConstructorOnlySchemaPayloadUsesDirectEmissionStorage) {
  ConstructorOnlyEmitOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.invoke_once());

  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data.value, 77);
  EXPECT_EQ(captured[0].metadata.payload_kind, PayloadKind::kSerializedBytes);
  EXPECT_EQ(captured[0].metadata.byte_size, 0);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: OPERATOR-014
// REQ: OPERATOR-021
TEST(OperatorBindingConformance, InternalPostEntryValueMoveExceptionMapsToFailure) {
  ThrowingMoveEmitOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(fixture.captured(op.output_).empty());

  op.mode = ThrowingMoveEmitOperator::Mode::kSuccess;
  ASSERT_TRUE(fixture.invoke_once());
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].metadata.sequence, 1);
}

// REQ: OPERATOR-014
// REQ: OPERATOR-021
TEST(OperatorBindingConformance, InternalPostEntrySampleMoveExceptionMapsToFailure) {
  ThrowingMoveEmitOperator op;
  op.mode = ThrowingMoveEmitOperator::Mode::kSample;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(fixture.captured(op.output_).empty());

  op.mode = ThrowingMoveEmitOperator::Mode::kSuccess;
  ASSERT_TRUE(fixture.invoke_once());
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].metadata.sequence, 1);
}

// REQ: OPERATOR-021
TEST(OperatorBindingConformance, BoundCaptureMoveFailureDoesNotConsumeSequence) {
  LateThrowingMoveEmitOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto failed = fixture.invoke_once();
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(fixture.captured(op.output_).empty());

  op.fail_capture = false;
  ASSERT_TRUE(fixture.invoke_once());
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].metadata.sequence, 1);
}

// REQ: OPERATOR-021
TEST(OperatorBindingConformance, PreEntryValueAndSampleMoveExceptionsRemainCallerVisible) {
  for (const ThrowingMoveEmitOperator::Mode mode : {
           ThrowingMoveEmitOperator::Mode::kPreEntryValueMove,
           ThrowingMoveEmitOperator::Mode::kPreEntrySampleMove,
       }) {
    ThrowingMoveEmitOperator op;
    op.mode = mode;
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_output(op.output_));

    EXPECT_THROW(static_cast<void>(fixture.invoke_once()), std::runtime_error);
    EXPECT_TRUE(fixture.captured(op.output_).empty());
    op.mode = ThrowingMoveEmitOperator::Mode::kSuccess;
    ASSERT_TRUE(fixture.invoke_once());
    const auto captured = fixture.captured(op.output_);
    ASSERT_EQ(captured.size(), 1);
    EXPECT_EQ(captured[0].metadata.sequence, 1);
  }
}

// REQ: OPERATOR-021
TEST(OperatorBindingConformance, PreEntryValueAndSampleCopyExceptionsRemainCallerVisible) {
  for (const ThrowingCopyEmitOperator::Mode mode : {
           ThrowingCopyEmitOperator::Mode::kValue,
           ThrowingCopyEmitOperator::Mode::kSample,
       }) {
    ThrowingCopyEmitOperator op;
    op.mode = mode;
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_output(op.output_));

    EXPECT_THROW(static_cast<void>(fixture.invoke_once()), std::runtime_error);
    EXPECT_TRUE(fixture.captured(op.output_).empty());
    op.mode = ThrowingCopyEmitOperator::Mode::kSuccess;
    ASSERT_TRUE(fixture.invoke_once());
    const auto captured = fixture.captured(op.output_);
    ASSERT_EQ(captured.size(), 1);
    EXPECT_EQ(captured[0].metadata.sequence, 1);
  }
}

// REQ: OPERATOR-014
// REQ: OPERATOR-023
// REQ: OPERATOR-012
TEST(OperatorBindingConformance, ReceiveDataSemanticCopyExceptionPropagatesAndRetiresActivation) {
  ThrowingCopyReceiveOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.bind_value(op.input_, ThrowingCopySchemaPayload{41, true}));

  EXPECT_THROW(static_cast<void>(fixture.invoke_once()), std::runtime_error);
  const auto stale_input = op.input_.receive_data();
  ASSERT_FALSE(stale_input);
  EXPECT_EQ(stale_input.error().code, ErrorCode::kStaleEndpoint);
  const auto retired_output = op.output_.emit(ThrowingCopySchemaPayload{42, false});
  ASSERT_FALSE(retired_output);
  EXPECT_EQ(retired_output.error().code, ErrorCode::kStaleEndpoint);

  ASSERT_TRUE(fixture.bind_value(op.input_, ThrowingCopySchemaPayload{43, false}));
  ASSERT_TRUE(fixture.invoke_once());
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data.value, 43);
}
#endif

// REQ: OPERATOR-004
TEST(OperatorBindingConformance, FreshSetupPassesRejectOnlyTheirOwnDuplicateNames) {
  DuplicateNameOperator duplicate;
  test::SyntheticOperatorFixture duplicate_fixture{duplicate};
  const auto duplicate_result = duplicate_fixture.declare_ports();
  ASSERT_FALSE(duplicate_result);
  EXPECT_EQ(duplicate_result.error().code, ErrorCode::kDuplicateName);

  DataReadOperator valid;
  test::SyntheticOperatorFixture first{valid};
  test::SyntheticOperatorFixture second{valid};
  EXPECT_TRUE(first.declare_ports());
  EXPECT_TRUE(second.declare_ports());
}

// REQ: OPERATOR-004
TEST(OperatorBindingConformance, SyntheticDeclarationRejectsInvalidFiniteBuilderBounds) {
  constexpr std::array invalid_kinds{
      InvalidBuilderOperator::Kind::kInputDepth,
      InvalidBuilderOperator::Kind::kOutputEmits,
      InvalidBuilderOperator::Kind::kTensorEmits,
      InvalidBuilderOperator::Kind::kTensorRank,
      InvalidBuilderOperator::Kind::kTensorBytes,
      InvalidBuilderOperator::Kind::kReadableCapacity,
      InvalidBuilderOperator::Kind::kSpinCapacity,
      InvalidBuilderOperator::Kind::kNotificationCapacity,
      InvalidBuilderOperator::Kind::kNotificationSenders,
      InvalidBuilderOperator::Kind::kDoorbellCapacity,
  };
  for (const InvalidBuilderOperator::Kind kind : invalid_kinds) {
    InvalidBuilderOperator op{kind};
    test::SyntheticOperatorFixture fixture{op};
    const auto result = fixture.declare_ports();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  }
}

// REQ: OPERATOR-014
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, ExplicitSamplePreservesSemanticMetadata) {
  //   fixture --[Sample<uint32>=9]--> ForwardingOperator --[Sample<uint32>]--> fixture:captured
  ForwardingOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const SampleMetadata metadata{
      .sequence = 101,
      .capture_timestamp_ns = 700,
      .source_clock_id = ClockId{.value = 3},
      .publish_timestamp_ns = 999,
      .publish_clock_id = ClockId{.value = 8},
      .trace_id = 12,
      .frame_id = 44,
      .byte_size = 1000,
      .payload_kind = PayloadKind::kSerializedBytes,
      .flags = SampleFlags::kDegraded,
  };
  ASSERT_TRUE(fixture.bind_sample(op.input_, Sample<std::uint32_t>{9, metadata}));
  ASSERT_TRUE(fixture.invoke_once());

  // Key assertion: capture/trace/frame/flags are forwarded; sequence is reset to 1 by the emit
  //               path and publish timestamps are cleared.
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 9);
  EXPECT_EQ(captured[0].metadata.sequence, 1);
  EXPECT_EQ(captured[0].metadata.capture_time(), metadata.capture_time());
  EXPECT_EQ(captured[0].metadata.publish_timestamp_ns, 0);
  EXPECT_FALSE(captured[0].metadata.publish_clock_id.valid());
  EXPECT_EQ(captured[0].metadata.trace_id, metadata.trace_id);
  EXPECT_EQ(captured[0].metadata.frame_id, metadata.frame_id);
  EXPECT_EQ(captured[0].metadata.byte_size, sizeof(std::uint32_t));
  EXPECT_EQ(captured[0].metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_EQ(captured[0].metadata.flags, metadata.flags);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, NoncontiguousBatchPreservesDeclaredOrder) {
  //   fixture --[batch of 3 non-contiguous refs]--> BatchSumOperator --[sum]--> fixture:captured
  BatchSumOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const std::array storage{
      Sample<std::uint32_t>{2, SampleMetadata{.sequence = 7}},
      Sample<std::uint32_t>{100, SampleMetadata{.sequence = 100}},
      Sample<std::uint32_t>{3, SampleMetadata{.sequence = 2}},
      Sample<std::uint32_t>{200, SampleMetadata{.sequence = 200}},
      Sample<std::uint32_t>{5, SampleMetadata{.sequence = 9}},
  };
  const std::array<const Sample<std::uint32_t>*, 3> selected{
      &storage[0],
      &storage[2],
      &storage[4],
  };
  ASSERT_TRUE(
      fixture.bind_batch_refs(op.input_, std::span<const Sample<std::uint32_t>* const>{selected}));
  ASSERT_TRUE(fixture.invoke_once());

  // Key assertion: observed sequences match selection order (not storage order) and sum is 10.
  EXPECT_EQ(op.observed_sequences, (std::array<std::uint64_t, 3>{7, 2, 9}));
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 10);
}

// REQ: OPERATOR-019
TEST(OperatorBindingConformance, BatchBindingAdmitsEmptyOneAndDeclaredMaximumOnly) {
  for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
    BatchViewProbeOperator op;
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    std::vector<Sample<std::uint32_t>> samples;
    samples.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      samples.emplace_back(static_cast<std::uint32_t>(index),
                           SampleMetadata{.sequence = index + 1});
    }
    ASSERT_TRUE(fixture.bind_batch(op.input_, std::span<const Sample<std::uint32_t>>{samples}));
    ASSERT_TRUE(fixture.invoke_once());
    if (!op.retained_batch_.has_value()) {
      ADD_FAILURE() << "compute did not retain the selected batch";
      continue;
    }
    EXPECT_EQ(op.retained_batch_->size(), count);
    EXPECT_EQ(op.retained_iterator_.has_value(), count != 0);
    for (std::size_t index = 0; index < count; ++index) {
      EXPECT_EQ(op.first_addresses_[index], op.second_addresses_[index]);
      EXPECT_EQ(op.first_sequences_[index], index + 1);
      EXPECT_EQ(op.second_sequences_[index], index + 1);
    }
  }

  BatchViewProbeOperator overflow_op;
  test::SyntheticOperatorFixture overflow_fixture{overflow_op};
  ASSERT_TRUE(overflow_fixture.declare_ports());
  const std::array overflow{
      Sample<std::uint32_t>{1, {}},
      Sample<std::uint32_t>{2, {}},
      Sample<std::uint32_t>{3, {}},
      Sample<std::uint32_t>{4, {}},
  };
  const auto overflow_result = overflow_fixture.bind_batch(
      overflow_op.input_, std::span<const Sample<std::uint32_t>>{overflow});
  ASSERT_FALSE(overflow_result);
  EXPECT_EQ(overflow_result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-020
TEST(OperatorBindingConformance, SelectedBatchRejectsMalformedGenerationWitness) {
  for (const bool omit_generation : {true, false}) {
    BatchFailureOperator op;
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_state(
        op.input_,
        detail::SyntheticInputState{
            .selector_shape = detail::SyntheticSelectorShape::kBatch,
            .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
            .advertises_value = true,
            .omit_batch_active_generation = omit_generation,
            .mismatch_batch_generation = !omit_generation,
        }));

    const auto result = fixture.invoke_once();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  }
}

#if GTEST_HAS_DEATH_TEST
// REQ: OPERATOR-019
// REQ: OPERATOR-024
TEST(OperatorBindingConformance, BatchBorrowAfterActivationRetirementTerminatesInEveryBuild) {
  BatchViewProbeOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  const std::array samples{Sample<std::uint32_t>{7, SampleMetadata{.sequence = 1}}};
  ASSERT_TRUE(fixture.bind_batch(op.input_, std::span<const Sample<std::uint32_t>>{samples}));
  ASSERT_TRUE(fixture.invoke_once());
  ASSERT_TRUE(op.retained_batch_.has_value());
  // The fatal assertions establish the optional witnesses. Keep their extraction outside the
  // death-test statement so the statement tests only activation-retirement enforcement.
  const auto* const retained_batch =
      &op.retained_batch_.value();  // NOLINT(bugprone-unchecked-optional-access)

  EXPECT_DEATH(
      {
        const auto value = retained_batch->front().data;
        (void)value;
      },
      "");
  ASSERT_TRUE(op.retained_iterator_.has_value());
  const auto* const retained_iterator =
      &op.retained_iterator_.value();  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_DEATH(
      {
        const auto value = (**retained_iterator).data;
        (void)value;
      },
      "");
}
#endif

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, ActivationContextIsOneShotAndBorrowed) {
  // Key setup: a full SyntheticActivationOptions is built with every non-default field set,
  //            including a named ManualClock, CUDA stream token, trace, and cancellation.
  ContextProbeOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  detail::ClockRegistryBuilder clock_builder;
  const ClockKey clock_key = ClockKey::named("fixture-clock");
  ASSERT_TRUE(clock_builder.add_clock(clock_key,
                                      std::make_shared<ManualClock>(123),
                                      detail::ClockCapabilityOptions{
                                          .can_read_now = true,
                                          .steady = true,
                                      }));
  ASSERT_TRUE(clock_builder.set_default(clock_key));
  auto registry_result = clock_builder.build();
  ASSERT_TRUE(registry_result);
  auto registry = std::move(*registry_result);
  auto clock_result = registry.resolve(ClockKey::default_clock());
  ASSERT_TRUE(clock_result);

  std::uint8_t stream_token{};
  auto* const stream = reinterpret_cast<cudaStream_t>(&stream_token);

  const detail::SyntheticActivationOptions options{
      .tag = Tag{.sequence = 8, .generation = 2},
      .tag_time = TagTime{.logical_ns = 90, .microstep = 4},
      .activation_time = TimePoint{.timestamp_ns = 1000, .clock_id = ClockId{.value = 5}},
      .clock = *clock_result,
      .cuda_stream = stream,
      .trace = TraceContext{.trace_id = 11, .span_id = 12, .parent_span_id = 13},
      .activation =
          ActivationStatus{ActivationReason::kDeadlineExceeded, OnViolation::kSafeDefault},
      .cancellation_requested = true,
  };
  ASSERT_TRUE(fixture.invoke_once(options));

  // Key assertion: all fields match the supplied options; a second default invocation resets them.
  EXPECT_EQ(op.callback_count, 1);
  EXPECT_EQ(op.observed_tag, options.tag);
  EXPECT_EQ(op.observed_tag_time, options.tag_time);
  EXPECT_EQ(op.observed_activation_time, options.activation_time);
  EXPECT_EQ(op.observed_clock_id, clock_result->id());
  EXPECT_EQ(op.observed_stream, stream);
  EXPECT_EQ(op.observed_trace, options.trace);
  EXPECT_EQ(op.observed_activation.reason, options.activation.reason);
  EXPECT_EQ(op.observed_activation.policy, options.activation.policy);
  EXPECT_TRUE(op.observed_stop);
  EXPECT_EQ(op.observed_params_generation, 0);

  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_EQ(op.callback_count, 2);
  EXPECT_EQ(op.observed_tag, (Tag{.sequence = 1, .generation = 1}));
  EXPECT_EQ(op.observed_tag_time, TagTime{});
  EXPECT_EQ(op.observed_activation_time, TimePoint{});
  EXPECT_EQ(op.observed_clock_id, ClockId{});
  EXPECT_EQ(op.observed_stream, cudaStream_t{});
  EXPECT_EQ(op.observed_trace, TraceContext{});
  EXPECT_EQ(op.observed_activation.reason, ActivationReason::kNominal);
  EXPECT_EQ(op.observed_activation.policy, OnViolation::kEmitWitness);
  EXPECT_FALSE(op.observed_stop);
  EXPECT_EQ(op.observed_params_generation, 0);
}

// REQ: OPERATOR-005
// REQ: OPERATOR-023
// REQ: PARAM-010
TEST(OperatorBindingConformance, ConfiguredParamsUsePositiveDefaultAndPermitExplicitInitialZero) {
  ParameterProbeOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_EQ(op.observed_generation, 1);
  EXPECT_EQ(op.observed_params, &op.params());

  const detail::SyntheticActivationOptions generation_seven{
      .params_generation = ParamGeneration{7},
  };
  ASSERT_TRUE(fixture.invoke_once(generation_seven));
  EXPECT_EQ(op.observed_generation, 7);
  EXPECT_EQ(op.observed_params, &op.params());

  // The synthetic fixture preserves its configured-parameter default of one. An explicit zero is
  // additionally accepted so HSDK5-004 can exercise the real generation-zero startup contract
  // before HSDK5-009 owns plan/session reconstruction.
  const detail::SyntheticActivationOptions initial_zero{
      .params_generation = ParamGeneration{0},
  };
  ASSERT_TRUE(fixture.invoke_once(initial_zero));
  EXPECT_EQ(op.observed_generation, 0);
}

// REQ: OPERATOR-002
// REQ: OPERATOR-005
// REQ: OPERATOR-023
TEST(OperatorBindingConformance, NoParamsRejectsNonzeroSyntheticGeneration) {
  ContextProbeOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  const detail::SyntheticActivationOptions invalid{
      .params_generation = ParamGeneration{1},
  };
  const auto result = fixture.invoke_once(invalid);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(op.callback_count, 0);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, CallbackFailureRetainsAlreadyStagedOutputForInspection) {
  EmitThenFailOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 7);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, BurstOverflowIsBoundedAndPreservesAcceptedPrefix) {
  BurstOverflowOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  const auto widened = fixture.bind_output(op.output_, 3);
  ASSERT_FALSE(widened);
  EXPECT_EQ(widened.error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 2);
  EXPECT_EQ(captured[0].data, 1);
  EXPECT_EQ(captured[1].data, 2);
}

// REQ: OPERATOR-016
TEST(OperatorBindingConformance, DefaultOutputBurstAdmitsExactlyOnePublication) {
  DefaultBurstOverflowOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 1);
}

// REQ: OPERATOR-023
// REQ: OPERATOR-016
TEST(OperatorBindingConformance, ExplicitOutputBurstOverrideMayOnlyTightenDeclaration) {
  BurstOverflowOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_, 1));

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  const auto captured = fixture.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 1);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, EmptyOutputIsAValidRepeatableActivationResult) {
  EmptyOutputOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_TRUE(fixture.captured(op.output_).empty());
  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_TRUE(fixture.captured(op.output_).empty());
  EXPECT_EQ(op.callback_count, 2);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, DoubleBindAndWrongTypeLookupAreRejectedWithoutRtti) {
  DoublingOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));

  const auto duplicate = fixture.bind_output(op.output_);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);

  Output<float> foreign_output{"foreign"};
  const auto wrong_port = fixture.bind_output(foreign_output);
  ASSERT_FALSE(wrong_port);
  EXPECT_EQ(wrong_port.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(fixture.captured_as<float>(op.output_).empty());
}

// REQ: OPERATOR-023
// REQ: OPERATOR-012
TEST(OperatorBindingConformance, UnboundInputFailurePropagatesWithoutCallbackRecovery) {
  ReceiveFailureOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInputUndeclared);
}

// REQ: OPERATOR-012
TEST(OperatorBindingConformance, InputGuardsUseOneCanonicalPrecedence) {
  // Key setup: each lambda invocation builds a SyntheticInputState with exactly one guard
  //            condition active to verify the strict stale > closed > not-selected > empty
  //            > backend-error precedence order.
  const auto invoke_scalar = [](detail::SyntheticInputState state) {
    ReceiveFailureOperator op;
    test::SyntheticOperatorFixture fixture{op};
    if (!fixture.declare_ports() || !fixture.bind_state(op.input_, state)) {
      return expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
    }
    return fixture.invoke_once();
  };

  auto stale = invoke_scalar(detail::SyntheticInputState{
      .selector_shape = detail::SyntheticSelectorShape::kScalar,
      .selection =
          InputSelection{
              .state = InputState::kNotSelected,
              .reason = SelectionReason::kNotSelectedForActivation,
          },
      .stale_generation = true,
      .closed = true,
  });
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);

  auto closed = invoke_scalar(detail::SyntheticInputState{
      .selector_shape = detail::SyntheticSelectorShape::kScalar,
      .selection =
          InputSelection{
              .state = InputState::kNotSelected,
              .reason = SelectionReason::kNotSelectedForActivation,
          },
      .closed = true,
  });
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ErrorCode::kClosed);

  auto not_selected = invoke_scalar(detail::SyntheticInputState{
      .selector_shape = detail::SyntheticSelectorShape::kScalar,
      .selection =
          InputSelection{
              .state = InputState::kNotSelected,
              .reason = SelectionReason::kNotSelectedForActivation,
          },
  });
  ASSERT_FALSE(not_selected);
  EXPECT_EQ(not_selected.error().code, ErrorCode::kInputNotSelected);

  auto empty = invoke_scalar(detail::SyntheticInputState{
      .selector_shape = detail::SyntheticSelectorShape::kScalar,
      .selection =
          InputSelection{
              .state = InputState::kAbsent,
              .reason = SelectionReason::kNeverSeen,
          },
  });
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, ErrorCode::kInputEmpty);

  auto backend = invoke_scalar(detail::SyntheticInputState{
      .selector_shape = detail::SyntheticSelectorShape::kScalar,
      .selection =
          InputSelection{
              .state = InputState::kSelected,
              .reason = SelectionReason::kNone,
          },
      .advertises_value = true,
      .backend_error = ErrorCode::kReceiveError,
  });
  // Key assertion: each state maps to its expected error code in declared precedence order.
  ASSERT_FALSE(backend);
  EXPECT_EQ(backend.error().code, ErrorCode::kReceiveError);
}

// REQ: OPERATOR-020
TEST(OperatorBindingConformance, ReceiveAndReceiveDataShareClassificationPrecedence) {
  const auto check = [](detail::SyntheticInputState state, ErrorCode expected_code) {
    DataReadOperator op;
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_state(op.input_, state));
    const auto sample = op.input_.receive();
    const auto data = op.input_.receive_data();
    ASSERT_FALSE(sample);
    ASSERT_FALSE(data);
    EXPECT_EQ(sample.error().code, expected_code);
    EXPECT_EQ(data.error().code, expected_code);
  };

  {
    DataReadOperator op;
    const auto sample = op.input_.receive();
    const auto data = op.input_.receive_data();
    ASSERT_FALSE(sample);
    ASSERT_FALSE(data);
    EXPECT_EQ(sample.error().code, ErrorCode::kInputUndeclared);
    EXPECT_EQ(data.error().code, ErrorCode::kInputUndeclared);
  }

  check(
      detail::SyntheticInputState{
          .selection =
              InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
          .stale_generation = true,
          .closed = true,
      },
      ErrorCode::kStaleEndpoint);
  check(
      detail::SyntheticInputState{
          .selection =
              InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
          .closed = true,
      },
      ErrorCode::kClosed);
  check(
      detail::SyntheticInputState{
          .selection =
              InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
      },
      ErrorCode::kInputNotSelected);
  check(
      detail::SyntheticInputState{
          .selector_shape = detail::SyntheticSelectorShape::kBatch,
          .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
          .advertises_value = true,
      },
      ErrorCode::kInputSelectionCardinalityMismatch);
  check(
      detail::SyntheticInputState{
          .selection = InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      },
      ErrorCode::kInputEmpty);
  check(
      detail::SyntheticInputState{
          .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
          .advertises_value = true,
          .backend_error = ErrorCode::kReceiveError,
      },
      ErrorCode::kReceiveError);
}

// REQ: OPERATOR-012
TEST(OperatorBindingConformance, SyntheticBindingsRejectContradictorySelectionStates) {
  ReceiveFailureOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  const auto missing_advertises_value = fixture.bind_state(
      op.input_,
      detail::SyntheticInputState{
          .selector_shape = detail::SyntheticSelectorShape::kScalar,
          .selection = InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
          .advertises_value = true,
      });
  ASSERT_FALSE(missing_advertises_value);
  EXPECT_EQ(missing_advertises_value.error().code, ErrorCode::kInvalidArgument);

  const auto fresh_has_failure_reason = fixture.bind_state(
      op.input_,
      detail::SyntheticInputState{
          .selector_shape = detail::SyntheticSelectorShape::kScalar,
          .selection = InputSelection{InputState::kSelected, SelectionReason::kTooStale},
          .advertises_value = true,
      });
  ASSERT_FALSE(fresh_has_failure_reason);
  EXPECT_EQ(fresh_has_failure_reason.error().code, ErrorCode::kInvalidArgument);

  const auto reused_has_no_failure_reason = fixture.bind_state(
      op.input_,
      detail::SyntheticInputState{
          .selector_shape = detail::SyntheticSelectorShape::kScalar,
          .selection = InputSelection{InputState::kReused, SelectionReason::kNone},
          .advertises_value = true,
      });
  ASSERT_FALSE(reused_has_no_failure_reason);
  EXPECT_EQ(reused_has_no_failure_reason.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-012
TEST(OperatorBindingConformance, MissingBatchDiffersFromSuccessfulEmptyBatch) {
  BatchFailureOperator missing_op;
  test::SyntheticOperatorFixture missing_fixture{missing_op};
  ASSERT_TRUE(missing_fixture.declare_ports());
  ASSERT_TRUE(missing_fixture.bind_state(
      missing_op.input_,
      detail::SyntheticInputState{
          .selector_shape = detail::SyntheticSelectorShape::kBatch,
          .selection = InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      }));
  const auto missing = missing_fixture.invoke_once();
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kInputEmpty);

  BatchFailureOperator empty_op;
  test::SyntheticOperatorFixture empty_fixture{empty_op};
  ASSERT_TRUE(empty_fixture.declare_ports());
  const std::array<Sample<std::uint32_t>, 0> empty_samples{};
  ASSERT_TRUE(empty_fixture.bind_batch(empty_op.input_,
                                       std::span<const Sample<std::uint32_t>>{empty_samples}));
  EXPECT_TRUE(empty_fixture.invoke_once());
}

// REQ: OPERATOR-012
TEST(OperatorBindingConformance, SelectorShapeNotRealizedCountControlsAccessor) {
  ReceiveFailureOperator scalar_accessor;
  test::SyntheticOperatorFixture scalar_fixture{scalar_accessor};
  ASSERT_TRUE(scalar_fixture.declare_ports());
  const std::array<Sample<std::uint32_t>, 0> empty_batch{};
  ASSERT_TRUE(scalar_fixture.bind_batch(scalar_accessor.input_,
                                        std::span<const Sample<std::uint32_t>>{empty_batch}));
  const auto scalar_result = scalar_fixture.invoke_once();
  ASSERT_FALSE(scalar_result);
  EXPECT_EQ(scalar_result.error().code, ErrorCode::kInputSelectionCardinalityMismatch);

  BatchFailureOperator batch_accessor;
  test::SyntheticOperatorFixture batch_fixture{batch_accessor};
  ASSERT_TRUE(batch_fixture.declare_ports());
  ASSERT_TRUE(batch_fixture.bind_value(batch_accessor.input_, std::uint32_t{7}));
  const auto batch_result = batch_fixture.invoke_once();
  ASSERT_FALSE(batch_result);
  EXPECT_EQ(batch_result.error().code, ErrorCode::kInputSelectionCardinalityMismatch);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, CompletedOutputMustBeTakenBeforeNextActivation) {
  //   fixture --[uint32=3 then 5]--> DoublingOperator --[6 then 10]--> fixture:captured
  DoublingOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_output(op.output_));
  ASSERT_TRUE(fixture.bind_value(op.input_, std::uint32_t{3}));
  ASSERT_TRUE(fixture.invoke_once());

  auto first = fixture.take_captured(op.output_);
  ASSERT_TRUE(first);
  ASSERT_EQ(first->size(), 1);
  EXPECT_EQ((*first)[0].data, 6);

  ASSERT_TRUE(fixture.bind_value(op.input_, std::uint32_t{5}));
  ASSERT_TRUE(fixture.invoke_once());
  const auto second = fixture.captured(op.output_);
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(second[0].data, 10);
  EXPECT_EQ(op.callback_count, 2);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, FixtureDestructionUnbindsOnlyItsExactPortSlots) {
  DoublingOperator op;
  {
    test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_value(op.input_, std::uint32_t{3}));
    ASSERT_TRUE(fixture.bind_output(op.output_));
  }

  const auto input_after_retirement = op.input_.receive();
  ASSERT_FALSE(input_after_retirement);
  EXPECT_EQ(input_after_retirement.error().code, ErrorCode::kInputUndeclared);
  EXPECT_EQ(op.input_.selection().state, InputState::kAbsent);

  const auto output_after_retirement = op.output_.emit(9);
  ASSERT_FALSE(output_after_retirement);
  EXPECT_EQ(output_after_retirement.error().code, ErrorCode::kNotReady);
}

// REQ: OPERATOR-023
TEST(OperatorBindingConformance, OlderFixtureDestructionPreservesNewerRebinding) {
  //   older fixture --[uint32=1]--> DoublingOperator (older destroyed)
  //   newer fixture --[uint32=11]--> DoublingOperator --[22]--> newer fixture:captured
  DoublingOperator op;
  auto older = std::make_unique<test::SyntheticOperatorFixture<DoublingOperator>>(op);
  ASSERT_TRUE(older->declare_ports());
  ASSERT_TRUE(older->bind_value(op.input_, std::uint32_t{1}));
  ASSERT_TRUE(older->bind_output(op.output_));

  test::SyntheticOperatorFixture newer{op};
  ASSERT_TRUE(newer.declare_ports());
  ASSERT_TRUE(newer.bind_value(op.input_, std::uint32_t{11}));
  ASSERT_TRUE(newer.bind_output(op.output_));
  older.reset();

  ASSERT_TRUE(newer.invoke_once());
  const auto captured = newer.captured(op.output_);
  ASSERT_EQ(captured.size(), 1);
  EXPECT_EQ(captured[0].data, 22);
}

}  // namespace
}  // namespace holoscan
