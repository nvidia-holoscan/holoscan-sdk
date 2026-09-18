/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include <holoscan/core/port.hpp>

#include "core/tensor_output_loan_internal.hpp"

namespace holoscan {

struct SingularTensorSchema {};
struct BatchTensorSchema {};

template <>
consteval SchemaIdentity schema_identity<SingularTensorSchema>() {
  return make_schema_identity(make_schema_family_identity("holoscan.test.SingularTensorSchema"),
                              make_flatbuffer_file_identifier("STSC"));
}

template <>
struct DescriptorTensorSchemaNativeTraits<SingularTensorSchema> {
  using root_type = SingularTensorSchema;
  static constexpr std::size_t metadata_capacity = 32U;
  static constexpr std::size_t tensor_count = 1U;
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{1U};

  [[nodiscard]] static const Tensor* tensor_at(const SingularTensorSchema&, std::size_t) noexcept {
    return nullptr;
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<BatchTensorSchema> {
  using root_type = BatchTensorSchema;
  static constexpr std::size_t metadata_capacity = 64U;
  static constexpr std::size_t tensor_count = 3U;
  static constexpr std::array<std::uint16_t, 3U> tensor_field_ids{1U, 3U, 7U};

  [[nodiscard]] static const Tensor* tensor_at(const BatchTensorSchema&, std::size_t) noexcept {
    return nullptr;
  }
};

namespace {

struct BatchTensorFields {
  static constexpr TensorFieldSelector<BatchTensorSchema> first{1U, 0U};
  static constexpr TensorFieldSelector<BatchTensorSchema> second{3U, 1U};
  static constexpr TensorFieldSelector<BatchTensorSchema> third{7U, 2U};
};

// Mock pool slot tracking the complete TensorOutputLoan state machine: live, writing,
// write_committed, cancel_pending, cancelled, abandoned, and published. All vtable callbacks
// below read/write this struct. Representative test: PrivateBindingConstructsWritesAndPublishes.
struct TensorLoanState {
  std::array<std::int64_t, 2> shape{2, 4};
  std::array<std::byte, 8> storage{};
  DLDataType dtype{.code = kDLUInt, .bits = 8, .lanes = 1};
  MemoryKind memory_kind{MemoryKind::kHost};
  bool live{};
  bool writing{};
  bool write_committed{};
  bool cancel_pending{};
  bool cancelled{};
  bool abandoned{};
  bool published{};
  std::uint32_t allocations{};
  std::uint32_t cuda_write_attempts{};
  std::uint32_t host_write_attempts{};
};

// Write-guard destructor callback. Clears the writing flag; if a cancel was deferred while the
// guard was active (cancel_pending), completes the cancellation rather than marking abandoned.
void abandon_write(void* state) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  loan.writing = false;
  if (loan.cancel_pending) {
    loan.live = false;
    loan.cancel_pending = false;
    loan.cancelled = true;
  } else if (loan.live) {
    loan.live = false;
    loan.abandoned = true;
  }
}

// TensorOutputWriteGuard vtable: data pointer, byte size, optional stream, and commit.
// commit_write enforces the deferred-cancel protocol: if cancel_pending was set, it finalizes
// cancellation and returns an error instead of marking write_committed.
void* write_data(const void* state) noexcept {
  return const_cast<std::byte*>(static_cast<const TensorLoanState*>(state)->storage.data());
}

std::size_t write_byte_size(const void* state) noexcept {
  return static_cast<const TensorLoanState*>(state)->storage.size();
}

std::optional<::cuda::stream_ref> write_stream(const void*) noexcept {
  return std::nullopt;
}

expected<void, Error> commit_write(void* state) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  if (loan.cancel_pending) {
    loan.live = false;
    loan.writing = false;
    loan.cancel_pending = false;
    loan.cancelled = true;
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!loan.live || !loan.writing) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.writing = false;
  loan.write_committed = true;
  return {};
}

// Vtable wiring the mock write-guard callbacks into a TensorOutputWriteGuardOps table.
constexpr detail::TensorOutputWriteGuardOps kWriteOps{
    .abandon = abandon_write,
    .data = write_data,
    .byte_size = write_byte_size,
    .stream = write_stream,
    .commit = commit_write,
};

// Loan destructor callback — invoked when a TensorOutputLoan is destroyed without being emitted
// or cancelled. Transitions the slot from live to abandoned so the pool can detect the leak.
void abandon_loan(void* state) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  if (loan.live) {
    loan.live = false;
    loan.abandoned = true;
  }
}

TensorShapeView loan_shape(const void* state) noexcept {
  return static_cast<const TensorLoanState*>(state)->shape;
}

TensorByteStridesView loan_byte_strides(const void*) noexcept {
  return std::nullopt;
}

DLDataType loan_dtype(const void* state) noexcept {
  return static_cast<const TensorLoanState*>(state)->dtype;
}

DeviceId loan_device_id(const void*) noexcept {
  return DeviceId{};
}

MemoryKind loan_memory_kind(const void* state) noexcept {
  return static_cast<const TensorLoanState*>(state)->memory_kind;
}

std::size_t loan_byte_size(const void* state) noexcept {
  return static_cast<const TensorLoanState*>(state)->storage.size();
}

// begin_cuda_write always returns kNotSupported — the mock pool has no device memory.
// begin_host_write transitions the live slot to writing and vends a guard backed by kWriteOps.
expected<TensorOutputWriteGuard, Error> begin_cuda_write(void* state, ::cuda::stream_ref) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  ++loan.cuda_write_attempts;
  return make_unexpected(Error{ErrorCode::kNotSupported});
}

expected<TensorOutputWriteGuard, Error> begin_host_write(void* state) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  ++loan.host_write_attempts;
  if (loan.memory_kind == MemoryKind::kCudaDevice) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (!loan.live || loan.writing || loan.write_committed) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.writing = true;
  return detail::TensorOutputLoanAccess::write_guard(state, &kWriteOps);
}

// Cancellation callback. If a write guard is outstanding, sets cancel_pending instead of retiring
// the slot immediately; the guard's commit or abandon path will complete the cancellation later.
expected<void, Error> cancel_loan(void* state) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  if (!loan.live) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (loan.writing) {
    // The facade is consumed even though cancellation cannot synchronously retire an active
    // writer. Tombstone the slot until that sole guard commits or is destroyed, preventing both
    // a finite-slot leak and premature reuse beneath the outstanding guard.
    loan.cancel_pending = true;
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.live = false;
  loan.cancelled = true;
  return {};
}

// Vtable wiring all loan metadata and write callbacks into a TensorOutputLoanOps table.
constexpr detail::TensorOutputLoanOps kLoanOps{
    .abandon = abandon_loan,
    .shape = loan_shape,
    .byte_strides = loan_byte_strides,
    .dtype = loan_dtype,
    .device_id = loan_device_id,
    .memory_kind = loan_memory_kind,
    .byte_size = loan_byte_size,
    .write = begin_cuda_write,
    .write_host = begin_host_write,
    .cancel = cancel_loan,
};

TensorSchemaOutputLoan<BatchTensorSchema> make_batch_loan(std::array<TensorLoanState, 3U>& states) {
  detail::TensorSchemaLoanBundleStorage storage;
  storage.field_count = static_cast<std::uint32_t>(states.size());
  for (std::size_t index = 0U; index < states.size(); ++index) {
    states[index].live = true;
    storage.field_ids[index] =
        DescriptorTensorSchemaNativeTraits<BatchTensorSchema>::tensor_field_ids[index];
    storage.fields[index].emplace(detail::TensorOutputLoanAccess::loan(&states[index], &kLoanOps));
  }
  return detail::TensorSchemaOutputLoanAccess::make<BatchTensorSchema>(std::move(storage));
}

// Pool allocation callback. Rejects if the slot is already live or the requested shape/dtype
// does not match; otherwise transitions the slot to live and increments the allocation count.
expected<TensorOutputLoan, Error> allocate_tensor(void* state, TensorLoanRequest request) noexcept {
  auto& loan = *static_cast<TensorLoanState*>(state);
  if (loan.live) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  if (request.shape.size() != loan.shape.size() || request.shape[0] != loan.shape[0] ||
      request.shape[1] != loan.shape[1] || request.dtype.code != loan.dtype.code ||
      request.dtype.bits != loan.dtype.bits || request.dtype.lanes != loan.dtype.lanes ||
      request.byte_strides.has_value()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  loan.live = true;
  loan.writing = false;
  loan.write_committed = false;
  loan.cancel_pending = false;
  loan.cancelled = false;
  loan.abandoned = false;
  loan.published = false;
  loan.cuda_write_attempts = 0U;
  loan.host_write_attempts = 0U;
  ++loan.allocations;
  return detail::TensorOutputLoanAccess::loan(state, &kLoanOps);
}

// Pool emit callback. Validates binding, liveness, and that a write was committed; then marks
// the slot published and consumes the loan facade to model the private Runtime handoff.
expected<void, Error> emit_tensor_loan(void* state, TensorOutputLoan loan, const void* descriptor,
                                       EmitOptions) noexcept {
  auto& allocation = *static_cast<TensorLoanState*>(state);
  if (descriptor != nullptr || !detail::TensorOutputLoanAccess::bound_to(loan, state) ||
      !allocation.live || allocation.writing || !allocation.write_committed) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  allocation.live = false;
  allocation.published = true;
  // The consumed loan's destructor observes the terminal state and therefore performs no second
  // retirement. This models the private Runtime handoff without exposing a Tensor extraction API.
  (void)loan;
  return {};
}

// Vtable wiring allocate and emit into a TensorOutputBindingOps table consumed by Output<Tensor>.
constexpr detail::TensorOutputBindingOps kOutputOps{
    .allocate = allocate_tensor,
    .emit_loan = emit_tensor_loan,
};

// Builds a TensorLoanRequest whose shape and dtype match the given state, with no explicit byte
// strides (contiguous layout implied). Used to drive allocate_tensor in all tests.
TensorLoanRequest request_for(const TensorLoanState& state) {
  return TensorLoanRequest{
      .shape = state.shape,
      .dtype = state.dtype,
      .byte_strides = std::nullopt,
  };
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, UnboundOutputRemainsDisabled) {
  Output<Tensor> output{"tensor"};
  const TensorLoanState state;

  auto result = output.allocate_tensor(request_for(state));

  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(state.allocations, 0);
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, PrivateBindingConstructsWritesAndPublishesOneBoundedLoan) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  // Key setup: private binding wired; allocate verifies that all metadata accessors are correct
  //            before any write is attempted.
  auto loan_result = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(loan_result);
  TensorOutputLoan loan = std::move(*loan_result);
  EXPECT_EQ(loan.shape().size(), 2);
  EXPECT_EQ(loan.shape()[0], 2);
  EXPECT_EQ(loan.shape()[1], 4);
  EXPECT_EQ(loan.dtype().code, kDLUInt);
  EXPECT_EQ(loan.memory_kind(), MemoryKind::kHost);
  EXPECT_EQ(loan.device_id(), DeviceId{});
  EXPECT_EQ(loan.byte_size(), state.storage.size());
  EXPECT_FALSE(loan.byte_strides().has_value());

  auto guard_result = loan.write_host();
  ASSERT_TRUE(guard_result);
  TensorOutputWriteGuard guard = std::move(*guard_result);
  ASSERT_NE(guard.data(), nullptr);
  EXPECT_EQ(guard.byte_size(), state.storage.size());
  EXPECT_FALSE(guard.stream().has_value());
  guard.data_as<std::uint8_t>()[3] = 0x5A;
  ASSERT_TRUE(std::move(guard).commit());

  // Key assertion: byte written through the guard survives in state.storage after successful emit.
  EXPECT_TRUE(output.emit(std::move(loan)));
  EXPECT_TRUE(state.published);
  EXPECT_FALSE(state.live);
  EXPECT_FALSE(state.abandoned);
  EXPECT_EQ(std::to_integer<std::uint8_t>(state.storage[3]), 0x5A);
}

// REQ: OPERATOR-040
TEST(TensorOutputLoan, ShapeOnlyAllocationUsesBoundExactDtype) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  // Key setup: bind the runtime allocator with the exact dtype projected from the output spec.
  detail::TensorOutputPortBindingAccess::bind(
      output, {.state = &state, .ops = &kOutputOps, .tensor_dtype = state.dtype});

  // Key assertion: shapes-only allocation supplies that dtype to the unchanged allocator callback.
  const auto allocated = output.allocate_tensor(state.shape);
  ASSERT_TRUE(allocated) << allocated.error().message;
  EXPECT_EQ(allocated->dtype().code, state.dtype.code);
  EXPECT_EQ(allocated->dtype().bits, state.dtype.bits);
  EXPECT_EQ(state.allocations, 1U);
}

// REQ: OPERATOR-040
TEST(TensorOutputLoan, ShapeOnlyAllocationRejectsUnresolvedDtype) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  // Key setup: bind an otherwise valid output allocator without an exact contract dtype.
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  // Key assertion: shapes-only allocation fails before calling the allocator or guessing dtype.
  const auto rejected = output.allocate_tensor(state.shape);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rejected.error().message, "TENSOR_OUTPUT_DTYPE_UNRESOLVED");
  EXPECT_EQ(state.allocations, 0U);
}

// REQ: OPERATOR-040
TEST(TensorOutputLoan, SingularDescriptorShapeOnlyAllocationUsesBoundExactDtype) {
  Output<SingularTensorSchema> output{"image"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(
      output, {.state = &state, .ops = &kOutputOps, .tensor_dtype = state.dtype});

  const auto allocated = output.allocate_tensor(state.shape);

  ASSERT_TRUE(allocated) << allocated.error().message;
  EXPECT_EQ(allocated->dtype().code, state.dtype.code);
  EXPECT_EQ(allocated->dtype().bits, state.dtype.bits);
  EXPECT_EQ(allocated->dtype().lanes, state.dtype.lanes);
  EXPECT_EQ(state.allocations, 1U);
}

// REQ: OPERATOR-040
TEST(TensorOutputLoan, SingularDescriptorShapeOnlyAllocationRejectsUnresolvedDtype) {
  Output<SingularTensorSchema> output{"image"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  const auto rejected = output.allocate_tensor(state.shape);

  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rejected.error().message, "TENSOR_OUTPUT_DTYPE_UNRESOLVED");
  EXPECT_EQ(state.allocations, 0U);
}

// The fused helpers preserve the checked acquisition/fill/commit state machine while removing
// the application-side guard plumbing from the common single-fill path.
// REQ: OPERATOR-027
TEST(TensorOutputLoan, FusedWriteAndCommitPreservesCheckedOutcomes) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  auto loan = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(loan);
  // Key setup: CUDA write path returns kNotSupported; fill callback must not be invoked.
  std::uint8_t stream_token{};
  auto* const stream = reinterpret_cast<cudaStream_t>(&stream_token);
  std::uint32_t fill_count = 0U;
  auto unsupported = loan->write_and_commit(
      ::cuda::stream_ref{stream}, [&fill_count](const TensorOutputWriteGuard&) { ++fill_count; });
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(fill_count, 0U);

  // Key setup: fused write_host_and_commit acquires the guard, invokes the fill callback, and
  //            commits atomically — the byte written inside the lambda must appear in storage.
  auto committed = loan->write_host_and_commit([&fill_count](const TensorOutputWriteGuard& writer) {
    ++fill_count;
    writer.data_as<std::uint8_t>()[3] = 0x6B;
  });
  ASSERT_TRUE(committed);
  EXPECT_EQ(fill_count, 1U);
  EXPECT_TRUE(output.emit(std::move(*loan)));
  EXPECT_EQ(std::to_integer<std::uint8_t>(state.storage[3]), 0x6B);

  {
    // Key assertion: when the fill callback returns an error, the guard is abandoned automatically
    //                and the slot transitions to abandoned rather than committed.
    auto rejected_loan = output.allocate_tensor(request_for(state));
    ASSERT_TRUE(rejected_loan);
    auto rejected = rejected_loan->write_host_and_commit(
        [&fill_count](const TensorOutputWriteGuard&) -> expected<void, Error> {
          ++fill_count;
          return make_unexpected(Error{ErrorCode::kFailure});
        });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kFailure);
    EXPECT_EQ(fill_count, 2U);
    EXPECT_FALSE(state.live);
    EXPECT_TRUE(state.abandoned);
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  {
    auto throwing_loan = output.allocate_tensor(request_for(state));
    ASSERT_TRUE(throwing_loan);
    EXPECT_THROW(
        static_cast<void>(throwing_loan->write_host_and_commit(
            [](const TensorOutputWriteGuard&) { throw std::runtime_error{"fill failed"}; })),
        std::runtime_error);
    EXPECT_FALSE(state.live);
    EXPECT_TRUE(state.abandoned);
  }
#endif

  auto replacement = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(std::move(*replacement).cancel());
}

TEST(TensorSchemaOutputLoan, BatchWritePreflightsSelectorsBeforeCallbacks) {
  std::array<TensorLoanState, 3U> states;
  auto bundle = make_batch_loan(states);
  std::uint32_t callback_count = 0U;

  // Key setup: submit the static field count while duplicating one selector value.
  auto duplicate =
      bundle.write_all(host_field(BatchTensorFields::first,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::first,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::third,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }));

  // Key assertion: no callback or write phase starts before the complete selector preflight.
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(duplicate.error().message,
            "TENSOR_SCHEMA_BATCH_WRITE_DUPLICATE_FIELD (field_id=1, tensor_index=0)");
  EXPECT_EQ(callback_count, 0U);
  EXPECT_TRUE(std::ranges::all_of(states, [](const auto& state) {
    return state.live && !state.writing && !state.write_committed;
  }));

  auto written =
      bundle.write_all(host_field(BatchTensorFields::first,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::second,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::third,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }));
  EXPECT_TRUE(written);
  EXPECT_EQ(callback_count, 3U);
  EXPECT_TRUE(std::ranges::all_of(
      states, [](const auto& state) { return state.write_committed && !state.writing; }));
}

TEST(TensorSchemaOutputLoan, BatchWriteReportsInvalidSelectorsBeforeCallbacks) {
  std::array<TensorLoanState, 3U> states;
  auto bundle = make_batch_loan(states);
  std::uint32_t callback_count = 0U;

  // Key setup: use the correct schema type but an unknown field identity at a valid Tensor index.
  const TensorFieldSelector<BatchTensorSchema> unknown{9U, 0U};
  auto rejected = bundle.write_all(
      host_field(unknown, [&](const TensorOutputWriteGuard&) { ++callback_count; }),
      host_field(BatchTensorFields::second,
                 [&](const TensorOutputWriteGuard&) { ++callback_count; }),
      host_field(BatchTensorFields::third,
                 [&](const TensorOutputWriteGuard&) { ++callback_count; }));

  // Key assertion: the field-qualified diagnostic is stable and the reusable bundle is untouched.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rejected.error().message,
            "TENSOR_SCHEMA_BATCH_WRITE_FIELD_SELECTOR_INVALID (field_id=9, tensor_index=0)");
  EXPECT_EQ(callback_count, 0U);
  EXPECT_TRUE(std::ranges::all_of(states, [](const auto& state) {
    return state.live && !state.writing && !state.write_committed;
  }));
}

TEST(TensorSchemaOutputLoan, BatchWriteDelegatesPlacementChecksToLoanRuntime) {
  std::uint8_t stream_token{};
  auto* const stream = reinterpret_cast<cudaStream_t>(&stream_token);

  {
    std::array<TensorLoanState, 3U> states;
    auto bundle = make_batch_loan(states);
    std::uint32_t callback_count = 0U;

    // Key setup: request a CUDA write for a host field; the public bundle has no placement rule.
    auto rejected =
        bundle.write_all(device_field(BatchTensorFields::first,
                                      ::cuda::stream_ref{stream},
                                      [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                         host_field(BatchTensorFields::second,
                                    [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                         host_field(BatchTensorFields::third,
                                    [&](const TensorOutputWriteGuard&) { ++callback_count; }));

    // Key assertion: acquisition reaches the authoritative loan Runtime and preserves its code.
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kNotSupported);
    EXPECT_EQ(rejected.error().message,
              "TENSOR_SCHEMA_BATCH_WRITE_FIELD_ACQUIRE_FAILED (field_id=1, tensor_index=0)");
    EXPECT_EQ(states[0].cuda_write_attempts, 1U);
    EXPECT_EQ(callback_count, 0U);
  }

  {
    std::array<TensorLoanState, 3U> states;
    states[0].memory_kind = MemoryKind::kCudaDevice;
    auto bundle = make_batch_loan(states);
    std::uint32_t callback_count = 0U;

    // Key setup: request host access to a CUDA-device field.
    auto rejected =
        bundle.write_all(host_field(BatchTensorFields::first,
                                    [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                         host_field(BatchTensorFields::second,
                                    [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                         host_field(BatchTensorFields::third,
                                    [&](const TensorOutputWriteGuard&) { ++callback_count; }));

    // Key assertion: the runtime-owned host-access check rejects before any callback runs.
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kNotSupported);
    EXPECT_EQ(rejected.error().message,
              "TENSOR_SCHEMA_BATCH_WRITE_FIELD_ACQUIRE_FAILED (field_id=1, tensor_index=0)");
    EXPECT_EQ(states[0].host_write_attempts, 1U);
    EXPECT_EQ(callback_count, 0U);
  }
}

TEST(TensorSchemaOutputLoan, BatchWriteRejectsMovedFromBundleWithStableDiagnostic) {
  std::array<TensorLoanState, 3U> states;
  auto bundle = make_batch_loan(states);
  auto live_bundle = std::move(bundle);
  std::uint32_t callback_count = 0U;

  // Key setup: invoke the valid but empty moved-from bundle with a complete static callback set.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  auto rejected =
      bundle.write_all(host_field(BatchTensorFields::first,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::second,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }),
                       host_field(BatchTensorFields::third,
                                  [&](const TensorOutputWriteGuard&) { ++callback_count; }));

  // Key assertion: no callback runs and the live destination retains all field authority.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(rejected.error().message, "TENSOR_SCHEMA_BATCH_WRITE_BUNDLE_INVALID");
  EXPECT_EQ(callback_count, 0U);
  EXPECT_EQ(live_bundle.size(), 3U);
  EXPECT_TRUE(std::ranges::all_of(states, [](const auto& state) { return state.live; }));
}

TEST(TensorSchemaOutputLoan, BatchWriteStopsAfterFirstCallbackFailure) {
  std::array<TensorLoanState, 3U> states;
  auto bundle = make_batch_loan(states);
  std::array<std::uint32_t, 3U> callback_counts{};

  // Key setup: fail the middle callback after every field write phase has been preflighted.
  auto written =
      bundle.write_all(host_field(BatchTensorFields::first,
                                  [&](const TensorOutputWriteGuard&) { ++callback_counts[0]; }),
                       host_field(BatchTensorFields::second,
                                  [&](const TensorOutputWriteGuard&) -> expected<void, Error> {
                                    ++callback_counts[1];
                                    return make_unexpected(Error{ErrorCode::kFailure});
                                  }),
                       host_field(BatchTensorFields::third,
                                  [&](const TensorOutputWriteGuard&) { ++callback_counts[2]; }));

  // Key assertion: the first write is committed, the later callback is skipped, and every active
  // uncommitted field is abandoned without publication.
  ASSERT_FALSE(written);
  EXPECT_EQ(written.error().code, ErrorCode::kFailure);
  EXPECT_EQ(callback_counts, (std::array<std::uint32_t, 3U>{1U, 1U, 0U}));
  EXPECT_TRUE(states[0].write_committed);
  EXPECT_TRUE(states[1].abandoned);
  EXPECT_TRUE(states[2].abandoned);
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, CancellationAndDestructionRetirePrivateAuthority) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  {
    auto loan = output.allocate_tensor(request_for(state));
    ASSERT_TRUE(loan);
    EXPECT_TRUE(std::move(*loan).cancel());
    EXPECT_TRUE(state.cancelled);
  }

  {
    auto loan = output.allocate_tensor(request_for(state));
    ASSERT_TRUE(loan);
  }
  EXPECT_TRUE(state.abandoned);
  EXPECT_FALSE(state.live);
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, CancellationDuringWriteRetiresAfterTheWriteGuard) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  auto loan_result = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(loan_result);
  TensorOutputLoan loan = std::move(*loan_result);
  auto guard_result = loan.write_host();
  ASSERT_TRUE(guard_result);
  TensorOutputWriteGuard guard = std::move(*guard_result);

  // Key setup: cancel() while write guard is outstanding; slot stays live with cancel_pending set
  //            rather than being retired immediately.
  const auto cancellation = std::move(loan).cancel();
  ASSERT_FALSE(cancellation);
  EXPECT_EQ(cancellation.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(state.live);
  EXPECT_TRUE(state.cancel_pending);

  const auto unavailable = output.allocate_tensor(request_for(state));
  ASSERT_FALSE(unavailable);
  EXPECT_EQ(unavailable.error().code, ErrorCode::kResourceExhausted);

  // Key assertion: committing the guard with cancel_pending finalizes cancellation and returns an
  //                error; the slot is retired as cancelled, not as committed or abandoned.
  const auto stale_commit = std::move(guard).commit();
  ASSERT_FALSE(stale_commit);
  EXPECT_EQ(stale_commit.error().code, ErrorCode::kInvalidArgument);
  EXPECT_FALSE(state.live);
  EXPECT_FALSE(state.cancel_pending);
  EXPECT_TRUE(state.cancelled);

  auto replacement = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(std::move(*replacement).cancel());
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, InvalidWriteTransitionsAndMissingCudaCapabilityAreDistinct) {
  Output<Tensor> output{"tensor"};
  TensorLoanState state;
  detail::TensorOutputPortBindingAccess::bind(output, {.state = &state, .ops = &kOutputOps});

  auto loan_result = output.allocate_tensor(request_for(state));
  ASSERT_TRUE(loan_result);
  TensorOutputLoan loan = std::move(*loan_result);

  std::uint8_t stream_token{};
  auto* const stream = reinterpret_cast<cudaStream_t>(&stream_token);
  const auto unsupported = loan.write(::cuda::stream_ref{stream});
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().code, ErrorCode::kNotSupported);

  auto guard_result = loan.write_host();
  ASSERT_TRUE(guard_result);
  const auto concurrent = loan.write_host();
  ASSERT_FALSE(concurrent);
  EXPECT_EQ(concurrent.error().code, ErrorCode::kInvalidArgument);

  TensorOutputWriteGuard guard = std::move(*guard_result);
  ASSERT_TRUE(std::move(guard).commit());
  const auto after_commit = loan.write_host();
  ASSERT_FALSE(after_commit);
  EXPECT_EQ(after_commit.error().code, ErrorCode::kInvalidArgument);
  // An rvalue-qualified commit consumes the guard's protocol state without moving the C++ object;
  // the deliberate second terminal call validates that consumed state.
  const auto second_commit = std::move(guard).commit();  // NOLINT(bugprone-use-after-move)
  ASSERT_FALSE(second_commit);
  EXPECT_EQ(second_commit.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-010
// REQ: OPERATOR-026
TEST(TensorOutputLoan, WrongOutputPublicationRejectsAndConsumesTheLoan) {
  Output<Tensor> first{"first"};
  Output<Tensor> second{"second"};
  TensorLoanState first_state;
  TensorLoanState second_state;
  detail::TensorOutputPortBindingAccess::bind(first, {.state = &first_state, .ops = &kOutputOps});
  detail::TensorOutputPortBindingAccess::bind(second, {.state = &second_state, .ops = &kOutputOps});

  auto loan_result = first.allocate_tensor(request_for(first_state));
  ASSERT_TRUE(loan_result);
  TensorOutputLoan loan = std::move(*loan_result);
  auto guard_result = loan.write_host();
  ASSERT_TRUE(guard_result);
  ASSERT_TRUE(std::move(*guard_result).commit());

  const auto wrong_output = second.emit(std::move(loan));
  ASSERT_FALSE(wrong_output);
  EXPECT_EQ(wrong_output.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(first_state.abandoned);
  EXPECT_FALSE(first_state.live);
  EXPECT_FALSE(second_state.published);
}

}  // namespace
}  // namespace holoscan
