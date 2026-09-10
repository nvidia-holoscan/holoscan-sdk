/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/sample.hpp>

namespace holoscan {

// POD type whose default constructor is not noexcept. Verifies that Output<T>::allocate() is
// suppressed (no loan slot can be constructed in-place without risk of throwing) while the
// Sample<T> emit path remains available.
// Representative test: RuntimeGenericLoanCapabilityMatchesPublicOverloadParticipation.
struct ThrowingDefaultLoanPod {
  ThrowingDefaultLoanPod() noexcept(false) {}

  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<ThrowingDefaultLoanPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.ThrowingDefaultLoanPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ThrowingDefaultLoanPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &ThrowingDefaultLoanPod::value, offsetof(ThrowingDefaultLoanPod, value)));
};

// POD type with deleted copy- and move-assignment operators. Verifies that Output<T>::allocate()
// is suppressed (in-place slot assignment is impossible) while the Sample<T> emit path still works.
// Representative test: RuntimeGenericLoanCapabilityMatchesPublicOverloadParticipation.
struct NonAssignableLoanPod {
  NonAssignableLoanPod() = default;
  NonAssignableLoanPod(const NonAssignableLoanPod&) = default;
  NonAssignableLoanPod(NonAssignableLoanPod&&) = default;
  NonAssignableLoanPod& operator=(const NonAssignableLoanPod&) = delete;
  NonAssignableLoanPod& operator=(NonAssignableLoanPod&&) = delete;

  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<NonAssignableLoanPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.NonAssignableLoanPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<NonAssignableLoanPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &NonAssignableLoanPod::value, offsetof(NonAssignableLoanPod, value)));
};

namespace {

template <typename Port>
concept HasLoad = requires(Port& port) { port.load(std::uint32_t{1}); };

template <typename Port>
concept HasRetrieve = requires(Port& port) { port.retrieve(); };

template <typename Port>
concept HasDataMember = requires(const Port& port) { port.data(); };

template <typename Port>
concept HasAllocate = requires(Port& port) { port.allocate(); };

template <typename Port>
concept HasAllocateTensor =
    requires(Port& port, TensorLoanRequest request) { port.allocate_tensor(request); };

template <typename Port>
concept HasAcquireTensor =
    requires(Port& port, TensorLoanRequest request) { port.acquire_tensor(request); };

template <typename Loan, typename Guard>
concept HasFusedTensorWrite = requires(Loan& loan, ::cuda::stream_ref stream) {
  {
    loan.write_and_commit(stream, [](const Guard&) {})
  } -> std::same_as<expected<void, Error>>;
  {
    loan.write_host_and_commit([](const Guard&) -> expected<void, Error> { return {}; })
  } -> std::same_as<expected<void, Error>>;
};

template <typename Port, typename SampleT>
concept HasSampleEmit = requires(Port& port, SampleT sample) { port.emit(std::move(sample)); };

template <typename Port, typename SampleT>
concept HasEmitSampleVerb =
    requires(Port& port, SampleT sample) { port.emit_sample(std::move(sample)); };

template <typename Port, typename SampleT>
concept HasSampleEmitOptions =
    requires(Port& port, SampleT sample) { port.emit(std::move(sample), EmitOptions{}); };

template <typename Loan>
concept HasEmplace = requires(Loan& loan) { loan.emplace(std::uint32_t{1}); };

template <typename Loan>
concept HasOwningTensorExtraction = requires(Loan& loan) {
  loan.tensor();
  *loan;
};

using ScalarInput = Input<std::uint32_t>;
using ScalarOutput = Output<std::uint32_t>;
using ScalarSample = Sample<std::uint32_t>;
using ScalarBatch = SelectedBatchView<std::uint32_t>;
using ScalarLoan = LoanedSample<std::uint32_t>;
using ScalarValueEmit = expected<void, Error> (ScalarOutput::*)(std::uint32_t, EmitOptions);
using ScalarSampleEmit = expected<void, Error> (ScalarOutput::*)(ScalarSample);
using ScalarLoanEmit = expected<void, Error> (ScalarOutput::*)(ScalarLoan&&, EmitOptions) noexcept;
using TensorLoanEmit = expected<void, Error> (Output<Tensor>::*)(TensorOutputLoan, EmitOptions);
using TensorValueEmit = expected<void, Error> (Output<Tensor>::*)(Tensor, EmitOptions);
using TensorSampleEmit = expected<void, Error> (Output<Tensor>::*)(Sample<Tensor>);

// Minimal visitor callable accepted by Input<T>::for_each_selected(); exercises the basic
// const-Sample& overload used in the static_assert checks below.
struct ScalarVisitor {
  expected<void, Error> operator()(const ScalarSample&) const { return {}; }
};

// Visitor with a deleted rvalue overload; confirms that for_each_selected() only passes lvalue
// references to the visitor and never moves from or consumes it.
struct LvalueOnlyScalarVisitor {
  expected<void, Error> operator()(const ScalarSample&) & { return {}; }
  expected<void, Error> operator()(const ScalarSample&) && = delete;
};

static_assert(std::default_initializable<ScalarInput>);
static_assert(std::constructible_from<ScalarInput, std::string_view>);
static_assert(std::default_initializable<ScalarOutput>);
static_assert(std::constructible_from<ScalarOutput, std::string_view>);
static_assert(!std::copy_constructible<ScalarInput>);
static_assert(!std::move_constructible<ScalarInput>);
static_assert(!std::copy_constructible<ScalarOutput>);
static_assert(!std::move_constructible<ScalarOutput>);

static_assert(std::same_as<decltype(std::declval<ScalarInput&>().selection()), InputSelection>);
static_assert(noexcept(std::declval<const ScalarInput&>().selection()));
static_assert(
    std::same_as<decltype(std::declval<ScalarInput&>().receive()), expected<ScalarSample, Error>>);
static_assert(std::same_as<decltype(std::declval<ScalarInput&>().receive_data()),
                           expected<std::uint32_t, Error>>);
static_assert(std::same_as<decltype(std::declval<const ScalarInput&>().selected_batch()),
                           expected<ScalarBatch, Error>>);
static_assert(noexcept(std::declval<const ScalarInput&>().selected_batch()));
static_assert(
    std::same_as<decltype(std::declval<const ScalarInput&>().for_each_selected(ScalarVisitor{})),
                 expected<void, Error>>);
static_assert(std::same_as<decltype(std::declval<const ScalarInput&>().for_each_selected(
                               LvalueOnlyScalarVisitor{})),
                           expected<void, Error>>);

static_assert(std::ranges::random_access_range<ScalarBatch>);
static_assert(!std::ranges::contiguous_range<ScalarBatch>);
static_assert(!std::ranges::borrowed_range<ScalarBatch>);
static_assert(std::same_as<std::ranges::range_reference_t<ScalarBatch>, const ScalarSample&>);
static_assert(!HasDataMember<ScalarBatch>);

// REQ: OPERATOR-009
static_assert(std::is_nothrow_move_constructible_v<ScalarLoan>);
static_assert(!std::is_copy_constructible_v<ScalarLoan>);
static_assert(!std::is_copy_assignable_v<ScalarLoan>);
static_assert(!std::is_move_assignable_v<ScalarLoan>);
static_assert(std::is_nothrow_destructible_v<ScalarLoan>);
static_assert(std::same_as<decltype(std::declval<ScalarLoan&>().value()), std::uint32_t&>);
static_assert(std::same_as<decltype(std::declval<ScalarLoan&>().operator->()), std::uint32_t*>);
static_assert(std::same_as<decltype(*std::declval<ScalarLoan&>()), std::uint32_t&>);
static_assert(noexcept(std::declval<ScalarLoan&>().cancel()));
static_assert(!HasEmplace<ScalarLoan>);  // Deliberately retained for post-baseline retrieval.

// REQ: OPERATOR-014
// REQ: OPERATOR-021
static_assert(HasSampleEmit<ScalarOutput, ScalarSample>);
static_assert(!HasEmitSampleVerb<ScalarOutput, ScalarSample>);
static_assert(!HasSampleEmitOptions<ScalarOutput, ScalarSample>);
static_assert(
    std::same_as<decltype(static_cast<ScalarValueEmit>(&ScalarOutput::emit)), ScalarValueEmit>);
static_assert(
    std::same_as<decltype(static_cast<ScalarSampleEmit>(&ScalarOutput::emit)), ScalarSampleEmit>);
static_assert(
    std::same_as<decltype(std::declval<ScalarOutput&>().allocate()), expected<ScalarLoan, Error>>);
static_assert(noexcept(std::declval<ScalarOutput&>().allocate()));
static_assert(
    std::same_as<decltype(static_cast<ScalarLoanEmit>(&ScalarOutput::emit)), ScalarLoanEmit>);
static_assert(!noexcept(std::declval<ScalarOutput&>().emit(std::declval<std::uint32_t>())));
static_assert(!noexcept(std::declval<ScalarOutput&>().emit(std::declval<ScalarSample>())));
static_assert(noexcept(std::declval<ScalarOutput&>().emit(std::declval<ScalarLoan&&>(),
                                                          EmitOptions{})));

// The public overload and Runtime pool inventory must use one exact eligibility boundary. A direct
// emit remains available for these admitted PODs even when no no-throw constructed/assignable loan
// slot can exist.
// REQ: OPERATOR-009
static_assert(PublishedPayload<ThrowingDefaultLoanPod>);
static_assert(std::default_initializable<ThrowingDefaultLoanPod>);
static_assert(std::assignable_from<ThrowingDefaultLoanPod&, ThrowingDefaultLoanPod>);
static_assert(!std::is_nothrow_default_constructible_v<ThrowingDefaultLoanPod>);
static_assert(!HasAllocate<Output<ThrowingDefaultLoanPod>>);
static_assert(HasSampleEmit<Output<ThrowingDefaultLoanPod>, Sample<ThrowingDefaultLoanPod>>);

static_assert(PublishedPayload<NonAssignableLoanPod>);
static_assert(std::is_nothrow_default_constructible_v<NonAssignableLoanPod>);
static_assert(!std::assignable_from<NonAssignableLoanPod&, NonAssignableLoanPod>);
static_assert(!HasAllocate<Output<NonAssignableLoanPod>>);
static_assert(HasSampleEmit<Output<NonAssignableLoanPod>, Sample<NonAssignableLoanPod>>);

// REQ: OPERATOR-013
static_assert(!HasLoad<ScalarInput>);
static_assert(!HasRetrieve<ScalarOutput>);
static_assert(!HasAllocate<Output<Tensor>>);
static_assert(HasAllocateTensor<Output<Tensor>>);
static_assert(!HasAcquireTensor<Output<Tensor>>);
static_assert(std::same_as<decltype(std::declval<Output<Tensor>&>().allocate_tensor(
                               std::declval<TensorLoanRequest>())),
                           expected<TensorOutputLoan, Error>>);

// REQ: OPERATOR-014
// REQ: OPERATOR-021
static_assert(HasSampleEmit<Output<Tensor>, Sample<Tensor>>);
static_assert(!HasEmitSampleVerb<Output<Tensor>, Sample<Tensor>>);
static_assert(!HasSampleEmitOptions<Output<Tensor>, Sample<Tensor>>);
static_assert(
    std::same_as<decltype(static_cast<TensorLoanEmit>(&Output<Tensor>::emit)), TensorLoanEmit>);
static_assert(
    std::same_as<decltype(static_cast<TensorValueEmit>(&Output<Tensor>::emit)), TensorValueEmit>);
static_assert(
    std::same_as<decltype(static_cast<TensorSampleEmit>(&Output<Tensor>::emit)), TensorSampleEmit>);
static_assert(!noexcept(std::declval<Output<Tensor>&>().emit(std::declval<Tensor>())));
static_assert(!noexcept(std::declval<Output<Tensor>&>().emit(std::declval<Sample<Tensor>>())));
static_assert(!noexcept(std::declval<Output<Tensor>&>().emit(std::declval<TensorOutputLoan&&>(),
                                                             EmitOptions{})));

// REQ: OPERATOR-010
// REQ: OPERATOR-026
static_assert(std::is_nothrow_move_constructible_v<TensorOutputLoan>);
static_assert(!std::is_copy_constructible_v<TensorOutputLoan>);
static_assert(!std::is_move_assignable_v<TensorOutputLoan>);
static_assert(std::is_nothrow_destructible_v<TensorOutputLoan>);
static_assert(!HasOwningTensorExtraction<TensorOutputLoan>);
static_assert(std::is_nothrow_move_constructible_v<TensorOutputWriteGuard>);
static_assert(!std::is_copy_constructible_v<TensorOutputWriteGuard>);
static_assert(!std::is_move_assignable_v<TensorOutputWriteGuard>);
static_assert(std::is_nothrow_destructible_v<TensorOutputWriteGuard>);
// REQ: OPERATOR-027
static_assert(HasFusedTensorWrite<TensorOutputLoan, TensorOutputWriteGuard>);
static_assert(std::same_as<decltype(std::move(std::declval<TensorOutputWriteGuard&>()).commit()),
                           expected<void, Error>>);

// REQ: OPERATOR-014
// REQ: OPERATOR-022
TEST(PortValues, SelectionStatesPreserveNotSelectedAndMissingDistinction) {
  constexpr InputSelection default_selection{};
  constexpr InputSelection selected{.state = InputState::kSelected,
                                    .reason = SelectionReason::kNone};
  constexpr InputSelection reused{.state = InputState::kReused,
                                  .reason = SelectionReason::kTooStale};
  constexpr InputSelection absent{.state = InputState::kAbsent,
                                  .reason = SelectionReason::kNeverSeen};
  constexpr InputSelection not_selected{
      .state = InputState::kNotSelected,
      .reason = SelectionReason::kNotSelectedForActivation,
  };

  EXPECT_EQ(selected.state, InputState::kSelected);
  EXPECT_EQ(default_selection.state, InputState::kAbsent);
  EXPECT_EQ(default_selection.reason, SelectionReason::kNeverSeen);
  EXPECT_EQ(reused.reason, SelectionReason::kTooStale);
  EXPECT_EQ(absent.state, InputState::kAbsent);
  EXPECT_NE(absent.state, not_selected.state);
  EXPECT_EQ(not_selected.reason, SelectionReason::kNotSelectedForActivation);
}

// REQ: OPERATOR-014
TEST(PortValues, PublishedPortTypesAreStableMemberSlotsNotEndpointOwners) {
  EXPECT_TRUE((std::is_standard_layout_v<InputSelection>));
  EXPECT_TRUE((std::is_trivially_copyable_v<InputSelection>));
  EXPECT_EQ(sizeof(InputSelection), 2);
}

// REQ: OPERATOR-009
TEST(PortValues, RuntimeGenericLoanCapabilityMatchesPublicOverloadParticipation) {
  // Key setup: runtime_port_type_ops<T>() returns a vtable the Runtime uses to decide whether a
  //   generic loan slot can be allocated for type T in a shared channel pool.
  // Key assertion: std::uint32_t qualifies (trivial no-throw construction and assignment);
  //   ThrowingDefaultLoanPod and NonAssignableLoanPod do not, signaled by slot_size_bytes == 0.
  const auto* scalar = detail::runtime_port_type_ops<std::uint32_t>();
  const auto* throwing_default = detail::runtime_port_type_ops<ThrowingDefaultLoanPod>();
  const auto* nonassignable = detail::runtime_port_type_ops<NonAssignableLoanPod>();

  ASSERT_NE(scalar, nullptr);
  ASSERT_NE(throwing_default, nullptr);
  ASSERT_NE(nonassignable, nullptr);
  EXPECT_TRUE(scalar->generic_loan_supported);
  EXPECT_EQ(scalar->generic_loan_slot_size_bytes, sizeof(std::uint32_t));
  EXPECT_FALSE(throwing_default->generic_loan_supported);
  EXPECT_EQ(throwing_default->generic_loan_slot_size_bytes, 0U);
  EXPECT_FALSE(nonassignable->generic_loan_supported);
  EXPECT_EQ(nonassignable->generic_loan_slot_size_bytes, 0U);
}

}  // namespace
}  // namespace holoscan
