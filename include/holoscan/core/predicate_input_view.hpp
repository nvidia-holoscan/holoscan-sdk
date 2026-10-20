/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <span>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port_types.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

/// @brief Pair one predicate-visible port identity with its frozen selection outcome.
struct PredicateInputSelection {
  PortId port;               ///< Plan-local logical input identity.
  InputSelection selection;  ///< Selection adopted for predicate evaluation.
};

namespace detail {

template <PublishedPayload T>
struct PredicateInputBinding {
  const Sample<T>* sample{};
  InputSelection selection{InputState::kAbsent, SelectionReason::kNeverSeen};
  const std::uint64_t* active_generation{};
  std::uint64_t expected_generation{};
};

struct PredicateEvaluationBinding {
  TimePoint reference_time{};
  Tag tag{};
  TagTime tag_time{};
  std::span<const PredicateInputSelection> selections{};
  const std::uint64_t* active_generation{};
  std::uint64_t expected_generation{};
};

struct PredicateBindingAccess;
struct RuntimePredicateBindingAccess;
}  // namespace detail

/// @brief Borrow one immutable scalar sample during predicate evaluation.
///
/// Runtime creates a view only for a selected or reused scalar input. The view owns neither the
/// sample nor its read claim and expires with the predicate evaluation or fired activation that
/// adopted the pinned sample. It exposes no mutable payload, endpoint, transport handle, or
/// ownership extension.
template <PublishedPayload T>
class PredicateInputView {
 public:
  /// @brief Copy the same predicate-scoped immutable sample view.
  /// @param other View whose borrowed binding and generation evidence are copied.
  PredicateInputView(const PredicateInputView& other) noexcept = default;

  /// @brief Replace this view with another predicate-scoped borrowed binding.
  /// @param other View whose borrowed binding and generation evidence are copied.
  /// @return This view after replacement.
  PredicateInputView& operator=(const PredicateInputView& other) noexcept = default;

  /// @brief Move the predicate-scoped borrowed binding from another view.
  /// @param other View whose trivially borrowed binding is copied by the move.
  PredicateInputView(PredicateInputView&& other) noexcept = default;

  /// @brief Replace this view by moving another predicate-scoped borrowed binding.
  /// @param other View whose trivially borrowed binding is copied by the move.
  /// @return This view after replacement.
  PredicateInputView& operator=(PredicateInputView&& other) noexcept = default;

  /// @brief Destroy this view without retaining or releasing the pinned sample.
  ~PredicateInputView() noexcept = default;

  /// @brief Return the pinned immutable sample.
  /// @return Predicate-evaluation-scoped sample reference.
  /// @pre The view's generation remains active and the binding contains a scalar sample.
  [[nodiscard]] const Sample<T>& sample() const noexcept {
    validate();
    assert(binding_.sample != nullptr);
    return *binding_.sample;
  }

  /// @brief Return the frozen selection outcome for this sample.
  /// @return Selected or reused state and its reason.
  [[nodiscard]] InputSelection selection() const noexcept {
    validate();
    return binding_.selection;
  }

 private:
  explicit PredicateInputView(detail::PredicateInputBinding<T> binding) noexcept
      : binding_(binding) {}

  void validate() const noexcept {
    assert(binding_.active_generation != nullptr);
    assert(*binding_.active_generation == binding_.expected_generation);
  }

  detail::PredicateInputBinding<T> binding_;

  friend struct detail::PredicateBindingAccess;
  friend struct detail::RuntimePredicateBindingAccess;
};

/// @brief Borrow deterministic evidence frozen for one predicate evaluation.
///
/// Selection entries are ordered by the compiled predicate contract: an input-driven implicit
/// driver first, followed by additional declared predicate inputs; a clock-driven predicate has no
/// implicit driver entry. The context is allocation-free and expires with its evaluation. It has
/// no runtime clock lookup, graph mutation, or waiting authority.
class PredicateEvaluationContext {
 public:
  /// @brief Copy the same evaluation-scoped frozen evidence view.
  /// @param other Context whose borrowed binding and generation evidence are copied.
  PredicateEvaluationContext(const PredicateEvaluationContext& other) noexcept = default;

  /// @brief Replace this context with another evaluation-scoped borrowed binding.
  /// @param other Context whose borrowed binding and generation evidence are copied.
  /// @return This context after replacement.
  PredicateEvaluationContext& operator=(const PredicateEvaluationContext& other) noexcept = default;

  /// @brief Move the evaluation-scoped borrowed binding from another context.
  /// @param other Context whose trivially borrowed binding is copied by the move.
  PredicateEvaluationContext(PredicateEvaluationContext&& other) noexcept = default;

  /// @brief Replace this context by moving another evaluation-scoped borrowed binding.
  /// @param other Context whose trivially borrowed binding is copied by the move.
  /// @return This context after replacement.
  PredicateEvaluationContext& operator=(PredicateEvaluationContext&& other) noexcept = default;

  /// @brief Destroy this context without retaining any evaluation evidence.
  ~PredicateEvaluationContext() noexcept = default;

  /// @brief Return the predicate reference coordinate.
  /// @return Frozen source or scheduled-clock time used for this evaluation.
  [[nodiscard]] TimePoint reference_time() const noexcept {
    validate();
    return binding_.reference_time;
  }

  /// @brief Return the candidate activation identity associated with this evaluation.
  /// @return Frozen candidate tag.
  [[nodiscard]] Tag tag() const noexcept {
    validate();
    return binding_.tag;
  }

  /// @brief Return the deterministic logical candidate coordinate.
  /// @return Frozen tag time.
  [[nodiscard]] TagTime tag_time() const noexcept {
    validate();
    return binding_.tag_time;
  }

  /// @brief Return declaration-ordered predicate input outcomes.
  /// @return Immutable borrowed span valid only for this evaluation generation.
  [[nodiscard]] std::span<const PredicateInputSelection> selections() const noexcept {
    validate();
    return binding_.selections;
  }

 private:
  explicit PredicateEvaluationContext(detail::PredicateEvaluationBinding binding) noexcept
      : binding_(binding) {}

  void validate() const noexcept {
    assert(binding_.active_generation != nullptr);
    assert(*binding_.active_generation == binding_.expected_generation);
  }

  detail::PredicateEvaluationBinding binding_;

  friend struct detail::PredicateBindingAccess;
};

namespace detail {

/// @brief Construct one Runtime-owned, generation-checked predicate input view.
struct RuntimePredicateBindingAccess {
  template <PublishedPayload T>
  [[nodiscard]] static expected<PredicateInputView<T>, Error> input(
      PredicateInputBinding<T> binding) noexcept {
    const bool fresh = binding.selection.state == InputState::kSelected &&
                       binding.selection.reason == SelectionReason::kNone;
    const bool reused = binding.selection.state == InputState::kReused &&
                        binding.selection.reason != SelectionReason::kNone &&
                        binding.selection.reason != SelectionReason::kNotSelectedForActivation;
    const bool valid_generation = binding.expected_generation != 0U &&
                                  binding.active_generation != nullptr &&
                                  *binding.active_generation == binding.expected_generation;
    if (binding.sample == nullptr || (!fresh && !reused) || !valid_generation) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return PredicateInputView<T>{binding};
  }
};

template <PublishedPayload T>
[[nodiscard]] expected<PredicateInputView<T>, Error> make_runtime_predicate_input_view(
    PredicateInputBinding<T> binding) noexcept {
  return RuntimePredicateBindingAccess::input<T>(binding);
}

}  // namespace detail

}  // namespace holoscan
