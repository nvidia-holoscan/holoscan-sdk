/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>

// This non-installed header is the only convenient test facade in baseline.  It is
// deliberately layered over the same private late-binding seam used by the
// runtime; no queue or execution verb is added to a public port or Operator.
#include <core/operator_parameter_binding.hpp>
#include <core/operator_test_binding.hpp>

namespace holoscan::test {

template <typename Op>
class SyntheticOperatorFixture {
 public:
  explicit SyntheticOperatorFixture(Op& op) : op_(op) {}

  [[nodiscard]] expected<void, Error> declare_ports() { return binding_.declare_ports(op_); }

  [[nodiscard]] expected<void, Error> bind_initial_params(typename Op::Params params) {
    return detail::OperatorParameterBinding::bind_initial(op_, std::move(params));
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_value(Input<T>& port, T value,
                                                 InputSelection selection = {
                                                     .state = InputState::kSelected,
                                                     .reason = SelectionReason::kNone,
                                                 }) {
    // A value fixture owns no source-time fact.  Canonical absent capture time
    // is preserved instead of substituting activation time or now().
    return binding_.bind_scalar(port, Sample<T>{std::move(value), SampleMetadata{}}, selection);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_sample(Input<T>& port, Sample<T> sample,
                                                  InputSelection selection = {
                                                      .state = InputState::kSelected,
                                                      .reason = SelectionReason::kNone,
                                                  }) {
    return binding_.bind_scalar(port, std::move(sample), selection);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_sample_state(Input<T>& port, Sample<T> sample,
                                                        detail::SyntheticInputState state) {
    return binding_.bind_scalar_state(port, std::move(sample), state);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_batch(Input<T>& port, std::span<const Sample<T>> samples,
                                                 InputSelection selection = {
                                                     .state = InputState::kSelected,
                                                     .reason = SelectionReason::kNone,
                                                 }) {
    return binding_.bind_batch(port, samples, selection);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_batch_refs(Input<T>& port,
                                                      std::span<const Sample<T>* const> samples,
                                                      InputSelection selection = {
                                                          .state = InputState::kSelected,
                                                          .reason = SelectionReason::kNone,
                                                      }) {
    return binding_.bind_batch_refs(port, samples, selection);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_state(Input<T>& port,
                                                 detail::SyntheticInputState state) {
    return binding_.bind_state(port, state);
  }

  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor>)
  [[nodiscard]] expected<void, Error> bind_output(
      Output<T>& port, std::optional<std::uint32_t> max_emits = std::nullopt) {
    return binding_.bind_output(port, max_emits);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> fail_next_loan_publication_indeterminate(
      Output<T>& port) noexcept {
    return binding_.fail_next_loan_publication_indeterminate(port);
  }

  [[nodiscard]] expected<void, Error> invoke_once(detail::SyntheticActivationOptions options = {}) {
    return binding_.invoke_once(op_, options);
  }

  template <PublishedPayload T>
  [[nodiscard]] std::span<const Sample<T>> captured(const Output<T>& port) const noexcept {
    return binding_.captured(port);
  }

  template <PublishedPayload T>
  [[nodiscard]] std::span<const Sample<T>> captured_as(const OutputBase& port) const noexcept {
    return binding_.captured_as<T>(port);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<detail::SyntheticCapturedSamples<T>, Error> take_captured(
      const Output<T>& port) {
    return binding_.take_captured(port);
  }

 private:
  Op& op_;
  detail::SyntheticOperatorBinding binding_;
};

}  // namespace holoscan::test
