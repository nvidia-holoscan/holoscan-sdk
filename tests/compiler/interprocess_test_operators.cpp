/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "interprocess_test_operators.hpp"

#include <iostream>
#include <utility>

namespace holoscan::test {
namespace {

using namespace std::chrono_literals;

}  // namespace

ScalarProcessBoundarySource::ScalarProcessBoundarySource(ScalarProcessBoundaryHooks hooks)
    : hooks_(hooks) {}

void ScalarProcessBoundarySource::setup(OperatorSpec& spec) {
  spec.output(output, "output").max_emits_per_compute(1U);
}

Contract ScalarProcessBoundarySource::contract() const {
  Contract result;
  result.trigger(OnClock{.period = 20ms});
  return result;
}

void ScalarProcessBoundarySource::start() {
  hooks_.source_started();
}

expected<void, Error> ScalarProcessBoundarySource::compute(ExecutionContext& context) {
  // LatestWithin selection requires a genuine event-time coordinate on every sample.
  const TimePoint capture_time = context.activation_time();
  if (!capture_time.valid()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto emitted = output.emit(hooks_.payload, EmitOptions{.capture_time = capture_time});
  // Retryable or indeterminate publication consumes this tick; only structural errors fail.
  if (!emitted && emitted.error().code != ErrorCode::kNotReady &&
      emitted.error().code != ErrorCode::kBackpressured &&
      emitted.error().code != ErrorCode::kPublicationIndeterminate) {
    return make_unexpected(std::move(emitted).error());
  }
  return {};
}

ScalarProcessBoundarySink::ScalarProcessBoundarySink(ScalarProcessBoundaryHooks hooks)
    : hooks_(hooks) {}

void ScalarProcessBoundarySink::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(4U);
}

Contract ScalarProcessBoundarySink::contract() const {
  Contract result;
  result.trigger(OnAnyInput{}).input(input, LatestWithin{.max_age = 1s}).on_missing(input, Skip{});
  return result;
}

void ScalarProcessBoundarySink::start() {
  hooks_.sink_started();
}

expected<void, Error> ScalarProcessBoundarySink::compute(ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  hooks_.sample_received(sample->data);
  return {};
}

FanoutProcessBoundarySource::FanoutProcessBoundarySource(FanoutProcessBoundaryHooks hooks)
    : hooks_(hooks) {}

void FanoutProcessBoundarySource::setup(OperatorSpec& spec) {
  spec.output(output, "output").max_emits_per_compute(1U);
}

Contract FanoutProcessBoundarySource::contract() const {
  Contract result;
  result.trigger(OnClock{.period = 1ms});
  return result;
}

void FanoutProcessBoundarySource::start() {
  hooks_.source_started();
}

expected<void, Error> FanoutProcessBoundarySource::compute(ExecutionContext&) {
  if (next_value_ > hooks_.sample_count) {
    return {};
  }
  auto published = output.emit(next_value_);
  // Retain the current sequence value until the shared publisher accepts it.
  if (!published && (published.error().code == ErrorCode::kNotReady ||
                     published.error().code == ErrorCode::kBackpressured)) {
    return {};
  }
  if (!published) {
    std::cerr << "fanout source publication failed at value " << next_value_ << ": "
              << runtime_error_message(published.error()) << '\n';
    return make_unexpected(std::move(published).error());
  }
  ++next_value_;
  hooks_.sample_emitted();
  return {};
}

FanoutProcessBoundarySink::FanoutProcessBoundarySink(FanoutProcessBoundaryHooks hooks,
                                                     const std::size_t sink_index)
    : hooks_(hooks), sink_index_(sink_index) {}

void FanoutProcessBoundarySink::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(4U);
}

Contract FanoutProcessBoundarySink::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void FanoutProcessBoundarySink::start() {
  hooks_.sink_started(sink_index_);
}

expected<void, Error> FanoutProcessBoundarySink::compute(ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  hooks_.sample_received(sink_index_, sample->data);
  return {};
}

}  // namespace holoscan::test
