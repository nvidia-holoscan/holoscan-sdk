/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: a single operator driven by SyntheticOperatorFixture proves that
// Input<T>/Output<T> ports carry data and that ExecutionContext exposes the
// full activation context (tag, tag-time, CUDA stream, trace, cancellation).

#include <cstdint>
#include <utility>

#include <holoscan/core/expected_macros.hpp>

#include "operator_test_fixture.hpp"

// Topology: [synthetic input] → SmokeOperator → [captured output]

namespace {

// REQ: OPERATOR-014
// Records every ExecutionContext field on each invocation so the test harness
// can verify that the fixture delivers the full activation context correctly.
class SmokeOperator final : public holoscan::Operator<> {
 public:
  holoscan::Input<std::uint32_t> input_{"input"};
  holoscan::Output<std::uint32_t> output_{"output"};
  std::uint32_t callbacks{};
  holoscan::Tag observed_tag{};
  holoscan::TagTime observed_tag_time{};
  holoscan::TimePoint observed_activation_time{};
  holoscan::TraceContext observed_trace{};
  cudaStream_t observed_stream{};
  bool observed_cancellation{};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    ++callbacks;
    if (!context.tag().valid()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    observed_tag = context.tag();
    observed_tag_time = context.tag_time();
    observed_activation_time = context.activation_time();
    observed_trace = context.trace();
    observed_stream = context.cuda_stream().get();
    observed_cancellation = context.cancellation().stop_requested();
    HOLOSCAN_TRY(auto sample, input_.receive());
    sample.data *= 2;
    return output_.emit(std::move(sample));
  }
};

}  // namespace

int main() {
  SmokeOperator op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  if (!fixture.declare_ports() || !fixture.bind_output(op.output_)) {
    return 1;  // exit 1: fixture setup (port declaration or output capture binding) failed
  }

  const holoscan::SampleMetadata metadata{
      .sequence = 12,
      .capture_timestamp_ns = 300,
      .source_clock_id = holoscan::ClockId{.value = 2},
      .frame_id = 7,
  };
  std::uint8_t stream_token{};
  auto* const stream = reinterpret_cast<cudaStream_t>(&stream_token);
  const holoscan::detail::SyntheticActivationOptions activation{
      .tag = holoscan::Tag{.sequence = 4, .generation = 2},
      .tag_time = holoscan::TagTime{.logical_ns = 1, .microstep = 3},
      .activation_time =
          holoscan::TimePoint{
              .timestamp_ns = 310,
              .clock_id = holoscan::ClockId{.value = 1},
          },
      .cuda_stream = stream,
      .trace = holoscan::TraceContext{.trace_id = 9, .span_id = 10, .parent_span_id = 8},
      .cancellation_requested = true,
  };
  if (!fixture.bind_sample(op.input_, holoscan::Sample<std::uint32_t>{21, metadata}) ||
      !fixture.invoke_once(activation)) {
    return 2;  // exit 2: sample injection or synthetic invocation failed
  }

  const auto captured = fixture.captured(op.output_);
  if (op.callbacks != 1 || captured.size() != 1 || captured[0].data != 42) {
    return 3;  // exit 3: compute() invocation count wrong, or input→output data transform failed
  }
  if (captured[0].metadata.capture_time() != metadata.capture_time() ||
      captured[0].metadata.frame_id != metadata.frame_id) {
    return 4;  // exit 4: sample metadata not propagated through the operator unchanged
  }
  if (op.observed_tag != activation.tag || op.observed_tag_time != activation.tag_time ||
      op.observed_activation_time != activation.activation_time ||
      op.observed_trace != activation.trace || op.observed_stream != stream ||
      !op.observed_cancellation) {
    return 5;  // exit 5: ExecutionContext did not deliver the full activation descriptor
  }
  return 0;
}
