/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

#include "image_generated.h"
#include "tensor_packet_generated.h"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

template <typename Native>
struct DescriptorOutputScenario;

// REQ: OPERATOR-031
// REQ: OPERATOR-035
// REQ: CHAN-107
// REQ: CHAN-122
// REQ: TENSOR-SCHEMA-005
// REQ: TENSOR-SCHEMA-006
template <>
struct DescriptorOutputScenario<schema::ImageT> {
  static constexpr std::string_view name{"Image"};
  static constexpr std::array<std::uint8_t, 8U> bytes{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};

  [[nodiscard]] static schema::ImageT descriptor() {
    schema::ImageT value{};
    value.height = 1;
    value.width = 8;
    value.encoding = schema::ImageEncoding_CUSTOM;
    return value;
  }

  [[nodiscard]] static EmitOptions emit_options() {
    return EmitOptions{
        .capture_time = TimePoint{.timestamp_ns = 1234, .clock_id = ClockId{1U}},
        .frame_id = 9U,
        .trace = TraceContext::root(0xCAFEU),
    };
  }

  [[nodiscard]] static bool matches(const Sample<schema::ImageT>& sample) {
    return sample.data.height == 1 && sample.data.width == 8 &&
           sample.data.encoding == schema::ImageEncoding_CUSTOM &&
           sample.metadata.capture_timestamp_ns == 1234 && sample.metadata.trace_id == 0xCAFEU &&
           sample.metadata.frame_id == 9U && sample.metadata.sequence == 1U;
  }
};

// REQ: CHAN-120
// REQ: OPERATOR-035
template <>
struct DescriptorOutputScenario<schema::TensorPacketT> {
  static constexpr std::string_view name{"TensorPacket"};
  static constexpr std::array<std::uint8_t, 8U> bytes{7U, 6U, 5U, 4U, 3U, 2U, 1U, 0U};

  [[nodiscard]] static schema::TensorPacketT descriptor() {
    schema::TensorPacketT value{};
    value.sequence = 42U;
    value.source = "third-party-camera";
    value.scale = 0.5;
    return value;
  }

  [[nodiscard]] static EmitOptions emit_options() {
    return EmitOptions{
        .capture_time = TimePoint{.timestamp_ns = 4321, .clock_id = ClockId{1U}},
        .frame_id = 11U,
        .trace = TraceContext::root(0xBEEFU),
    };
  }

  [[nodiscard]] static bool matches(const Sample<schema::TensorPacketT>& sample) {
    return sample.data.sequence == 42U && sample.data.source == "third-party-camera" &&
           sample.data.scale == 0.5 && sample.metadata.capture_timestamp_ns == 4321 &&
           sample.metadata.trace_id == 0xBEEFU && sample.metadata.frame_id == 11U &&
           sample.metadata.sequence == 1U;
  }
};

// Captures provider pointer identity and validates descriptor, Tensor, and Sample metadata.
template <typename Native>
struct RuntimeDescriptorOutputProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    source_pointer = nullptr;
    observed = false;
    valid = false;
    error.reset();
  }

  static void record_source(const void* pointer) {
    const std::lock_guard lock(mutex);
    source_pointer = pointer;
  }

  static void record_sink(const Sample<Native>& sample) {
    const std::lock_guard lock(mutex);
    const Tensor* const tensor = sample.data.data.get();
    const std::uint8_t* const bytes = tensor == nullptr ? nullptr : tensor->data_as<std::uint8_t>();
    valid = tensor != nullptr && tensor->data() == source_pointer && tensor->nbytes() == 8 &&
            tensor->ndim() == 1 && std::ranges::equal(tensor->shape_span(), kShape) &&
            tensor->dtype().code == kDLUInt && tensor->dtype().bits == 8U &&
            tensor->dtype().lanes == 1U && bytes != nullptr &&
            std::ranges::equal(std::span<const std::uint8_t>{bytes, 8U},
                               DescriptorOutputScenario<Native>::bytes) &&
            DescriptorOutputScenario<Native>::matches(sample);
    observed = true;
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      observed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed; });
  }

  static constexpr std::array<std::int64_t, 1U> kShape{8};
  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline const void* source_pointer{};
  static inline bool observed{};
  static inline bool valid{};
  static inline std::optional<ErrorCode> error;
};

// Allocates a descriptor Tensor from the output's finite provider-backed pool.
template <typename Native>
class RuntimeDescriptorOutputSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        })
        .tensor_allocation(TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 8U})
        .payload_options(PayloadOptions{
            .schema = schema_identity<Native>(),
            .max_tensor_byte_span = 8U,
            .interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
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
    emitted_ = true;

    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = RuntimeDescriptorOutputProbeState<Native>::kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return fail(loan.error().code);
    }
    const auto written = loan->write_host_and_commit([](const TensorOutputWriteGuard& writer) {
      RuntimeDescriptorOutputProbeState<Native>::record_source(writer.data());
      std::ranges::copy(DescriptorOutputScenario<Native>::bytes, writer.data_as<std::uint8_t>());
    });
    if (!written) {
      return fail(written.error().code);
    }
    const auto emitted = output.emit_tensor(std::move(*loan),
                                            DescriptorOutputScenario<Native>::descriptor(),
                                            DescriptorOutputScenario<Native>::emit_options());
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  Output<Native> output;

 private:
  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDescriptorOutputProbeState<Native>::fail(code);
    return make_unexpected(Error{code});
  }

  bool emitted_{};
};

// Verifies that descriptor metadata and the provider Tensor arrive without a data copy.
template <typename Native>
class RuntimeDescriptorOutputSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
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
      RuntimeDescriptorOutputProbeState<Native>::fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    RuntimeDescriptorOutputProbeState<Native>::record_sink(*sample);
    return {};
  }

  Input<Native> input;
};

// Thread-safe probe for the fanout stream test. Tracks two application CUDA
// streams: blocked_stream is held via a host callback to force the source into
// pool exhaustion while quick_stream completes immediately. Detects premature
// slot reuse by comparing subsequent allocation pointers against the first.

// Graph topology:
//   descriptor source --[queue_depth=1, direct retained]--> descriptor sink
// The allocation declaration materializes one provider-owned pool for either generated schema.
template <typename Native>
[[nodiscard]] ExecutionPlan make_runtime_descriptor_output_plan() {
  Graph graph{"runtime-descriptor-output"};
  auto source = graph.op<RuntimeDescriptorOutputSourceOperator<Native>>("source");
  auto sink = graph.op<RuntimeDescriptorOutputSinkOperator<Native>>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

template <typename Native>
class DescriptorOutputScenarioTest : public testing::Test {};

using DescriptorOutputNatives = testing::Types<schema::ImageT, schema::TensorPacketT>;

struct DescriptorOutputScenarioNames {
  template <typename Native>
  static std::string GetName(int) {
    return std::string{DescriptorOutputScenario<Native>::name};
  }
};

TYPED_TEST_SUITE(DescriptorOutputScenarioTest, DescriptorOutputNatives,
                 DescriptorOutputScenarioNames);

TYPED_TEST(DescriptorOutputScenarioTest, PreservesProviderBackingAndMetadata) {
  using Probe = RuntimeDescriptorOutputProbeState<TypeParam>;
  Probe::reset();
  const ExecutionPlan plan = make_runtime_descriptor_output_plan<TypeParam>();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_rank, 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 8U);
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().payload_identity.payload_kind,
            PayloadKind::kDescriptorBacked);

  RunSession session = run_async(plan);
  const bool observed = Probe::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(Probe::mutex);
  EXPECT_FALSE(Probe::error.has_value());
  EXPECT_TRUE(Probe::valid);
}

}  // namespace
}  // namespace holoscan
