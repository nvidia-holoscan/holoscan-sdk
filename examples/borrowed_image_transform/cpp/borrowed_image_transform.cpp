/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {
using namespace holoscan;
constexpr std::array<std::int64_t, 1> kShape{1};

void declare_output(OperatorSpec& spec, Output<schema::ImageT>& output) {
  spec.output(output)
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                           .dtype = DLDataType{kDLUInt, 64, 1},
                                                           .rank = 1},
                                        .bounds = tensor_bounds(sizeof(std::uint64_t)),
                                        .storage = TensorOutputStorage::kRuntimePool});
}
void declare_input(OperatorSpec& spec, Input<schema::ImageT>& input) {
  spec.input(input)
      .tensor_field(schema::ImageTensorFields::data)
      .expects_tensor(TensorInputSpec{.representation = {.memory_kind = MemoryKind::kHost,
                                                         .dtype = DLDataType{kDLUInt, 64, 1},
                                                         .rank = 1}});
}

class Source final : public Operator<> {
 public:
  Source() {
    image_.width = 1;
    image_.height = 1;
    image_.header = std::make_shared<schema::HeaderT>();
    image_.header->frame_id = "camera_front";
  }
  void setup(OperatorSpec& spec) override { declare_output(spec, output); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 1});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto loan, output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}}));
    HOLOSCAN_TRY(loan.write_host_and_commit(
        [](const TensorOutputWriteGuard& writer) { *writer.data_as<std::uint64_t>() = 41; }));
    return output.emit_tensor(std::move(loan),
                              image_,
                              EmitOptions{.capture_time = context.activation_time(),
                                          .frame_id = 31,
                                          .trace = TraceContext::root(77)});
  }
  Output<schema::ImageT> output{"output"};

 private:
  schema::ImageT image_;
};

class MetadataTransform final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input);
    declare_output(spec, output);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  void start() override {
    if (scratch_)
      return;  // Reuse prepared capacity across an idempotent start/restart.
    auto created = sensor_io::ImageDescriptorScratch::create();
    if (!created)
      throw RuntimeError(created.error());
    scratch_.emplace(std::move(*created));
  }
  expected<void, Error> compute(ExecutionContext&) override {
    if (!scratch_)
      return make_unexpected(Error{ErrorCode::kNotReady});
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& image = view.data();  // No owning Image or Sample copy.
    if (!image.data)
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    const Tensor& pixels = *image.data;  // Borrowed pointees must also remain immutable.
    HOLOSCAN_TRY(scratch_->copy_metadata_from(image));
    scratch_->image().roi_offset_x =
        3;  // Persistent scratch contains metadata, never input pixels.

    HOLOSCAN_TRY(auto loan, output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}}));
    HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
      *writer.data_as<std::uint64_t>() = *pixels.data_as<std::uint64_t>() + 1;
    }));
    // The loan supplies output pixels; the descriptor supplies metadata. The sample envelope
    // carries provenance independently of the Image Header's sensor facts.
    return output.emit_tensor(
        std::move(loan), scratch_->image(), sensor_io::derive_from(view.metadata()));
  }
  Input<schema::ImageT> input{"input"};
  Output<schema::ImageT> output{"output"};

 private:
  std::optional<sensor_io::ImageDescriptorScratch> scratch_;
};

class Sink final : public Operator<> {
 public:
  explicit Sink(std::shared_ptr<bool> verified) : verified_(std::move(verified)) {}
  void setup(OperatorSpec& spec) override { declare_input(spec, input); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& image = view.data();
    *verified_ = image.data && image.header && *image.data->data_as<std::uint64_t>() == 42 &&
                 image.roi_offset_x == 3 && image.header->frame_id == "camera_front" &&
                 view.metadata().frame_id == 31 && view.metadata().trace_id == 77;
    return *verified_ ? expected<void, Error>{} : make_unexpected(Error{ErrorCode::kFailure});
  }
  Input<schema::ImageT> input{"input"};

 private:
  std::shared_ptr<bool> verified_;
};
}  // namespace

int main() {
  try {
    auto verified = std::make_shared<bool>(false);
    holoscan::Graph graph{"borrowed_image_transform"};
    auto source = graph.op<Source>("source");
    auto transform = graph.op<MetadataTransform>("transform");
    auto sink = graph.op<Sink>("sink", verified);
    graph.add_flow(source->output, transform->input);
    graph.add_flow(transform->output, sink->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    const auto plan = holoscan::compile(graph);
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      return 1;
    }
    if (holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged) != 0 || !*verified)
      return 1;
    std::cout << "borrowed image transform: pixels, metadata, and provenance verified\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
