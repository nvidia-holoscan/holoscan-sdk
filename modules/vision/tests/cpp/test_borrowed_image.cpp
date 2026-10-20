// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>

#include "common/allocation_counter.hpp"
#include "common/cuda_test_utils.hpp"

namespace holoscan::vision {
namespace {
using namespace std::chrono_literals;
constexpr DLDataType kU8{kDLUInt, 8, 1};
constexpr std::array<std::int64_t, 3> kShape{4, 4, 1};
constexpr std::uint64_t kMessages = 64;

template <typename Op>
// Keep each operator's failed preparation, retry, idempotent start, and restart checks in one
// test scenario so the same instance exercises its complete lifecycle.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void check_start_retry() {
  std::size_t scratch_allocations{};
  bool scratch_created = false;
  {
    const test::ScopedAllocationCounter counter;
    auto scratch = sensor_io::ImageDescriptorScratch::create();
    scratch_created = static_cast<bool>(scratch);
    scratch_allocations = counter.count();
  }
  ASSERT_TRUE(scratch_created);
  // Exercise every scratch allocation and the first C++ allocation after scratch preparation.
  // Native NVCV allocations are qualified separately by the isolated warmup test executable.
  for (std::size_t fail_after = 0; fail_after <= scratch_allocations; ++fail_after) {
    SCOPED_TRACE(fail_after);
    Op operation;
    bool failed = false;
    ErrorCode code = ErrorCode::kFailure;
    {
      const test::ScopedAllocationFailure failure(fail_after);
      try {
        operation.start();
      } catch (const RuntimeError& error) {
        failed = true;
        code = error.code();
      } catch (const std::bad_alloc&) {
        failed = true;
        code = ErrorCode::kResourceExhausted;
      }
    }
    ASSERT_TRUE(failed);
    EXPECT_EQ(code, ErrorCode::kResourceExhausted);
    ASSERT_NO_THROW(operation.start());
    bool restart_failed = false;
    {
      const test::ScopedAllocationExhaustion failure(0);
      try {
        operation.start();
        operation.stop();
        operation.start();
      } catch (...) {
        restart_failed = true;
      }
    }
    EXPECT_FALSE(restart_failed);
  }
}

// REQ: VISION-008
TEST(VisionBorrowedImage, EveryImageOperatorRetriesPartialStartAndReusesPreparedState) {
  if (!test::has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  check_start_retry<ResizeOp>();
  check_start_retry<ConvertToOp>();
  check_start_retry<CopyMakeBorderOp>();
  check_start_retry<CustomCropOp>();
  check_start_retry<NormalizeOp>();
  check_start_retry<ReformatOp>();
  check_start_retry<RemapOp>();
  // BayerDemosaicOp is deliberately not exercised by check_start_retry<Op>(): unlike the other
  // seven Image operators, its start() has no CV-CUDA operation to construct after preparing the
  // scratch (NPP's Bayer kernels are stateless free functions), so there is no "first allocation
  // after scratch preparation" for the fail_after == scratch_allocations case to fail. Its own
  // scratch-creation failure path is exercised indirectly by every BayerDemosaicOpValidation case,
  // which all call start() and would themselves fail loudly if scratch preparation broke.
}

struct Progress {
  std::array<std::atomic<std::uint64_t>, 2> received{};
  std::atomic<const void*> input_pointer{};
};

class RepeatedImageSource final : public Operator<> {
 public:
  explicit RepeatedImageSource(std::shared_ptr<Progress> progress)
      : progress_(std::move(progress)), header_(std::make_shared<schema::HeaderT>()) {
    header_->frame_id.assign(128, 'c');
    header_->capture_timestamp_ns = 123;
    image_.width = 4;
    image_.height = 4;
    image_.encoding = schema::ImageEncoding_MONO8;
    image_.significant_bits = 7;
    image_.roi_offset_x = 7;
    image_.roi_offset_y = 9;
    image_.gain = 2.5F;
  }
  void setup(OperatorSpec& spec) override {
    spec.output(output)
        .tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice, .dtype = kU8, .rank = 3},
            .bounds = tensor_bounds(16),
            .storage = TensorOutputStorage::kRuntimePool});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (sent_ == kMessages || progress_->received[0].load() != sent_ ||
        progress_->received[1].load() != sent_)
      return {};
    auto allocated = output.allocate_tensor({.shape = kShape, .dtype = kU8});
    if (!allocated && allocated.error().code == ErrorCode::kResourceExhausted)
      return {};
    if (!allocated)
      return make_unexpected(allocated.error());
    auto& loan = *allocated;
    HOLOSCAN_TRY(loan.write_and_commit(
        context.cuda_stream(), [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          progress_->input_pointer.store(writer.data());
          if (cudaMemsetAsync(
                  writer.data(), static_cast<int>(sent_ + 1), 16, context.cuda_stream().get()) !=
              cudaSuccess)
            return make_unexpected(Error{ErrorCode::kFailure});
          return {};
        }));
    header_->device_sequence = sent_ + 1;
    image_.header = sent_ % 2 == 0 ? header_ : nullptr;
    auto emitted = output.emit_tensor(std::move(loan),
                                      image_,
                                      EmitOptions{.capture_time = context.activation_time(),
                                                  .frame_id = sent_ + 1,
                                                  .trace = TraceContext::root(77)});
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (!emitted)
      return make_unexpected(emitted.error());
    ++sent_;
    return {};
  }
  Output<schema::ImageT> output{"output"};

 private:
  std::shared_ptr<Progress> progress_;
  std::shared_ptr<schema::HeaderT> header_;
  schema::ImageT image_;
  std::uint64_t sent_{};
};

class RepeatedImageSink final : public Operator<> {
 public:
  RepeatedImageSink(std::shared_ptr<Progress> progress, std::size_t index)
      : progress_(std::move(progress)), index_(index) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input)
        .queue_depth(1)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice}});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& image = view.data();
    const auto sequence = progress_->received[index_].load() + 1;
    const bool transformed = index_ == 0;
    const bool has_header = sequence % 2 == 1;
    const auto dimension = transformed ? 2 : 4;
    if (!image.data || image.width != dimension || image.height != dimension ||
        image.roi_offset_x != (transformed ? 0 : 7) ||
        image.roi_offset_y != (transformed ? 0 : 9) || image.gain != 2.5F ||
        image.encoding != schema::ImageEncoding_MONO8 || image.significant_bits != 7 ||
        !image.plane_layouts.empty() || !image.header || view.metadata().sequence != sequence ||
        view.metadata().frame_id != sequence || view.metadata().trace_id != 77 ||
        !view.metadata().source_clock_id.valid())
      return make_unexpected(Error{ErrorCode::kFailure});
    if (has_header &&
        (image.header->frame_id.size() != 128 ||
         !std::ranges::all_of(image.header->frame_id, [](char value) { return value == 'c'; }) ||
         image.header->device_sequence != sequence || image.header->capture_timestamp_ns != 123))
      return make_unexpected(Error{ErrorCode::kFailure});
    // Existing prepared receive slots materialize absent wire Header/Time as empty owners.
    // Scratch preserves the received metadata; this migration does not change decode semantics.
    if (!has_header && (!image.header->frame_id.empty() || image.header->device_sequence != 0 ||
                        image.header->capture_timestamp_ns != 0))
      return make_unexpected(Error{ErrorCode::kFailure});
    const bool source_pointer = image.data->data() == progress_->input_pointer.load();
    if (source_pointer == transformed)
      return make_unexpected(Error{ErrorCode::kFailure});
    std::array<std::uint8_t, 16> pixels{};
    const auto bytes = static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension);
    HOLOSCAN_TRY(image.data->copy_to_host_async(pixels.data(), bytes, context.cuda_stream().get()));
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess ||
        !std::all_of(
            pixels.begin(), pixels.begin() + bytes, [&](auto value) { return value == sequence; }))
      return make_unexpected(Error{ErrorCode::kFailure});
    progress_->received[index_].store(sequence);
    return {};
  }
  Input<schema::ImageT> input{"input"};

 private:
  std::shared_ptr<Progress> progress_;
  std::size_t index_;
};

// REQ: VISION-008
// REQ: VISION-005
// REQ: VISION-007
TEST(VisionBorrowedImage, ResizeRepeatedFanoutPreservesSourceAndOptionalMetadata) {
  if (!test::has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  auto progress = std::make_shared<Progress>();
  Graph graph{"borrowed-resize"};
  auto source = graph.op<RepeatedImageSource>("source", progress);
  auto resize = graph.op<ResizeOp>("resize");
  resize.set_params(
      ResizeOpParams{.width = 2, .height = 2, .interpolation = InterpolationMode::kNearest});
  auto sink = graph.op<RepeatedImageSink>("sink", progress, 0);
  auto sibling = graph.op<RepeatedImageSink>("sibling", progress, 1);
  graph.add_flow(source->output, resize->input);
  graph.add_flow(source->output, sibling->input);
  graph.add_flow(resize->output, sink->input);
  graph.partition("source-lane").add(source);
  graph.partition("resize-lane").add(resize);
  graph.partition("sink-lane").add(sink);
  graph.partition("sibling-lane").add(sibling);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  for (auto name : {"source", "resize"})
    options.deployment.bind_tensor_output_device(
        {.operator_path = name, .output_port = "output", .device = DeviceId{0}});
  const auto plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // No explicit slot/grant overrides: progress uses only compiler-admitted storage.
  auto session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while ((progress->received[0].load() != kMessages || progress->received[1].load() != kMessages) &&
         session.is_running() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  session.request_stop();
  session.wait();
  EXPECT_EQ(progress->received[0].load(), kMessages);
  EXPECT_EQ(progress->received[1].load(), kMessages);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}
}  // namespace
}  // namespace holoscan::vision
