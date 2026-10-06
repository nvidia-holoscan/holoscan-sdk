// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;

struct Arguments {
  std::uint64_t seconds{0};  // --seconds: stop after this long (0: run until Ctrl-C)
  int device{0};
  ::holoscan::visualization::ImageFormat format{
      ::holoscan::visualization::ImageFormat::kR8G8B8Unorm};
  bool cuda{false};
};

/// The wire encoding for the formats this example can emit. HolovizOp checks the frame's encoding
/// against the layer's declared ImageFormat, so these must agree.
[[nodiscard]] holoscan::schema::ImageEncoding wire_encoding(
    ::holoscan::visualization::ImageFormat format) {
  switch (format) {
    case ::holoscan::visualization::ImageFormat::kR8G8B8Unorm:
      return holoscan::schema::ImageEncoding_RGB8;
    case ::holoscan::visualization::ImageFormat::kNV12:
      return holoscan::schema::ImageEncoding_NV12;
    case ::holoscan::visualization::ImageFormat::kYUYV422:
      return holoscan::schema::ImageEncoding_YUYV;
    default:
      // No single-plane wire encoding for this format; CUSTOM states no organization and is
      // exempt from the operator's encoding check.
      return holoscan::schema::ImageEncoding_CUSTOM;
  }
}

class ImageSource final : public holoscan::Operator<> {
 public:
  explicit ImageSource(Arguments arguments) : arguments_(arguments) {
    const bool rgb = arguments.format == ::holoscan::visualization::ImageFormat::kR8G8B8Unorm;
    const bool nv12 = arguments.format == ::holoscan::visualization::ImageFormat::kNV12;
    shape_ = nv12 ? std::vector<int64_t>{kHeight * 3 / 2, kWidth}
                  : std::vector<int64_t>{kHeight, kWidth, rgb ? 3 : 2};
    bytes_.resize(size_t{kWidth} * kHeight * (rgb ? 6 : nv12 ? 3 : 4) / 2);
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t luma = static_cast<uint8_t>(16 + x * 219 / (kWidth - 1));
        const size_t pixel = size_t{y} * kWidth + x;
        if (rgb) {
          bytes_[pixel * 3] = static_cast<uint8_t>(x * 255 / (kWidth - 1));
          bytes_[pixel * 3 + 1] = static_cast<uint8_t>(y * 255 / (kHeight - 1));
          bytes_[pixel * 3 + 2] = 128;
        } else if (nv12) {
          bytes_[pixel] = luma;
          if (y < kHeight / 2) {
            bytes_[size_t{kWidth} * kHeight + pixel] = x % 2 ? 173 : 81;
          }
        } else {
          bytes_[pixel * 2] = luma;
          bytes_[pixel * 2 + 1] = x % 2 ? 173 : 81;
        }
      }
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = arguments_.cuda ? holoscan::MemoryKind::kCudaDevice
                                                              : holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = static_cast<uint8_t>(shape_.size())},
            .bounds = holoscan::tensor_bounds(bytes_.size()),
            .storage = holoscan::TensorOutputStorage::kRuntimePool})
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 33ms});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto loan = frame.allocate_tensor({.shape = shape_, .dtype = DLDataType{kDLUInt, 8, 1}});
    if (!loan) {
      return holoscan::make_unexpected(loan.error());
    }
    auto writer = arguments_.cuda ? loan->write(context.cuda_stream()) : loan->write_host();
    if (!writer) {
      return holoscan::make_unexpected(writer.error());
    }
    if (arguments_.cuda) {
      const auto error = cudaMemcpyAsync(writer->data(),
                                         bytes_.data(),
                                         bytes_.size(),
                                         cudaMemcpyHostToDevice,
                                         context.cuda_stream().get());
      if (error != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(error)});
      }
    } else {
      std::memcpy(writer->data(), bytes_.data(), bytes_.size());
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(committed.error());
    }
    // emit_tensor() ignores the descriptor's own `data` field in favor of the loan, so only the
    // semantic fields need to be set. HolovizOp validates these against the declared layer.
    // width/height are the visible image size for every encoding; NV12 additionally locates its
    // two planes (Y, then the interleaved UV rows packed right after it) through plane_layouts,
    // which the schema requires for multi-planar encodings.
    holoscan::schema::ImageT descriptor{};
    descriptor.height = static_cast<std::int32_t>(kHeight);
    descriptor.width = static_cast<std::int32_t>(kWidth);
    descriptor.encoding = wire_encoding(arguments_.format);
    descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
    if (arguments_.format == ::holoscan::visualization::ImageFormat::kNV12) {
      constexpr std::uint64_t kLumaBytes = std::uint64_t{kWidth} * kHeight;
      descriptor.plane_layouts.emplace_back(std::uint64_t{0}, kWidth, kLumaBytes);
      descriptor.plane_layouts.emplace_back(kLumaBytes, kWidth, kLumaBytes / 2);
    }
    return frame.emit_tensor(std::move(*loan),
                             descriptor,
                             holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  Arguments arguments_;
  std::vector<int64_t> shape_;
  std::vector<uint8_t> bytes_;
};

/// Parse the whole of @p text as a nonnegative decimal integer of type T. Rejects empty input,
/// trailing text, a sign, and values outside T's range.
template <typename T>
[[nodiscard]] bool parse_nonnegative(std::string_view text, T& value) noexcept {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && value >= T{0};
}

/// --seconds is bounded so that it stays representable as std::chrono::nanoseconds.
constexpr std::uint64_t kMaxSeconds = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::nanoseconds::max()).count());

void print_usage() {
  std::cerr << "usage: image_formats [--format rgb|nv12|yuyv] [--memory host|cuda]"
               " [--seconds N (0: run until Ctrl-C)] [--device D]\n";
}

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    const bool has_value = i + 1 < argc;
    if (arg == "--seconds" && has_value) {
      const std::string_view text{argv[++i]};
      std::uint64_t value{};
      if (!parse_nonnegative(text, value) || value > kMaxSeconds) {
        std::cerr << "invalid --seconds value: '" << text << "' (expected a nonnegative integer)\n";
        print_usage();
        return false;
      }
      out.seconds = value;
    } else if (arg == "--device" && has_value) {
      const std::string_view text{argv[++i]};
      int value{};
      if (!parse_nonnegative(text, value)) {
        std::cerr << "invalid --device value: '" << text
                  << "' (expected a nonnegative CUDA device ordinal)\n";
        print_usage();
        return false;
      }
      out.device = value;
    } else if (arg == "--format" && has_value) {
      const std::string_view format{argv[++i]};
      if (format == "rgb") {
        out.format = ::holoscan::visualization::ImageFormat::kR8G8B8Unorm;
      } else if (format == "nv12") {
        out.format = ::holoscan::visualization::ImageFormat::kNV12;
      } else if (format == "yuyv") {
        out.format = ::holoscan::visualization::ImageFormat::kYUYV422;
      } else {
        std::cerr << "invalid --format value: '" << format << "' (expected rgb, nv12, or yuyv)\n";
        print_usage();
        return false;
      }
    } else if (arg == "--memory" && has_value) {
      const std::string_view memory{argv[++i]};
      if (memory != "host" && memory != "cuda") {
        std::cerr << "invalid --memory value: '" << memory << "' (expected host or cuda)\n";
        print_usage();
        return false;
      }
      out.cuda = memory == "cuda";
    } else {
      print_usage();
      return false;
    }
  }
  return true;
}

[[nodiscard]] ::holoscan::visualization::SinkDesc make_sink() {
  return ::holoscan::visualization::WindowSinkDesc{
      .title = "image_formats", .width = kWidth, .height = kHeight};
}

}  // namespace

int main(int argc, char** argv) {
  // Own SIGINT/SIGTERM before anything else starts a thread (CUDA, Vulkan, the runtime worker):
  // the session bound to this domain turns the first signal into a cooperative stop. Where the
  // domain is unavailable (non-Linux reports kNotSupported) the run falls back to the default
  // process signal disposition.
  auto signals = holoscan::ManagedShutdownSignals::create();

  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    return 1;
  }

  holoscan::Graph graph{"image-formats"};

  const auto source = graph.op<ImageSource>("source", arguments);
  ::holoscan::visualization::LayerStack stack;
  ::holoscan::visualization::ImageLayerDesc image{
      .format = arguments.format, .max_width = kWidth, .max_height = kHeight};
  if (arguments.format != ::holoscan::visualization::ImageFormat::kR8G8B8Unorm) {
    image.yuv_model = ::holoscan::visualization::YUVModelConversion::kBT709;
    image.yuv_range = ::holoscan::visualization::YUVRange::kLimited;
  }
  stack.add_image_layer("image", image);
  const auto viz = graph.op<holoscan::visualization::HolovizOp>(
      "holoviz",
      std::move(stack),
      make_sink(),
      ::holoscan::visualization::ClockTrigger{
          std::chrono::nanoseconds{16'666'667}},  // 60 Hz display
      std::chrono::nanoseconds{100ms});
  // Depth-1 "latest frame" mailbox: never backpressure the source while the window starts up.
  graph.add_flow(source->frame,
                 viz->image_layer_input(0),
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U});

  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // CUDA tensor pools must be placed explicitly; compile rejects an implicit device 0.
  holoscan::CompileOptions options;
  if (arguments.cuda) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = "source",
                                              .output_port = "frame",
                                              .device = holoscan::DeviceId{arguments.device}});
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // HolovizOp is clock-driven here (ClockTrigger), and a clock-triggered operator has no
  // end-of-stream completion: bounding the source with complete_after() would not end the run.
  // A bounded run is therefore a timed wait followed by a cooperative stop, which the runtime
  // also performs for a managed Ctrl-C; both end as kStopped. wait_for() only observes terminal
  // publication, so the session is always joined with wait() before the signal domain closes:
  // wait() also retires the managed binding, without which close() may report kNotReady.
  holoscan::RunSession session =
      signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  if (arguments.seconds != 0U && !session.wait_for(std::chrono::seconds{arguments.seconds})) {
    session.request_stop();
  }
  session.wait();
  if (signals) {
    if (const auto closed = signals->close(); !closed) {
      std::cerr << "image_formats: managed signal domain close failed: "
                << static_cast<int>(closed.error().code) << '\n';
    }
  }

  const holoscan::RunTermination termination = session.termination();
  if (termination != holoscan::RunTermination::kStopped &&
      termination != holoscan::RunTermination::kCompleted) {
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "diagnostic: " << diagnostic.token
                << " vertex=" << diagnostic.source.vertex.value << '\n';
    }
    std::cerr << "image_formats failed: termination=" << static_cast<int>(termination) << '\n';
    return 2;
  }
  std::cout << "image_formats complete\n";
  return 0;
}
