/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 1U> kFrameShape{64};
constexpr std::array<std::int64_t, 1U> kOverlayShape{8};
constexpr std::array<holoscan::TensorAxisBounds, 1U> kOverlayAxes{{
    {.max_extent = static_cast<std::uint64_t>(kOverlayShape.front()), .max_abs_stride_bytes = 1U},
}};
constexpr std::string_view kOverlayEndpointPrefix{"retained-tensor-compositor.overlay"};
constexpr std::string_view kControlEndpointPrefix{"retained-tensor-compositor.control"};
constexpr std::string_view kDefaultRoute{"holomq://retained-tensor-compositor.overlay"};
constexpr std::uint32_t kControlValue = 0xC0DEU;
constexpr std::size_t kPrerollFrames = 3U;
constexpr int kNoCudaDeviceReturnCode = 77;

struct CompositionObservation {
  class ComputeActivity {
   public:
    ComputeActivity() noexcept { compute_active.store(true, std::memory_order_release); }
    ~ComputeActivity() {
      compute_active.store(false, std::memory_order_release);
      ready.notify_all();
    }

    ComputeActivity(const ComputeActivity&) = delete;
    ComputeActivity& operator=(const ComputeActivity&) = delete;
  };

  static void reset() {
    const std::lock_guard lock{mutex};
    base_frames.store(0U, std::memory_order_relaxed);
    compute_active.store(false, std::memory_order_relaxed);
    frame_quiescence_requested.store(false, std::memory_order_relaxed);
    frame_quiesced.store(false, std::memory_order_relaxed);
    activations = 0U;
    selected = 0U;
    reused = 0U;
    selected_sequence.reset();
    reused_same_sequence = true;
    payloads_valid = true;
  }

  static void record_base_frame() {
    base_frames.fetch_add(1U, std::memory_order_release);
    ready.notify_all();
  }

  static void record_composition(holoscan::InputState state, std::uint64_t sequence,
                                 std::uint32_t control_value, bool payload_valid) {
    {
      const std::lock_guard lock{mutex};
      ++activations;
      payloads_valid = payloads_valid && payload_valid && control_value == kControlValue;
      if (state == holoscan::InputState::kSelected) {
        ++selected;
        selected_sequence = sequence;
      } else if (state == holoscan::InputState::kReused) {
        ++reused;
        reused_same_sequence =
            reused_same_sequence && selected_sequence.has_value() && *selected_sequence == sequence;
      } else {
        payloads_valid = false;
      }
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_preroll() {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [] { return base_frames.load(std::memory_order_acquire) >= kPrerollFrames; });
  }

  [[nodiscard]] static bool wait_for_selected_and_reused() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 5s, [] { return selected > 0U && reused > 0U; });
  }

  static void request_frame_quiescence() {
    frame_quiescence_requested.store(true, std::memory_order_release);
  }

  [[nodiscard]] static bool quiescence_requested() {
    return frame_quiescence_requested.load(std::memory_order_acquire);
  }

  [[nodiscard]] static bool compositor_active() {
    return compute_active.load(std::memory_order_acquire);
  }

  static void record_frame_quiesced() {
    frame_quiesced.store(true, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_frame_quiesced() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 5s, [] { return frame_quiesced.load(std::memory_order_acquire); });
  }

  [[nodiscard]] static std::size_t activation_count() {
    const std::lock_guard lock{mutex};
    return activations;
  }

  [[nodiscard]] static bool valid() {
    const std::lock_guard lock{mutex};
    return activations >= 2U && selected > 0U && reused > 0U && selected_sequence.has_value() &&
           reused_same_sequence && payloads_valid;
  }

  [[nodiscard]] static std::pair<std::size_t, std::size_t> selection_counts() {
    const std::lock_guard lock{mutex};
    return {selected, reused};
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> base_frames{};
  static inline std::atomic<bool> compute_active{};
  static inline std::atomic<bool> frame_quiescence_requested{};
  static inline std::atomic<bool> frame_quiesced{};
  static inline std::size_t activations{};
  static inline std::size_t selected{};
  static inline std::size_t reused{};
  static inline std::optional<std::uint64_t> selected_sequence;
  static inline bool reused_same_sequence{true};
  static inline bool payloads_valid{true};
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> fail(
    holoscan::ErrorCode code = holoscan::ErrorCode::kFailure) {
  return holoscan::make_unexpected(holoscan::Error{code});
}

template <holoscan::MemoryKind Kind>
class FrameSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorPortLayout{
            .memory_kind = Kind,
            .dtype = kByteDtype,
            .rank = 1U,
        })
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U,
            .max_byte_span = static_cast<std::uint64_t>(kFrameShape.front()),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (CompositionObservation::quiescence_requested()) {
      const std::size_t activations = CompositionObservation::activation_count();
      if (!CompositionObservation::compositor_active() &&
          activations == quiescence_activation_count_) {
        CompositionObservation::record_frame_quiesced();
      } else {
        quiescence_activation_count_ = activations;
      }
      return {};
    }

    const std::uint8_t value = static_cast<std::uint8_t>((sequence_++ % 255U) + 1U);
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }

    holoscan::expected<void, holoscan::Error> committed;
    if constexpr (Kind == holoscan::MemoryKind::kCudaDevice) {
      cudaStream_t stream = context.cuda_stream().get();
      if (stream == nullptr) {
        return fail(holoscan::ErrorCode::kNotSupported);
      }
      committed = loan->write_and_commit(
          context.cuda_stream(),
          [stream, value](const holoscan::TensorOutputWriteGuard& writer)
              -> holoscan::expected<void, holoscan::Error> {
            if (writer.byte_size() != static_cast<std::size_t>(kFrameShape.front()) ||
                cudaMemsetAsync(writer.data(), value, writer.byte_size(), stream) != cudaSuccess) {
              return fail();
            }
            return {};
          });
    } else {
      committed =
          loan->write_host_and_commit([value](const holoscan::TensorOutputWriteGuard& writer)
                                          -> holoscan::expected<void, holoscan::Error> {
            if (writer.byte_size() != static_cast<std::size_t>(kFrameShape.front())) {
              return fail(holoscan::ErrorCode::kInvalidArgument);
            }
            std::fill_n(writer.data_as<std::uint8_t>(), writer.byte_size(), value);
            return {};
          });
    }
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    auto emitted = frame.emit(std::move(*loan),
                              holoscan::EmitOptions{.capture_time = context.activation_time()});
    if (!emitted) {
      return emitted;
    }
    CompositionObservation::record_base_frame();
    return {};
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::uint32_t sequence_{};
  std::size_t quiescence_activation_count_{std::numeric_limits<std::size_t>::max()};
};

template <holoscan::MemoryKind FrameKind>
class RetainedCompositor final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = FrameKind,
            .dtype = kByteDtype,
            .rank = 1U,
        });
    spec.input(overlay, "overlay")
        .queue_depth(2U)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kHost,
            .dtype = kByteDtype,
            .rank = 1U,
        });
    spec.input(control, "control").queue_depth(2U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame})
        .input(overlay, holoscan::LatestWithin{.max_age = 2s})
        .on_missing(overlay, holoscan::ReuseLast{.max_age = 2s})
        .input(control, holoscan::LatestWithin{.max_age = 2s})
        .on_missing(control, holoscan::ReuseLast{.max_age = 2s});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const CompositionObservation::ComputeActivity activity;
    auto frame_sample = frame.receive();
    if (!frame_sample) {
      return holoscan::make_unexpected(std::move(frame_sample).error());
    }
    auto overlay_sample = overlay.receive();
    if (!overlay_sample) {
      return holoscan::make_unexpected(std::move(overlay_sample).error());
    }
    auto control_sample = control.receive();
    if (!control_sample) {
      return holoscan::make_unexpected(std::move(control_sample).error());
    }

    const DLDeviceType expected_frame_device =
        FrameKind == holoscan::MemoryKind::kCudaDevice ? kDLCUDA : kDLCPU;
    const holoscan::Tensor& frame_tensor = frame_sample->data;
    const holoscan::Tensor& overlay_tensor = overlay_sample->data;
    bool valid = frame_tensor.device().device_type == expected_frame_device &&
                 frame_tensor.nbytes() == kFrameShape.front() &&
                 frame_tensor.shape_span().size() == 1U &&
                 frame_tensor.shape_span().front() == kFrameShape.front() &&
                 overlay_tensor.device().device_type == kDLCPU &&
                 overlay_tensor.nbytes() == kOverlayShape.front() &&
                 overlay_tensor.shape_span().size() == 1U &&
                 overlay_tensor.shape_span().front() == kOverlayShape.front();
    if (const auto* bytes = overlay_tensor.data_as<std::uint8_t>(); bytes == nullptr) {
      valid = false;
    } else {
      for (std::size_t index = 0U; index < static_cast<std::size_t>(kOverlayShape.front());
           ++index) {
        valid = valid && bytes[index] == 0xA5U;
      }
    }
    CompositionObservation::record_composition(
        overlay.selection().state, overlay_sample->metadata.sequence, control_sample->data, valid);
    return valid ? holoscan::expected<void, holoscan::Error>{}
                 : fail(holoscan::ErrorCode::kInvalidArgument);
  }

  holoscan::Input<holoscan::Tensor> frame;
  holoscan::Input<holoscan::Tensor> overlay;
  holoscan::Input<std::uint32_t> control;
};

struct Arguments {
  bool cuda{};
  int device{};
  bool device_authored{};
  bool route_authored{};
  std::string route{kDefaultRoute};
  std::string overlay_endpoint{kOverlayEndpointPrefix};
  std::string control_endpoint{kControlEndpointPrefix};
};

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& result) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--cuda") {
      result.cuda = true;
    } else if ((argument == "--device" || argument == "--route") && index + 1 < argc) {
      if (argument == "--device") {
        try {
          const std::string authored_device{argv[++index]};
          std::size_t parsed_characters = 0U;
          result.device = std::stoi(authored_device, &parsed_characters);
          if (parsed_characters != authored_device.size()) {
            return false;
          }
        } catch (const std::exception&) {
          return false;
        }
        result.device_authored = true;
      } else {
        result.route = argv[++index];
        result.route_authored = true;
      }
    } else {
      return false;
    }
  }
  return result.cuda || !result.device_authored;
}

[[nodiscard]] int cuda_device_status(int device) {
  int count = 0;
  const cudaError_t probe = cudaGetDeviceCount(&count);
  if (probe == cudaErrorNoDevice || probe == cudaErrorInsufficientDriver ||
      (probe == cudaSuccess && count == 0)) {
    static_cast<void>(cudaGetLastError());
    std::cout << "retained_tensor_compositor: skipped: CUDA device unavailable\n";
    return kNoCudaDeviceReturnCode;
  }
  if (probe != cudaSuccess) {
    std::cerr << "CUDA device discovery failed: " << cudaGetErrorString(probe) << '\n';
    return 2;
  }
  if (device < 0 || device >= count) {
    std::cerr << "authored CUDA device ordinal " << device << " is outside [0, " << count << ")\n";
    return 2;
  }
  const cudaError_t selected = cudaSetDevice(device);
  if (selected != cudaSuccess) {
    std::cerr << "CUDA device selection failed for ordinal " << device << ": "
              << cudaGetErrorString(selected) << '\n';
    return 2;
  }
  const cudaError_t initialized = cudaFree(nullptr);
  if (initialized != cudaSuccess) {
    std::cerr << "CUDA context initialization failed for ordinal " << device << ": "
              << cudaGetErrorString(initialized) << '\n';
    return 2;
  }
  return 0;
}

template <holoscan::MemoryKind FrameKind>
[[nodiscard]] holoscan::ExecutionPlan compile_compositor(const Arguments& arguments) {
  holoscan::Graph graph{"retained-tensor-compositor"};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>(
      "runtime-clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
  graph.set_default_clock(clock);

  holoscan::TopicOptions overlay_options;
  overlay_options.endpoint_name = arguments.overlay_endpoint;
  overlay_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  overlay_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  overlay_options.tensor = holoscan::LogicalTensorRequirements{
      .memory_kind = holoscan::MemoryKind::kHost,
      .dtype = kByteDtype,
      .rank = 1U,
      .logical_bounds =
          holoscan::LogicalTensorBounds{
              .max_rank = 1U,
              .max_byte_span = static_cast<std::uint64_t>(kOverlayShape.front()),
              .axes = kOverlayAxes,
          },
      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  const auto overlay =
      graph.import_topic<holoscan::Tensor>(arguments.route, std::move(overlay_options));
  holoscan::TopicOptions control_options;
  control_options.endpoint_name = arguments.control_endpoint;
  control_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  const auto control =
      graph.import_topic<std::uint32_t>(arguments.route + ".control", std::move(control_options));
  const auto source = graph.op<FrameSource<FrameKind>>("frame-source");
  const auto compositor = graph.op<RetainedCompositor<FrameKind>>("compositor");
  graph.add_flow(source->frame,
                 compositor->frame,
                 holoscan::ConnectionOptions{
                     .queue_policy = holoscan::QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });
  graph.add_flow(
      overlay.output(), compositor->overlay, holoscan::ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(
      control.output(), compositor->control, holoscan::ConnectionOptions{.queue_depth = 2U});

  holoscan::CompileOptions options;
  if constexpr (FrameKind == holoscan::MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "frame-source",
        .output_port = "frame",
        .device = holoscan::DeviceId{arguments.device},
    });
  }
  return holoscan::compile(graph, std::move(options));
}

template <typename T>
[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_subscriber(
    holoscan::ChannelPublisher<T>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return refreshed;
    }
    if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> publish_overlay(
    holoscan::ChannelPublisher<holoscan::Tensor>& publisher) {
  auto loan = publisher.loan_tensor(
      holoscan::TensorLoanRequest{.shape = kOverlayShape, .dtype = kByteDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto committed = loan->write_host_and_commit([](const holoscan::TensorLoanWriteGuard& writer)
                                                   -> holoscan::expected<void, holoscan::Error> {
    if (writer.byte_size() != static_cast<std::size_t>(kOverlayShape.front())) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    std::fill_n(writer.data_as<std::uint8_t>(), writer.byte_size(), std::uint8_t{0xA5U});
    return {};
  });
  if (!committed) {
    return committed;
  }
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  auto capture_time = publisher.make_capture_time(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  if (!capture_time) {
    return holoscan::make_unexpected(std::move(capture_time).error());
  }
  auto emitted = publisher.emit(std::move(*loan),
                                holoscan::EmitOptions{.capture_time = std::move(*capture_time)});
  if (!emitted) {
    return holoscan::make_unexpected(std::move(emitted).error());
  }
  if (*emitted == 0U) {
    return fail(holoscan::ErrorCode::kNotReady);
  }
  return {};
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> publish_control(
    holoscan::ChannelPublisher<std::uint32_t>& publisher) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  auto emitted = publisher.emit_captured(
      kControlValue, std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  if (!emitted) {
    return holoscan::make_unexpected(std::move(emitted).error());
  }
  if (*emitted == 0U) {
    return fail(holoscan::ErrorCode::kNotReady);
  }
  return {};
}

void print_session_diagnostics(const holoscan::RunSession& session) {
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::cerr << "  diagnostic=" << diagnostic.token << " vertex=" << diagnostic.source.vertex.value
              << '\n';
  }
}

template <typename Publisher>
[[nodiscard]] bool close_publisher_once(Publisher& publisher, std::string_view label,
                                        std::size_t run_index) {
  auto closed = publisher.close();
  if (!closed) {
    std::cerr << "run " << run_index << ": ordered " << label
              << " EOS failed: " << holoscan::runtime_error_message(closed.error()) << '\n';
    return false;
  }

  auto closed_again = publisher.close();
  if (closed_again || closed_again.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "run " << run_index << ": repeated " << label << " close did not report kClosed\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool run_once(const holoscan::ExecutionPlan& plan, const Arguments& arguments,
                            std::size_t run_index) {
  CompositionObservation::reset();
  auto channel_runtime = holoscan::ChannelRuntime::create();
  if (!channel_runtime) {
    std::cerr << "ChannelRuntime creation failed: "
              << holoscan::runtime_error_message(channel_runtime.error()) << '\n';
    return false;
  }
  auto publisher_result = channel_runtime->create_publisher<holoscan::Tensor>(
      plan.endpoint_catalog(), arguments.overlay_endpoint);
  auto control = channel_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(),
                                                                  arguments.control_endpoint);
  if (!publisher_result || !control) {
    std::cerr << "external publisher creation failed";
    if (!publisher_result) {
      std::cerr << ": overlay=" << holoscan::runtime_error_message(publisher_result.error());
    }
    if (!control) {
      std::cerr << ": control=" << holoscan::runtime_error_message(control.error());
    }
    std::cerr << '\n';
    return false;
  }
  auto publisher = std::move(*publisher_result);

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto stop = [&session]() {
    session.request_stop();
    session.wait();
  };
  auto connected = wait_for_subscriber(publisher);
  auto control_connected = wait_for_subscriber(*control);
  if (!connected) {
    std::cerr << "run " << run_index << ": overlay connection failed: "
              << holoscan::runtime_error_message(connected.error()) << '\n';
    stop();
    print_session_diagnostics(session);
    return false;
  }
  if (!control_connected) {
    std::cerr << "run " << run_index << ": control connection failed: "
              << holoscan::runtime_error_message(control_connected.error()) << '\n';
    stop();
    print_session_diagnostics(session);
    return false;
  }
  if (!CompositionObservation::wait_for_preroll()) {
    std::cerr << "run " << run_index << ": frame preroll timed out\n";
    stop();
    print_session_diagnostics(session);
    return false;
  }
  std::this_thread::sleep_for(20ms);
  if (CompositionObservation::activation_count() != 0U) {
    std::cerr << "run " << run_index << ": retained overlay leaked across session start\n";
    stop();
    return false;
  }

  auto published = publish_overlay(publisher);
  auto control_published = publish_control(*control);
  if (!published || !control_published || !CompositionObservation::wait_for_selected_and_reused()) {
    std::cerr << "run " << run_index << ": overlay/control publish or retained selection failed";
    if (!published) {
      std::cerr << ": " << holoscan::runtime_error_message(published.error());
    }
    if (!control_published) {
      std::cerr << ": control=" << holoscan::runtime_error_message(control_published.error());
    }
    std::cerr << '\n';
    stop();
    print_session_diagnostics(session);
    return false;
  }
  CompositionObservation::request_frame_quiescence();
  if (!CompositionObservation::wait_for_frame_quiesced()) {
    std::cerr << "run " << run_index << ": frame source did not reach the EOS safe point\n";
    stop();
    return false;
  }
  if (!close_publisher_once(publisher, "Tensor", run_index) ||
      !close_publisher_once(*control, "control", run_index)) {
    stop();
    return false;
  }
  session.request_stop();
  session.wait();
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty() || !CompositionObservation::valid()) {
    std::cerr << "run " << run_index << ": session did not stop with valid retained selections\n";
    print_session_diagnostics(session);
    return false;
  }

  const auto [selected, reused] = CompositionObservation::selection_counts();
  std::cout << "run=" << run_index << " pre_overlay_activations=0 selected=" << selected
            << " reused=" << reused << " tensor_eos=closed control_eos=closed reset=clean\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: retained_tensor_compositor [--cuda [--device <ordinal>]] "
                 "[--route holomq://<service>]\n";
    return 2;
  }
  const std::string run_id =
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  arguments.overlay_endpoint += "-" + run_id;
  arguments.control_endpoint += "-" + run_id;
  if (!arguments.route_authored) {
    arguments.route += "-" + run_id;
  }
  if (!arguments.route.starts_with("holomq://")) {
    std::cerr << "route must use the holomq:// scheme\n";
    return 2;
  }
  if (arguments.cuda) {
    const int status = cuda_device_status(arguments.device);
    if (status != 0) {
      return status;
    }
  }

  const holoscan::ExecutionPlan plan =
      arguments.cuda ? compile_compositor<holoscan::MemoryKind::kCudaDevice>(arguments)
                     : compile_compositor<holoscan::MemoryKind::kHost>(arguments);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 3;
  }
  if (!run_once(plan, arguments, 1U) || !run_once(plan, arguments, 2U)) {
    return 4;
  }

  std::cout << "route=" << arguments.route
            << " frame_memory=" << (arguments.cuda ? "cuda_device" : "host");
  if (arguments.cuda) {
    std::cout << " device=" << arguments.device;
  }
  std::cout << '\n';
  std::cout << "retained Tensor compositor complete\n";
  return 0;
}
