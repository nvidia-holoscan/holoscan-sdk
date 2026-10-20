/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
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
constexpr holoscan::TensorRepresentation kHostByteFrame{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kByteDtype,
    .rank = 1U,
};
constexpr std::array<std::int64_t, 1U> kFrameShape{64};
constexpr std::uint32_t kCameraOnlyControl = 64U;
constexpr std::uint32_t kMaximumReferenceControl = 200U;
constexpr std::string_view kImuEndpoint{"baseline.camera_imu.imu"};
constexpr std::string_view kControlEndpoint{"baseline.camera_imu.control"};

class CameraSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kHostByteFrame,
            .bounds = holoscan::tensor_bounds(64U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    // The equivalent explicit form is write_host(), fill through the returned
    // TensorOutputWriteGuard, then consume that guard with std::move(*writer).commit().
    auto fill = [this](const holoscan::TensorOutputWriteGuard& writer)
        -> holoscan::expected<void, holoscan::Error> {
      auto* bytes = writer.data_as<std::uint8_t>();
      if (bytes == nullptr || std::cmp_not_equal(writer.byte_size(), kFrameShape.front())) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
      }
      std::fill_n(bytes, writer.byte_size(), static_cast<std::uint8_t>(sequence_++));
      return {};
    };
    auto committed = loan->write_host_and_commit(fill);
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return frame.emit(std::move(*loan),
                      holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::uint32_t sequence_{};
};

class WindowFusion final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera, "camera")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kHostByteFrame});
    spec.input(imu, "imu").queue_depth(32U);
    spec.output(control, "control");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnWindow{.anchor = camera, .max_wait = 20ms, .max_pending_windows = 2U})
        .input(camera, holoscan::Exact{})
        .input(imu,
               holoscan::BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 16U})
        .on_missing(imu, holoscan::SafeDefault{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto frame = camera.receive();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }

    std::uint32_t integrated = 0U;
    if (imu.selection().state == holoscan::InputState::kSelected) {
      auto batch = imu.selected_batch();
      if (!batch) {
        return holoscan::make_unexpected(std::move(batch).error());
      }
      for (const auto& sample : *batch) {
        integrated += sample.data;
      }
    }
    const std::uint32_t frame_bytes = static_cast<std::uint32_t>(frame->data.nbytes());
    auto published = control.emit(frame_bytes + integrated);
    if (!published && published.error().code == holoscan::ErrorCode::kNotReady) {
      // External attachment is not a startup barrier. A proven pre-submit
      // rejection may be dropped here; a later camera activation retries with
      // fresh control state without duplicating an accepted publication.
      return {};
    }
    return published;
  }

  holoscan::Input<holoscan::Tensor> camera;
  holoscan::Input<std::uint32_t> imu;
  holoscan::Output<std::uint32_t> control;
};

class FusionGraph final : public holoscan::Graph {
 public:
  holoscan::Output<std::uint32_t> control{"control"};

 protected:
  void compose() override {
    const holoscan::ClockKey clock = add_clock<holoscan::RealtimeClock>(
        "capture-clock", holoscan::RealtimeClock::Config{.use_time_since_epoch = true});
    set_default_clock(clock);

    holoscan::TopicOptions imu_options;
    imu_options.endpoint_name = kImuEndpoint;
    imu_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto imu =
        import_topic<std::uint32_t>("holomq://" + std::string{kImuEndpoint}, imu_options);

    const auto camera = op<CameraSource>("camera");
    const auto fusion = op<WindowFusion>("window-fusion");
    add_flow(camera->frame, fusion->camera, holoscan::ConnectionOptions{.queue_depth = 4U});
    add_flow(imu.output(), fusion->imu, holoscan::ConnectionOptions{.queue_depth = 32U});
    export_output(control, fusion->control);
  }
};

class CameraImuGraph final : public holoscan::Graph {
 public:
  CameraImuGraph() : Graph("holoscan-camera-imu-control") {}

 protected:
  void compose() override {
    set_default_clock(add_clock<holoscan::RealtimeClock>("runtime-clock"));

    holoscan::TopicOptions control_options;
    control_options.endpoint_name = kControlEndpoint;
    const auto control =
        export_topic<std::uint32_t>("holomq://" + std::string{kControlEndpoint}, control_options);

    const auto fusion = embed<FusionGraph>("fusion");
    add_flow(fusion->control, control.input(), holoscan::ConnectionOptions{.queue_depth = 8U});
  }
};

[[nodiscard]] bool wait_for_connections(holoscan::ChannelPublisher<std::uint32_t>& publisher,
                                        holoscan::ChannelSubscriber<std::uint32_t>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sender_refreshed = publisher.update_connections();
    auto receiver_refreshed = subscriber.has_samples();
    if (!sender_refreshed || !receiver_refreshed) {
      return false;
    }
    if (publisher.has_subscribers() && subscriber.has_publishers()) {
      ++consecutive_observations;
      if (consecutive_observations == 2U) {
        return true;
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

[[nodiscard]] holoscan::expected<holoscan::Sample<std::uint32_t>, holoscan::Error>
wait_for_integrated_control(holoscan::ChannelSubscriber<std::uint32_t>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto sample = subscriber.receive_sample();
      if (sample) {
        if (sample->data > kCameraOnlyControl && sample->data <= kMaximumReferenceControl) {
          return sample;
        }
        if (sample->data != kCameraOnlyControl) {
          return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
        }
      } else if (sample.error().code != holoscan::ErrorCode::kNotReady) {
        return holoscan::make_unexpected(std::move(sample).error());
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_imu_until_connected(
    holoscan::ChannelPublisher<std::uint32_t>& publisher, std::uint32_t value,
    std::int64_t capture_timestamp_ns) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    auto emitted = publisher.emit_captured(value, capture_timestamp_ns);
    if (emitted) {
      return emitted;
    }
    if (!emitted && emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    // Connection observation is advisory. Retry only proven pre-submit kNotReady and
    // kBackpressured outcomes. Partial, indeterminate, and other code-specific failures returned
    // above are terminal here.
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] int run_reference(const holoscan::ExecutionPlan& plan) {
  auto client_runtime = holoscan::ChannelRuntime::create();
  if (!client_runtime) {
    return 5;
  }
  auto imu = client_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), kImuEndpoint);
  auto control =
      client_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), kControlEndpoint);
  if (!imu || !control) {
    return 6;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto stop = [&session](int result) {
    session.request_stop();
    session.wait();
    return result;
  };
  if (!wait_for_connections(*imu, *control)) {
    std::cerr << "external endpoint attachment failed: imu_subscribers=" << imu->has_subscribers()
              << " control_publishers=" << control->has_publishers()
              << " termination=" << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << "@vertex=" << diagnostic.source.vertex.value << ',';
    }
    std::cerr << '\n';
    return stop(7);
  }

  for (std::uint32_t index = 0U; index < 16U; ++index) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto capture_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    auto emitted = emit_imu_until_connected(*imu, index + 1U, capture_ns);
    if (!emitted) {
      std::cerr << "IMU publish failed: " << holoscan::error_code_name(emitted.error().code) << " "
                << emitted.error().message
                << " termination=" << static_cast<int>(session.termination()) << " diagnostics=";
      for (const auto& diagnostic : session.diagnostics()) {
        std::cerr << diagnostic.token << "@vertex=" << diagnostic.source.vertex.value << ',';
      }
      std::cerr << '\n';
      return stop(8);
    }
    std::this_thread::sleep_for(5ms);
  }

  auto sample = wait_for_integrated_control(*control);
  auto closed = imu->close();
  if (!closed) {
    std::cerr << "IMU close failed: " << holoscan::error_code_name(closed.error().code) << " "
              << closed.error().message << '\n';
    return stop(9);
  }
  if (!sample) {
    return stop(10);
  }
  const std::uint32_t value = sample->data;
  const int stopped = stop(0);
  const auto control_closed = control->close();
  const auto control_after_close = control->receive_sample();
  if (stopped != 0 || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty() || !control_closed || control_after_close ||
      control_after_close.error().code != holoscan::ErrorCode::kClosed) {
    return 11;
  }
  std::cout << "control=" << value << " sequence=" << sample->metadata.sequence << '\n';
  std::cout << "camera/IMU runtime complete\n";
  return 0;
}

struct Arguments {
  bool emit_artifacts{};
  bool plan_json{};
  std::filesystem::path artifact_root;
  std::string deployment_id{"default"};
};

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& result) {
  if (argc == 1) {
    return true;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--plan-json") {
    result.plan_json = true;
    return true;
  }
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--emit-artifacts") {
      result.emit_artifacts = true;
    } else if ((argument == "--artifact-root" || argument == "--deployment-id") &&
               index + 1 < argc) {
      if (argument == "--artifact-root") {
        result.artifact_root = argv[++index];
      } else {
        result.deployment_id = argv[++index];
      }
    } else {
      return false;
    }
  }
  return result.emit_artifacts;
}

}  // namespace

// REQ: SYSTEM-004
int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: camera_imu_control [--plan-json | --emit-artifacts "
                 "[--artifact-root <path>] [--deployment-id <id>]]\n";
    return 2;
  }

  CameraImuGraph graph;
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 3;
  }
  if (arguments.plan_json) {
    std::cout << plan.json() << '\n';
    return 0;
  }
  if (arguments.emit_artifacts) {
    auto emitted = holoscan::emit_artifacts(
        plan,
        holoscan::ArtifactEmitOptions{.root = arguments.artifact_root,
                                      .deployment_id = std::move(arguments.deployment_id)});
    if (!emitted) {
      std::cerr << "artifact publication failed: "
                << holoscan::error_code_name(emitted.error().code) << '\n';
      return 4;
    }
    std::cout << "catalog=" << emitted->catalog_path << " digest=" << emitted->catalog_digest
              << " changed=" << emitted->changed << '\n';
    std::cout << "artifact-only publication complete\n";
    return 0;
  }
  return run_reference(plan);
}
