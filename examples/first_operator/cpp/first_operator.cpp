/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// The camera runs 4x faster than the IMU so every imu-selection outcome is observable:
// fresh (LatestWithin admits), reused (ReuseLast re-serves), missing (both bounds exceeded).
constexpr std::chrono::nanoseconds kCameraPeriod = 5ms;
constexpr std::chrono::nanoseconds kIMUPeriod = 20ms;
// The IMU goes silent for these of its own clock ticks (an ~80 ms stall). ReuseLast{20ms}
// bridges the first part of the gap; beyond that the imu input reports a missing selection.
constexpr std::uint32_t kIMUStallFirstTick = 4U;
constexpr std::uint32_t kIMUStallLastTick = 7U;
// ~240 ms of camera activity: several fresh/reused cycles plus the whole stall window.
constexpr std::size_t kFusedActivationsToObserve = 48U;

// Operators run on runtime threads, so results are collected here and the summary is printed
// from main() after shutdown; interleaved per-activation lines are illustrative only.
struct RunObservation {
  static void reset() noexcept {
    doubled_count.store(0U, std::memory_order_relaxed);
    fused_count.store(0U, std::memory_order_relaxed);
    fresh_count.store(0U, std::memory_order_relaxed);
    reused_count.store(0U, std::memory_order_relaxed);
    missing_count.store(0U, std::memory_order_relaxed);
  }

  static void record_doubled() noexcept { doubled_count.fetch_add(1U, std::memory_order_relaxed); }

  static void record_fused(holoscan::InputState imu_state) {
    {
      const std::lock_guard lock(mutex);
      if (imu_state == holoscan::InputState::kSelected) {
        fresh_count.fetch_add(1U, std::memory_order_relaxed);
      } else if (imu_state == holoscan::InputState::kReused) {
        reused_count.fetch_add(1U, std::memory_order_relaxed);
      } else {
        missing_count.fetch_add(1U, std::memory_order_relaxed);
      }
      fused_count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_fused(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return fused_count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> doubled_count{};
  static inline std::atomic<std::size_t> fused_count{};
  static inline std::atomic<std::size_t> fresh_count{};
  static inline std::atomic<std::size_t> reused_count{};
  static inline std::atomic<std::size_t> missing_count{};
};

class CameraSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(frame, "frame"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kCameraPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // Stamp capture time explicitly: LatestWithin/ReuseLast ages are measured in event time,
    // so an unstamped sample would never be admitted by an age-bounded selector.
    return frame.emit(++next_, holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<std::uint32_t> frame;

 private:
  std::uint32_t next_{};
};

class IMUSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(sample, "sample"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kIMUPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t tick = tick_++;
    if (tick >= kIMUStallFirstTick && tick <= kIMUStallLastTick) {
      return {};  // simulated sensor stall: emit nothing and let downstream policies take over
    }
    return sample.emit((++emitted_) * 10U,
                       holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<std::uint32_t> sample;

 private:
  std::uint32_t tick_{};
  std::uint32_t emitted_{};
};

// The contract-less teaching operator: with at least one connected input and no contract()
// override, compilation applies the OnAnyInput{} default, so compute() runs once per accepted
// sample. That default is exactly right for a stateless per-sample transform like this one.
class Doubler final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(8U);
    spec.output(doubled, "doubled");
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();  // full Sample<T>: payload plus SampleMetadata
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    sample->data *= 2U;
    // Forwarding the received Sample preserves the camera's capture timestamp, so downstream
    // age-based selection still sees the original acquisition time, not the re-emit time.
    return doubled.emit(std::move(*sample));
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> doubled;
};

// The contracted teaching operator: camera-driven fusion with an age-bounded imu side channel.
class SensorFusion final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera, "camera").queue_depth(8U);
    spec.input(imu, "imu").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{camera})
        .input(imu, holoscan::LatestWithin{.max_age = 5ms})
        .on_missing(imu, holoscan::ReuseLast{.max_age = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto cam_value = camera.receive_data();  // payload only: fusion needs no camera metadata
    if (!cam_value) {
      return holoscan::make_unexpected(std::move(cam_value).error());
    }
    const holoscan::InputState imu_state = imu.selection().state;
    RunObservation::record_fused(imu_state);
    if (imu.is_missing()) {
      // Both age bounds exceeded (or no imu sample admitted yet): there is no value to read.
      std::cout << ("fused: cam=" + std::to_string(*cam_value) + " imu=none (missing)\n");
      return {};
    }
    auto imu_value = imu.receive_data();
    if (!imu_value) {
      return holoscan::make_unexpected(std::move(imu_value).error());
    }
    const bool reused = imu_state == holoscan::InputState::kReused;
    std::cout << ("fused: cam=" + std::to_string(*cam_value) +
                  " imu=" + std::to_string(*imu_value) + (reused ? " (reused)\n" : " (fresh)\n"));
    return {};
  }

  holoscan::Input<std::uint32_t> camera;
  holoscan::Input<std::uint32_t> imu;
};

// Also contract-less: OnAnyInput{} is the right default for a per-sample terminal consumer.
class DoubledSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    RunObservation::record_doubled();
    std::cout << ("doubled: " + std::to_string(*value) + "\n");
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

}  // namespace

int main() {
  RunObservation::reset();

  holoscan::Graph graph{"first-operator"};
  const auto camera = graph.op<CameraSource>("camera");
  const auto imu = graph.op<IMUSource>("imu");
  const auto doubler = graph.op<Doubler>("doubler");
  const auto sink = graph.op<DoubledSink>("doubled-sink");
  const auto fusion = graph.op<SensorFusion>("fusion");
  graph.add_flow(camera->frame, doubler->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(doubler->doubled, sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(camera->frame, fusion->camera, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(imu->sample, fusion->imu, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("camera-processing").add(camera).add(doubler).add(sink).add(fusion);
  graph.partition("imu-ingress").add(imu);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = RunObservation::wait_for_fused(kFusedActivationsToObserve);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not stop cleanly\n";
    return 2;
  }

  // Printed after wait(): these lines are ordered and complete, unlike the interleaved
  // per-activation lines above, so tests should anchor only on this summary.
  const std::size_t doubled = RunObservation::doubled_count.load(std::memory_order_relaxed);
  const std::size_t fused = RunObservation::fused_count.load(std::memory_order_relaxed);
  const std::size_t fresh = RunObservation::fresh_count.load(std::memory_order_relaxed);
  const std::size_t reused = RunObservation::reused_count.load(std::memory_order_relaxed);
  const std::size_t missing = RunObservation::missing_count.load(std::memory_order_relaxed);
  std::cout << "imu selection over " << fused << " fused activations: fresh=" << fresh
            << " reused=" << reused << " missing=" << missing << '\n';
  std::cout << "first_operator complete: doubled=" << doubled << " fused_fresh=" << fresh
            << " fused_reused=" << reused << " fused_missing=" << missing << '\n';
  return 0;
}
