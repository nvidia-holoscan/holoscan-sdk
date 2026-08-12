/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kWindowsToObserve = 6U;
constexpr std::chrono::nanoseconds kCameraPeriod = 40ms;
constexpr std::chrono::nanoseconds kImuPeriod = 10ms;

// One frozen record per OnWindow activation: the camera anchor plus the imu
// samples the runtime selected around it (or the reason no bracket completed).
struct WindowRecord {
  std::uint32_t anchor_value{};
  std::int64_t anchor_time_ns{};
  bool bracketed{};
  holoscan::SelectionReason reason{holoscan::SelectionReason::kNone};
  std::vector<std::uint32_t> imu_values;
  std::int64_t first_imu_time_ns{};
  std::int64_t last_imu_time_ns{};
};

// Operators record here instead of printing: interleaved std::cout from
// concurrent compute() calls is not line-atomic. main() prints after wait().
struct WindowObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    records.clear();
  }

  static void record(WindowRecord record) {
    {
      const std::lock_guard lock(mutex);
      records.push_back(std::move(record));
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return records.size() >= expected; });
  }

  [[nodiscard]] static std::vector<WindowRecord> snapshot() {
    const std::lock_guard lock(mutex);
    return records;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<WindowRecord> records;
};

class ClockedSource final : public holoscan::Operator<> {
 public:
  explicit ClockedSource(std::chrono::nanoseconds period) noexcept : period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ++next_;
    // BracketedWindow selects by event time: every sample must carry a genuine
    // capture time in a comparable clock domain (here the graph default clock).
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t next_{};
};

class BracketWatcher final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera, "camera").queue_depth(4U);
    // Bracket selection replays retained history; the imu queue must be deep
    // enough to keep predecessors alive until each window closes.
    spec.input(imu, "imu").queue_depth(32U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnWindow{.anchor = camera, .max_wait = 20ms, .max_pending_windows = 2U})
        .input(camera, holoscan::Exact{})
        .input(imu, holoscan::BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 8U})
        .on_missing(imu, holoscan::SafeDefault{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto anchor = camera.receive();
    if (!anchor) {
      return holoscan::make_unexpected(std::move(anchor).error());
    }

    WindowRecord record;
    record.anchor_value = anchor->data;
    record.anchor_time_ns = anchor->metadata.capture_timestamp_ns;

    const holoscan::InputSelection selection = imu.selection();
    if (selection.state == holoscan::InputState::kSelected) {
      auto batch = imu.selected_batch();
      if (!batch) {
        return holoscan::make_unexpected(std::move(batch).error());
      }
      record.bracketed = true;
      record.imu_values.reserve(batch->size());
      for (const auto& sample : *batch) {
        if (record.imu_values.empty()) {
          record.first_imu_time_ns = sample.metadata.capture_timestamp_ns;
        }
        record.last_imu_time_ns = sample.metadata.capture_timestamp_ns;
        record.imu_values.push_back(sample.data);
      }
    } else {
      // SafeDefault{} entered the operator-authored degraded path: no imu data
      // was fabricated, and selection.reason says why the bracket is missing.
      record.reason = selection.reason;
    }
    WindowObservation::record(std::move(record));
    return {};
  }

  holoscan::Input<std::uint32_t> camera;
  holoscan::Input<std::uint32_t> imu;
};

[[nodiscard]] std::string_view reason_name(holoscan::SelectionReason reason) noexcept {
  switch (reason) {
    case holoscan::SelectionReason::kNeverSeen:
      return "never-seen";
    case holoscan::SelectionReason::kInsufficientHistory:
      return "insufficient-history";
    case holoscan::SelectionReason::kWindowTimeout:
      return "window-timeout";
    case holoscan::SelectionReason::kWindowSampleLimit:
      return "window-sample-limit";
    case holoscan::SelectionReason::kWindowPendingLimit:
      return "window-pending-limit";
    default:
      return "other";
  }
}

}  // namespace

int main() {
  WindowObservation::reset();

  holoscan::Graph graph{"ea1-on-window-bracket"};
  const auto camera = graph.op<ClockedSource>("camera", kCameraPeriod);
  const auto imu = graph.op<ClockedSource>("imu", kImuPeriod);
  const auto watcher = graph.op<BracketWatcher>("watcher");
  graph.add_flow(camera->output, watcher->camera, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(imu->output, watcher->imu, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.partition("camera-source").add(camera);
  graph.partition("imu-history-source").add(imu);
  graph.partition("window-watcher").add(watcher);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = WindowObservation::wait_for(kWindowsToObserve);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not observe " << kWindowsToObserve << " windows cleanly\n";
    return 2;
  }

  const std::vector<WindowRecord> records = WindowObservation::snapshot();
  std::size_t bracketed = 0U;
  std::size_t safe_defaults = 0U;
  bool straddled = false;
  for (const WindowRecord& record : records) {
    std::cout << "[on_window] anchor=" << record.anchor_value;
    if (record.bracketed) {
      ++bracketed;
      std::cout << " bracket=[";
      for (std::size_t index = 0; index < record.imu_values.size(); ++index) {
        std::cout << (index == 0U ? "" : ",") << record.imu_values[index];
      }
      std::cout << "] samples=" << record.imu_values.size() << '\n';
      if (record.imu_values.size() >= 2U && record.first_imu_time_ns < record.anchor_time_ns &&
          record.last_imu_time_ns > record.anchor_time_ns) {
        straddled = true;  // at least one earlier and one later imu sample
      }
    } else {
      ++safe_defaults;
      std::cout << " bracket=safe-default reason=" << reason_name(record.reason) << '\n';
    }
  }
  std::cout << "[on_window] windows=" << records.size() << " bracketed=" << bracketed
            << " safe_default=" << safe_defaults << '\n';
  if (bracketed == 0U || !straddled) {
    std::cerr << "no bracket straddled its anchor with earlier and later imu samples\n";
    return 3;
  }
  std::cout << "[on_window] verified: a bracket straddles its anchor with earlier and later"
               " imu samples\n";
  std::cout << "on_window bracket demo complete\n";
  return 0;
}
