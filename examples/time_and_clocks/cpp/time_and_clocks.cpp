/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/time_point.hpp>

namespace {

using namespace std::chrono_literals;

// Each source emits exactly kFrameCount frames, one per 5ms tick, then idles.
constexpr std::uint32_t kFrameCount = 12U;
constexpr std::chrono::nanoseconds kFramePeriod = 5ms;
// The synthetic "hardware" exposure happens a fixed latency before the driver
// (our compute callback) sees the frame — the realistic camera shape. The raw
// device timestamp becomes the sample's physical capture coordinate after the
// driver labels it with the graph clock domain.
constexpr std::chrono::nanoseconds kExposureLatency = 500us;
// Frame 3 simulates a frame that sat in a stalled driver queue for 5 seconds
// before the driver handed it over: its device timestamp is 5 s older than the
// moment we received and published it.
constexpr std::uint32_t kStaleFrame = 3U;
constexpr std::chrono::nanoseconds kStaleOffset = 5s;
// The sink's application-level freshness budget on capture age.
constexpr std::chrono::nanoseconds kMaxCaptureAge = 500ms;
// How many per-frame correlation lines each operator prints while running.
constexpr std::uint64_t kPrintedFrames = 2U;

// Operators run on runtime threads; main() blocks on this observation object
// (mutex + condition_variable + counters) and stops the run when every frame
// has been accounted for. Static because graph.op<> captures must be owned
// copyable values — operators never borrow pointers into main's stack.
struct TimeObservation {
  static constexpr std::size_t kHw = 0;   // slot for the hardware-stamped chain
  static constexpr std::size_t kAct = 1;  // slot for the activation-stamped chain

  static void reset() noexcept {
    for (auto& slot : accepted)
      slot.store(0U, std::memory_order_relaxed);
    for (auto& slot : dropped)
      slot.store(0U, std::memory_order_relaxed);
    restamped.store(0U, std::memory_order_relaxed);
    domain_mismatch.store(0U, std::memory_order_relaxed);
    for (auto& stamp : hw_stamps)
      stamp.store(0, std::memory_order_relaxed);
  }

  static void record(std::size_t slot, bool fresh) {
    {
      const std::lock_guard lock(mutex);
      (fresh ? accepted[slot] : dropped[slot]).fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool complete() noexcept {
    return accepted[kHw].load(std::memory_order_acquire) +
                   dropped[kHw].load(std::memory_order_acquire) >=
               kFrameCount &&
           accepted[kAct].load(std::memory_order_acquire) +
                   dropped[kAct].load(std::memory_order_acquire) >=
               kFrameCount;
  }

  [[nodiscard]] static bool wait_until_complete() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return complete(); });
  }

  // Frames recorded across both chains — the --manual harness uses this to
  // advance virtual time in lock-step with observed scheduler progress.
  [[nodiscard]] static std::size_t total_recorded() noexcept {
    return accepted[kHw].load(std::memory_order_acquire) +
           dropped[kHw].load(std::memory_order_acquire) +
           accepted[kAct].load(std::memory_order_acquire) +
           dropped[kAct].load(std::memory_order_acquire);
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<std::atomic<std::size_t>, 2> accepted{};
  static inline std::array<std::atomic<std::size_t>, 2> dropped{};
  static inline std::atomic<std::size_t> restamped{};
  static inline std::atomic<std::size_t> domain_mismatch{};
  // What the hardware-stamp source actually emitted, indexed by frame_id, so
  // the sink can prove the forwarder did not restamp anything.
  static inline std::array<std::atomic<std::int64_t>, kFrameCount> hw_stamps{};
};

// Source #1: a camera-shaped driver. A real driver hands you a raw hardware
// timestamp (an integer from the device) that predates the callback. at_ns()
// labels that integer with the activation clock's domain without reading or
// changing the clock, producing the physical capture coordinate. Runtime keeps
// that coordinate distinct from the monotonic TagTime used to order dispatch.
class HardwareStampSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFramePeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= kFrameCount) {
      return {};  // exhausted: keep ticking idly; main() observes and stops the run
    }
    const std::uint32_t frame = next_++;
    const holoscan::TimePoint activation = context.activation_time();
    // The device's own exposure timestamp — one exposure latency earlier, and
    // 5 s stale for the frame that sat in the stalled driver queue. A raw
    // integer, exactly as a driver would hand it over; it travels as payload.
    std::int64_t device_ns = activation.timestamp_ns - kExposureLatency.count();
    if (frame == kStaleFrame) {
      device_ns -= kStaleOffset.count();  // simulate a stalled driver queue
    }
    const holoscan::TimePoint capture = context.clock().at_ns(device_ns);
    TimeObservation::hw_stamps[frame].store(capture.timestamp_ns, std::memory_order_release);
    auto emitted =
        out.emit(device_ns,
                 holoscan::EmitOptions{.capture_time = capture, .frame_id = std::uint64_t{frame}});
    if (!emitted) {
      std::cerr << "hw-source emit frame=" << frame
                << " failed: " << holoscan::runtime_error_message(emitted.error()) << '\n';
    }
    return emitted;
  }

  holoscan::Output<std::int64_t> out;

 private:
  std::uint32_t next_{};
};

// Source #2: has no device timestamp at all, so its payload carries the only
// truthful time it has — the activation coordinate itself. Never fabricate an
// earlier acquisition time you do not have; downstream, this chain shows a
// capture age near zero.
class ActivationStampSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFramePeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= kFrameCount) {
      return {};
    }
    const std::uint32_t frame = next_++;
    const holoscan::TimePoint capture = context.activation_time();
    auto emitted =
        out.emit(capture.timestamp_ns,
                 holoscan::EmitOptions{.capture_time = capture, .frame_id = std::uint64_t{frame}});
    if (!emitted) {
      std::cerr << "act-source emit frame=" << frame
                << " failed: " << holoscan::runtime_error_message(emitted.error()) << '\n';
    }
    return emitted;
  }

  holoscan::Output<std::int64_t> out;

 private:
  std::uint32_t next_{};
};

// The forward-don't-restamp discipline: an intermediate operator that passes a
// sample along re-emits the *sample* (the Sample overload of emit deliberately
// has no EmitOptions parameter) so capture_time and its source clock ride
// through unchanged. Runtime replaces only the publication-side fields.
// Restamping here would silently turn "when the pixels were captured" into
// "when this operator ran" for every consumer downstream.
class ForwardGate final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
    spec.output(out, "out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});  // OnEach pins its own driver sample
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (sample->metadata.frame_id < kPrintedFrames) {
      // Tag identifies *this activation of this operator*; capture_time
      // identifies *the sample*. The tag changes at every hop, the capture
      // time must not.
      const holoscan::Tag tag = context.tag();
      const holoscan::TagTime tag_time = context.tag_time();
      std::cout << "forward: frame=" << sample->metadata.frame_id << " tag=g" << tag.generation
                << ":s" << tag.sequence << " tag_time=" << tag_time.logical_ns << "ns/m"
                << tag_time.microstep << " capture_ns=" << sample->metadata.capture_timestamp_ns
                << " (forwarded unchanged)\n";
    }
    return out.emit(std::move(*sample));  // forward the sample — do not restamp
  }

  holoscan::Input<std::int64_t> in;
  holoscan::Output<std::int64_t> out;
};

// The sink *measures* and *labels*: clock.now() measures and is fallible (a
// clock generation can be gone or unreadable); clock.at_ns() labels the raw
// device integer from the payload with this clock's domain and is infallible —
// no clock read happens. Subtraction is only meaningful inside one clock
// domain, so the sink checks clock_id before every subtraction. Freshness is
// an application policy, so it lives in application code — the runtime
// delivers the sample; the sink decides 5-second-old pixels are trash.
class AgeGaugeSink final : public holoscan::Operator<> {
 public:
  AgeGaugeSink(std::string label, std::size_t slot) noexcept
      : label_(std::move(label)), slot_(slot) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    HOLOSCAN_TRY(const holoscan::TimePoint now, context.clock().now());
    const holoscan::TimePoint capture = sample->metadata.capture_time();
    if (!capture.valid() || capture.clock_id != now.clock_id) {
      // Never subtract coordinates from different clock domains.
      TimeObservation::domain_mismatch.fetch_add(1U, std::memory_order_relaxed);
      TimeObservation::record(slot_, false);
      return {};
    }
    if (slot_ == TimeObservation::kHw && sample->metadata.frame_id < kFrameCount &&
        capture.timestamp_ns !=
            TimeObservation::hw_stamps[sample->metadata.frame_id].load(std::memory_order_acquire)) {
      TimeObservation::restamped.fetch_add(1U, std::memory_order_relaxed);
    }
    // at_ns() labels the raw device integer with this clock's domain — no
    // clock read, no failure mode. Then check the domain before subtracting.
    const holoscan::TimePoint device = context.clock().at_ns(sample->data);
    if (device.clock_id != capture.clock_id) {
      TimeObservation::domain_mismatch.fetch_add(1U, std::memory_order_relaxed);
      TimeObservation::record(slot_, false);
      return {};
    }
    // The payload and metadata describe the same physical timestamp. Their
    // delta is deterministic data validation, not a live clock measurement.
    const std::int64_t device_delta_ns = capture.timestamp_ns - device.timestamp_ns;
    // Capture age is a genuine measurement against the live clock (varies).
    const std::int64_t age_ns = now.timestamp_ns - capture.timestamp_ns;
    const bool fresh = age_ns <= kMaxCaptureAge.count();
    if (!fresh) {
      std::cout << label_ << ": DROP frame=" << sample->metadata.frame_id
                << " capture_age_ms=" << age_ns / 1'000'000
                << " (over the 500 ms freshness budget)\n";
    } else if (sample->metadata.frame_id < kPrintedFrames) {
      const holoscan::Tag tag = context.tag();
      std::cout << label_ << ": frame=" << sample->metadata.frame_id << " tag=g" << tag.generation
                << ":s" << tag.sequence << " device_delta_us=" << device_delta_ns / 1'000
                << " capture_age_us=" << age_ns / 1'000 << " capture_ns=" << capture.timestamp_ns
                << '\n';
    }
    TimeObservation::record(slot_, fresh);
    return {};
  }

  holoscan::Input<std::int64_t> in;

 private:
  std::string label_;
  std::size_t slot_;
};

[[nodiscard]] holoscan::ExecutionPlan build_plan(bool manual_mode,
                                                 std::shared_ptr<holoscan::ManualClock>& manual) {
  // The graph is byte-for-byte identical in both modes; only the clock
  // declaration differs. That is the whole point of the --manual variant.
  holoscan::Graph graph{"time-and-clocks"};
  const auto hw_source = graph.op<HardwareStampSource>("hw-source");
  const auto act_source = graph.op<ActivationStampSource>("act-source");
  const auto forward = graph.op<ForwardGate>("forward");
  const auto hw_sink =
      graph.op<AgeGaugeSink>("sink-hw", std::string{"sink-hw"}, TimeObservation::kHw);
  const auto act_sink =
      graph.op<AgeGaugeSink>("sink-act", std::string{"sink-act"}, TimeObservation::kAct);
  graph.add_flow(hw_source->out, forward->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(forward->out, hw_sink->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(act_source->out, act_sink->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("hardware-time-path").add(hw_source).add(forward).add(hw_sink);
  graph.partition("activation-time-path").add(act_source).add(act_sink);

  // bind_shared_clock() is the one clock-authoring verb that retains the live
  // instance you constructed (add_clock() replays a factory and discards your
  // object), so main() keeps a handle to the very clock the graph schedules
  // against — proven below by advancing the epoch before the run and reading
  // the final virtual time after it.
  if (manual_mode) {
    manual = std::make_shared<holoscan::ManualClock>(0);
    manual->advance_by(1s);  // pre-run epoch shift: safe while nothing runs
    graph.set_default_clock(graph.bind_shared_clock("wall", manual));
  } else {
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("wall"));
  }
  return holoscan::compile(graph);
}

void drive_manual_clock(const std::shared_ptr<holoscan::ManualClock>& manual) {
  // A ManualClock bound into a session does not advance itself: without
  // these advance_by() calls the graph fires exactly one tick per source at
  // the start coordinate and then idles forever. The harness therefore *is*
  // the time authority. The admitted kCoalesceLateTicks policy would collapse
  // a multi-period jump to the greatest due tick; this example instead feeds
  // time in lock-step so one authored period produces one observable frame.
  const auto harness_deadline = std::chrono::steady_clock::now() + 8s;
  while (!TimeObservation::complete() && std::chrono::steady_clock::now() < harness_deadline) {
    const std::size_t before = TimeObservation::total_recorded();
    manual->advance_by(kFramePeriod);
    const auto tick_deadline = std::chrono::steady_clock::now() + 200ms;
    while (TimeObservation::total_recorded() < before + 2U && !TimeObservation::complete() &&
           std::chrono::steady_clock::now() < tick_deadline) {
      std::this_thread::sleep_for(1ms);
    }
  }
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diag : session.diagnostics()) {
    std::cerr << "  diagnostic: " << diag.token << " severity=" << static_cast<int>(diag.severity)
              << " required=" << diag.required << " provided=" << diag.provided << '\n';
  }
}

[[nodiscard]] bool validate_run(bool complete, const holoscan::RunSession& session,
                                std::string_view variant) {
  if (!complete) {
    std::cerr << "time_and_clocks[" << variant << "]: timed out waiting for " << kFrameCount
              << " frames per chain (termination=" << static_cast<int>(session.termination())
              << " hw=" << TimeObservation::accepted[TimeObservation::kHw].load() << '+'
              << TimeObservation::dropped[TimeObservation::kHw].load()
              << " act=" << TimeObservation::accepted[TimeObservation::kAct].load() << '+'
              << TimeObservation::dropped[TimeObservation::kAct].load() << ")\n";
    print_diagnostics(session);
    return false;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "time_and_clocks[" << variant << "]: unexpected termination state (termination="
              << static_cast<int>(session.termination()) << ")\n";
    print_diagnostics(session);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool manual_mode = false;
  if (argc == 2 && std::string_view(argv[1]) == "--manual") {
    manual_mode = true;
  } else if (argc != 1) {
    std::cerr << "usage: time_and_clocks [--manual]\n";
    return 3;
  }

  TimeObservation::reset();

  std::shared_ptr<holoscan::ManualClock> manual;
  const holoscan::ExecutionPlan plan = build_plan(manual_mode, manual);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);

  if (manual_mode) {
    drive_manual_clock(manual);
  }

  const bool complete = TimeObservation::wait_until_complete();
  session.request_stop();
  session.wait();

  if (manual_mode) {
    // The live handle still belongs to main(): read how far the virtual clock
    // actually ran (roughly 12 frames x 5 ms past the 1 s pre-run epoch).
    std::cout << "manual: virtual clock ended at " << manual->timestamp()
              << " ns; no wall-clock frame period was waited on\n";
  }

  const char* const variant = manual_mode ? "manual" : "realtime";
  if (!validate_run(complete, session, variant)) {
    return 2;
  }
  const bool preserved = TimeObservation::restamped.load(std::memory_order_relaxed) == 0U &&
                         TimeObservation::domain_mismatch.load(std::memory_order_relaxed) == 0U;

  std::cout << "time_and_clocks[" << variant
            << "]: hw accepted=" << TimeObservation::accepted[TimeObservation::kHw].load()
            << " dropped=" << TimeObservation::dropped[TimeObservation::kHw].load()
            << " act accepted=" << TimeObservation::accepted[TimeObservation::kAct].load()
            << " forward_preserved=" << (preserved ? "yes" : "no") << " termination=stopped\n";
  return preserved ? 0 : 2;
}
