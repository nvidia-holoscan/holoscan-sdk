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
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

// This example uses Holoscan's experimental expert-level raw-POD escape hatch.
// Prefer a generated FlatBuffers schema for ordinary application-defined messages.
//
// The application's own plain-old-data control message. Members are ordered by
// DECREASING alignment (8 -> 4 -> 2 -> 1) so the compiler inserts zero padding:
// the POD contract requires the declared fields to tile the object exactly, and
// there is no pad-field mechanism to paper over an alignment gap.
//
// Two deliberate substitutions keep the struct admissible:
//   - `exposure_locked` is a std::uint8_t carrying 0/1 — `bool` is not admitted
//     as a POD field (its object representation is not pinned by the wire
//     contract), although plain `bool` remains fine as a standalone scalar port.
//   - `reserved` makes the tail bytes explicit: alignof(FusionControl) is 8, so
//     sizeof must be a multiple of 8; without these named bytes the compiler
//     would add invisible tail padding and the layout check would reject it.
struct FusionControl {
  std::uint64_t capture_timestamp_ns{};       // alignment 8
  std::uint32_t frame_id{};                   // alignment 4
  std::int32_t exposure_delta{};              // alignment 4
  float confidence{};                         // alignment 4
  std::array<std::uint16_t, 2> roi_center{};  // alignment 2 (fixed arrays are admitted)
  std::uint8_t exposure_locked{};             // alignment 1 (bool replacement: 0 or 1)
  std::array<std::uint8_t, 7> reserved{};     // alignment 1 (explicit, named tail bytes)
};

static_assert(sizeof(FusionControl) == 32U, "FusionControl must stay padding-free");
static_assert(alignof(FusionControl) == 8U);
static_assert(std::is_standard_layout_v<FusionControl>);
static_assert(std::is_trivially_copyable_v<FusionControl>);

// Cautionary counter-example — deliberately NOT compiled. Swapping the order so
// a 1-byte member precedes a 4-byte member forces implicit padding:
//
//   struct PaddedControl {
//     std::uint8_t exposure_locked{};  // bytes [0, 1)
//     // 3 bytes of implicit compiler padding here — bytes [1, 4) belong to no field
//     std::uint32_t frame_id{};        // bytes [4, 8)
//   };
//
// No ChannelPodLayout can describe PaddedControl: `pod_field` offsets must be
// contiguous (each field starts exactly where the previous one ended), so the
// Boolean inspection through `complete_channel_pod_contract<PaddedControl>()`
// evaluates false. Typed authoring invokes
// `assert_complete_channel_pod_contract<PaddedControl>()` when
// `OperatorSpec::output` materializes the `Output<PaddedControl>` identity, so
// that declaration fails with a focused diagnostic. An untrusted pointer-free
// record that bypasses typed authoring is still rejected during composition:
// GRAPH_POD_LAYOUT_INVALID (ErrorCode::kInvalidArgument).

namespace holoscan {

// Identity: a globally scoped dotted name plus a nonzero version. The guide's
// inline walkthrough ships version 1 with three fields; this example extends
// the payload (timestamp, ROI, lock flag), so the version bumps to 2 — the same
// name + version must always mean the same bytes, even across processes.
template <>
struct ChannelTypeIdentity<FusionControl> {
  static constexpr std::string_view name = "com.example.FusionControl";
  static constexpr std::uint32_t version = 2U;
};

// Layout: ordered fields that cover every object byte exactly once. Each
// pod_field pairs the member pointer with its real offsetof(...) value; the
// compile-time contract verifies the pointer actually selects those bytes.
template <>
struct ChannelPodLayout<FusionControl> {
  static constexpr auto fields = pod_fields(
      pod_field("capture_timestamp_ns", &FusionControl::capture_timestamp_ns,
                offsetof(FusionControl, capture_timestamp_ns)),
      pod_field("frame_id", &FusionControl::frame_id, offsetof(FusionControl, frame_id)),
      pod_field("exposure_delta", &FusionControl::exposure_delta,
                offsetof(FusionControl, exposure_delta)),
      pod_field("confidence", &FusionControl::confidence, offsetof(FusionControl, confidence)),
      pod_field("roi_center", &FusionControl::roi_center, offsetof(FusionControl, roi_center)),
      pod_field("exposure_locked", &FusionControl::exposure_locked,
                offsetof(FusionControl, exposure_locked)),
      pod_field("reserved", &FusionControl::reserved, offsetof(FusionControl, reserved)));
};

}  // namespace holoscan

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kControlPeriod = 20ms;
constexpr std::uint32_t kControlLimit = 6U;

// Operators run on runtime threads; main() must not poll or sleep. The consumer
// records every decoded struct here, and main() blocks on the condition
// variable until the expected count arrives (or a 10s safety cap expires).
struct ControlObservation {
  static void reset() noexcept {
    count.store(0U, std::memory_order_relaxed);
    last_frame.store(0U, std::memory_order_relaxed);
  }

  static void record(const FusionControl& value) {
    {
      const std::lock_guard lock(mutex);
      last_frame.store(value.frame_id, std::memory_order_relaxed);
      count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint32_t> last_frame{};
};

// The producer treats FusionControl exactly like a scalar payload: declare an
// Output<FusionControl>, state a temporal contract, emit values. No base class,
// no serialization code — the runtime derives the codec from the two traits.
class FusionStage final : public holoscan::Operator<> {
 public:
  explicit FusionStage(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(control, "control");  // max_emits_per_compute defaults to 1
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kControlPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= limit_) {
      return {};  // exhausted: keep ticking idly; main() observes the count and stops the run
    }
    const std::uint32_t frame = ++next_;
    // The payload field below remains an application-level coordinate. Exact
    // selection also requires genuine Sample metadata, so stamp the public
    // EmitOptions capture coordinate from this activation's admitted clock.
    const FusionControl value{
        .capture_timestamp_ns = static_cast<std::uint64_t>(frame) * 20'000'000ULL,
        .frame_id = frame,
        .exposure_delta = static_cast<std::int32_t>(frame) - 3,
        .confidence = 0.90F,
        .roi_center = {static_cast<std::uint16_t>(960U + frame), 540U},
        .exposure_locked = static_cast<std::uint8_t>(frame % 2U),
        .reserved = {},  // explicit tail bytes stay zero
    };
    return control.emit(value, holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<FusionControl> control;

 private:
  std::uint32_t limit_;
  std::uint32_t next_{};
};

// The consumer mirrors the producer with an Input<FusionControl>. receive()
// yields an expected<Sample<FusionControl>, Error> whose .data member is the
// struct value — already decoded, ready to use field by field.
class ExposureController final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(control, "control").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // Spell out the ordinary input-ready default to make its transaction
    // visible: an arrival proposes the activation, Exact selects one sample,
    // and Skip suppresses compute if selection cannot produce that sample.
    result.trigger(holoscan::OnAnyInput{})
        .input(control, holoscan::Exact{})
        .on_missing(control, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = control.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const FusionControl& value = sample->data;
    // Sole printer while the graph runs: one line of decoded fields per sample.
    // uint8_t streams as a character, so widen it before printing.
    std::cout << "FusionControl frame=" << value.frame_id << " t_ns=" << value.capture_timestamp_ns
              << " exposure_delta=" << value.exposure_delta << " confidence=" << std::fixed
              << std::setprecision(2) << value.confidence << " roi=" << value.roi_center[0] << 'x'
              << value.roi_center[1] << " locked=" << static_cast<unsigned>(value.exposure_locked)
              << '\n';
    ControlObservation::record(value);
    return {};
  }

  holoscan::Input<FusionControl> control;
};

// Labeled post-stop failure report: print the retained runtime diagnostics so
// a failed run explains itself instead of just timing out silently.
void print_session_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << "pod_payloads: diagnostic token=" << diagnostic.token
              << " severity=" << static_cast<unsigned>(diagnostic.severity)
              << " service=" << diagnostic.service << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << " vertex=" << diagnostic.source.vertex.value
              << '\n';
  }
}

}  // namespace

int main() {
  ControlObservation::reset();

  // Wiring is byte-for-byte the same as a scalar pipeline: the POD type rides
  // the ordinary typed ports and add_flow() — no codec opt-in anywhere.
  holoscan::Graph graph{"pod-payloads"};
  const auto fusion = graph.op<FusionStage>("fusion", kControlLimit);
  const auto controller = graph.op<ExposureController>("controller");
  graph.add_flow(
      fusion->control, controller->control, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = ControlObservation::wait_for(kControlLimit);
  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "pod_payloads: timed out waiting for " << kControlLimit << " control samples"
              << " (termination=" << static_cast<unsigned>(session.termination()) << ")\n";
    print_session_diagnostics(session);
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "pod_payloads: unexpected termination state (termination="
              << static_cast<unsigned>(session.termination()) << ")\n";
    print_session_diagnostics(session);
    return 2;
  }

  std::cout << "pod_payloads: decoded " << ControlObservation::count.load(std::memory_order_relaxed)
            << " FusionControl samples (last frame="
            << ControlObservation::last_frame.load(std::memory_order_relaxed)
            << ", size=" << sizeof(FusionControl) << " bytes, zero padding) termination=stopped\n";
  return 0;
}
