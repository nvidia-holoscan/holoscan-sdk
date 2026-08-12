/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The anatomy operator intentionally uses OnEach: one published snapshot is the
// semantic driver for one anatomy activation. Its BurstSource uses OnClock to
// generate the independent burst and side inputs that demonstrate DrainAll and
// LatestWithin without acquiring trigger authority. Runtime tracks their capture
// coordinates separately from monotonic TagTime scheduling order, so forwarding a
// selected Sample preserves physical provenance even when that capture coordinate
// predates the callback.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
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

// Every 20ms the source publishes one burst of four frames on the batch
// lane, one snapshot of the newest frame on the driver lane, and -- on every
// OTHER burst only -- a copy of that snapshot on the side lane. The anatomy
// operator is driven by the snapshot lane, so activations alternate
// deterministically between "side data selected" and the SafeDefault branch:
// LatestWithin admission is consume-once, and a 25-per-second side lane
// under a 50-per-second driver leaves every second activation with nothing
// new to select. The period is test-fast: the lesson is the accessor
// anatomy, not the rate.
constexpr std::chrono::nanoseconds kSourcePeriod = 20ms;
constexpr std::uint32_t kBurstSize = 4U;
constexpr std::uint32_t kSideLaneEvery = 2U;
constexpr std::uint64_t kRestampFrameBase = 1000000U;
constexpr std::chrono::nanoseconds kLongComputeBudget = 2ms;
constexpr std::size_t kObservationTarget = 12U;

[[nodiscard]] constexpr std::string_view state_label(holoscan::InputState state) noexcept {
  switch (state) {
    case holoscan::InputState::kSelected:
      return "selected";
    case holoscan::InputState::kReused:
      return "reused";
    case holoscan::InputState::kSafeDefault:
      return "safe-default";
    case holoscan::InputState::kAbsent:
      return "absent";
    case holoscan::InputState::kNotSelected:
      return "not-selected";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view reason_label(holoscan::SelectionReason reason) noexcept {
  switch (reason) {
    case holoscan::SelectionReason::kNone:
      return "none";
    case holoscan::SelectionReason::kNeverSeen:
      return "never-seen";
    case holoscan::SelectionReason::kTooStale:
      return "too-stale";
    case holoscan::SelectionReason::kEndOfStream:
      return "end-of-stream";
    case holoscan::SelectionReason::kNotConnected:
      return "not-connected";
    default:
      return "other";
  }
}

// Routes the two emit flavors to their verifying sinks. A ctor argument must
// be an owned copyable value, and a plain enum satisfies that.
enum class Route : std::uint8_t { kPreserved = 0, kRestamped = 1 };

// Operators run on runtime threads; every observation lands in these static
// structs and main() prints the labeled summary only after session.wait().
struct AnatomyObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    has_sample_snapshot = false;
    receive_value = 0U;
    frame_id = 0U;
    capture_ns = 0;
    sequence = 0U;
    metadata_valid = false;
    data_value = 0U;
    has_batch_snapshot = false;
    first_batch_size = 0U;
    batch_first_frame = 0U;
    batch_last_frame = 0U;
    selected_ticks = 0U;
    missing_ticks = 0U;
    first_missing = holoscan::InputSelection{};
    drained_total = 0U;
    empty_batches = 0U;
    cancellation_seen.store(false, std::memory_order_relaxed);
  }

  static void record_missing(holoscan::InputSelection selection) {
    const std::lock_guard lock(mutex);
    if (missing_ticks == 0U) {
      first_missing = selection;
    }
    ++missing_ticks;
  }

  static void record_scalar(std::uint32_t value, const holoscan::SampleMetadata& metadata,
                            std::uint32_t payload_value) {
    const std::lock_guard lock(mutex);
    ++selected_ticks;
    if (!has_sample_snapshot) {
      has_sample_snapshot = true;
      receive_value = value;
      frame_id = metadata.frame_id;
      capture_ns = metadata.capture_timestamp_ns;
      sequence = metadata.sequence;
      metadata_valid = metadata.valid();
      data_value = payload_value;
    }
  }

  static void record_batch(std::size_t size, std::uint64_t first_frame, std::uint64_t last_frame) {
    const std::lock_guard lock(mutex);
    drained_total += size;
    if (size == 0U) {
      ++empty_batches;
      return;
    }
    if (!has_batch_snapshot) {
      has_batch_snapshot = true;
      first_batch_size = size;
      batch_first_frame = first_frame;
      batch_last_frame = last_frame;
    }
  }

  static inline std::mutex mutex;
  static inline bool has_sample_snapshot{};
  static inline std::uint32_t receive_value{};
  static inline std::uint64_t frame_id{};
  static inline std::int64_t capture_ns{};
  static inline std::uint64_t sequence{};
  static inline bool metadata_valid{};
  static inline std::uint32_t data_value{};
  static inline bool has_batch_snapshot{};
  static inline std::size_t first_batch_size{};
  static inline std::uint64_t batch_first_frame{};
  static inline std::uint64_t batch_last_frame{};
  static inline std::size_t selected_ticks{};
  static inline std::size_t missing_ticks{};
  static inline holoscan::InputSelection first_missing{};
  static inline std::size_t drained_total{};
  static inline std::size_t empty_batches{};
  static inline std::atomic<bool> cancellation_seen{};
};

struct SinkObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    preserved_count = 0U;
    restamped_count = 0U;
    violations = 0U;
  }

  static void record(Route route, const holoscan::SampleMetadata& metadata) {
    {
      const std::lock_guard lock(mutex);
      const bool fresh_frame = metadata.frame_id >= kRestampFrameBase;
      const bool stamped = metadata.capture_timestamp_ns != 0;
      if (route == Route::kPreserved) {
        ++preserved_count;
        // Forwarded samples must still carry the source's frame identity
        // (1..N, far below the restamp base) and its capture time.
        if (metadata.frame_id == 0U || fresh_frame || !stamped) {
          ++violations;
        }
      } else {
        ++restamped_count;
        // Restamped samples must carry the anatomy operator's fresh identity.
        if (!fresh_frame || !stamped) {
          ++violations;
        }
      }
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] {
      return preserved_count >= expected && restamped_count >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t preserved_count{};
  static inline std::size_t restamped_count{};
  static inline std::size_t violations{};
};

// Emits, per 20ms tick: one burst of kBurstSize freshly stamped frames on
// the batch lane, a copy of the newest frame on the side lane on every other
// burst (emitted BEFORE the driver so a driver arrival always finds its side
// copy already queued), and the driver snapshot last. The burst output opts
// into burst emission explicitly: max_emits_per_compute defaults to 1, and
// the compiler budgets downstream queue capacity from this declaration --
// this example is the one place the builder call belongs.
class BurstSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out").max_emits_per_compute(kBurstSize);
    spec.output(side_out, "side_out");
    spec.output(snapshot_out, "snapshot_out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kSourcePeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    ++burst_index_;
    for (std::uint32_t index = 0U; index < kBurstSize; ++index) {
      const std::uint64_t frame = ++next_frame_;
      // Fresh stamping: EmitOptions supplies genuine capture time and frame
      // identity for a brand-new sample; Runtime fills the publication
      // sequence/time fields itself.
      auto emitted = out.emit(
          static_cast<std::uint32_t>(frame),
          holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = frame});
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    const holoscan::EmitOptions snapshot_options{.capture_time = context.activation_time(),
                                                 .frame_id = next_frame_};
    if (burst_index_ % kSideLaneEvery == 0U) {
      // Publishing every other burst intentionally leaves alternating driver
      // activations without a new side sample, exercising SafeDefault.
      auto side = side_out.emit(static_cast<std::uint32_t>(next_frame_), snapshot_options);
      if (!side) {
        return holoscan::make_unexpected(std::move(side).error());
      }
    }
    auto snapshot = snapshot_out.emit(static_cast<std::uint32_t>(next_frame_), snapshot_options);
    if (!snapshot) {
      return holoscan::make_unexpected(std::move(snapshot).error());
    }
    return {};
  }

  holoscan::Output<std::uint32_t> out;
  holoscan::Output<std::uint32_t> side_out;
  holoscan::Output<std::uint32_t> snapshot_out;

 private:
  std::uint64_t next_frame_{};
  std::uint64_t burst_index_{};
};

// The lesson operator: an OnEach snapshot driver plus two side inputs fed by
// the SAME upstream source, each read through a different accessor, plus
// both emit flavors and a cooperative cancellation poll inside a
// deliberately long compute.
class AnatomyOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(sample_in, "sample_in").queue_depth(8U);
    spec.input(data_in, "data_in").queue_depth(8U);
    spec.input(batch_in, "batch_in").queue_depth(32U);
    spec.output(preserved, "preserved");
    spec.output(restamped, "restamped");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // OnEach{sample_in} drives the operator once per snapshot; the driver is
    // guaranteed selected and takes no selector and no on_missing. DrainAll
    // batches everything that arrived since the last activation; it cannot
    // fail -- an empty batch is a valid selection -- so it takes no
    // on_missing either. The LatestWithin side selector CAN fail: scalar
    // admission is consume-once, and the side lane publishes every other
    // burst, so every second activation finds nothing new. It therefore
    // declares SafeDefault: compute still runs, and selection() reports the
    // branch. (Skip would suppress the whole activation instead.)
    result.trigger(holoscan::OnEach{sample_in})
        .input(data_in, holoscan::LatestWithin{.max_age = 200ms})
        .on_missing(data_in, holoscan::SafeDefault{})
        .input(batch_in, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    // (1) receive(): the full owning Sample<T> -- payload plus
    // SampleMetadata -- carrying a finite read claim so it may outlive this
    // callback. The driver input is always selected, so this read cannot
    // find the input missing.
    auto sample = sample_in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }

    // (2) selected_batch(): the DrainAll view -- immutable,
    // activation-scoped, random access. An empty view is success, not an
    // error.
    auto batch = batch_in.selected_batch();
    if (!batch) {
      return holoscan::make_unexpected(std::move(batch).error());
    }
    std::uint64_t first_frame = 0U;
    std::uint64_t last_frame = 0U;
    for (const holoscan::Sample<std::uint32_t>& drained : *batch) {
      if (first_frame == 0U) {
        first_frame = drained.metadata.frame_id;
      }
      last_frame = drained.metadata.frame_id;
    }
    AnatomyObservation::record_batch(batch->size(), first_frame, last_frame);

    // (3) Introspection before the side read: the selection was frozen
    // before compute ran, and on the SafeDefault branch no value exists --
    // receive_data() here would return an accessor error. Record the frozen
    // selection (state + reason) and continue: one side input's silence
    // does not silence the operator.
    if (data_in.is_missing()) {
      AnatomyObservation::record_missing(data_in.selection());
    } else {
      // (4) receive_data(): payload only. No metadata, no read claim.
      // Reading a frozen selection is not a queue pop. On alternating
      // activations the side lane's copy of the same snapshot is selected,
      // so its payload matches the driver sample exactly.
      auto payload = data_in.receive_data();
      if (!payload) {
        return holoscan::make_unexpected(std::move(payload).error());
      }
      AnatomyObservation::record_scalar(sample->data, sample->metadata, *payload);
    }

    // (5) Forwarding emit: emit(Sample<T>) preserves capture time, source
    // clock, frame id, trace, and flags as forwarding evidence. The overload
    // deliberately has no EmitOptions parameter -- the sample already
    // carries the semantic metadata to forward. The driver sample is used
    // because it is the artifact this operator transforms; its physical
    // capture coordinate remains distinct from downstream TagTime order.
    const std::uint32_t value = sample->data;
    auto forwarded = preserved.emit(std::move(*sample));
    if (!forwarded) {
      return holoscan::make_unexpected(std::move(forwarded).error());
    }

    // (6) Restamping emit: emit(T, EmitOptions) starts a fresh metadata
    // record; this operator claims new frame identity and stamps the
    // activation's own frozen time.
    auto fresh =
        restamped.emit(value,
                       holoscan::EmitOptions{.capture_time = context.activation_time(),
                                             .frame_id = kRestampFrameBase + ++restamp_count_});
    if (!fresh) {
      return holoscan::make_unexpected(std::move(fresh).error());
    }

    // (7) A long compute must poll cooperative cancellation. This bounded
    // loop stands in for chunked heavy work; each chunk checks the
    // callback-scoped latch so request_stop() never waits out the full loop.
    const auto deadline = std::chrono::steady_clock::now() + kLongComputeBudget;
    while (std::chrono::steady_clock::now() < deadline) {
      if (context.cancellation().stop_requested()) {
        AnatomyObservation::cancellation_seen.store(true, std::memory_order_relaxed);
        break;
      }
      // One chunk of real work would run here.
    }
    return {};
  }

  holoscan::Input<std::uint32_t> sample_in;
  holoscan::Input<std::uint32_t> data_in;
  holoscan::Input<std::uint32_t> batch_in;
  holoscan::Output<std::uint32_t> preserved;
  holoscan::Output<std::uint32_t> restamped;

 private:
  std::uint64_t restamp_count_{};
};

// Terminal observer for one of the two emit flavors; validates that the
// metadata that arrives matches the route's stamping story.
class MetadataSink final : public holoscan::Operator<> {
 public:
  explicit MetadataSink(Route route) noexcept : route_(route) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SinkObservation::record(route_, sample->metadata);
    return {};
  }

  holoscan::Input<std::uint32_t> input;

 private:
  Route route_;
};

}  // namespace

// REQ: SYSTEM-014
int main() {
  AnatomyObservation::reset();
  SinkObservation::reset();

  holoscan::Graph graph{"operator-anatomy"};
  const auto source = graph.op<BurstSource>("source");
  const auto anatomy = graph.op<AnatomyOperator>("anatomy");
  const auto preserved_sink = graph.op<MetadataSink>("preserved_sink", Route::kPreserved);
  const auto restamped_sink = graph.op<MetadataSink>("restamped_sink", Route::kRestamped);

  // One source feeds all three anatomy inputs. The snapshot lane is the
  // OnEach driver, so its edge keeps the default blocking policy -- a
  // dropped driver arrival would be a lost activation. The side lane is
  // observational (LatestWithin admits at most one arrival per activation),
  // so its edge opts into bounded kDropOldest. The fast burst lane goes ONLY to the
  // DrainAll input, which empties it every activation, sized for a few
  // bursts of headroom.
  graph.add_flow(
      source->snapshot_out, anatomy->sample_in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->side_out,
                 anatomy->data_in,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 8U});
  graph.add_flow(source->out, anatomy->batch_in, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.add_flow(
      anatomy->preserved, preserved_sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(
      anatomy->restamped, restamped_sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = SinkObservation::wait_for(kObservationTarget);
  session.request_stop();
  session.wait();

  // Copy every observation under its lock before judging or printing.
  std::size_t preserved_count = 0U;
  std::size_t restamped_count = 0U;
  std::size_t violations = 0U;
  {
    const std::lock_guard lock(SinkObservation::mutex);
    preserved_count = SinkObservation::preserved_count;
    restamped_count = SinkObservation::restamped_count;
    violations = SinkObservation::violations;
  }
  bool has_sample_snapshot = false;
  bool has_batch_snapshot = false;
  std::uint32_t receive_value = 0U;
  std::uint64_t frame_id = 0U;
  std::int64_t capture_ns = 0;
  std::uint64_t sequence = 0U;
  bool metadata_valid = false;
  std::uint32_t data_value = 0U;
  std::size_t first_batch_size = 0U;
  std::uint64_t batch_first_frame = 0U;
  std::uint64_t batch_last_frame = 0U;
  std::size_t selected_ticks = 0U;
  std::size_t missing_ticks = 0U;
  holoscan::InputSelection first_missing{};
  std::size_t drained_total = 0U;
  std::size_t empty_batches = 0U;
  {
    const std::lock_guard lock(AnatomyObservation::mutex);
    has_sample_snapshot = AnatomyObservation::has_sample_snapshot;
    has_batch_snapshot = AnatomyObservation::has_batch_snapshot;
    receive_value = AnatomyObservation::receive_value;
    frame_id = AnatomyObservation::frame_id;
    capture_ns = AnatomyObservation::capture_ns;
    sequence = AnatomyObservation::sequence;
    metadata_valid = AnatomyObservation::metadata_valid;
    data_value = AnatomyObservation::data_value;
    first_batch_size = AnatomyObservation::first_batch_size;
    batch_first_frame = AnatomyObservation::batch_first_frame;
    batch_last_frame = AnatomyObservation::batch_last_frame;
    selected_ticks = AnatomyObservation::selected_ticks;
    missing_ticks = AnatomyObservation::missing_ticks;
    first_missing = AnatomyObservation::first_missing;
    drained_total = AnatomyObservation::drained_total;
    empty_batches = AnatomyObservation::empty_batches;
  }
  const bool cancellation_seen =
      AnatomyObservation::cancellation_seen.load(std::memory_order_relaxed);

  const bool scalar_consistent = !has_sample_snapshot || data_value == receive_value;
  const bool healthy = complete && session.termination() == holoscan::RunTermination::kStopped &&
                       session.diagnostics().empty() && has_sample_snapshot && has_batch_snapshot &&
                       drained_total > 0U && scalar_consistent && metadata_valid &&
                       missing_ticks > 0U && violations == 0U;
  if (!healthy) {
    std::cerr << "operator_anatomy failed: complete=" << complete
              << " termination=" << static_cast<int>(session.termination())
              << " diagnostics=" << session.diagnostics().size()
              << " sample_snapshot=" << has_sample_snapshot
              << " batch_snapshot=" << has_batch_snapshot << " drained=" << drained_total
              << " selected=" << selected_ticks << " missing=" << missing_ticks
              << " scalar_consistent=" << scalar_consistent << " violations=" << violations << '\n';
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "diagnostic: " << diagnostic.token
                << " vertex=" << diagnostic.source.vertex.value << '\n';
    }
    return 2;
  }

  std::cout << "operator_anatomy: accessor summary\n";
  std::cout << "receive()        : value=" << receive_value << " frame_id=" << frame_id
            << " capture_time_ns=" << capture_ns << " sequence=" << sequence
            << " valid=" << metadata_valid << '\n';
  std::cout << "receive_data()   : value=" << data_value
            << " matches_receive=" << (data_value == receive_value)
            << " (payload only: no metadata, no read claim)\n";
  std::cout << "selected_batch() : first_batch=" << first_batch_size << " frames ["
            << batch_first_frame << ".." << batch_last_frame << "] drained_total=" << drained_total
            << " empty_batches=" << empty_batches << '\n';
  std::cout << "selection()      : selected=" << selected_ticks << " safe_default=" << missing_ticks
            << " first_missing state=" << state_label(first_missing.state)
            << " reason=" << reason_label(first_missing.reason) << '\n';
  std::cout << "forward emit     : preserved=" << preserved_count
            << " source frame ids kept (violations=0)\n";
  std::cout << "restamp emit     : restamped=" << restamped_count
            << " fresh frame ids >= " << kRestampFrameBase << " (violations=0)\n";
  std::cout << "cancellation     : stop_requested observed during long compute = "
            << (cancellation_seen ? "yes" : "no") << '\n';
  std::cout << "operator_anatomy complete: accessors verified termination=stopped\n";
  return 0;
}
