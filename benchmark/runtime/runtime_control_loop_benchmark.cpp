/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Robot-control-loop benchmark: a 1 kHz four-operator control chain and a 137 Hz target producer
// with depth-one drop-oldest edges and side inputs, the shape of the Isaac OS robot control
// benchmark. The same executable runs the graph ("split"), a fused single-operator control
// ("fused"), and a two-thread baseline without the SDK ("no-framework"), so every mode shares the
// work functions, payload sizes, timestamps, and statistics. See README.md, "Robot control loop".

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>
#endif

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "channel/process_identity.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/temporal_input_cache.hpp"
#include "runtime_control_loop_benchmark_traits.hpp"
#include "runtime_graph_benchmark_common.hpp"

namespace holoscan {

// Payloads with the robotics sizes: joint state 80 B, pose 80 B, setpoints 176 B, command 192 B.
struct ControlJointState {
  std::uint64_t sequence{};
  std::int64_t scheduled_ns{};  // scheduled activation, CLOCK_MONOTONIC
  std::int64_t entry_ns{};      // callback entry
  std::uint64_t checksum{};
  std::array<double, 6> position{};
};

struct ControlTargetPose {
  std::uint64_t version{};
  std::uint64_t joint_sequence{};  // newest joint state the target consumed
  std::int64_t scheduled_ns{};     // target activation
  std::uint64_t checksum{};
  std::array<double, 6> pose{};
};

struct ControlSetpoints {
  std::uint64_t sequence{};
  std::int64_t scheduled_ns{};
  std::int64_t entry_ns{};
  std::uint64_t target_version{};
  std::uint64_t checksum{};
  std::uint64_t reserved{};
  std::array<double, 16> q{};
};

struct ControlCommand {
  std::uint64_t sequence{};
  std::int64_t scheduled_ns{};
  std::int64_t entry_ns{};
  std::uint64_t target_version{};
  std::uint64_t checksum{};
  std::uint64_t reserved{};
  std::array<double, 18> q{};
};

static_assert(sizeof(ControlJointState) == 80U);
static_assert(sizeof(ControlTargetPose) == 80U);
static_assert(sizeof(ControlSetpoints) == 176U);
static_assert(sizeof(ControlCommand) == 192U);

template <>
struct ChannelTypeIdentity<ControlJointState> {
  static constexpr std::string_view name = "holoscan.benchmark.ControlJointState";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ControlJointState> {
  using P = ControlJointState;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &P::sequence, offsetof(P, sequence)),
                 pod_field("scheduled_ns", &P::scheduled_ns, offsetof(P, scheduled_ns)),
                 pod_field("entry_ns", &P::entry_ns, offsetof(P, entry_ns)),
                 pod_field("checksum", &P::checksum, offsetof(P, checksum)),
                 pod_field("position", &P::position, offsetof(P, position)));
};

template <>
struct ChannelTypeIdentity<ControlTargetPose> {
  static constexpr std::string_view name = "holoscan.benchmark.ControlTargetPose";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ControlTargetPose> {
  using P = ControlTargetPose;
  static constexpr auto fields =
      pod_fields(pod_field("version", &P::version, offsetof(P, version)),
                 pod_field("joint_sequence", &P::joint_sequence, offsetof(P, joint_sequence)),
                 pod_field("scheduled_ns", &P::scheduled_ns, offsetof(P, scheduled_ns)),
                 pod_field("checksum", &P::checksum, offsetof(P, checksum)),
                 pod_field("pose", &P::pose, offsetof(P, pose)));
};

template <>
struct ChannelTypeIdentity<ControlSetpoints> {
  static constexpr std::string_view name = "holoscan.benchmark.ControlSetpoints";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ControlSetpoints> {
  using P = ControlSetpoints;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &P::sequence, offsetof(P, sequence)),
                 pod_field("scheduled_ns", &P::scheduled_ns, offsetof(P, scheduled_ns)),
                 pod_field("entry_ns", &P::entry_ns, offsetof(P, entry_ns)),
                 pod_field("target_version", &P::target_version, offsetof(P, target_version)),
                 pod_field("checksum", &P::checksum, offsetof(P, checksum)),
                 pod_field("reserved", &P::reserved, offsetof(P, reserved)),
                 pod_field("q", &P::q, offsetof(P, q)));
};

template <>
struct ChannelTypeIdentity<ControlCommand> {
  static constexpr std::string_view name = "holoscan.benchmark.ControlCommand";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<ControlCommand> {
  using P = ControlCommand;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &P::sequence, offsetof(P, sequence)),
                 pod_field("scheduled_ns", &P::scheduled_ns, offsetof(P, scheduled_ns)),
                 pod_field("entry_ns", &P::entry_ns, offsetof(P, entry_ns)),
                 pod_field("target_version", &P::target_version, offsetof(P, target_version)),
                 pod_field("checksum", &P::checksum, offsetof(P, checksum)),
                 pod_field("reserved", &P::reserved, offsetof(P, reserved)),
                 pod_field("q", &P::q, offsetof(P, q)));
};

namespace {

using benchmarking::BenchmarkMonotonicClock;
using benchmarking::BenchmarkSupportThreadControl;
using benchmarking::kTensorByteDtype;
using benchmarking::make_preinitialized_byte_tensor;
using benchmarking::PreinitializedTensorStorage;
using benchmarking::SupportThreadBinding;
using benchmarking::SupportThreadRole;
using benchmarking::tensor_memory_kind;
using benchmarking::TensorStorage;

constexpr std::int64_t kNsPerSecond = 1'000'000'000LL;

// How the joint state travels on ReadJointState's outputs. Only the three operators that touch it
// are templated over the codec; the rest of the loop and the no-framework run stay POD.
enum class JointPayload : std::uint8_t {
  kPod,
  kSchema,
  kTensorHost,
  kTensorPinned,
  kTensorDevice,
  kDescriptor,
  kDescriptorBorrow,
};

enum class Mode : std::uint8_t { kSplit, kFused, kNoFramework };
enum class Topology : std::uint8_t {
  kRcb,                // one joint-state output to MotionPlanner and the cross-partition target
  kSplitOutputs,       // two joint-state outputs, one per consumer: the robotics workaround
  kSingle,             // the target runs open-loop; joint state has one consumer
  kSamePartitionPort,  // open-loop target; joint state also feeds a second MotionPlanner input
  kRcbNoTargetInput,   // as kRcb, without MotionPlanner's target input
};

// The selector of the two side inputs: MotionPlanner's target pose and the target's joint state.
// LatestWithin and ReuseLast select by capture time, so producers stamp it only in those arms.
// MotionPlanner's freshness-reconciled side input keeps the control partition's causal chain for
// the later hops; --disable-boundary-causal-chain is the control.
// REQ: RUNTIME-086
enum class SideSelector : std::uint8_t { kDrainAll, kLatestWithin, kReuseLast };
enum class SideScope : std::uint8_t { kBoth, kMotion, kTarget };

// Stage work in the fixed order of the chain: read, plan, safety, write, target.
enum Stage : std::size_t { kRead, kPlan, kSafety, kWrite, kTarget, kStageCount };
constexpr std::array<std::string_view, kStageCount> kStageNames{
    "read", "plan", "safety", "write", "target"};
constexpr std::array<std::uint32_t, kStageCount> kRoboticsWorkNs{300U, 2'400U, 400U, 200U, 1'000U};

// A deterministic expectation checked after the run: `path op value`, for example
// `phase.transport_mutex_acquisitions==0` or `plan.joint_state==drop_oldest:direct`.
struct Guard {
  enum class Op : std::uint8_t { kEq, kNe, kGe, kLe, kGt, kLt };
  std::string text;  // as authored
  std::string path;  // phase.X, per_tick.X, partition.<role>.X, count.X, or plan.joint_state
  Op op{Op::kEq};
  double value{};        // numeric guards
  std::string expected;  // plan guards compare strings
};

struct Options {
  Mode mode{Mode::kSplit};
  Topology topology{Topology::kRcb};
  std::uint32_t control_rate_hz{1'000U};
  std::uint32_t target_rate_hz{137U};
  std::uint32_t samples{10'000U};
  std::uint32_t warmup{1'000U};
  std::uint32_t timeout_ms{0U};  // zero derives a timeout from the run length
  std::array<std::uint32_t, kStageCount> work_ns{kRoboticsWorkNs};
  std::optional<std::array<std::uint64_t, kStageCount>> work_iterations;
  int control_cpu{-1};
  int target_cpu{-1};
  int controller_cpu{-1};
  int pump_cpu{-1};
  std::uint32_t control_fifo_priority{};
  std::uint32_t target_fifo_priority{};
  bool busy_poll_control{};
  bool force_holomq{};
  bool force_holomq_event_wake{};
  // Keeps the joint-state fanout on HoloMQ instead of compile()'s direct fanout (A/B control).
  bool force_holomq_fanout{};
  // Pump delivery instead of compile()'s delivery inside emit() (A/B control).
  bool force_pump_delivery{};
  // Keeps the Read->Motion hop on the ordinary scan instead of continuing it (A/B control).
  bool disable_side_input_continuation{};
  bool disable_boundary_causal_chain{};
  // Policy of the control chain's edges, Read->Motion through Safety->Write. The target's edges
  // stay drop-oldest, so a blocking chain makes the joint-state fanout a mixed-policy family.
  QueuePolicy control_policy{QueuePolicy::kDropOldest};
  // Also export the joint state as an external topic with an optional peer, so its fanout family
  // has an exported member.
  bool export_joint_state{};
  JointPayload joint_payload{JointPayload::kPod};
  SideSelector side_selector{SideSelector::kDrainAll};
  SideScope side_scope{SideScope::kBoth};
  std::uint32_t latest_within_us{20'000U};  // LatestWithin max age, over two target periods
  bool counters{};
  bool mlock{};
  bool require_yield_shim{};
  std::optional<std::uint64_t> timer_slack_ns;  // unset keeps the inherited kernel default
  std::string output;
  std::string observations;
  // Checked after the run; any failure makes the run exit 1 (see Guard).
  std::vector<Guard> expectations;
  // Compile, report the plan, check plan guards, and exit without running.
  bool plan_only{};
};

[[nodiscard]] std::string_view joint_payload_name(const JointPayload payload) {
  switch (payload) {
    case JointPayload::kSchema:
      return "schema";
    case JointPayload::kTensorHost:
      return "tensor-host";
    case JointPayload::kTensorPinned:
      return "tensor-pinned";
    case JointPayload::kTensorDevice:
      return "tensor-device";
    case JointPayload::kDescriptor:
      return "descriptor";
    case JointPayload::kDescriptorBorrow:
      return "descriptor-borrow";
    case JointPayload::kPod:
      break;
  }
  return "pod";
}

[[nodiscard]] std::string_view mode_name(const Mode mode) {
  switch (mode) {
    case Mode::kSplit:
      return "split";
    case Mode::kFused:
      return "fused";
    case Mode::kNoFramework:
      return "no-framework";
  }
  return "unknown";
}

[[nodiscard]] std::string_view topology_name(const Topology topology) {
  switch (topology) {
    case Topology::kRcb:
      return "rcb";
    case Topology::kSplitOutputs:
      return "split-outputs";
    case Topology::kSingle:
      return "single";
    case Topology::kSamePartitionPort:
      return "same-partition-port";
    case Topology::kRcbNoTargetInput:
      return "rcb-no-target-input";
  }
  return "unknown";
}

// Whether the target consumes joint state (a closed loop) and whether MotionPlanner reads targets.
[[nodiscard]] bool target_reads_joint_state(const Options& options) {
  return options.mode != Mode::kFused && options.topology != Topology::kSingle &&
         options.topology != Topology::kSamePartitionPort;
}
[[nodiscard]] bool motion_reads_second_state(const Options& options) {
  return options.topology == Topology::kSamePartitionPort;
}
[[nodiscard]] bool motion_reads_target(const Options& options) {
  return options.topology != Topology::kRcbNoTargetInput;
}
// Whether MotionPlanner's target input (motion) or the target's joint-state input selects
// by capture time instead of DrainAll.
[[nodiscard]] bool scalar_side(const Options& options, const bool motion) {
  return options.side_selector != SideSelector::kDrainAll &&
         (options.side_scope == SideScope::kBoth ||
          options.side_scope == (motion ? SideScope::kMotion : SideScope::kTarget));
}
// Producers stamp capture time whenever a side input selects by it.
[[nodiscard]] bool stamp_capture(const Options& options) {
  return options.side_selector != SideSelector::kDrainAll;
}
// Declare one side input with the configured selector.
template <typename T>
void declare_side(Contract& contract, const Input<T>& input, const Options& options,
                  const bool motion) {
  if (!scalar_side(options, motion)) {
    contract.input(input, DrainAll{});
    return;
  }
  contract.input(input,
                 LatestWithin{.max_age = std::chrono::microseconds{options.latest_within_us}});
  if (options.side_selector == SideSelector::kReuseLast) {
    contract.on_missing(input, ReuseLast{.max_age = std::chrono::milliseconds{50}});
  } else {
    contract.on_missing(input, SafeDefault{});
  }
}

// ------------------------------------------------------------------------------------------------
// Calibrated work: a dependent floating-point chain sized in iterations, so stages of 0.2-2.4 us
// cost no clock reads and replay identically across builds.

// Preserve the call boundary with GCC noipa or Clang noinline. The memory clobber keeps
// the measured loop beside its surrounding timestamp reads.
#if defined(__GNUC__) && !defined(__clang__)
[[gnu::noipa]]
#else
[[gnu::noinline]]
#endif
double work_kernel(const std::uint64_t iterations, double value) noexcept {
  for (std::uint64_t index = 0U; index < iterations; ++index) {
    value = value * 1.000000001 + 1.0e-9;
  }
  asm volatile("" : : "g"(&value) : "memory");
  return value;
}

struct Work {
  std::array<std::uint64_t, kStageCount> iterations{};
  double ns_per_iteration{};

  [[nodiscard]] double run(const Stage stage, const double seed) const noexcept {
    return work_kernel(iterations[stage], seed);
  }
};

[[nodiscard]] Work calibrate(const Options& options) {
  Work work;
  constexpr std::uint64_t kTrialIterations = 200'000U;
  std::vector<double> per_iteration;
  volatile double sink = 0.0;
  for (int trial = 0; trial < 31; ++trial) {
    const auto start = BenchmarkMonotonicClock::now_ns();
    sink = work_kernel(kTrialIterations, 1.0 + trial);
    const auto elapsed = BenchmarkMonotonicClock::now_ns() - start;
    per_iteration.push_back(static_cast<double>(elapsed) / static_cast<double>(kTrialIterations));
  }
  static_cast<void>(sink);
  std::nth_element(per_iteration.begin(),
                   per_iteration.begin() + static_cast<std::ptrdiff_t>(per_iteration.size() / 2U),
                   per_iteration.end());
  work.ns_per_iteration = per_iteration[per_iteration.size() / 2U];
  for (std::size_t stage = 0U; stage < kStageCount; ++stage) {
    work.iterations[stage] =
        options.work_iterations
            ? (*options.work_iterations)[stage]
            : static_cast<std::uint64_t>(
                  std::llround(options.work_ns[stage] / std::max(work.ns_per_iteration, 1e-3)));
  }
  return work;
}

// ------------------------------------------------------------------------------------------------
// Records. Each array has one writer thread, and the main thread reads it only after the run.

struct TickRecord {
  std::int64_t scheduled{};  // scheduled activation
  std::int64_t entry{};      // callback entry
  std::int64_t done{};       // command completion
  std::int64_t read_emit{};  // joint state published (emit returned), for staleness
  // Split mode: entry of each later chain stage, which locates a stall inside the active path.
  std::int64_t motion_entry{};
  std::int64_t safety_entry{};
  std::int64_t write_entry{};
  std::uint64_t target_version{};
  bool written{};
};

struct TargetRecord {
  std::int64_t scheduled{};  // target activation, clock domain until analysis
  std::int64_t emitted{};    // target emit started; delivery latency runs from here
  std::int64_t published{};  // target emit returned; later ticks could select this version
  std::int64_t first_use{};  // first use by the control lane
  std::uint64_t first_use_sequence{};
  std::uint64_t joint_sequence{};
};

// Per-thread sched_yield counts from control_loop_yield_shim, when it is preloaded.
using YieldSnapshotFunction = std::size_t (*)(std::uint64_t*, std::uint64_t*, std::size_t);
constexpr std::size_t kYieldThreads = 256U;

struct YieldCounts {
  std::array<std::uint64_t, kYieldThreads> tids{};
  std::array<std::uint64_t, kYieldThreads> counts{};
  std::size_t size{};
  bool taken{};

  void take(const YieldSnapshotFunction snapshot) noexcept {
    if (snapshot != nullptr && !taken) {
      size = snapshot(tids.data(), counts.data(), kYieldThreads);
      taken = true;
    }
  }
  [[nodiscard]] std::uint64_t count(const std::uint64_t tid) const noexcept {
    for (std::size_t index = 0U; index < size; ++index) {
      if (tids[index] == tid) {
        return counts[index];
      }
    }
    return 0U;
  }
};

struct State {
  explicit State(Options value, Work calibrated)
      : options(std::move(value)), work(calibrated), ticks(options.samples) {
    const std::int64_t control_period = kNsPerSecond / options.control_rate_hz;
    const std::int64_t target_period = kNsPerSecond / options.target_rate_hz;
    const auto total_ticks = static_cast<std::int64_t>(options.warmup + options.samples) + 64;
    targets.resize(static_cast<std::size_t>(total_ticks * control_period / target_period + 64));
  }

  [[nodiscard]] std::int64_t control_period_ns() const {
    return kNsPerSecond / options.control_rate_hz;
  }
  [[nodiscard]] std::int64_t target_period_ns() const {
    return kNsPerSecond / options.target_rate_hz;
  }
  [[nodiscard]] TickRecord* slot(const std::uint64_t sequence) {
    if (sequence <= options.warmup || sequence > options.warmup + options.samples)
      return nullptr;
    return &ticks[sequence - options.warmup - 1U];
  }
  [[nodiscard]] TargetRecord* target(const std::uint64_t version) {
    return version != 0U && version < targets.size() ? &targets[version] : nullptr;
  }
  // The record of a publication. Storage is sized from the nominal run length, so a target lane
  // that keeps publishing through control overruns can outrun it; count each publication without a
  // record so the run is rejected instead of reporting KPIs over the stored prefix.
  [[nodiscard]] TargetRecord* published_target(const std::uint64_t version) {
    TargetRecord* const record = target(version);
    if (record == nullptr) {
      target_observation_overflows.fetch_add(1U, std::memory_order_relaxed);
    }
    return record;
  }

  Options options;
  Work work;
  std::vector<TickRecord> ticks;
  std::vector<TargetRecord> targets;
  // Clock-domain to CLOCK_MONOTONIC offset for activation times; measured once per run.
  std::int64_t clock_offset_ns{};
  std::int64_t clock_offset_uncertainty_ns{};
  bool clock_offset_valid{};
  bool target_times_native{};  // no-framework target times are CLOCK_MONOTONIC
  std::uint64_t control_tid{};
  std::uint64_t target_tid{};
  std::int64_t control_timer_slack_ns{-1};
  std::int64_t target_timer_slack_ns{-1};
  std::atomic<std::uint64_t> checksum_failures{};             // written by either lane
  std::atomic<std::uint64_t> dropped_publications{};          // tolerated drop-oldest results
  std::atomic<std::uint64_t> target_observation_overflows{};  // publications without a record
  std::uint64_t grid_errors{};
  std::uint64_t last_target_version_seen{};
  std::uint64_t target_version_regressions{};
  // Scalar side-input selections, per side: selected, reused, or absent (safe default).
  std::uint64_t motion_side_selected{};
  std::uint64_t motion_side_reused{};
  std::uint64_t motion_side_absent{};
  std::uint64_t target_side_selected{};
  std::uint64_t target_side_reused{};
  std::uint64_t target_side_absent{};
  // Scalar target-side reads made through the selected-sample view of a borrowed codec.
  std::uint64_t target_side_borrowed{};
  std::uint64_t target_publications{};
  // The joint-state output's process-local delivery, from the compiled plan (joint_state_delivery).
  std::string joint_state_delivery{"none"};
  std::atomic<bool> finished{};
  YieldSnapshotFunction yield_snapshot{};
  bool yield_noop{};
  YieldCounts yield_begin;  // taken by the control lane at the first measured tick
  YieldCounts yield_end;    // taken by the control lane after the last measured command

  void begin_window(const std::uint64_t sequence) noexcept {
    if (sequence == options.warmup + 1U) {
      yield_begin.take(yield_snapshot);
    }
  }
  void end_window(const std::uint64_t sequence) noexcept {
    if (sequence == options.warmup + options.samples) {
      yield_end.take(yield_snapshot);
    }
  }
};

State* g_state{};

// Checksum the header words only, so validation costs a few instructions per stage.
template <typename T>
[[nodiscard]] std::uint64_t header_checksum(const T& payload) {
  std::array<std::uint64_t, 3> words{};
  std::memcpy(words.data(), &payload, sizeof(words));
  return words[0] ^ (words[1] << 1U) ^ (words[2] << 2U) ^ 0xC0DE'F00D'5A5A'A5A5ULL;
}

template <typename T>
void seal(T& payload) {
  payload.checksum = header_checksum(payload);
}

template <typename T>
[[nodiscard]] bool sealed(const T& payload) {
  return payload.checksum == header_checksum(payload);
}

// Apply the requested timer slack to the calling thread and return the slack in effect.
std::int64_t apply_timer_slack(const std::optional<std::uint64_t> slack_ns) {
#if defined(__linux__)
  if (slack_ns) {
    static_cast<void>(prctl(PR_SET_TIMERSLACK, static_cast<unsigned long>(*slack_ns), 0, 0, 0));
  }
  return prctl(PR_GET_TIMERSLACK, 0, 0, 0, 0);
#else
  static_cast<void>(slack_ns);
  return -1;
#endif
}

void name_thread(const char* name) {
#if defined(__linux__)
  static_cast<void>(pthread_setname_np(pthread_self(), name));
#else
  static_cast<void>(name);
#endif
}

[[nodiscard]] expected<void, Error> tolerate_drop(expected<void, Error> published, State& state) {
  if (!published && (published.error().code == ErrorCode::kBackpressured ||
                     published.error().code == ErrorCode::kPublicationIndeterminate)) {
    state.dropped_publications.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }
  return published;
}

// Measure the activation clock's offset from CLOCK_MONOTONIC with the tightest of 64 brackets.
void measure_clock_offset(ExecutionContext& context, State& state) {
  std::int64_t best_width = std::numeric_limits<std::int64_t>::max();
  for (int trial = 0; trial < 64; ++trial) {
    const auto before = BenchmarkMonotonicClock::now_ns();
    auto now = context.clock().now();
    const auto after = BenchmarkMonotonicClock::now_ns();
    if (!now)
      return;
    if (after - before < best_width) {
      best_width = after - before;
      state.clock_offset_ns = before + (after - before) / 2 - now->timestamp_ns;
    }
  }
  state.clock_offset_uncertainty_ns = best_width / 2;
  state.clock_offset_valid = true;
}

// ------------------------------------------------------------------------------------------------
// Graph operators. The control chain shares one partition; its first activation measures the
// clock offset. Each operator records only in its own lane's arrays.

// The joint state as the 80-byte POD itself.
struct PodJointCodec {
  using Wire = ControlJointState;
  static constexpr bool kBorrowedInput = false;
  template <typename Port>
  static void configure_output(Port& /*port*/) {}
  template <typename Port>
  static void configure_input(Port& /*port*/) {}
  [[nodiscard]] static expected<void, Error> emit(Output<Wire>& output,
                                                  const ControlJointState& joint,
                                                  ExecutionContext& /*context*/,
                                                  const EmitOptions& options) {
    return output.emit(joint, options);
  }
  [[nodiscard]] static ControlJointState decode(const Sample<Wire>& sample) { return sample.data; }
};

// The joint state as an owning generated schema (control_loop_benchmark.JointStateMessage).
struct SchemaJointCodec {
  using Wire = control_loop_benchmark::JointStateMessageT;
  static constexpr bool kBorrowedInput = false;
  template <typename Port>
  static void configure_output(Port& /*port*/) {}
  template <typename Port>
  static void configure_input(Port& /*port*/) {}
  [[nodiscard]] static expected<void, Error> emit(Output<Wire>& output,
                                                  const ControlJointState& joint,
                                                  ExecutionContext& /*context*/,
                                                  const EmitOptions& options) {
    return output.emit(Wire{.sequence = joint.sequence,
                            .scheduled_ns = joint.scheduled_ns,
                            .entry_ns = joint.entry_ns,
                            .checksum = joint.checksum,
                            .p0 = joint.position[0],
                            .p1 = joint.position[1],
                            .p2 = joint.position[2],
                            .p3 = joint.position[3],
                            .p4 = joint.position[4],
                            .p5 = joint.position[5]},
                       options);
  }
  [[nodiscard]] static ControlJointState decode(const Sample<Wire>& sample) {
    const Wire& wire = sample.data;
    return ControlJointState{
        .sequence = wire.sequence,
        .scheduled_ns = wire.scheduled_ns,
        .entry_ns = wire.entry_ns,
        .checksum = wire.checksum,
        .position = {wire.p0, wire.p1, wire.p2, wire.p3, wire.p4, wire.p5},
    };
  }
};

// Host copies of device joint states, one per Tensor pool backing and keyed by its address. A
// backing's copy is written before the sample on it is published, and is rewritten only when the
// pool reuses the backing, which it does only after every receiver has released that sample. So a
// retained sample's copy cannot change under it, however far the producer runs ahead. A lookup
// that finds no copy for the backing, or a copy of another frame, fails closed with an unsealed
// state.
class DeviceJointShadow {
 public:
  // Pool backings per output stay far below this.
  static constexpr std::size_t kCapacity = 256U;

  // Record `joint` for `backing`, claiming a free copy for a new backing. Return false when every
  // copy belongs to another backing.
  [[nodiscard]] bool record(const void* const backing, const ControlJointState& joint) noexcept {
    for (Copy& copy : copies_) {
      const void* claimed = nullptr;
      if (copy.backing.load(std::memory_order_acquire) == backing ||
          copy.backing.compare_exchange_strong(claimed, backing, std::memory_order_acq_rel)) {
        copy.joint = joint;
        return true;
      }
    }
    return false;
  }

  // Return the copy recorded for `backing` when it holds `frame`, or an unsealed state.
  [[nodiscard]] ControlJointState lookup(const void* const backing,
                                         const std::uint64_t frame) const noexcept {
    // Copies are claimed in order and never released, so the first free one ends the search.
    for (const Copy& copy : copies_) {
      const void* const claimed = copy.backing.load(std::memory_order_acquire);
      if (claimed == nullptr) {
        break;
      }
      if (claimed == backing) {
        if (copy.joint.sequence == frame) {
          return copy.joint;
        }
        break;
      }
    }
    ControlJointState missing{.sequence = frame};
    missing.checksum = ~header_checksum(missing);
    return missing;
  }

 private:
  struct Copy {
    std::atomic<const void*> backing{};
    ControlJointState joint{};
  };
  std::array<Copy, kCapacity> copies_{};
};

// Deterministic checks of DeviceJointShadow, run by --self-test-device-shadow: a held frame keeps
// decoding as itself while other backings publish far past any sequence ring, and a stale or
// unknown lookup fails closed.
[[nodiscard]] bool device_joint_shadow_self_test() {
  const auto sealed_joint = [](const std::uint64_t sequence) {
    ControlJointState joint{.sequence = sequence};
    seal(joint);
    return joint;
  };
  auto shadow = std::make_unique<DeviceJointShadow>();
  std::array<std::byte, 8U> backings{};
  const std::byte unknown{};
  bool passed = shadow->record(&backings[0], sealed_joint(1U));
  for (std::uint64_t sequence = 2U; sequence <= 10'000U; ++sequence) {
    passed = passed && shadow->record(&backings[1U + sequence % 7U], sealed_joint(sequence));
  }
  const ControlJointState held = shadow->lookup(&backings[0], 1U);
  passed = passed && held.sequence == 1U && sealed(held);
  // The pool reuses a backing only once its frame is released. A lookup of the old frame after
  // that, or of a backing never recorded, must not return a sealed state.
  passed = passed && shadow->record(&backings[0], sealed_joint(10'001U));
  passed = passed && !sealed(shadow->lookup(&backings[0], 1U));
  passed = passed && sealed(shadow->lookup(&backings[0], 10'001U));
  passed = passed && !sealed(shadow->lookup(&unknown, 10'001U));
  return passed;
}

// The joint state's 80 bytes in a rank-1 byte Tensor from the runtime pool, in host, pinned, or
// device memory. Receivers of a device Tensor read the joint state from a host copy the producer
// keeps per pool backing (DeviceJointShadow), so the measurement adds no device-to-host
// synchronization. That copy stands in for the device bytes: it does not show that the device
// buffer holds them, or that a receiver's CUDA work on the buffer finished.
// On rcb the Tensor and descriptor arms are drop-oldest fanout families stored by direct retention
// with one shared backing (CHAN-162). Under --force-holomq-fanout the host arms are delivered
// inside emit() (RUNTIME-085); under --force-pump-delivery the pump serves them, and its
// subscribers stage like inline ones (CHAN-161). The matrix rows guard the delivery kind, pump
// waits, data integrity, nonempty side-input observations, and side-input staleness.
// REQ: CHAN-161
// REQ: CHAN-162
// REQ: RUNTIME-085
template <TensorStorage Storage>
struct TensorJointCodec {
  using Wire = Tensor;
  static constexpr bool kBorrowedInput = false;
  static constexpr std::size_t kBytes = sizeof(ControlJointState);

  [[nodiscard]] static TensorRepresentation representation() {
    return TensorRepresentation{
        .memory_kind = tensor_memory_kind(Storage), .dtype = kTensorByteDtype, .rank = 1U};
  }
  template <typename Port>
  static void configure_output(Port& port) {
    port.produces_tensor(TensorOutputSpec{.representation = representation(),
                                          .bounds = tensor_bounds(kBytes),
                                          .storage = TensorOutputStorage::kRuntimePool});
  }
  template <typename Port>
  static void configure_input(Port& port) {
    port.expects_tensor(TensorInputSpec{.representation = representation()});
  }
  [[nodiscard]] static expected<void, Error> emit(Output<Wire>& output,
                                                  const ControlJointState& joint,
                                                  ExecutionContext& context, EmitOptions options) {
    constexpr std::array<std::int64_t, 1U> kShape{static_cast<std::int64_t>(kBytes)};
    auto loan =
        output.allocate_tensor(TensorLoanRequest{.shape = kShape, .dtype = kTensorByteDtype});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    expected<void, Error> committed;
    if constexpr (Storage == TensorStorage::kDevice) {
      // A pageable source is staged before cudaMemcpyAsync returns, so `joint` may go out of scope.
      committed = loan->write_and_commit(
          context.cuda_stream(),
          [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (!shadow().record(writer.data(), joint)) {
              return make_unexpected(
                  Error{ErrorCode::kResourceExhausted, "device joint shadow is full"});
            }
            if (cudaMemcpyAsync(writer.data(),
                                &joint,
                                kBytes,
                                cudaMemcpyHostToDevice,
                                context.cuda_stream().get()) != cudaSuccess) {
              return make_unexpected(Error{ErrorCode::kFailure, "cudaMemcpyAsync failed"});
            }
            return {};
          });
    } else {
      committed = loan->write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
        std::memcpy(writer.data(), &joint, kBytes);
      });
    }
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    options.frame_id = joint.sequence;
    return output.emit(std::move(*loan), options);
  }
  [[nodiscard]] static ControlJointState decode(const Sample<Wire>& sample) {
    if constexpr (Storage == TensorStorage::kDevice) {
      return shadow().lookup(sample.data.data(), sample.metadata.frame_id);
    } else {
      ControlJointState joint{};
      std::memcpy(&joint, sample.data.data(), kBytes);
      return joint;
    }
  }

 private:
  // Written by the producer before each emit; the Channel publication orders it before any read.
  [[nodiscard]] static DeviceJointShadow& shadow() {
    static DeviceJointShadow table;
    return table;
  }
};

// The joint state as a descriptor-backed schema published the owning way: each publication builds
// a fresh host Tensor holding the 80 bytes and emits an owning ImageT around it, which the runtime
// copies into its pool. Receivers take owning samples. This is the baseline for the borrowing arm.
struct DescriptorJointCodec {
  using Wire = schema::ImageT;
  static constexpr bool kBorrowedInput = false;
  static constexpr std::size_t kBytes = sizeof(ControlJointState);

  [[nodiscard]] static TensorRepresentation representation() {
    return TensorRepresentation{
        .memory_kind = MemoryKind::kHost, .dtype = kTensorByteDtype, .rank = 1U};
  }
  template <typename Port>
  static void configure_output(Port& port) {
    port.produces_tensor(TensorOutputSpec{
        .representation = representation(),
        .bounds = tensor_bounds(kBytes),
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    });
  }
  template <typename Port>
  static void configure_input(Port& port) {
    port.expects_tensor(TensorInputSpec{.representation = representation()});
  }
  [[nodiscard]] static expected<void, Error> emit(Output<Wire>& output,
                                                  const ControlJointState& joint,
                                                  ExecutionContext& /*context*/,
                                                  EmitOptions options) {
    std::shared_ptr<PreinitializedTensorStorage> storage;
    auto tensor =
        make_preinitialized_byte_tensor(kBytes, TensorStorage::kPageableHost, 0, 0U, storage, 2U);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    std::memcpy(storage->data(), &joint, kBytes);
    schema::ImageT image;
    image.height = 1;
    image.width = static_cast<std::int32_t>(kBytes);
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.data = std::move(*tensor);
    options.frame_id = joint.sequence;
    return output.emit(std::move(image), options);
  }
  [[nodiscard]] static ControlJointState decode(const Sample<Wire>& sample) {
    ControlJointState joint{};
    if (sample.data.data != nullptr) {
      std::memcpy(&joint, sample.data.data->data(), kBytes);
    }
    return joint;
  }
};

// The joint state as a descriptor-backed schema published the way the vision operators publish
// images: the 80 bytes go into a runtime-pool Tensor loan for ImageT.data, and the metadata comes
// from one reused ImageDescriptorScratch, so publication allocates nothing. Receivers read through
// borrowed selected-sample views instead of copying the ImageT.
struct DescriptorBorrowJointCodec {
  using Wire = schema::ImageT;
  static constexpr bool kBorrowedInput = true;
  static constexpr std::size_t kBytes = sizeof(ControlJointState);

  [[nodiscard]] static TensorRepresentation representation() {
    return TensorRepresentation{
        .memory_kind = MemoryKind::kHost, .dtype = kTensorByteDtype, .rank = 1U};
  }
  template <typename Port>
  static void configure_output(Port& port) {
    port.tensor_field(schema::ImageTensorFields::data)
        .produces_tensor(TensorOutputSpec{.representation = representation(),
                                          .bounds = tensor_bounds(kBytes),
                                          .storage = TensorOutputStorage::kRuntimePool});
  }
  template <typename Port>
  static void configure_input(Port& port) {
    port.tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{.representation = representation()});
  }
  [[nodiscard]] static expected<void, Error> emit(Output<Wire>& output,
                                                  const ControlJointState& joint,
                                                  ExecutionContext& /*context*/,
                                                  EmitOptions options) {
    auto* const descriptor = scratch();
    if (descriptor == nullptr) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted, "image descriptor scratch"});
    }
    constexpr std::array<std::int64_t, 1U> kShape{static_cast<std::int64_t>(kBytes)};
    auto loan =
        output.allocate_tensor(TensorLoanRequest{.shape = kShape, .dtype = kTensorByteDtype});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto committed = loan->write_host_and_commit(
        [&](const TensorOutputWriteGuard& writer) { std::memcpy(writer.data(), &joint, kBytes); });
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    options.frame_id = joint.sequence;
    return output.emit_tensor(std::move(*loan), descriptor->image(), options);
  }
  [[nodiscard]] static ControlJointState decode_image(const schema::ImageT& image) {
    ControlJointState joint{};
    if (image.data != nullptr) {
      std::memcpy(&joint, image.data->data(), kBytes);
    }
    return joint;
  }
  [[nodiscard]] static ControlJointState decode(const Sample<Wire>& sample) {
    return decode_image(sample.data);
  }
  template <typename View>
  [[nodiscard]] static ControlJointState decode_view(const View& view) {
    return decode_image(view.data());
  }

 private:
  // Created once, with fixed 1 x 80 custom-encoding metadata; image().data stays null.
  [[nodiscard]] static sensor_io::ImageDescriptorScratch* scratch() {
    static std::optional<sensor_io::ImageDescriptorScratch> instance = [] {
      std::optional<sensor_io::ImageDescriptorScratch> created;
      if (auto made = sensor_io::ImageDescriptorScratch::create()) {
        created.emplace(std::move(*made));
        created->image().height = 1;
        created->image().width = static_cast<std::int32_t>(kBytes);
        created->image().encoding = schema::ImageEncoding_CUSTOM;
      }
      return created;
    }();
    return instance ? &*instance : nullptr;
  }
};

// Receive the trigger's joint state: through a borrowed view when the codec publishes borrowable
// payloads, otherwise by receiving the owning sample.
template <typename Codec>
[[nodiscard]] expected<ControlJointState, Error> receive_joint(Input<typename Codec::Wire>& input) {
  if constexpr (Codec::kBorrowedInput) {
    auto view = input.selected_sample();
    if (!view) {
      return make_unexpected(std::move(view).error());
    }
    return Codec::decode_view(*view);
  } else {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return Codec::decode(*sample);
  }
}

// Visit what a side input selected: the whole batch for DrainAll, otherwise the selected or reused
// sample. Each scalar outcome is counted in `selected`, `reused`, or `absent` (safe default). A
// borrowed codec reads a scalar through the selected-sample view, as its trigger input does, and
// counts each such read in `borrowed`; an owning codec receives a copy.
template <bool kBorrowScalar = false, typename T, typename Visit>
[[nodiscard]] expected<void, Error> read_side(Input<T>& input, const bool scalar,
                                              std::uint64_t& selected, std::uint64_t& reused,
                                              std::uint64_t& absent, std::uint64_t& borrowed,
                                              const Visit& visit) {
  if (!scalar) {
    auto batch = input.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    for (const Sample<T>& sample : *batch) {
      visit(sample);
    }
    return {};
  }
  const InputState selection = input.selection().state;
  if (selection != InputState::kSelected && selection != InputState::kReused) {
    ++absent;
    return {};
  }
  ++(selection == InputState::kSelected ? selected : reused);
  if constexpr (kBorrowScalar) {
    auto view = input.selected_sample();
    if (!view) {
      return make_unexpected(std::move(view).error());
    }
    ++borrowed;
    visit(view->sample());
  } else {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    visit(*sample);
  }
  return {};
}

template <typename Codec>
class ReadJointState final : public Operator<> {
 public:
  Output<typename Codec::Wire> out;
  Output<typename Codec::Wire> side_out;  // only in the split-outputs topology

  void setup(OperatorSpec& spec) override {
    Codec::configure_output(spec.output(out, "out_joint_state").max_emits_per_compute(1U));
    if (g_state->options.topology == Topology::kSplitOutputs) {
      Codec::configure_output(
          spec.output(side_out, "out_joint_state_side").max_emits_per_compute(1U));
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::nanoseconds{g_state->control_period_ns()}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const auto entry = BenchmarkMonotonicClock::now_ns();
    State& state = *g_state;
    if (!state.clock_offset_valid) {
      name_thread("cl-control");
      state.control_tid = benchmarking::benchmark_thread_identity().value;
      state.control_timer_slack_ns = apply_timer_slack(state.options.timer_slack_ns);
      measure_clock_offset(context, state);
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError, "invalid activation time"});
    }
    const std::int64_t scheduled = activation.timestamp_ns + state.clock_offset_ns;
    state.begin_window(sequence_ + 1U);
    if (!epoch_) {
      epoch_ = activation.timestamp_ns;
    } else if ((activation.timestamp_ns - *epoch_) % state.control_period_ns() != 0) {
      ++state.grid_errors;
    }
    ControlJointState joint{.sequence = ++sequence_, .scheduled_ns = scheduled, .entry_ns = entry};
    const double value = state.work.run(kRead, static_cast<double>(sequence_));
    for (double& position : joint.position) {
      position = value;
    }
    seal(joint);
    // LatestWithin and ReuseLast need capture times; DrainAll arms stay capture-less, like the
    // robot's joint-state reader.
    const EmitOptions emit_options{.capture_time =
                                       stamp_capture(state.options) ? activation : TimePoint{}};
    auto published = tolerate_drop(Codec::emit(out, joint, context, emit_options), state);
    if (!published) {
      return published;
    }
    if (state.options.topology == Topology::kSplitOutputs) {
      published = tolerate_drop(Codec::emit(side_out, joint, context, emit_options), state);
      if (!published) {
        return published;
      }
    }
    if (auto* slot = state.slot(joint.sequence)) {
      slot->scheduled = scheduled;
      slot->entry = entry;
      slot->read_emit = BenchmarkMonotonicClock::now_ns();
    }
    return {};
  }

 private:
  std::uint64_t sequence_{};
  std::optional<std::int64_t> epoch_;
};

template <typename Codec>
class MotionPlanner final : public Operator<> {
 public:
  Input<typename Codec::Wire> in_state;
  Input<typename Codec::Wire> in_state_side;  // only in the same-partition-port topology
  Input<ControlTargetPose> in_target;
  Output<ControlSetpoints> out;

  void setup(OperatorSpec& spec) override {
    Codec::configure_input(spec.input(in_state, "in_joint_state").queue_depth(1U));
    if (motion_reads_second_state(g_state->options)) {
      Codec::configure_input(spec.input(in_state_side, "in_joint_state_side").queue_depth(1U));
    }
    if (motion_reads_target(g_state->options)) {
      spec.input(in_target, "in_target").queue_depth(1U);
    }
    spec.output(out, "out_setpoints").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in_state});
    if (motion_reads_second_state(g_state->options)) {
      result.input(in_state_side, DrainAll{});
    }
    if (motion_reads_target(g_state->options)) {
      declare_side(result, in_target, g_state->options, true);
    }
    return result;
  }

  // Keep side-input selection and version accounting next to the control-stage timestamps;
  // all outcomes belong to the same measured callback.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const auto entry = BenchmarkMonotonicClock::now_ns();
    State& state = *g_state;
    auto received = receive_joint<Codec>(in_state);
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    const ControlJointState joint = *received;
    if (!sealed(joint)) {
      state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
    }
    if (auto* slot = state.slot(joint.sequence)) {
      slot->motion_entry = entry;
    }
    if (motion_reads_second_state(state.options)) {
      auto batch = in_state_side.selected_batch();
      if (!batch) {
        return make_unexpected(std::move(batch).error());
      }
      for (const Sample<typename Codec::Wire>& side : *batch) {
        if (!sealed(Codec::decode(side))) {
          state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
        }
      }
    }
    if (motion_reads_target(state.options)) {
      const bool scalar = scalar_side(state.options, true);
      std::uint64_t borrowed = 0U;
      auto read = read_side(in_target,
                            scalar,
                            state.motion_side_selected,
                            state.motion_side_reused,
                            state.motion_side_absent,
                            borrowed,
                            [&](const Sample<ControlTargetPose>& target) {
                              if (!sealed(target.data)) {
                                state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
                              }
                              // A drained pose is new exactly once; a selected or reused one may
                              // repeat the current version, and only an older one regresses.
                              if (target.data.version < target_version_ ||
                                  (!scalar && target.data.version == target_version_)) {
                                ++state.target_version_regressions;
                                return;
                              }
                              if (target.data.version == target_version_) {
                                return;
                              }
                              target_version_ = target.data.version;
                              if (auto* record = state.target(target_version_);
                                  record != nullptr && record->first_use == 0) {
                                record->first_use = BenchmarkMonotonicClock::now_ns();
                                record->first_use_sequence = joint.sequence;
                              }
                            });
      if (!read) {
        return read;
      }
    }
    ControlSetpoints setpoints{.sequence = joint.sequence,
                               .scheduled_ns = joint.scheduled_ns,
                               .entry_ns = joint.entry_ns,
                               .target_version = target_version_};
    const double value = state.work.run(kPlan, joint.position[0]);
    for (double& q : setpoints.q) {
      q = value;
    }
    seal(setpoints);
    return tolerate_drop(out.emit(setpoints), state);
  }

 private:
  std::uint64_t target_version_{};
};

class SafetyCheck final : public Operator<> {
 public:
  Input<ControlSetpoints> in;
  Output<ControlCommand> out;

  void setup(OperatorSpec& spec) override {
    spec.input(in, "in_setpoints").queue_depth(1U);
    spec.output(out, "out_commands").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const auto entry = BenchmarkMonotonicClock::now_ns();
    State& state = *g_state;
    auto sample = in.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const ControlSetpoints& setpoints = sample->data;
    if (!sealed(setpoints)) {
      state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
    }
    if (auto* slot = state.slot(setpoints.sequence)) {
      slot->safety_entry = entry;
    }
    ControlCommand command{.sequence = setpoints.sequence,
                           .scheduled_ns = setpoints.scheduled_ns,
                           .entry_ns = setpoints.entry_ns,
                           .target_version = setpoints.target_version};
    const double value = state.work.run(kSafety, setpoints.q[0]);
    for (double& q : command.q) {
      q = value;
    }
    seal(command);
    return tolerate_drop(out.emit(command), state);
  }
};

// Stand-in for the motor register the command is written to.
alignas(64) std::array<double, 18> g_output_register{};

class WriteMotorCommand final : public Operator<> {
 public:
  Input<ControlCommand> in;

  void setup(OperatorSpec& spec) override { spec.input(in, "in_commands").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const auto entry = BenchmarkMonotonicClock::now_ns();
    State& state = *g_state;
    auto sample = in.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    const ControlCommand& command = sample->data;
    if (!sealed(command)) {
      state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
    }
    const double value = state.work.run(kWrite, command.q[0]);
    std::copy(std::begin(command.q), std::end(command.q), g_output_register.begin());
    g_output_register[0] = value;
    const auto done = BenchmarkMonotonicClock::now_ns();
    if (auto* slot = state.slot(command.sequence)) {
      slot->write_entry = entry;
      slot->done = done;
      slot->target_version = command.target_version;
      slot->written = true;
    }
    state.end_window(command.sequence);
    if (command.sequence >= state.options.warmup + state.options.samples &&
        !state.finished.exchange(true)) {
      context.request_stop();
    }
    return {};
  }
};

template <typename Codec>
class TargetProducer final : public Operator<> {
 public:
  Input<typename Codec::Wire> in_state;
  Output<ControlTargetPose> out;

  void setup(OperatorSpec& spec) override {
    if (target_reads_joint_state(g_state->options)) {
      Codec::configure_input(spec.input(in_state, "in_joint_state").queue_depth(1U));
    }
    if (motion_reads_target(g_state->options)) {
      spec.output(out, "out_target").max_emits_per_compute(1U);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::nanoseconds{g_state->target_period_ns()}});
    if (target_reads_joint_state(g_state->options)) {
      declare_side(result, in_state, g_state->options, false);
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    State& state = *g_state;
    if (!named_) {
      named_ = true;
      name_thread("cl-target");
      state.target_tid = benchmarking::benchmark_thread_identity().value;
      state.target_timer_slack_ns = apply_timer_slack(state.options.timer_slack_ns);
    }
    if (target_reads_joint_state(state.options)) {
      auto read = read_side<Codec::kBorrowedInput>(
          in_state,
          scalar_side(state.options, false),
          state.target_side_selected,
          state.target_side_reused,
          state.target_side_absent,
          state.target_side_borrowed,
          [&](const Sample<typename Codec::Wire>& sample) {
            const ControlJointState joint = Codec::decode(sample);
            if (!sealed(joint)) {
              state.checksum_failures.fetch_add(1U, std::memory_order_relaxed);
            }
            joint_sequence_ = std::max(joint_sequence_, joint.sequence);
          });
      if (!read) {
        return read;
      }
    }
    const TimePoint activation = context.activation_time();
    ControlTargetPose pose{.version = ++version_,
                           .joint_sequence = joint_sequence_,
                           .scheduled_ns = activation.timestamp_ns};
    const double value = state.work.run(kTarget, static_cast<double>(version_));
    for (double& axis : pose.pose) {
      axis = value;
    }
    seal(pose);
    // Stamp before emitting: delivery inside emit() lets MotionPlanner read this version on its own
    // lane before emit() returns. Freshness uses the return instead, when every later selection can
    // see this version.
    const std::int64_t emitted = BenchmarkMonotonicClock::now_ns();
    if (motion_reads_target(state.options)) {
      auto published = tolerate_drop(
          out.emit(
              pose,
              EmitOptions{.capture_time = stamp_capture(state.options) ? activation : TimePoint{}}),
          state);
      if (!published) {
        return published;
      }
    }
    const std::int64_t returned = BenchmarkMonotonicClock::now_ns();
    ++state.target_publications;
    if (auto* record = state.published_target(version_)) {
      record->scheduled = pose.scheduled_ns;
      record->emitted = emitted;
      record->published = returned;
      record->joint_sequence = joint_sequence_;
    }
    return {};
  }

 private:
  std::uint64_t version_{};
  std::uint64_t joint_sequence_{};
  bool named_{};
};

// Read, plan, safety, and write in one callback: the fused control of the robotics experiment.
class FusedControl final : public Operator<> {
 public:
  Input<ControlTargetPose> in_target;

  void setup(OperatorSpec& spec) override { spec.input(in_target, "in_target").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::nanoseconds{g_state->control_period_ns()}})
        .input(in_target, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const auto entry = BenchmarkMonotonicClock::now_ns();
    State& state = *g_state;
    if (!state.clock_offset_valid) {
      name_thread("cl-control");
      state.control_tid = benchmarking::benchmark_thread_identity().value;
      state.control_timer_slack_ns = apply_timer_slack(state.options.timer_slack_ns);
      measure_clock_offset(context, state);
    }
    const std::uint64_t sequence = ++sequence_;
    state.begin_window(sequence);
    const std::int64_t scheduled = context.activation_time().timestamp_ns + state.clock_offset_ns;
    double value = state.work.run(kRead, static_cast<double>(sequence));
    auto batch = in_target.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    for (const Sample<ControlTargetPose>& target : *batch) {
      if (target.data.version <= target_version_) {
        ++state.target_version_regressions;
        continue;
      }
      target_version_ = target.data.version;
      if (auto* record = state.target(target_version_);
          record != nullptr && record->first_use == 0) {
        record->first_use = BenchmarkMonotonicClock::now_ns();
        record->first_use_sequence = sequence;
      }
    }
    value = state.work.run(kPlan, value);
    value = state.work.run(kSafety, value);
    value = state.work.run(kWrite, value);
    g_output_register[0] = value;
    const auto done = BenchmarkMonotonicClock::now_ns();
    if (auto* slot = state.slot(sequence)) {
      *slot = TickRecord{.scheduled = scheduled,
                         .entry = entry,
                         .done = done,
                         .read_emit = entry,
                         .target_version = target_version_,
                         .written = true};
    }
    state.end_window(sequence);
    if (sequence >= state.options.warmup + state.options.samples &&
        !state.finished.exchange(true)) {
      context.request_stop();
    }
    return {};
  }

 private:
  std::uint64_t sequence_{};
  std::uint64_t target_version_{};
};

[[nodiscard]] ThreadPlacement placement(const int cpu, const std::uint32_t fifo_priority,
                                        const bool busy_poll) {
  ThreadPlacement result = benchmarking::make_placement(cpu, busy_poll);
  if (fifo_priority != 0U) {
    result.scheduling.policy = SchedulingPolicy::kFIFO;
    result.scheduling.priority = static_cast<int>(fifo_priority);
  }
  return result;
}

// Keep topology wiring and its placement/transport overrides together so every benchmark
// variant starts from the same authored control chain.
template <typename Codec>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ExecutionPlan make_plan_with(const Options& options) {
  Graph graph{"control-loop-" + std::to_string(BenchmarkMonotonicClock::now_ns())};
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const ConnectionOptions latest{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  // A blocking control chain makes the joint-state fanout a mixed-policy family, which is
  // delivered by direct retention unless --force-holomq-fanout keeps it on HoloMQ (CHAN-167).
  // REQ: CHAN-167
  const ConnectionOptions control_edge{.queue_policy = options.control_policy, .queue_depth = 1U};
  auto control = graph.partition("control");
  if (options.mode == Mode::kFused) {
    auto fused = graph.op<FusedControl>("fused_control");
    control.add(fused);
    auto target = graph.op<TargetProducer<Codec>>("target_producer");
    graph.partition("target").add(target);
    graph.add_flow(target->out, fused->in_target, latest);
  } else {
    auto read = graph.op<ReadJointState<Codec>>("read_joint_state");
    auto motion = graph.op<MotionPlanner<Codec>>("motion_planner");
    auto safety = graph.op<SafetyCheck>("safety_check");
    auto write = graph.op<WriteMotorCommand>("write_motor_command");
    auto target = graph.op<TargetProducer<Codec>>("target_producer");
    control.add(read).add(motion).add(safety).add(write);
    graph.partition("target").add(target);
    graph.add_flow(read->out, motion->in_state, control_edge);
    if (motion_reads_second_state(options)) {
      graph.add_flow(read->out, motion->in_state_side, latest);
    }
    if (target_reads_joint_state(options)) {
      graph.add_flow(options.topology == Topology::kSplitOutputs ? read->side_out : read->out,
                     target->in_state,
                     latest);
    }
    if (motion_reads_target(options)) {
      graph.add_flow(target->out, motion->in_target, latest);
    }
    // Exports are rejected for Tensor and descriptor payloads (parse()); a descriptor schema has no
    // export overload without a Tensor topic contract. The exported member keeps its own service,
    // and the local recipients of the joint-state family stay direct (CHAN-165).
    // REQ: CHAN-165
    if constexpr (!std::is_same_v<typename Codec::Wire, schema::ImageT>) {
      if (options.export_joint_state) {
        // A process-unique name keeps concurrent runs from sharing the external service.
        const std::string endpoint = "control_loop_joint_state_" +
                                     std::to_string(channel_internal::current_process_identity()) +
                                     "_" + std::to_string(BenchmarkMonotonicClock::now_ns());
        TopicOptions topic;
        topic.endpoint_name = endpoint;
        topic.bounds.subscriber =
            SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
        const auto exported =
            graph.export_topic<typename Codec::Wire>("holomq://" + endpoint, std::move(topic));
        graph.add_flow(read->out, exported.input(), latest);
      }
    }
    graph.add_flow(motion->out, safety->in, control_edge);
    graph.add_flow(safety->out, write->in, control_edge);
  }
  CompileOptions compile_options;
  if (options.joint_payload == JointPayload::kTensorDevice) {
    for (const std::string_view port : {"out_joint_state", "out_joint_state_side"}) {
      if (port == "out_joint_state_side" && options.topology != Topology::kSplitOutputs) {
        continue;
      }
      compile_options.deployment.bind_tensor_output_device(
          TensorOutputDevicePlacement{.operator_path = "read_joint_state",
                                      .output_port = std::string{port},
                                      .device = DeviceId{0}});
    }
  }
  compile_options.deployment.bind_thread_placement(
      "control",
      placement(options.control_cpu, options.control_fifo_priority, options.busy_poll_control));
  compile_options.deployment.bind_thread_placement(
      "target", placement(options.target_cpu, options.target_fifo_priority, false));
  if (options.controller_cpu >= 0) {
    compile_options.deployment.bind_support_thread_affinity(
        holoscan::SupportThreadRole::kController,
        CPUAffinity{static_cast<std::uint32_t>(options.controller_cpu)});
  }
  if (options.pump_cpu >= 0) {
    compile_options.deployment.bind_support_thread_affinity(
        holoscan::SupportThreadRole::kChannelPump,
        CPUAffinity{static_cast<std::uint32_t>(options.pump_cpu)});
  }
  if (options.force_holomq_event_wake) {
    return detail::compile_with_process_local_holomq_event_wake(graph, std::move(compile_options));
  }
  if (options.force_pump_delivery) {
    return detail::compile_with_process_local_pump_delivery(
        graph, std::move(compile_options), options.force_holomq);
  }
  if (options.force_holomq) {
    return detail::compile_with_process_local_holomq(graph, std::move(compile_options));
  }
  if (options.force_holomq_fanout) {
    return detail::compile_with_process_local_holomq_fanout(graph, std::move(compile_options));
  }
  return compile(graph, std::move(compile_options));
}

// ------------------------------------------------------------------------------------------------
// No-framework baseline: two plain threads, absolute-deadline sleeps on a shared epoch, and
// wait-free latest-value handoffs in both directions (the equivalent of depth-one drop-oldest).

// The writer, reader, and exchange index occupy separate cache lines to avoid false sharing
// in the wait-free baseline; the resulting padding is intentional.
template <typename T>
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class LatestValue {
 public:
  void publish(const T& value) noexcept {
    slots_[back_] = value;
    const std::uint32_t previous =
        middle_.exchange(back_ | kFresh, std::memory_order_acq_rel) & kIndexMask;
    back_ = previous;
  }

  // Return the newest published value, or nothing when none arrived since the last read.
  [[nodiscard]] std::optional<T> read() noexcept {
    if ((middle_.load(std::memory_order_relaxed) & kFresh) == 0U) {
      return std::nullopt;
    }
    const std::uint32_t previous = middle_.exchange(front_, std::memory_order_acq_rel);
    front_ = previous & kIndexMask;
    return slots_[front_];
  }

 private:
  static constexpr std::uint32_t kFresh = 4U;
  static constexpr std::uint32_t kIndexMask = 3U;
  std::array<T, 3> slots_{};
  alignas(64) std::atomic<std::uint32_t> middle_{1U};
  alignas(64) std::uint32_t back_{0U};   // writer-owned
  alignas(64) std::uint32_t front_{2U};  // reader-owned
};

struct NativeThreadEvidence {
  int requested_cpu{-1};
  std::uint32_t requested_priority{};
  int observed_cpu{-1};
  int observed_policy{-1};
  int observed_priority{-1};
  std::string error;
};

void place_native_thread(const int cpu, const std::uint32_t priority,
                         NativeThreadEvidence& evidence) {
  evidence.requested_cpu = cpu;
  evidence.requested_priority = priority;
#if defined(__linux__)
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      evidence.error = "affinity";
    }
  }
  if (priority != 0U) {
    sched_param parameter{};
    parameter.sched_priority = static_cast<int>(priority);
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameter) != 0) {
      evidence.error = "scheduling";
    }
  }
  int policy = 0;
  sched_param observed{};
  if (pthread_getschedparam(pthread_self(), &policy, &observed) == 0) {
    evidence.observed_policy = policy;
    evidence.observed_priority = observed.sched_priority;
  }
  evidence.observed_cpu = sched_getcpu();
#endif
}

void sleep_until_ns(const std::int64_t deadline) {
#if defined(__linux__)
  const timespec value{.tv_sec = deadline / kNsPerSecond, .tv_nsec = deadline % kNsPerSecond};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &value, nullptr) == EINTR) {
  }
#else
  std::this_thread::sleep_for(
      std::chrono::nanoseconds{deadline - BenchmarkMonotonicClock::now_ns()});
#endif
}

// Selects native ticks the way OnClock coalescing does (CONTRACT-008): each activation takes the
// greatest grid index already due, and never one it handled before. A late wake or an overrun
// therefore runs the newest due tick at once instead of sleeping to the one after it.
[[nodiscard]] constexpr std::int64_t select_native_tick(const std::int64_t last_tick,
                                                        const std::int64_t now,
                                                        const std::int64_t epoch,
                                                        const std::int64_t period) noexcept {
  const std::int64_t due = now >= epoch ? (now - epoch) / period : -1;
  return std::max(last_tick + 1, due);
}

class NativeTickSchedule {
 public:
  NativeTickSchedule(const std::int64_t epoch, const std::int64_t period) noexcept
      : epoch_(epoch), period_(period) {}

  // The deadline of the next unhandled tick, which the thread sleeps to when it is still ahead.
  [[nodiscard]] std::int64_t next_deadline() const noexcept {
    return epoch_ + ((last_tick_ + 1) * period_);
  }

  // Sleeps to the next unhandled tick if it is still ahead, then takes the greatest due tick and
  // returns its scheduled time.
  [[nodiscard]] std::int64_t wait() {
    if (const std::int64_t deadline = next_deadline();
        BenchmarkMonotonicClock::now_ns() < deadline) {
      sleep_until_ns(deadline);
    }
    last_tick_ = select_native_tick(last_tick_, BenchmarkMonotonicClock::now_ns(), epoch_, period_);
    return epoch_ + (last_tick_ * period_);
  }

 private:
  std::int64_t epoch_;
  std::int64_t period_;
  std::int64_t last_tick_{-1};
};

// Compares native tick selection with the SDK's coalescing schedule. Each reading is when the
// native thread looks at the clock; it never looks before its next unhandled deadline, because it
// sleeps to that deadline first. Returns 0 when both select the same tick at every wake.
[[nodiscard]] int self_test_native_schedule() {
  constexpr std::int64_t kEpoch = 5'000'000;
  constexpr std::int64_t kPeriod = 1'000'000;
  // Offsets from the epoch: on time; a 0.4 ms late wake; a callback that ends at 2.4 ms, which
  // must run tick 2 at once rather than sleep to tick 3; a reading before tick 3 is due; a long
  // overrun into tick 9; and a reading before tick 10 is due.
  constexpr std::array<std::int64_t, 6> kReadings{
      0, 1'400'000, 2'400'000, 2'500'000, 9'700'000, 9'800'000};
  constexpr std::array<std::int64_t, 6> kExpected{0, 1, 2, 3, 9, 10};
  const ClockId clock{1U};
  detail::CoalescingClockSchedule sdk{TimePoint{kEpoch, clock}, std::chrono::nanoseconds{kPeriod}};
  std::int64_t last_tick = -1;
  bool passed = true;
  std::cout << "{\"native_schedule_self_test\":[";
  for (std::size_t index = 0U; index < kReadings.size(); ++index) {
    const std::int64_t wake =
        std::max(kEpoch + kReadings[index], kEpoch + ((last_tick + 1) * kPeriod));
    const std::int64_t native = select_native_tick(last_tick, wake, kEpoch, kPeriod);
    auto due = sdk.due(TimePoint{wake, clock});
    const auto due_tick = due.value_or(std::nullopt);
    const std::int64_t sdk_tick =
        due_tick.has_value() ? static_cast<std::int64_t>(due_tick->index) : std::int64_t{-1};
    const bool matches = native == sdk_tick && native == kExpected[index];
    passed = passed && matches;
    last_tick = native;
    std::cout << (index == 0U ? "" : ",") << "{\"reading_ns\":" << kReadings[index]
              << ",\"native_tick\":" << native << ",\"sdk_tick\":" << sdk_tick
              << ",\"matches\":" << (matches ? "true" : "false") << '}';
  }
  std::cout << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
  return passed ? 0 : 1;
}

struct NoFrameworkRun {
  NativeThreadEvidence control;
  NativeThreadEvidence target;
  bool completed{};
  bool timed_out{};
};

// Runs the native loops until every control tick completed, or until `timeout` elapses, whichever
// comes first. The control lane checks the deadline before and after each callback, the target lane
// starts no callback after the control lane finished or the deadline passed, and a run whose lanes
// finish joining after the deadline times out too.
// Keep both native lanes and their common deadline/join policy together to make the
// baseline termination and measurement boundaries auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] NoFrameworkRun run_no_framework(State& state,
                                              const std::chrono::milliseconds timeout) {
  NoFrameworkRun run;
  const std::int64_t give_up =
      BenchmarkMonotonicClock::now_ns() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count();
  state.target_times_native = true;
  LatestValue<ControlJointState> joints;
  LatestValue<ControlTargetPose> targets;
  const std::int64_t epoch = BenchmarkMonotonicClock::now_ns() + 10'000'000LL;
  const bool closed_loop = target_reads_joint_state(state.options);
  std::thread target_thread([&] {
    name_thread("cl-target");
    place_native_thread(state.options.target_cpu, state.options.target_fifo_priority, run.target);
    state.target_tid = benchmarking::benchmark_thread_identity().value;
    state.target_timer_slack_ns = apply_timer_slack(state.options.timer_slack_ns);
    std::uint64_t version = 0U;
    std::uint64_t joint_sequence = 0U;
    NativeTickSchedule schedule{epoch, state.target_period_ns()};
    while (!state.finished.load(std::memory_order_acquire)) {
      const std::int64_t deadline = schedule.wait();
      if (state.finished.load(std::memory_order_acquire) ||
          BenchmarkMonotonicClock::now_ns() >= give_up) {
        break;
      }
      if (closed_loop) {
        if (auto joint = joints.read()) {
          joint_sequence = std::max(joint_sequence, joint->sequence);
        }
      }
      ControlTargetPose pose{
          .version = ++version, .joint_sequence = joint_sequence, .scheduled_ns = deadline};
      const double value = state.work.run(kTarget, static_cast<double>(version));
      for (double& axis : pose.pose) {
        axis = value;
      }
      seal(pose);
      // Stamp around publishing, as the framework target does: the control lane can read this
      // version before publish() returns, and every later read sees it after.
      const std::int64_t emitted = BenchmarkMonotonicClock::now_ns();
      targets.publish(pose);
      const std::int64_t published = BenchmarkMonotonicClock::now_ns();
      ++state.target_publications;
      if (auto* record = state.published_target(version)) {
        record->scheduled = deadline;
        record->emitted = emitted;
        record->published = published;
        record->joint_sequence = joint_sequence;
      }
    }
  });
  std::thread control_thread([&] {
    name_thread("cl-control");
    place_native_thread(
        state.options.control_cpu, state.options.control_fifo_priority, run.control);
    state.control_tid = benchmarking::benchmark_thread_identity().value;
    state.control_timer_slack_ns = apply_timer_slack(state.options.timer_slack_ns);
    std::uint64_t sequence = 0U;
    std::uint64_t target_version = 0U;
    const std::uint64_t total = state.options.warmup + state.options.samples;
    NativeTickSchedule schedule{epoch, state.control_period_ns()};
    while (sequence < total) {
      // The timeout bounds the whole run, as it bounds a framework session: a tick that is not due
      // before it, a wake after it, and a callback that ends after it all time the run out.
      if (schedule.next_deadline() >= give_up) {
        run.timed_out = true;
        break;
      }
      const std::int64_t deadline = schedule.wait();
      const auto entry = BenchmarkMonotonicClock::now_ns();
      if (entry >= give_up) {
        run.timed_out = true;
        break;
      }
      ++sequence;
      state.begin_window(sequence);
      ControlJointState joint{.sequence = sequence, .scheduled_ns = deadline, .entry_ns = entry};
      double value = state.work.run(kRead, static_cast<double>(sequence));
      for (double& position : joint.position) {
        position = value;
      }
      seal(joint);
      if (closed_loop) {
        joints.publish(joint);
      }
      const auto read_emit = BenchmarkMonotonicClock::now_ns();
      if (auto target = targets.read(); target && target->version > target_version) {
        target_version = target->version;
        if (auto* record = state.target(target_version);
            record != nullptr && record->first_use == 0) {
          record->first_use = BenchmarkMonotonicClock::now_ns();
          record->first_use_sequence = sequence;
        }
      }
      value = state.work.run(kPlan, value);
      value = state.work.run(kSafety, value);
      value = state.work.run(kWrite, value);
      g_output_register[0] = value;
      const auto done = BenchmarkMonotonicClock::now_ns();
      if (auto* slot = state.slot(sequence)) {
        *slot = TickRecord{.scheduled = deadline,
                           .entry = entry,
                           .done = done,
                           .read_emit = read_emit,
                           .target_version = target_version,
                           .written = true};
      }
      state.end_window(sequence);
      if (done >= give_up) {
        run.timed_out = true;
        break;
      }
    }
    state.finished.store(true, std::memory_order_release);
  });
  control_thread.join();
  target_thread.join();
  // A target callback or sleep can outlast the control lane; the timeout bounds both.
  if (BenchmarkMonotonicClock::now_ns() >= give_up) {
    run.timed_out = true;
  }
  run.completed = !run.timed_out;
  return run;
}

// ------------------------------------------------------------------------------------------------
// Statistics and output.

struct Distribution {
  std::size_t count{};
  double p50{}, p90{}, p99{}, p999{}, p9999{}, max{}, min{}, mean{};
};

[[nodiscard]] double quantile(const std::vector<std::int64_t>& sorted, const double q) {
  const double position = q * static_cast<double>(sorted.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = std::min(lower + 1U, sorted.size() - 1U);
  const double fraction = position - static_cast<double>(lower);
  return static_cast<double>(sorted[lower]) +
         fraction * static_cast<double>(sorted[upper] - sorted[lower]);
}

[[nodiscard]] Distribution summarize(std::vector<std::int64_t> values) {
  Distribution result;
  result.count = values.size();
  if (values.empty()) {
    return result;
  }
  std::sort(values.begin(), values.end());
  double sum = 0.0;
  for (const auto value : values) {
    sum += static_cast<double>(value);
  }
  result.mean = sum / static_cast<double>(values.size());
  result.min = static_cast<double>(values.front());
  result.max = static_cast<double>(values.back());
  result.p50 = quantile(values, 0.5);
  result.p90 = quantile(values, 0.9);
  result.p99 = quantile(values, 0.99);
  result.p999 = quantile(values, 0.999);
  result.p9999 = quantile(values, 0.9999);
  return result;
}

void emit_distribution(std::ostream& output, const Distribution& value) {
  const auto number = [&](const double field, const bool present) {
    if (present) {
      output << field;
    } else {
      output << "null";
    }
  };
  output << "{\"samples\":" << value.count << ",\"min\":";
  number(value.min, value.count != 0U);
  output << ",\"mean\":";
  number(value.mean, value.count != 0U);
  output << ",\"p50\":";
  number(value.p50, value.count != 0U);
  output << ",\"p90\":";
  number(value.p90, value.count >= 10U);
  output << ",\"p99\":";
  number(value.p99, value.count >= 100U);
  output << ",\"p999\":";
  number(value.p999, value.count >= 1'000U);
  output << ",\"p9999\":";
  number(value.p9999, value.count >= 100'000U);
  output << ",\"max\":";
  number(value.max, value.count != 0U);
  output << '}';
}

struct Analysis {
  std::vector<std::int64_t> control_loop, active, entry, period, period_error, tprl, delivery,
      staleness, read_to_motion, motion_to_safety, safety_to_write;
  std::uint64_t worst_active_sequence{};
  const TickRecord* worst_active{};
  const TargetRecord* worst_stale_target{};
  const TickRecord* worst_stale_joint{};
  std::uint64_t worst_stale_version{};
  std::uint64_t worst_stale_newest_joint{};
  std::int64_t worst_staleness{-1};
  std::uint64_t measured_ticks{};
  std::uint64_t missing_ticks{};
  std::uint64_t late{};
  std::uint64_t coalesced{};
  std::uint64_t missed_freshness{};
  std::uint64_t measured_target_versions{};
  bool ordered{true};
};

void analyze_control_ticks(const State& state, Analysis& result) {
  const std::int64_t period = state.control_period_ns();
  const TickRecord* previous = nullptr;
  for (const TickRecord& tick : state.ticks) {
    if (!tick.written) {
      ++result.missing_ticks;
      previous = nullptr;
      continue;
    }
    ++result.measured_ticks;
    if (tick.entry < tick.scheduled || tick.done < tick.entry) {
      result.ordered = false;
    }
    result.control_loop.push_back(tick.done - tick.scheduled);
    result.active.push_back(tick.done - tick.entry);
    if (tick.motion_entry != 0 && tick.safety_entry != 0 && tick.write_entry != 0) {
      // Each hop runs from the previous stage's publication (or entry) to the next stage's entry.
      result.read_to_motion.push_back(tick.motion_entry - tick.read_emit);
      result.motion_to_safety.push_back(tick.safety_entry - tick.motion_entry);
      result.safety_to_write.push_back(tick.write_entry - tick.safety_entry);
    }
    if (result.worst_active == nullptr ||
        tick.done - tick.entry > result.worst_active->done - result.worst_active->entry) {
      result.worst_active = &tick;
      result.worst_active_sequence =
          state.options.warmup + 1U + static_cast<std::uint64_t>(&tick - state.ticks.data());
    }
    result.entry.push_back(tick.entry - tick.scheduled);
    if (tick.done > tick.scheduled + period) {
      ++result.late;
    }
    if (previous != nullptr) {
      result.period.push_back(tick.entry - previous->entry);
      result.period_error.push_back(std::llabs(tick.entry - previous->entry - period));
      if (tick.scheduled - previous->scheduled > period) {
        result.coalesced +=
            static_cast<std::uint64_t>((tick.scheduled - previous->scheduled) / period) - 1U;
      }
    }
    previous = &tick;
  }
}

void record_target_first_use(const State& state, const TargetRecord& target,
                             std::int64_t target_scheduled, Analysis& result) {
  if (target.first_use == 0) {
    return;
  }
  result.delivery.push_back(target.first_use - target.emitted);
  const auto first = target.first_use_sequence;
  if (first > state.options.warmup && first <= state.options.warmup + state.options.samples) {
    const TickRecord& used = state.ticks[first - state.options.warmup - 1U];
    if (used.written) {
      result.tprl.push_back(used.done - target_scheduled);
    }
  }
}

// Join target versions to control ticks with a monotonic cursor before measuring freshness.
[[nodiscard]] std::vector<std::pair<std::int64_t, std::uint64_t>> analyze_target_versions(
    const State& state, Analysis& result) {
  // Target versions whose activation lies inside the measured window.
  const std::int64_t window_start = state.ticks.front().scheduled;
  const std::int64_t window_end = state.ticks.back().scheduled;
  const std::int64_t target_offset = state.target_times_native ? 0 : state.clock_offset_ns;
  // (emit return, version) in version order
  std::vector<std::pair<std::int64_t, std::uint64_t>> publications;
  std::size_t joint_cursor = 0U;  // ticks before the activation
  std::uint64_t newest_joint = 0U;
  for (std::size_t version = 1U; version < state.targets.size(); ++version) {
    const TargetRecord& target = state.targets[version];
    if (target.scheduled == 0 || target.emitted == 0 || target.published == 0) {
      continue;
    }
    const std::int64_t target_scheduled = target.scheduled + target_offset;
    publications.emplace_back(target.published, version);
    while (joint_cursor < state.ticks.size() &&
           (!state.ticks[joint_cursor].written ||
            state.ticks[joint_cursor].read_emit < target_scheduled)) {
      if (state.ticks[joint_cursor].written) {
        newest_joint = state.options.warmup + 1U + joint_cursor;
      }
      ++joint_cursor;
    }
    if (target_scheduled < window_start || target_scheduled > window_end) {
      continue;
    }
    ++result.measured_target_versions;
    record_target_first_use(state, target, target_scheduled, result);
    if (target_reads_joint_state(state.options) && target.joint_sequence != 0U &&
        newest_joint != 0U && newest_joint >= target.joint_sequence) {
      // Ticks between the newest joint state published before the target activation and the
      // one consumed.
      const auto staleness = static_cast<std::int64_t>(newest_joint - target.joint_sequence);
      result.staleness.push_back(staleness);
      if (staleness > result.worst_staleness) {
        result.worst_staleness = staleness;
        result.worst_stale_target = &target;
        result.worst_stale_joint = &state.ticks[newest_joint - state.options.warmup - 1U];
        result.worst_stale_version = version;
        result.worst_stale_newest_joint = newest_joint;
      }
    }
  }
  return publications;
}

[[nodiscard]] Analysis analyze(const State& state) {
  Analysis result;
  analyze_control_ticks(state, result);
  const auto publications = analyze_target_versions(state, result);
  // Missed freshness: a control tick used an older target version than one whose emit returned
  // before the tick's scheduled activation, which precedes every selection for that tick. A target
  // still being emitted then does not count: the tick could select only the previous version.
  if (motion_reads_target(state.options) || state.options.mode != Mode::kSplit) {
    std::size_t cursor = 0U;
    std::uint64_t newest = 0U;
    for (const TickRecord& tick : state.ticks) {
      if (!tick.written) {
        continue;
      }
      while (cursor < publications.size() && publications[cursor].first < tick.scheduled) {
        newest = std::max(newest, publications[cursor].second);
        ++cursor;
      }
      if (tick.target_version < newest) {
        ++result.missed_freshness;
      }
    }
  }
  return result;
}

// Checks the two target timelines in analyze(), on native-clock records: delivery runs from the
// start of a target's emit, which a consumer can overlap, while a tick misses freshness only when
// it used an older version than one whose emit had returned before the tick. Returns 0 when every
// case matches.
[[nodiscard]] int self_test_target_freshness() {
  Options options;
  options.mode = Mode::kNoFramework;
  options.samples = 6U;
  options.warmup = 0U;
  State state{options, Work{}};
  state.target_times_native = true;
  constexpr std::int64_t kMs = 1'000'000;
  // Ticks on the 1 ms grid, with the target version each one used.
  constexpr std::array<std::uint64_t, 6> kTickTargets{1U, 1U, 1U, 3U, 3U, 3U};
  for (std::size_t index = 0U; index < state.ticks.size(); ++index) {
    TickRecord& tick = state.ticks[index];
    tick.scheduled = static_cast<std::int64_t>(index + 1U) * kMs;
    tick.entry = tick.scheduled + 5'000;
    tick.read_emit = tick.scheduled + 10'000;
    tick.done = tick.scheduled + 30'000;
    tick.target_version = kTickTargets[index];
    tick.written = true;
  }
  // Version 1 is published before every tick. Version 2's emit starts before tick 2 and returns
  // after it, so tick 2 could use only version 1; tick 3 uses version 1 although version 2 was
  // published, a true miss. Tick 4 reads version 3 inside its emit, before the emit returns.
  state.targets[1] = {.scheduled = 100'000, .emitted = 200'000, .published = 300'000};
  state.targets[2] = {.scheduled = 1'980'000, .emitted = 1'990'000, .published = 2'050'000};
  state.targets[3] = {.scheduled = 3'985'000,
                      .emitted = 3'990'000,
                      .published = 4'020'000,
                      .first_use = 4'010'000,
                      .first_use_sequence = 4U};
  // Synthetic older selection: sequence 5 was published before the scheduled target tick,
  // which consumed sequence 2. Report all three ticks without clamping to the smoke threshold.
  state.targets[4] = {
      .scheduled = 5'980'000, .emitted = 5'990'000, .published = 6'050'000, .joint_sequence = 2U};
  const Analysis analysis = analyze(state);
  const bool freshness = analysis.missed_freshness == 1U;
  const bool delivery = analysis.delivery.size() == 1U && analysis.delivery.front() == 20'000;
  const bool staleness = analysis.staleness.size() == 1U && analysis.staleness.front() == 3 &&
                         analysis.worst_staleness == 3 && analysis.worst_stale_version == 4U &&
                         analysis.worst_stale_target == &state.targets[4] &&
                         analysis.worst_stale_target->joint_sequence == 2U &&
                         analysis.worst_stale_newest_joint == 5U &&
                         analysis.worst_stale_joint == &state.ticks[4];
  const bool passed = freshness && delivery && staleness;
  std::cout << "{\"target_freshness_self_test\":{\"missed_freshness\":" << analysis.missed_freshness
            << ",\"delivery_ns\":" << (analysis.delivery.empty() ? -1 : analysis.delivery.front())
            << ",\"staleness_ticks\":"
            << (analysis.staleness.empty() ? -1 : analysis.staleness.front())
            << ",\"passed\":" << (passed ? "true" : "false") << "}}\n";
  return passed ? 0 : 1;
}

// Partition counters by JSON name; also the names `partition.<role>.X` guards accept.
constexpr const auto& kPartitionFields = detail::kRuntimePartitionCounterFields;

// Session phase counters by JSON name; also the names `phase.X` and `per_tick.X` guards accept.
constexpr const auto& kPhaseFields = detail::kRuntimePhaseCounterFields;

// Run counts by JSON name; also the names `count.X` guards accept.
constexpr std::array<std::string_view, 21U> kCountNames{"measured_ticks",
                                                        "missing_ticks",
                                                        "late",
                                                        "coalesced",
                                                        "missed_freshness",
                                                        "measured_target_versions",
                                                        "target_publications",
                                                        "dropped_publications",
                                                        "target_observation_overflows",
                                                        "checksum_failures",
                                                        "grid_errors",
                                                        "target_version_regressions",
                                                        "motion_side_selected",
                                                        "motion_side_reused",
                                                        "motion_side_absent",
                                                        "target_side_selected",
                                                        "target_side_reused",
                                                        "target_side_absent",
                                                        "target_side_borrowed",
                                                        "side_input_staleness_samples",
                                                        "side_input_staleness_max"};

[[nodiscard]] std::uint64_t count_value(const std::string_view name, const State& state,
                                        const Analysis& analysis) {
  const std::array<std::pair<std::string_view, std::uint64_t>, kCountNames.size()> values{{
      {"measured_ticks", analysis.measured_ticks},
      {"missing_ticks", analysis.missing_ticks},
      {"late", analysis.late},
      {"coalesced", analysis.coalesced},
      {"missed_freshness", analysis.missed_freshness},
      {"measured_target_versions", analysis.measured_target_versions},
      {"target_publications", state.target_publications},
      {"dropped_publications", state.dropped_publications.load()},
      {"target_observation_overflows", state.target_observation_overflows.load()},
      {"checksum_failures", state.checksum_failures.load()},
      {"grid_errors", state.grid_errors},
      {"target_version_regressions", state.target_version_regressions},
      {"motion_side_selected", state.motion_side_selected},
      {"motion_side_reused", state.motion_side_reused},
      {"motion_side_absent", state.motion_side_absent},
      {"target_side_selected", state.target_side_selected},
      {"target_side_reused", state.target_side_reused},
      {"target_side_absent", state.target_side_absent},
      {"target_side_borrowed", state.target_side_borrowed},
      {"side_input_staleness_samples", analysis.staleness.size()},
      {"side_input_staleness_max",
       analysis.staleness.empty()
           ? 0U
           : static_cast<std::uint64_t>(std::max<std::int64_t>(
                 0, *std::max_element(analysis.staleness.begin(), analysis.staleness.end())))},
  }};
  const auto found = std::find_if(
      values.begin(), values.end(), [&](const auto& entry) { return entry.first == name; });
  return found != values.end() ? found->second : 0U;
}

[[nodiscard]] std::string_view partition_role(const State& state, const std::uint64_t owner) {
  return owner == state.control_tid  ? std::string_view{"control"}
         : owner == state.target_tid ? std::string_view{"target"}
                                     : std::string_view{"other"};
}

// Per-partition owner accounting, labeled by matching owner threads to the measured lanes.
void emit_partition_counters(std::ostream& output, const State& state,
                             const detail::RuntimePhaseCounters& counters) {
  const auto snapshot = counters.partition_snapshot();
  const auto owners = counters.partition_owner_threads();
  output << ",\"partitions\":[";
  bool first = true;
  for (std::size_t partition = 0U; partition < owners.size(); ++partition) {
    if (owners[partition] == 0U) {
      continue;
    }
    output << (first ? "" : ",") << "{\"index\":" << partition
           << ",\"owner_tid\":" << owners[partition] << ",\"role\":\""
           << partition_role(state, owners[partition]) << '"';
    for (const auto& [name, counter] : kPartitionFields) {
      output << ",\"" << name
             << "\":" << snapshot[partition][detail::RuntimePhaseCounters::index(counter)];
    }
    output << '}';
    first = false;
  }
  output << ']';
}

void emit_phase_counters(std::ostream& output, const State& state,
                         const detail::RuntimePhaseCounters* counters) {
  output << "{\"enabled\":" << (counters != nullptr ? "true" : "false");
  if (counters != nullptr) {
    const auto snapshot = counters->snapshot();
    for (const auto& [name, counter] : kPhaseFields) {
      output << ",\"" << name << "\":" << snapshot[detail::RuntimePhaseCounters::index(counter)];
    }
    emit_partition_counters(output, state, *counters);
  }
  output << '}';
}

// Calls per thread inside the measured window, from the preloaded shim.
void emit_yield_counts(std::ostream& output, const State& state) {
  const bool measured = state.yield_begin.taken && state.yield_end.taken;
  output << "{\"shim_loaded\":" << (state.yield_snapshot != nullptr ? "true" : "false")
         << ",\"noop\":" << (state.yield_noop ? "true" : "false")
         << ",\"window_measured\":" << (measured ? "true" : "false");
  if (measured) {
    const auto delta = [&](const std::uint64_t tid) {
      return state.yield_end.count(tid) - state.yield_begin.count(tid);
    };
    std::uint64_t total{};
    output << ",\"threads\":[";
    for (std::size_t index = 0U; index < state.yield_end.size; ++index) {
      const std::uint64_t tid = state.yield_end.tids[index];
      total += delta(tid);
      output << (index == 0U ? "" : ",") << "{\"tid\":" << tid << ",\"calls\":" << delta(tid)
             << '}';
    }
    const std::uint64_t control = delta(state.control_tid);
    const std::uint64_t target = delta(state.target_tid);
    output << "],\"control\":" << control << ",\"target\":" << target << ",\"other\":"
           << total - control - (state.target_tid != state.control_tid ? target : 0U)
           << ",\"control_per_tick\":"
           << static_cast<double>(control) / std::max<double>(state.options.samples, 1.0);
  }
  output << '}';
}

void emit_native_thread(std::ostream& output, const NativeThreadEvidence& thread) {
  output << "{\"requested_cpu\":" << thread.requested_cpu
         << ",\"requested_fifo_priority\":" << thread.requested_priority
         << ",\"observed_cpu\":" << thread.observed_cpu
         << ",\"observed_policy\":" << thread.observed_policy
         << ",\"observed_priority\":" << thread.observed_priority << ",\"error\":\"" << thread.error
         << "\"}";
}

void emit_lane_evidence(std::ostream& output, const BenchmarkSupportThreadControl& threads) {
  output << '[';
  bool first = true;
  for (const auto& lane : threads.lane_evidence()) {
    if (!first) {
      output << ',';
    }
    first = false;
    const auto applied = lane.applied.value_or(detail::ExecutionLanePlacementSnapshot{});
    output << "{\"lane\":" << lane.execution_lane
           << ",\"disposition\":" << static_cast<unsigned>(lane.disposition) << ",\"affinity\":[";
    for (std::size_t index = 0U; index < applied.affinity.cores.size(); ++index) {
      if (index != 0U) {
        output << ',';
      }
      output << applied.affinity.cores[index];
    }
    output << "],\"policy\":" << static_cast<unsigned>(applied.policy)
           << ",\"priority\":" << applied.priority.value_or(0) << '}';
  }
  output << ']';
}

// The process-local delivery of ReadJointState's joint-state output: one `policy:kind` per
// Channel service, sorted and joined by '+'. The kind is direct (direct retention), emit
// (delivered inside emit()), pump (HoloMQ event wake), or cut (the post-compute activation cut).
[[nodiscard]] std::string joint_state_delivery(const ExecutionPlan& plan) {
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  if (!state) {
    return "unavailable";
  }
  std::vector<PortId> sources;
  for (const PortPlanRecord& port : state->ports) {
    if (port.direction == PortDirection::kOutput &&
        port.local_name.starts_with("out_joint_state")) {
      sources.push_back(port.id);
    }
  }
  std::vector<std::string> parts;
  for (const auto& service : state->process_local_channel_services) {
    if (std::find(sources.begin(), sources.end(), service.source) == sources.end()) {
      continue;
    }
    const std::string_view policy =
        service.queue_policy == QueuePolicy::kBlocking ? "blocking" : "drop_oldest";
    const std::string_view kind =
        service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kDirectRetained ? "direct"
        : service.emit_admitted                                                           ? "emit"
        : service.event_wake                                                              ? "pump"
                                                                                          : "cut";
    parts.push_back(std::string{policy} + ':' + std::string{kind});
  }
  std::sort(parts.begin(), parts.end());
  parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
  std::string joined;
  for (const std::string& part : parts) {
    joined += (joined.empty() ? "" : "+") + part;
  }
  return joined.empty() ? std::string{"none"} : joined;
}

// Whether a name table (pairs keyed by name, or plain names) contains `name`.
template <typename Table>
[[nodiscard]] bool names_field(const Table& table, const std::string_view name) {
  return std::any_of(table.begin(), table.end(), [&](const auto& entry) {
    if constexpr (std::is_convertible_v<decltype(entry), std::string_view>) {
      return entry == name;
    } else {
      return entry.first == name;
    }
  });
}

// Parse `path op value`; reject unknown paths so a typo or a counter missing from this build
// fails at startup instead of passing vacuously.
[[nodiscard]] std::optional<Guard> parse_guard(const std::string_view text) {
  constexpr std::array<std::pair<std::string_view, Guard::Op>, 6U> kOps{{
      {"==", Guard::Op::kEq},
      {"!=", Guard::Op::kNe},
      {">=", Guard::Op::kGe},
      {"<=", Guard::Op::kLe},
      {">", Guard::Op::kGt},
      {"<", Guard::Op::kLt},
  }};
  for (const auto& [symbol, op] : kOps) {
    const std::size_t at = text.find(symbol);
    if (at == std::string_view::npos) {
      continue;
    }
    Guard guard{.text = std::string{text}, .path = std::string{text.substr(0U, at)}, .op = op};
    const std::string_view rhs = text.substr(at + symbol.size());
    const std::string_view path = guard.path;
    if (path == "plan.joint_state") {
      if ((op != Guard::Op::kEq && op != Guard::Op::kNe) || rhs.empty()) {
        return std::nullopt;
      }
      guard.expected = std::string{rhs};
      return guard;
    }
    const auto known = [&](const std::string_view prefix, const auto& table) {
      return path.starts_with(prefix) && names_field(table, path.substr(prefix.size()));
    };
    if (!known("phase.", kPhaseFields) && !known("per_tick.", kPhaseFields) &&
        !known("partition.control.", kPartitionFields) &&
        !known("partition.target.", kPartitionFields) && !known("count.", kCountNames)) {
      return std::nullopt;
    }
    const auto parsed = std::from_chars(rhs.data(), rhs.data() + rhs.size(), guard.value);
    if (parsed.ec != std::errc() || parsed.ptr != rhs.data() + rhs.size()) {
      return std::nullopt;
    }
    return guard;
  }
  return std::nullopt;
}

struct GuardResult {
  const Guard* guard{};
  std::string actual;
  bool passed{};
};

[[nodiscard]] bool compare(const double actual, const Guard::Op op, const double expected) {
  switch (op) {
    case Guard::Op::kEq:
      return actual == expected;
    case Guard::Op::kNe:
      return actual != expected;
    case Guard::Op::kGe:
      return actual >= expected;
    case Guard::Op::kLe:
      return actual <= expected;
    case Guard::Op::kGt:
      return actual > expected;
    case Guard::Op::kLt:
      return actual < expected;
  }
  return false;
}

// Evaluate every expectation. Counter guards need `counters`; without them they fail.
// Dispatch the declared guard paths in one place, preserving unavailable-evidence behavior
// across plan, phase, partition, and measured-count expectations.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::vector<GuardResult> evaluate_guards(
    const State& state, const Analysis* analysis, const detail::RuntimePhaseCounters* counters) {
  std::vector<GuardResult> results;
  const auto phase_value = [&](const std::string_view name) -> std::optional<double> {
    if (counters == nullptr) {
      return std::nullopt;
    }
    const auto found = std::find_if(kPhaseFields.begin(),
                                    kPhaseFields.end(),
                                    [&](const auto& entry) { return entry.first == name; });
    return static_cast<double>(
        counters->snapshot()[detail::RuntimePhaseCounters::index(found->second)]);
  };
  const auto partition_value = [&](const std::string_view role,
                                   const std::string_view name) -> std::optional<double> {
    if (counters == nullptr) {
      return std::nullopt;
    }
    const auto found = std::find_if(kPartitionFields.begin(),
                                    kPartitionFields.end(),
                                    [&](const auto& entry) { return entry.first == name; });
    const auto snapshot = counters->partition_snapshot();
    const auto owners = counters->partition_owner_threads();
    double total = 0.0;
    bool seen = false;
    for (std::size_t partition = 0U; partition < owners.size(); ++partition) {
      if (owners[partition] != 0U && partition_role(state, owners[partition]) == role) {
        total += static_cast<double>(
            snapshot[partition][detail::RuntimePhaseCounters::index(found->second)]);
        seen = true;
      }
    }
    return seen ? std::optional<double>{total} : std::nullopt;
  };
  const double ticks = static_cast<double>(state.options.samples + state.options.warmup);
  for (const Guard& guard : state.options.expectations) {
    GuardResult result{.guard = &guard};
    const std::string_view path = guard.path;
    if (path == "plan.joint_state") {
      result.actual = state.joint_state_delivery;
      result.passed =
          (state.joint_state_delivery == guard.expected) == (guard.op == Guard::Op::kEq);
      results.push_back(std::move(result));
      continue;
    }
    std::optional<double> actual;
    if (path.starts_with("phase.")) {
      actual = phase_value(path.substr(6U));
    } else if (path.starts_with("per_tick.")) {
      const auto total = phase_value(path.substr(9U));
      actual = total ? std::optional<double>{*total / ticks} : std::nullopt;
    } else if (path.starts_with("partition.")) {
      const std::string_view rest = path.substr(10U);
      const std::size_t dot = rest.find('.');
      actual = partition_value(rest.substr(0U, dot), rest.substr(dot + 1U));
    } else if (path.starts_with("count.") && analysis != nullptr) {
      actual = static_cast<double>(count_value(path.substr(6U), state, *analysis));
    }
    if (actual) {
      std::ostringstream text;
      text << *actual;
      result.actual = text.str();
      result.passed = compare(*actual, guard.op, guard.value);
    } else {
      result.actual = "unavailable";
    }
    results.push_back(std::move(result));
  }
  return results;
}

[[nodiscard]] bool guards_passed(const std::vector<GuardResult>& results) {
  return std::all_of(
      results.begin(), results.end(), [](const auto& result) { return result.passed; });
}

// Escape `text` for a JSON string: guard text is authored on the command line.
[[nodiscard]] std::string json_escape(const std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char character : text) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20U) {
          constexpr std::string_view kHex = "0123456789abcdef";
          const auto code = static_cast<unsigned char>(character);
          escaped += "\\u00";
          escaped += kHex[code >> 4U];
          escaped += kHex[code & 0x0FU];
        } else {
          escaped += character;
        }
    }
  }
  return escaped;
}

void emit_guards(std::ostream& output, const State& state,
                 const std::vector<GuardResult>& results) {
  output << ",\"plan\":{\"joint_state\":\"" << json_escape(state.joint_state_delivery)
         << "\"},\"guards\":{\"passed\":" << (guards_passed(results) ? "true" : "false")
         << ",\"results\":[";
  for (std::size_t index = 0U; index < results.size(); ++index) {
    const GuardResult& result = results[index];
    output << (index == 0U ? "" : ",") << "{\"expect\":\"" << json_escape(result.guard->text)
           << "\",\"actual\":\"" << json_escape(result.actual)
           << "\",\"passed\":" << (result.passed ? "true" : "false") << '}';
  }
  output << "]}";
}

struct RunEvidence {
  std::optional<RunTermination> termination;
  std::vector<Diagnostic> diagnostics;
  const BenchmarkSupportThreadControl* threads{};
  const detail::RuntimePhaseCounters* counters{};
  std::optional<NoFrameworkRun> native;
  bool timed_out{};
  bool mlocked{};
};

[[nodiscard]] bool valid_run(const State& state, const Analysis& analysis,
                             const RunEvidence& evidence) {
  const bool framework_ok = evidence.native.has_value()
                                ? evidence.native->completed &&
                                      evidence.native->control.error.empty() &&
                                      evidence.native->target.error.empty()
                                : evidence.termination == RunTermination::kStopped &&
                                      evidence.diagnostics.empty() && state.clock_offset_valid;
  const bool yield_ok =
      !state.options.require_yield_shim || (state.yield_begin.taken && state.yield_end.taken);
  return !evidence.timed_out && framework_ok && yield_ok && analysis.missing_ticks == 0U &&
         analysis.measured_ticks == state.options.samples && analysis.ordered &&
         state.checksum_failures.load() == 0U && state.grid_errors == 0U &&
         state.target_version_regressions == 0U && state.target_publications != 0U &&
         state.target_observation_overflows.load() == 0U;
}

// Emit the result schema in field order; conditional evidence stays beside its null fallback
// so the complete benchmark record can be compared directly with the schema.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void emit_result(std::ostream& output, const State& state, const Analysis& analysis,
                 const RunEvidence& evidence, const bool valid,
                 const std::vector<GuardResult>& guards) {
  const auto& options = state.options;
  output << std::fixed << std::setprecision(3);
  output << "{\"schema\":\"holoscan.control_loop.v1\",\"valid\":" << (valid ? "true" : "false")
         << ",\"mode\":\"" << mode_name(options.mode) << "\",\"topology\":\""
         << topology_name(options.topology) << "\",\"control_rate_hz\":" << options.control_rate_hz
         << ",\"target_rate_hz\":" << options.target_rate_hz << ",\"samples\":" << options.samples
         << ",\"warmup\":" << options.warmup
         << ",\"placement_request\":{\"control_cpu\":" << options.control_cpu
         << ",\"target_cpu\":" << options.target_cpu
         << ",\"controller_cpu\":" << options.controller_cpu << ",\"pump_cpu\":" << options.pump_cpu
         << ",\"control_fifo_priority\":" << options.control_fifo_priority
         << ",\"target_fifo_priority\":" << options.target_fifo_priority
         << ",\"busy_poll_control\":" << (options.busy_poll_control ? "true" : "false")
         << "},\"controls\":{\"force_holomq\":" << (options.force_holomq ? "true" : "false")
         << ",\"force_holomq_event_wake\":" << (options.force_holomq_event_wake ? "true" : "false")
         << ",\"force_holomq_fanout\":" << (options.force_holomq_fanout ? "true" : "false")
         << ",\"force_pump_delivery\":" << (options.force_pump_delivery ? "true" : "false")
         << ",\"disable_side_input_continuation\":"
         << (options.disable_side_input_continuation ? "true" : "false")
         << ",\"disable_boundary_causal_chain\":"
         << (options.disable_boundary_causal_chain ? "true" : "false") << ",\"control_policy\":\""
         << (options.control_policy == QueuePolicy::kBlocking ? "blocking" : "drop_oldest") << '"'
         << ",\"export_joint_state\":" << (options.export_joint_state ? "true" : "false")
         << ",\"joint_payload\":\"" << joint_payload_name(options.joint_payload) << '"'
         << ",\"side_selector\":\""
         << (options.side_selector == SideSelector::kDrainAll       ? "drain_all"
             : options.side_selector == SideSelector::kLatestWithin ? "latest_within"
                                                                    : "reuse_last")
         << "\",\"side_selector_scope\":\""
         << (options.side_scope == SideScope::kBoth     ? "both"
             : options.side_scope == SideScope::kMotion ? "motion"
                                                        : "target")
         << "\",\"latest_within_us\":" << options.latest_within_us
         << ",\"mlock\":" << (evidence.mlocked ? "true" : "false") << "},\"work\":{";
  output << "\"ns_per_iteration\":" << state.work.ns_per_iteration;
  for (std::size_t stage = 0U; stage < kStageCount; ++stage) {
    output << ",\"" << kStageNames[stage] << "\":{\"target_ns\":" << options.work_ns[stage]
           << ",\"iterations\":" << state.work.iterations[stage] << '}';
  }
  output << "},\"clock_offset_ns\":" << state.clock_offset_ns
         << ",\"clock_offset_uncertainty_ns\":" << state.clock_offset_uncertainty_ns
         << ",\"kpis_ns\":{\"control_loop_latency\":";
  emit_distribution(output, summarize(analysis.control_loop));
  output << ",\"active_path\":";
  emit_distribution(output, summarize(analysis.active));
  output << ",\"entry_latency\":";
  emit_distribution(output, summarize(analysis.entry));
  output << ",\"control_period\":";
  emit_distribution(output, summarize(analysis.period));
  output << ",\"control_period_error\":";
  emit_distribution(output, summarize(analysis.period_error));
  output << ",\"target_pose_response_latency\":";
  emit_distribution(output, summarize(analysis.tprl));
  output << ",\"target_delivery\":";
  emit_distribution(output, summarize(analysis.delivery));
  output << ",\"hop_read_to_motion\":";
  emit_distribution(output, summarize(analysis.read_to_motion));
  output << ",\"hop_motion_to_safety\":";
  emit_distribution(output, summarize(analysis.motion_to_safety));
  output << ",\"hop_safety_to_write\":";
  emit_distribution(output, summarize(analysis.safety_to_write));
  output << '}';
  if (const TickRecord* worst = analysis.worst_active) {
    // Offsets from the scheduled activation, so the stalled hop of the slowest tick is visible.
    const auto offset = [&](const std::int64_t value) {
      return value != 0 ? value - worst->scheduled : std::int64_t{-1};
    };
    output << ",\"worst_active_tick\":{\"sequence\":" << analysis.worst_active_sequence
           << ",\"entry\":" << offset(worst->entry) << ",\"read_emit\":" << offset(worst->read_emit)
           << ",\"motion_entry\":" << offset(worst->motion_entry)
           << ",\"safety_entry\":" << offset(worst->safety_entry)
           << ",\"write_entry\":" << offset(worst->write_entry)
           << ",\"done\":" << offset(worst->done) << '}';
  }
  output << ",\"side_input_staleness_ticks\":";
  emit_distribution(output, summarize(analysis.staleness));
  output << ",\"worst_side_input_staleness\":";
  if (const TargetRecord* worst = analysis.worst_stale_target) {
    const auto scheduled =
        worst->scheduled + (state.target_times_native ? 0 : state.clock_offset_ns);
    output << "{\"target_version\":" << analysis.worst_stale_version
           << ",\"staleness_ticks\":" << analysis.worst_staleness
           << ",\"target_scheduled_ns\":" << scheduled << ",\"target_emit_ns\":" << worst->emitted
           << ",\"target_publish_ns\":" << worst->published
           << ",\"selected_joint_sequence\":" << worst->joint_sequence
           << ",\"newest_joint_sequence\":" << analysis.worst_stale_newest_joint
           << ",\"newest_joint_emit_ns\":" << analysis.worst_stale_joint->read_emit << '}';
  } else {
    output << "null";
  }
  output << ",\"counts\":{";
  for (std::size_t index = 0U; index < kCountNames.size(); ++index) {
    output << (index == 0U ? "\"" : ",\"") << kCountNames[index]
           << "\":" << count_value(kCountNames[index], state, analysis);
  }
  output << "},\"threads\":{\"control_tid\":" << state.control_tid
         << ",\"target_tid\":" << state.target_tid
         << ",\"control_timer_slack_ns\":" << state.control_timer_slack_ns
         << ",\"target_timer_slack_ns\":" << state.target_timer_slack_ns;
  if (evidence.native) {
    output << ",\"control\":";
    emit_native_thread(output, evidence.native->control);
    output << ",\"target\":";
    emit_native_thread(output, evidence.native->target);
  }
  output << '}';
  if (evidence.threads != nullptr) {
    output << ",\"support_threads\":";
    benchmarking::emit_support_thread_evidence(output, *evidence.threads);
    output << ",\"lane_startup\":";
    emit_lane_evidence(output, *evidence.threads);
  }
  output << ",\"sched_yield\":";
  emit_yield_counts(output, state);
  output << ",\"phase_counters\":";
  emit_phase_counters(output, state, evidence.counters);
  output << ",\n";
  benchmarking::emit_runtime_thread_usage(output, evidence.counters);
  output << "\"termination\":";
  if (evidence.termination) {
    output << static_cast<unsigned>(*evidence.termination);
  } else {
    output << "null";
  }
  output << ",\"timed_out\":" << (evidence.timed_out ? "true" : "false") << ",\"diagnostics\":";
  benchmarking::emit_runtime_diagnostics(output, evidence.diagnostics);
  emit_guards(output, state, guards);
  output << "}\n";
}

[[nodiscard]] bool write_observations(const std::string& path, const State& state) {
  std::ofstream file(path);
  if (!file) {
    return false;
  }
  file << "sequence,scheduled_ns,entry_ns,done_ns,read_emit_ns,motion_entry_ns,safety_entry_ns,"
          "write_entry_ns,target_version\n";
  for (std::size_t index = 0U; index < state.ticks.size(); ++index) {
    const TickRecord& tick = state.ticks[index];
    file << state.options.warmup + 1U + index << ',' << tick.scheduled << ',' << tick.entry << ','
         << tick.done << ',' << tick.read_emit << ',' << tick.motion_entry << ','
         << tick.safety_entry << ',' << tick.write_entry << ',' << tick.target_version << '\n';
  }
  file.flush();
  return static_cast<bool>(file);
}

// Keep the run lifetime, validation, guard evaluation, and artifact writes in one sequence;
// every early failure must still use the same captured evidence and cleanup.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] int run_benchmark(Options options) {
  RunEvidence evidence;
#if defined(__linux__)
  if (options.mlock) {
    evidence.mlocked = mlockall(MCL_CURRENT | MCL_FUTURE) == 0;
  }
#endif
  const Work work = calibrate(options);
  State state{std::move(options), work};
  g_state = &state;
#if defined(__linux__)
  state.yield_snapshot = reinterpret_cast<YieldSnapshotFunction>(
      dlsym(RTLD_DEFAULT, "control_loop_yield_shim_snapshot"));
  if (const auto noop =
          reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "control_loop_yield_shim_noop"))) {
    state.yield_noop = noop() != 0;
  }
#endif
  const std::int64_t run_ns =
      static_cast<std::int64_t>(state.options.warmup + state.options.samples) *
      state.control_period_ns();
  const auto timeout = std::chrono::milliseconds{
      state.options.timeout_ms != 0U ? static_cast<std::int64_t>(state.options.timeout_ms)
                                     : run_ns / 1'000'000 * 2 + 10'000};
  // The result stream serves the plan report and the run result alike.
  std::ofstream file;
  if (!state.options.output.empty()) {
    file.open(state.options.output);
    if (!file) {
      g_state = nullptr;
      return 2;
    }
  }
  std::ostream& output = file.is_open() ? static_cast<std::ostream&>(file) : std::cout;
  std::shared_ptr<BenchmarkSupportThreadControl> threads;
  std::shared_ptr<detail::RuntimePhaseCounters> counters;
  if (state.options.mode == Mode::kNoFramework) {
    evidence.native = run_no_framework(state, timeout);
    evidence.timed_out = evidence.native->timed_out;
  } else {
    if (state.options.joint_payload == JointPayload::kTensorDevice) {
      int devices = 0;
      if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cerr << "--joint-payload tensor-device needs a CUDA device\n";
        g_state = nullptr;
        return 77;
      }
    }
    auto plan = [&] {
      switch (state.options.joint_payload) {
        case JointPayload::kSchema:
          return make_plan_with<SchemaJointCodec>(state.options);
        case JointPayload::kTensorHost:
          return make_plan_with<TensorJointCodec<TensorStorage::kPageableHost>>(state.options);
        case JointPayload::kTensorPinned:
          return make_plan_with<TensorJointCodec<TensorStorage::kPinnedHost>>(state.options);
        case JointPayload::kTensorDevice:
          return make_plan_with<TensorJointCodec<TensorStorage::kDevice>>(state.options);
        case JointPayload::kDescriptor:
          return make_plan_with<DescriptorJointCodec>(state.options);
        case JointPayload::kDescriptorBorrow:
          return make_plan_with<DescriptorBorrowJointCodec>(state.options);
        case JointPayload::kPod:
          break;
      }
      return make_plan_with<PodJointCodec>(state.options);
    }();
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      g_state = nullptr;
      return 2;
    }
    state.joint_state_delivery = joint_state_delivery(plan);
    if (state.options.plan_only) {
      const auto guards = evaluate_guards(state, nullptr, nullptr);
      output << "{\"schema\":\"holoscan.control_loop.plan.v1\",\"topology\":\""
             << topology_name(state.options.topology) << '"';
      emit_guards(output, state, guards);
      output << "}\n";
      output.flush();
      g_state = nullptr;
      if (!output) {
        std::cerr << "control-loop plan report serialization failed\n";
        return 2;
      }
      return guards_passed(guards) ? 0 : 1;
    }
    threads = std::make_shared<BenchmarkSupportThreadControl>(std::vector<SupportThreadBinding>{
        {.role = SupportThreadRole::kController,
         .requested_cpu = state.options.controller_cpu,
         .required = true},
        {.role = SupportThreadRole::kChannelPump, .requested_cpu = state.options.pump_cpu}});
    plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, threads);
    plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, threads);
    if (state.options.disable_side_input_continuation) {
      plan = detail::RuntimeTestPlanAccess::disable_side_input_continuation_for_test(plan);
    }
    if (state.options.disable_boundary_causal_chain) {
      plan = detail::RuntimeTestPlanAccess::disable_boundary_causal_chain_for_test(plan);
    }
    if (state.options.counters) {
      counters = std::make_shared<detail::RuntimePhaseCounters>();
      plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    }
    auto session = run_async(plan);
    evidence.timed_out = !session.wait_for(timeout);
    if (evidence.timed_out) {
      session.request_stop();
    }
    session.wait();
    evidence.termination = session.termination();
    const auto diagnostics = session.diagnostics();
    evidence.diagnostics.assign(diagnostics.begin(), diagnostics.end());
    evidence.threads = threads.get();
    evidence.counters = counters.get();
  }
  const Analysis analysis = analyze(state);
  const bool valid = valid_run(state, analysis, evidence);
  if (!state.options.observations.empty() &&
      !write_observations(state.options.observations, state)) {
    std::cerr << "control-loop observations could not be written to " << state.options.observations
              << '\n';
    g_state = nullptr;
    return 2;
  }
  const auto guards = evaluate_guards(state, &analysis, evidence.counters);
  emit_result(output, state, analysis, evidence, valid, guards);
  output.flush();
  g_state = nullptr;
  // A failed stream leaves a truncated result that the runner cannot parse; never report success.
  if (!output) {
    std::cerr << "control-loop result serialization failed\n";
    return 2;
  }
  return valid && guards_passed(guards) ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------
// Command line.

[[nodiscard]] std::optional<std::int64_t> parse_integer(const std::string_view text) {
  std::int64_t value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

// Parse "read=300,plan=2400,..." into per-stage values; unnamed stages keep their defaults.
template <typename T>
[[nodiscard]] bool parse_stage_list(const std::string_view text, std::array<T, kStageCount>& out) {
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const auto end = std::min(text.find(',', begin), text.size());
    const auto item = text.substr(begin, end - begin);
    const auto equals = item.find('=');
    if (equals == std::string_view::npos) {
      return false;
    }
    const auto name = item.substr(0U, equals);
    const auto value = parse_integer(item.substr(equals + 1U));
    const auto stage = std::find(kStageNames.begin(), kStageNames.end(), name);
    if (!value || *value < 0 || stage == kStageNames.end()) {
      return false;
    }
    out[static_cast<std::size_t>(stage - kStageNames.begin())] = static_cast<T>(*value);
    begin = end + 1U;
  }
  return true;
}

// Keep command-line spellings next to their typed validation and assignments; the flat
// dispatcher makes the accepted benchmark variants and rejection rules visible together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<Options> parse(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view key{argv[index]};
    if (key == "--busy-poll-control") {
      options.busy_poll_control = true;
      continue;
    }
    if (key == "--force-holomq") {
      options.force_holomq = true;
      continue;
    }
    if (key == "--force-holomq-event-wake") {
      options.force_holomq_event_wake = true;
      continue;
    }
    if (key == "--force-holomq-fanout") {
      options.force_holomq_fanout = true;
      continue;
    }
    if (key == "--force-pump-delivery") {
      options.force_pump_delivery = true;
      continue;
    }
    if (key == "--disable-side-input-continuation") {
      options.disable_side_input_continuation = true;
      continue;
    }
    if (key == "--disable-boundary-causal-chain") {
      options.disable_boundary_causal_chain = true;
      continue;
    }
    if (key == "--counters") {
      options.counters = true;
      continue;
    }
    if (key == "--plan-only") {
      options.plan_only = true;
      continue;
    }
    if (key == "--export-joint-state") {
      options.export_joint_state = true;
      continue;
    }
    if (key == "--mlock") {
      options.mlock = true;
      continue;
    }
    if (key == "--require-yield-shim") {
      options.require_yield_shim = true;
      continue;
    }
    if (++index == argc) {
      return std::nullopt;
    }
    const std::string_view value{argv[index]};
    if (key == "--mode") {
      if (value == "split") {
        options.mode = Mode::kSplit;
      } else if (value == "fused") {
        options.mode = Mode::kFused;
      } else if (value == "no-framework") {
        options.mode = Mode::kNoFramework;
      } else {
        return std::nullopt;
      }
    } else if (key == "--topology") {
      constexpr std::array topologies{Topology::kRcb,
                                      Topology::kSplitOutputs,
                                      Topology::kSingle,
                                      Topology::kSamePartitionPort,
                                      Topology::kRcbNoTargetInput};
      const auto found = std::find_if(topologies.begin(), topologies.end(), [&](Topology item) {
        return topology_name(item) == value;
      });
      if (found == topologies.end()) {
        return std::nullopt;
      }
      options.topology = *found;
    } else if (key == "--work-profile") {
      if (value == "robotics") {
        options.work_ns = kRoboticsWorkNs;
      } else if (value == "zero") {
        options.work_ns = {};
      } else {
        return std::nullopt;
      }
    } else if (key == "--work-ns") {
      if (!parse_stage_list(value, options.work_ns)) {
        return std::nullopt;
      }
    } else if (key == "--work-iterations") {
      std::array<std::uint64_t, kStageCount> iterations{};
      if (!parse_stage_list(value, iterations)) {
        return std::nullopt;
      }
      options.work_iterations = iterations;
    } else if (key == "--control-policy") {
      if (value == "drop-oldest") {
        options.control_policy = QueuePolicy::kDropOldest;
      } else if (value == "blocking") {
        options.control_policy = QueuePolicy::kBlocking;
      } else {
        return std::nullopt;
      }
    } else if (key == "--joint-payload") {
      if (value == "pod") {
        options.joint_payload = JointPayload::kPod;
      } else if (value == "schema") {
        options.joint_payload = JointPayload::kSchema;
      } else if (value == "tensor-host") {
        options.joint_payload = JointPayload::kTensorHost;
      } else if (value == "tensor-pinned") {
        options.joint_payload = JointPayload::kTensorPinned;
      } else if (value == "tensor-device") {
        options.joint_payload = JointPayload::kTensorDevice;
      } else if (value == "descriptor") {
        options.joint_payload = JointPayload::kDescriptor;
      } else if (value == "descriptor-borrow") {
        options.joint_payload = JointPayload::kDescriptorBorrow;
      } else {
        return std::nullopt;
      }
    } else if (key == "--side-selector") {
      if (value == "drain-all") {
        options.side_selector = SideSelector::kDrainAll;
      } else if (value == "latest-within") {
        options.side_selector = SideSelector::kLatestWithin;
      } else if (value == "reuse-last") {
        options.side_selector = SideSelector::kReuseLast;
      } else {
        return std::nullopt;
      }
    } else if (key == "--side-selector-scope") {
      if (value == "both") {
        options.side_scope = SideScope::kBoth;
      } else if (value == "motion") {
        options.side_scope = SideScope::kMotion;
      } else if (value == "target") {
        options.side_scope = SideScope::kTarget;
      } else {
        return std::nullopt;
      }
    } else if (key == "--expect") {
      auto guard = parse_guard(value);
      if (!guard) {
        return std::nullopt;
      }
      options.expectations.push_back(std::move(*guard));
    } else if (key == "--output") {
      options.output = value;
    } else if (key == "--observations") {
      options.observations = value;
    } else {
      const auto number = parse_integer(value);
      const bool cpu_option = key == "--control-cpu" || key == "--target-cpu" ||
                              key == "--controller-cpu" || key == "--pump-cpu";
      if (!number || *number < -1 || *number > 100'000'000 || (!cpu_option && *number < 0)) {
        return std::nullopt;
      }
      const int cpu = static_cast<int>(*number);
      const auto count = static_cast<std::uint32_t>(std::max<std::int64_t>(*number, 0));
      if (key == "--control-cpu") {
        options.control_cpu = cpu;
      } else if (key == "--target-cpu") {
        options.target_cpu = cpu;
      } else if (key == "--controller-cpu") {
        options.controller_cpu = cpu;
      } else if (key == "--pump-cpu") {
        options.pump_cpu = cpu;
      } else if (key == "--control-rate-hz") {
        options.control_rate_hz = count;
      } else if (key == "--target-rate-hz") {
        options.target_rate_hz = count;
      } else if (key == "--samples") {
        options.samples = count;
      } else if (key == "--warmup") {
        options.warmup = count;
      } else if (key == "--timeout-ms") {
        options.timeout_ms = count;
      } else if (key == "--control-fifo-priority") {
        options.control_fifo_priority = count;
      } else if (key == "--target-fifo-priority") {
        options.target_fifo_priority = count;
      } else if (key == "--timer-slack-ns") {
        options.timer_slack_ns = count;
      } else if (key == "--latest-within-us") {
        options.latest_within_us = count;
      } else {
        return std::nullopt;
      }
    }
  }
  const bool valid_rates = options.control_rate_hz >= 1U && options.control_rate_hz <= 10'000U &&
                           options.target_rate_hz >= 1U && options.target_rate_hz <= 10'000U;
  if (!valid_rates || options.samples == 0U || options.samples > 10'000'000U ||
      options.warmup > 1'000'000U || options.control_fifo_priority > 99U ||
      options.target_fifo_priority > 99U ||
      (options.force_pump_delivery && options.force_holomq_event_wake) ||
      // The other HoloMQ controls already keep the fanout on HoloMQ.
      (options.force_holomq_fanout &&
       (options.force_holomq || options.force_holomq_event_wake || options.force_pump_delivery)) ||
      (options.mode != Mode::kSplit &&
       (options.force_holomq || options.force_holomq_event_wake || options.force_holomq_fanout ||
        options.force_pump_delivery || options.disable_side_input_continuation ||
        options.disable_boundary_causal_chain ||
        options.control_policy != QueuePolicy::kDropOldest || options.export_joint_state ||
        options.side_selector != SideSelector::kDrainAll ||
        options.joint_payload != JointPayload::kPod || options.busy_poll_control ||
        options.topology == Topology::kSamePartitionPort ||
        options.topology == Topology::kSplitOutputs ||
        options.topology == Topology::kRcbNoTargetInput))) {
    return std::nullopt;
  }
#if defined(__linux__)
  // A native lane places itself through a fixed cpu_set_t, which cannot name a CPU at or above
  // CPU_SETSIZE.
  if (options.mode == Mode::kNoFramework &&
      (options.control_cpu >= CPU_SETSIZE || options.target_cpu >= CPU_SETSIZE)) {
    return std::nullopt;
  }
#endif
  // Counter guards need --counters; a plan-only run checks plan guards only; the no-framework
  // loop has no plan.
  for (const Guard& guard : options.expectations) {
    const bool plan_guard = guard.path.starts_with("plan.");
    const bool counter_guard = !plan_guard && !guard.path.starts_with("count.");
    if ((counter_guard && !options.counters) || (options.plan_only && !plan_guard) ||
        (plan_guard && options.mode == Mode::kNoFramework)) {
      return std::nullopt;
    }
  }
  if (options.plan_only && options.mode == Mode::kNoFramework) {
    return std::nullopt;
  }
  // A Tensor export needs a Tensor topic contract this benchmark does not author.
  if (options.export_joint_state && options.joint_payload != JointPayload::kPod &&
      options.joint_payload != JointPayload::kSchema) {
    return std::nullopt;
  }
  return options;
}

}  // namespace
}  // namespace holoscan

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--self-test-native-schedule") {
    return holoscan::self_test_native_schedule();
  }
  if (argc == 2 && std::string_view(argv[1]) == "--self-test-target-freshness") {
    return holoscan::self_test_target_freshness();
  }
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout
        << "Robot control loop: 1 kHz read -> plan -> safety -> write chain and a 137 Hz target.\n"
           "  --mode split|fused|no-framework  (fused and no-framework accept rcb or single)\n"
           "  --topology rcb|split-outputs|single|same-partition-port|rcb-no-target-input\n"
           "  --control-rate-hz N --target-rate-hz N --samples N --warmup N --timeout-ms N\n"
           "  --work-profile robotics|zero --work-ns read=300,plan=2400,safety=400,write=200,"
           "target=1000\n"
           "  --work-iterations read=N,... (replay calibrated iteration counts)\n"
           "  --control-cpu N --target-cpu N --controller-cpu N --pump-cpu N\n"
           "  --control-fifo-priority N --target-fifo-priority N --busy-poll-control\n"
           "  --timer-slack-ns N (applied to the control and target threads; 1 is the minimum)\n"
           "  --force-holomq --force-holomq-event-wake --force-holomq-fanout\n"
           "  --force-pump-delivery --disable-side-input-continuation\n"
           "  --disable-boundary-causal-chain --counters --mlock\n"
           "  --control-policy drop-oldest|blocking (the control chain's edges; target edges stay\n"
           "    drop-oldest)\n"
           "  --export-joint-state (also export the joint state as an external topic)\n"
           "  --joint-payload pod|schema|tensor-host|tensor-pinned|tensor-device|\n"
           "    descriptor|descriptor-borrow (joint-state wire type)\n"
           "  --side-selector drain-all|latest-within|reuse-last\n"
           "  --side-selector-scope both|motion|target (default both)\n"
           "  --latest-within-us N (LatestWithin max age; producers stamp capture time)\n"
           "  --expect PATH<op>VALUE (repeatable; ops == != >= <= > <; PATH is phase.X, "
           "per_tick.X,\n"
           "    partition.control|target.X, count.X, or plan.joint_state; exit 1 when one fails)\n"
           "  --plan-only (compile, report plan.joint_state, check plan guards, and exit)\n"
           "  --require-yield-shim (with LD_PRELOAD=libcontrol_loop_yield_shim.so)\n"
           "  --output PATH --observations PATH.csv\n"
           "  --self-test-native-schedule (alone: compare native tick selection with OnClock)\n"
           "  --self-test-target-freshness (alone: check target delivery and freshness)\n"
           "  --self-test-device-shadow (alone: check device joint-state shadow decoding)\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--self-test-device-shadow") {
    const bool passed = holoscan::device_joint_shadow_self_test();
    std::cout << "{\"self_test\":\"device_shadow\",\"passed\":" << (passed ? "true" : "false")
              << "}\n";
    return passed ? 0 : 1;
  }
  auto options = holoscan::parse(argc, argv);
  if (!options) {
    std::cerr << "invalid control-loop options; see --help\n";
    return 2;
  }
  return holoscan::run_benchmark(std::move(*options));
}
