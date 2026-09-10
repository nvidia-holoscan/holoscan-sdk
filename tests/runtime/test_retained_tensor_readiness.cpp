/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

struct RetainedTensorReadinessProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    fusion_count = 0U;
    saw_selected = false;
    saw_safe_default = false;
    saw_capture_time_absent = false;
  }

  static void record_fusion(InputSelection observed) {
    {
      const std::lock_guard lock{mutex};
      saw_selected = saw_selected || observed.state == InputState::kSelected;
      saw_safe_default = saw_safe_default || observed.state == InputState::kSafeDefault;
      saw_capture_time_absent =
          saw_capture_time_absent || observed.reason == SelectionReason::kCaptureTimeAbsent;
      ++fusion_count;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_outcome(bool expect_selected,
                                             std::chrono::milliseconds timeout) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, timeout, [expect_selected] {
      return expect_selected ? saw_selected
                             : saw_safe_default && saw_capture_time_absent && fusion_count >= 3U;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::uint32_t fusion_count{};
  static inline bool saw_selected{};
  static inline bool saw_safe_default{};
  static inline bool saw_capture_time_absent{};
};

template <MemoryKind Kind>
[[nodiscard]] expected<void, Error> publish_byte(Output<Tensor>& output, ExecutionContext& context,
                                                 std::uint8_t value,
                                                 std::optional<TimePoint> capture_time) {
  constexpr std::array<std::int64_t, 1U> kShape{1};
  auto loan = output.allocate_tensor(TensorLoanRequest{
      .shape = kShape,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }

  auto committed = [&]() -> expected<void, Error> {
    if constexpr (Kind == MemoryKind::kCudaDevice) {
      cudaStream_t stream = context.cuda_stream().get();
      if (stream == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      return loan->write_and_commit(
          context.cuda_stream(),
          [stream, value](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (cudaMemsetAsync(writer.data(), value, 1U, stream) != cudaSuccess) {
              return make_unexpected(Error{ErrorCode::kFailure});
            }
            return {};
          });
    } else {
      return loan->write_host_and_commit([value](const TensorOutputWriteGuard& writer) noexcept {
        *writer.data_as<std::uint8_t>() = value;
      });
    }
  }();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }

  EmitOptions options;
  if (capture_time) {
    options.capture_time = *capture_time;
  }
  return output.emit(std::move(*loan), options);
}

template <MemoryKind Kind>
class RetainedTensorSource final : public Operator<> {
 public:
  RetainedTensorSource(std::chrono::nanoseconds period, std::uint8_t value, bool stamp_capture_time)
      : period_(period), value_(value), stamp_capture_time_(stamp_capture_time) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    return publish_byte<Kind>(
        output,
        context,
        value_,
        stamp_capture_time_ ? std::optional{context.activation_time()} : std::nullopt);
  }

  Output<Tensor> output;

 private:
  std::chrono::nanoseconds period_{};
  std::uint8_t value_{};
  bool stamp_capture_time_{};
};

template <MemoryKind Kind>
class RetainedTensorFusion final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation layout{
        .memory_kind = Kind,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    spec.input(base, "base")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{.representation = layout});
    spec.input(overlay, "overlay")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{.representation = layout});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{base})
        .input(overlay, LatestWithin{.max_age = 1s})
        .on_missing(overlay, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto base_sample = base.receive();
    if (!base_sample) {
      return make_unexpected(std::move(base_sample).error());
    }
    const InputSelection observed = overlay.selection();
    if (observed.state == InputState::kSelected) {
      auto overlay_sample = overlay.receive();
      if (!overlay_sample) {
        return make_unexpected(std::move(overlay_sample).error());
      }
    }
    RetainedTensorReadinessProbe::record_fusion(observed);
    return {};
  }

  Input<Tensor> base;
  Input<Tensor> overlay;
};

// Records publication, selection, and Tensor-materialization outcomes for a slow producer feeding
// a faster LatestWithin consumer. Used by
// RetainedTensorReadiness.HostStaleSamplesDoNotPinTemporalCacheAtCapacity and the corresponding
// device and aged-selection cases.
struct SlowTensorFreshnessProbe {
  static void reset() {
    successful_publishes.store(0U, std::memory_order_relaxed);
    activations.store(0U, std::memory_order_relaxed);
    selected.store(0U, std::memory_order_relaxed);
    safe_defaults.store(0U, std::memory_order_relaxed);
    receive_failures.store(0U, std::memory_order_relaxed);
    null_data_pointers.store(0U, std::memory_order_relaxed);
    copy_failures.store(0U, std::memory_order_relaxed);
    value_mismatches.store(0U, std::memory_order_relaxed);
    maximum_selected_value.store(0U, std::memory_order_relaxed);
  }

  static void record_publish() { successful_publishes.fetch_add(1U, std::memory_order_relaxed); }

  static std::uint32_t record_activation(const InputSelection selection, const bool received,
                                         const bool data_pointer_valid, const bool copied,
                                         const std::uint8_t expected_value,
                                         const std::uint8_t observed_value) {
    if (selection.state == InputState::kSelected) {
      selected.fetch_add(1U, std::memory_order_relaxed);
      receive_failures.fetch_add(static_cast<std::uint32_t>(!received), std::memory_order_relaxed);
      null_data_pointers.fetch_add(static_cast<std::uint32_t>(received && !data_pointer_valid),
                                   std::memory_order_relaxed);
      copy_failures.fetch_add(static_cast<std::uint32_t>(received && !copied),
                              std::memory_order_relaxed);
      value_mismatches.fetch_add(
          static_cast<std::uint32_t>(received && copied && observed_value != expected_value),
          std::memory_order_relaxed);
      std::uint32_t maximum = maximum_selected_value.load(std::memory_order_relaxed);
      while (maximum < observed_value && !maximum_selected_value.compare_exchange_weak(
                                             maximum, observed_value, std::memory_order_relaxed)) {
      }
    } else if (selection.state == InputState::kSafeDefault) {
      safe_defaults.fetch_add(1U, std::memory_order_relaxed);
    }
    const std::uint32_t count = activations.fetch_add(1U, std::memory_order_relaxed) + 1U;
    ready.notify_all();
    return count;
  }

  [[nodiscard]] static bool wait_for_activations(std::uint32_t count,
                                                 std::chrono::seconds timeout) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [count] { return activations.load(std::memory_order_relaxed) >= count; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> successful_publishes{};
  static inline std::atomic<std::uint32_t> activations{};
  static inline std::atomic<std::uint32_t> selected{};
  static inline std::atomic<std::uint32_t> safe_defaults{};
  static inline std::atomic<std::uint32_t> receive_failures{};
  static inline std::atomic<std::uint32_t> null_data_pointers{};
  static inline std::atomic<std::uint32_t> copy_failures{};
  static inline std::atomic<std::uint32_t> value_mismatches{};
  static inline std::atomic<std::uint32_t> maximum_selected_value{};
};

enum class SlowTensorRealization : std::uint8_t { kDirect, kForcedHoloMq };

// Publishes numbered Tensors at 30 Hz and optionally backdates an initial prefix so recovery can be
// distinguished from repeatedly selecting the original cache contents. Used by
// RetainedTensorReadiness.HostStaleSamplesDoNotPinTemporalCacheAtCapacity.
template <MemoryKind Kind, std::int64_t CaptureDelayNs, std::uint8_t StalePublishCount>
class SlowTensorFreshnessSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 33333333ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::uint8_t value = static_cast<std::uint8_t>(next_value_ + 1U);
    TimePoint capture_time = context.activation_time();
    if (value <= StalePublishCount) {
      capture_time.timestamp_ns -= CaptureDelayNs;
    }
    auto published = publish_byte<Kind>(output, context, value, capture_time);
    if (published) {
      next_value_ = value;
      SlowTensorFreshnessProbe::record_publish();
      return {};
    }
    if (published.error().code == ErrorCode::kBackpressured ||
        published.error().code == ErrorCode::kResourceExhausted) {
      return {};
    }
    return make_unexpected(std::move(published).error());
  }

  Output<Tensor> output;

 private:
  std::uint8_t next_value_{};
};

// Polls a LatestWithin + SafeDefault Tensor input at 60 Hz and validates the selected payload's
// pointer, copied value, and sequence identity. Used by
// RetainedTensorReadiness.HostAgedSelectionsRetainTensorBacking.
template <MemoryKind Kind, std::int64_t MaxAgeNs>
class FastTensorFreshnessConsumer final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(kQueueDepth)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = kDisplayPeriod})
        .input(input, LatestWithin{.max_age = std::chrono::nanoseconds{MaxAgeNs}})
        .on_missing(input, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const InputSelection selection = input.selection();
    bool received = false;
    bool pointer_valid = false;
    bool copied = false;
    std::uint8_t expected_value = 0U;
    std::uint8_t observed_value = 0U;
    if (selection.state == InputState::kSelected) {
      auto sample = input.receive();
      if (sample) {
        received = true;
        pointer_valid = sample->data.data() != nullptr;
        expected_value = static_cast<std::uint8_t>(sample->metadata.sequence);
        if (pointer_valid) {
          auto copied_result = sample->data.copy_to_host(
              &observed_value, sizeof(observed_value), context.cuda_stream());
          copied = copied_result.has_value();
        }
      }
    }
    const std::uint32_t count = SlowTensorFreshnessProbe::record_activation(
        selection, received, pointer_valid, copied, expected_value, observed_value);
    if (count == kActivationLimit) {
      context.request_completion();
    }
    return {};
  }

  Input<Tensor> input;

  static constexpr std::uint32_t kQueueDepth{8U};
  static constexpr std::chrono::nanoseconds kDisplayPeriod{16666666ns};
  static constexpr std::uint32_t kActivationLimit{72U};
};

// Runs the two-partition blocking direct-retained/HoloMQ fixture shared by the stale-cache recovery
// and aged Tensor lifetime cases. Used by
// RetainedTensorReadiness.HostStaleSamplesDoNotPinTemporalCacheAtCapacity.
template <MemoryKind Kind, std::int64_t CaptureDelayNs, std::int64_t MaxAgeNs,
          std::uint8_t StalePublishCount, SlowTensorRealization Delivery>
void run_slow_tensor_freshness_case() {
  using Consumer = FastTensorFreshnessConsumer<Kind, MaxAgeNs>;
  SlowTensorFreshnessProbe::reset();
  // Graph:
  //   [source @ 30 Hz; partition "source"]
  //       --[direct-retained or process-local HoloMQ; kBlocking; depth=8]-->
  //   [consumer @ 60 Hz; partition "consumer"; LatestWithin + SafeDefault]
  Graph graph{"slow-tensor-latest-within"};
  auto source =
      graph.op<SlowTensorFreshnessSource<Kind, CaptureDelayNs, StalePublishCount>>("source");
  auto consumer = graph.op<Consumer>("consumer");
  graph.add_flow(source->output,
                 consumer->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kBlocking,
                                   .queue_depth = Consumer::kQueueDepth});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime_clock"));

  CompileOptions options;
  if constexpr (Kind == MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = DeviceId{0},
    });
  }
  const ExecutionPlan plan =
      Delivery == SlowTensorRealization::kForcedHoloMq
          ? detail::compile_with_process_local_holomq(graph, std::move(options))
          : compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(plan_state);
  ASSERT_EQ(plan_state->process_local_channel_services.size(), 1U) << plan.json();
  EXPECT_EQ(plan_state->process_local_channel_services.front().delivery_kind,
            Delivery == SlowTensorRealization::kForcedHoloMq
                ? detail::ProcessLocalChannelDeliveryKind::kHoloMQ
                : detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  RunSession session = run_async(plan);
  const bool observed =
      SlowTensorFreshnessProbe::wait_for_activations(Consumer::kActivationLimit, 5s);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_GT(SlowTensorFreshnessProbe::successful_publishes.load(std::memory_order_relaxed),
            Consumer::kQueueDepth);
  EXPECT_GT(SlowTensorFreshnessProbe::safe_defaults.load(std::memory_order_relaxed), 0U);
  EXPECT_GT(SlowTensorFreshnessProbe::selected.load(std::memory_order_relaxed),
            Consumer::kQueueDepth);
  EXPECT_GT(SlowTensorFreshnessProbe::maximum_selected_value.load(std::memory_order_relaxed),
            Consumer::kQueueDepth);
  EXPECT_EQ(SlowTensorFreshnessProbe::receive_failures.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(SlowTensorFreshnessProbe::null_data_pointers.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(SlowTensorFreshnessProbe::copy_failures.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(SlowTensorFreshnessProbe::value_mismatches.load(std::memory_order_relaxed), 0U);
}

template <MemoryKind Kind>
void run_retained_tensor_readiness_case(bool stamp_overlay_capture_time) {
  RetainedTensorReadinessProbe::reset();
  Graph graph{"retained-tensor-readiness"};
  auto manual_clock = std::make_shared<ManualClock>(0);
  graph.set_default_clock(graph.bind_shared_clock("clock", manual_clock));
  const auto overlay =
      graph.op<RetainedTensorSource<Kind>>("overlay", 20ms, 0x22U, stamp_overlay_capture_time);
  const auto base = graph.op<RetainedTensorSource<Kind>>("base", 10ms, 0x11U, true);
  const auto fusion = graph.op<RetainedTensorFusion<Kind>>("fusion");
  graph.add_flow(overlay->output, fusion->overlay, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(base->output, fusion->base, ConnectionOptions{.queue_depth = 2U});
  graph.partition("base-fusion").add(base).add(fusion);
  graph.partition("overlay").add(overlay);

  CompileOptions options;
  if constexpr (Kind == MemoryKind::kCudaDevice) {
    for (const std::string_view path : {"overlay", "base"}) {
      options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = path,
          .output_port = "output",
          .device = DeviceId{0},
      });
    }
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  bool observed = RetainedTensorReadinessProbe::wait_for_outcome(stamp_overlay_capture_time, 250ms);
  for (std::uint32_t step = 0U; !observed && step < 8U; ++step) {
    manual_clock->advance_by(10ms);
    observed = RetainedTensorReadinessProbe::wait_for_outcome(stamp_overlay_capture_time, 250ms);
  }
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_GT(RetainedTensorReadinessProbe::fusion_count, 0U);
  if (stamp_overlay_capture_time) {
    EXPECT_TRUE(RetainedTensorReadinessProbe::saw_selected);
  } else {
    EXPECT_FALSE(RetainedTensorReadinessProbe::saw_selected);
    EXPECT_TRUE(RetainedTensorReadinessProbe::saw_safe_default);
    EXPECT_TRUE(RetainedTensorReadinessProbe::saw_capture_time_absent);
  }
}

// A process-local sample without capture time remains valid arrival evidence but is ineligible for
// LatestWithin. SafeDefault must resolve the side input without losing Tensor readiness credit.
// REQ: RUNTIME-017
// REQ: RUNTIME-034
TEST(RetainedTensorReadiness, HostMissingCaptureUsesSafeDefault) {
  run_retained_tensor_readiness_case<MemoryKind::kHost>(false);
}

// REQ: RUNTIME-017
// REQ: RUNTIME-034
TEST(RetainedTensorReadiness, HostCapturedOverlayIsSelected) {
  run_retained_tensor_readiness_case<MemoryKind::kHost>(true);
}

// REQ: RUNTIME-017
// REQ: RUNTIME-034
TEST(RetainedTensorReadiness, DeviceMissingCaptureUsesSafeDefault) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  run_retained_tensor_readiness_case<MemoryKind::kCudaDevice>(false);
}

// REQ: RUNTIME-017
// REQ: RUNTIME-034
TEST(RetainedTensorReadiness, DeviceCapturedOverlayIsSelected) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  run_retained_tensor_readiness_case<MemoryKind::kCudaDevice>(true);
}

// After more than eight 30 Hz delivered Tensor samples arrive outside a 60 Hz LatestWithin
// window, a later fresh sample must still be admitted and selected. SafeDefault does not consume
// the stale prefix, so observing the fresh suffix requires retention eviction independent of a
// prior successful selection.
// REQ: CONTRACT-021
// REQ: RUNTIME-017
// REQ: RUNTIME-043
// REQ: TENSOR-STORAGE-010
TEST(RetainedTensorReadiness, HostStaleSamplesDoNotPinTemporalCacheAtCapacity) {
  // Key setup: a 30 Hz host Tensor producer backdates its first 12 samples by 20 ms while a 60 Hz
  // LatestWithin + SafeDefault consumer has a 16.67 ms window and a blocking queue of depth 8; the
  // graph runs once with direct retention and once with forced process-local HoloMQ.
  // Key assertion: both realizations publish and select fresh suffix samples beyond the original
  // queue capacity, with valid backing and matching payload values.
  run_slow_tensor_freshness_case<MemoryKind::kHost,
                                 20'000'000,
                                 16'666'666,
                                 12U,
                                 SlowTensorRealization::kDirect>();
  run_slow_tensor_freshness_case<MemoryKind::kHost,
                                 20'000'000,
                                 16'666'666,
                                 12U,
                                 SlowTensorRealization::kForcedHoloMq>();
}

// REQ: CONTRACT-021
// REQ: RUNTIME-017
// REQ: RUNTIME-043
// REQ: TENSOR-STORAGE-010
TEST(RetainedTensorReadiness, DeviceStaleSamplesDoNotPinTemporalCacheAtCapacity) {
  // Key setup: the stale-prefix recovery graph uses CUDA-device Tensors across two process-local
  // partitions and runs with both direct retention and forced HoloMQ delivery.
  // Key assertion: fresh device samples pass the previously pinned capacity boundary in both
  // realizations and retain a valid, copyable payload with the expected value.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  run_slow_tensor_freshness_case<MemoryKind::kCudaDevice,
                                 20'000'000,
                                 16'666'666,
                                 12U,
                                 SlowTensorRealization::kDirect>();
  run_slow_tensor_freshness_case<MemoryKind::kCudaDevice,
                                 20'000'000,
                                 16'666'666,
                                 12U,
                                 SlowTensorRealization::kForcedHoloMq>();
}

// An accepted sample that is substantially older than one display period must still materialize
// the exact live Tensor backing named by its temporal identity. Metadata without its payload is
// not a valid selection.
// REQ: RUNTIME-017
// REQ: TENSOR-STORAGE-010
TEST(RetainedTensorReadiness, HostAgedSelectionsRetainTensorBacking) {
  // Key setup: a 30 Hz host Tensor producer stamps every sample 100 ms old while the 60 Hz
  // consumer admits samples through a 500 ms LatestWithin window, using direct retention and
  // forced process-local HoloMQ in turn.
  // Key assertion: both realizations continue selecting beyond queue depth with non-null backing,
  // successful host copies, and payload values matching their temporal sequence identities.
  run_slow_tensor_freshness_case<MemoryKind::kHost,
                                 100'000'000,
                                 500'000'000,
                                 255U,
                                 SlowTensorRealization::kDirect>();
  run_slow_tensor_freshness_case<MemoryKind::kHost,
                                 100'000'000,
                                 500'000'000,
                                 255U,
                                 SlowTensorRealization::kForcedHoloMq>();
}

// REQ: RUNTIME-017
// REQ: TENSOR-STORAGE-010
TEST(RetainedTensorReadiness, DeviceAgedSelectionsRetainTensorBacking) {
  // Key setup: the aged-selection fixture uses CUDA-device Tensors stamped 100 ms old and a 500 ms
  // LatestWithin window across two process-local partitions, with both direct retention and forced
  // HoloMQ delivery.
  // Key assertion: for both realizations, every received selection has non-null device backing,
  // copies successfully, and contains the value identified by its temporal sequence.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  run_slow_tensor_freshness_case<MemoryKind::kCudaDevice,
                                 100'000'000,
                                 500'000'000,
                                 255U,
                                 SlowTensorRealization::kDirect>();
  run_slow_tensor_freshness_case<MemoryKind::kCudaDevice,
                                 100'000'000,
                                 500'000'000,
                                 255U,
                                 SlowTensorRealization::kForcedHoloMq>();
}

}  // namespace
}  // namespace holoscan
