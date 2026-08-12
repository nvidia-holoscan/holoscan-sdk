/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
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
        .produces_tensor(TensorPortLayout{
            .memory_kind = Kind,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        })
        .tensor_allocation(TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 1U});
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
    const TensorPortLayout layout{
        .memory_kind = Kind,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    spec.input(base, "base").queue_depth(2U).expects_tensor(layout);
    spec.input(overlay, "overlay").queue_depth(2U).expects_tensor(layout);
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

}  // namespace
}  // namespace holoscan
