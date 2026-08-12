/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr std::array<std::int64_t, 1U> kFrameShape{16};
constexpr std::size_t kFrameBytes = 16U * sizeof(float);
constexpr holoscan::TensorPortLayout kHostFloatFrame{
    .memory_kind = holoscan::MemoryKind::kHost,
    .dtype = kFloat32,
    .rank = 1U,
};
constexpr std::size_t kFrameGoal = 6U;
constexpr std::size_t kDropGoal = 3U;
constexpr std::size_t kMaxHeldLoans = 64U;

// Operators record what happened here; main() prints the summary only after
// the session has fully stopped, so stdout lines are never interleaved.
struct RoundTripObservation {
  struct FrameRecord {
    std::uint32_t frame_id{};
    float first_element{};
    std::size_t byte_size{};
  };

  static void reset() {
    const std::lock_guard lock(mutex);
    frames.clear();
    drops.clear();
    read_only_code.reset();
    exhaust_overflow = false;
  }

  static void record_frame(FrameRecord record) {
    {
      const std::lock_guard lock(mutex);
      frames.push_back(record);
    }
    ready.notify_all();
  }

  static void record_drop(std::uint32_t frame_id, holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      drops.emplace_back(frame_id, code);
    }
    ready.notify_all();
  }

  static void record_read_only_code(holoscan::ErrorCode code) {
    const std::lock_guard lock(mutex);
    read_only_code = code;
  }

  static void record_exhaust_overflow() {
    {
      const std::lock_guard lock(mutex);
      exhaust_overflow = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_frames(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return frames.size() >= expected; });
  }

  [[nodiscard]] static bool wait_for_drops(std::size_t expected) {
    std::unique_lock lock(mutex);
    ready.wait_for(lock, 10s, [expected] { return drops.size() >= expected || exhaust_overflow; });
    return drops.size() >= expected;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<FrameRecord> frames;
  static inline std::vector<std::pair<std::uint32_t, holoscan::ErrorCode>> drops;
  static inline std::optional<holoscan::ErrorCode> read_only_code;
  static inline bool exhaust_overflow{false};
};

class FrameProducer final : public holoscan::Operator<> {
 public:
  explicit FrameProducer(bool exhaust_pool) : exhaust_pool_(exhaust_pool) {
    if (exhaust_pool_) {
      held_loans_.reserve(kMaxHeldLoans);
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    // produces_tensor states the logical facts consumers may rely on;
    // tensor_allocation opts this output into a bounded plan-owned host pool.
    spec.output(frame, "frame")
        .produces_tensor(kHostFloatFrame)
        .tensor_allocation(
            holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kFrameBytes});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t frame_id = ++frame_id_;

    // Step 1: borrow one pool slot. allocate_tensor never blocks and never
    // heap-allocates; an exhausted pool answers immediately.
    auto loan =
        frame.allocate_tensor(holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kFloat32});
    if (!loan) {
      if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
        // Drop-and-continue: record the refusal and keep producing. There is
        // no blocking loan variant to accidentally stall the pipeline on.
        RoundTripObservation::record_drop(frame_id, loan.error().code);
        return {};
      }
      return holoscan::make_unexpected(std::move(loan).error());
    }

    const float value = static_cast<float>(frame_id);
    if ((frame_id % 2U) == 1U) {
      // Steps 2+3 fused: write_host_and_commit runs the fill callback once and
      // commits the write phase. Best when the whole buffer is filled in one go.
      auto fill = [value](const holoscan::TensorOutputWriteGuard& writer)
          -> holoscan::expected<void, holoscan::Error> {
        auto* data = writer.data_as<float>();
        if (data == nullptr || writer.byte_size() != kFrameBytes) {
          return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
        }
        std::fill_n(data, writer.byte_size() / sizeof(float), value);
        return {};
      };
      auto committed = loan->write_host_and_commit(fill);
      if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    } else {
      // Steps 2+3 explicit: hold the write guard across a staged fill, then
      // consume it with commit(). Commit records readiness; it never publishes.
      auto writer = loan->write_host();
      if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
      }
      auto* data = writer->data_as<float>();
      if (data == nullptr || writer->byte_size() != kFrameBytes) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
      }
      std::fill_n(data, writer->byte_size() / sizeof(float), value);
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
      }
    }

    if (exhaust_pool_) {
      // Model a backlog of in-flight frames: keep committed loans alive instead
      // of publishing them. Each live loan pins one pool slot, so a later
      // allocate_tensor is refused with kResourceExhausted.
      if (held_loans_.size() >= kMaxHeldLoans) {
        RoundTripObservation::record_exhaust_overflow();
        return {};
      }
      held_loans_.push_back(std::move(*loan));
      return {};
    }

    // Step 4: publication stays explicit and separate from commit.
    return frame.emit(std::move(*loan),
                      holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  bool exhaust_pool_{};
  std::uint32_t frame_id_{};
  std::vector<holoscan::TensorOutputLoan> held_loans_;
};

class FrameInspector final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame").queue_depth(8U).expects_tensor(kHostFloatFrame);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = frame.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    holoscan::Tensor& tensor = sample->data;

    if (!write_probe_done_) {
      write_probe_done_ = true;
      // Received tensors are read-only: mutable access must be refused. The
      // refusal code is recorded once and reported by main() after shutdown.
      auto mutable_pointer = tensor.mutable_data();
      if (mutable_pointer) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
      }
      RoundTripObservation::record_read_only_code(mutable_pointer.error().code);
    }

    // Host access is explicit. For this host-pool tensor the call is cheap;
    // for device tensors it blocks the host thread, which is why it never
    // belongs on a steady-state hot path.
    auto synchronized = tensor.synchronize_for_host();
    if (!synchronized) {
      return holoscan::make_unexpected(std::move(synchronized).error());
    }
    const float* data = tensor.data_as<float>();
    if (data == nullptr || tensor.nbytes() != static_cast<std::int64_t>(kFrameBytes)) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    RoundTripObservation::record_frame(RoundTripObservation::FrameRecord{
        .frame_id = static_cast<std::uint32_t>(data[0]),
        .first_element = data[0],
        .byte_size = static_cast<std::size_t>(tensor.nbytes()),
    });
    return {};
  }

  holoscan::Input<holoscan::Tensor> frame;

 private:
  bool write_probe_done_{};
};

}  // namespace

int main(int argc, char** argv) {
  bool exhaust_pool = false;
  if (argc == 2 && std::string_view(argv[1]) == "--exhaust") {
    exhaust_pool = true;
  } else if (argc != 1) {
    std::cerr << "usage: tensors_101 [--exhaust]\n";
    return 2;
  }

  RoundTripObservation::reset();

  holoscan::Graph graph{"tensors-101"};
  const auto producer = graph.op<FrameProducer>("producer", exhaust_pool);
  const auto inspector = graph.op<FrameInspector>("inspector");
  graph.add_flow(producer->frame, inspector->frame, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = exhaust_pool ? RoundTripObservation::wait_for_drops(kDropGoal)
                                     : RoundTripObservation::wait_for_frames(kFrameGoal);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "tensors_101: run did not complete cleanly\n";
    return 2;
  }

  // The session is fully stopped: observation state is stable and the stdout
  // lines below cannot interleave with operator activity.
  if (exhaust_pool) {
    for (std::size_t index = 0; index < kDropGoal; ++index) {
      const auto& [frame_id, code] = RoundTripObservation::drops[index];
      if (code != holoscan::ErrorCode::kResourceExhausted) {
        std::cerr << "tensors_101: unexpected drop code " << holoscan::error_code_name(code)
                  << '\n';
        return 3;
      }
      std::cout << "frame " << frame_id << " dropped: " << holoscan::error_code_name(code) << '\n';
    }
    std::cout << "tensors_101: exhaust demo complete (drops=" << kDropGoal
              << ", code=kResourceExhausted)\n";
    return 0;
  }

  std::sort(RoundTripObservation::frames.begin(),
            RoundTripObservation::frames.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.frame_id < rhs.frame_id; });
  for (std::size_t index = 0; index < kFrameGoal; ++index) {
    const auto& record = RoundTripObservation::frames[index];
    const auto expected_id = static_cast<std::uint32_t>(index + 1U);
    if (record.frame_id != expected_id || record.first_element != static_cast<float>(expected_id) ||
        record.byte_size != kFrameBytes) {
      std::cerr << "tensors_101: frame " << expected_id << " failed verification\n";
      return 3;
    }
    std::cout << "frame " << record.frame_id << " ok ("
              << ((record.frame_id % 2U) == 1U ? "fused" : "explicit") << " writer)\n";
  }
  if (!RoundTripObservation::read_only_code.has_value() ||
      *RoundTripObservation::read_only_code != holoscan::ErrorCode::kPermissionDenied) {
    std::cerr << "tensors_101: read-only reception check failed\n";
    return 3;
  }
  std::cout << "read-only check: mutable_data() -> "
            << holoscan::error_code_name(*RoundTripObservation::read_only_code) << '\n';
  std::cout << "tensors_101: round trip complete (frames=" << kFrameGoal
            << ", bytes_per_frame=" << kFrameBytes << ")\n";
  return 0;
}
