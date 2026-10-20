/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>

namespace holoscan {
namespace {

using namespace std::chrono_literals;
constexpr DLDataType kByteType{kDLUInt, 8U, 1U};
constexpr ClockId kClockId{1U};  // the compiled graph has one clock

struct TickSelection {
  std::int64_t tick_ns{};
  std::uint64_t value{};
  std::size_t batch_size{};
};

// Hold real compute activations, leaving all selector admission and fence updates to RunSession.
struct TickFreshnessSchedule {
  template <typename Predicate>
  bool wait(std::unique_lock<std::mutex>& lock, Predicate predicate) {
    if (!cv.wait_for(lock, 5s, [&] { return stopped || predicate(); })) {
      timed_out = true;
      stopped = true;
      cv.notify_all();
    }
    return !stopped;
  }

  std::optional<std::uint64_t> next_publication() {
    std::unique_lock lock{mutex};
    if (!wait(lock, [&] { return requested > published; })) {
      return std::nullopt;
    }
    return requested;
  }

  void complete_publication(std::uint64_t value) {
    const std::lock_guard lock{mutex};
    published = value;
    cv.notify_all();
  }

  bool publish(ManualClock& clock, std::chrono::nanoseconds time, std::uint64_t value) {
    std::unique_lock lock{mutex};
    clock.advance_to(time.count());
    requested = value;
    cv.notify_all();
    return wait(lock, [&] { return published == value; });
  }

  void hold_consumer(TickSelection selection) {
    std::unique_lock lock{mutex};
    selections.push_back(selection);
    cv.notify_all();
    wait(lock, [&] { return released >= selections.size(); });
  }

  bool wait_for_consumer(std::size_t count) {
    std::unique_lock lock{mutex};
    return wait(lock, [&] { return selections.size() >= count; });
  }

  void release_consumer(std::size_t count) {
    const std::lock_guard lock{mutex};
    released = count;
    cv.notify_all();
  }

  void stop() {
    const std::lock_guard lock{mutex};
    stopped = true;
    cv.notify_all();
  }

  std::mutex mutex;
  std::condition_variable cv;
  std::uint64_t requested{};
  std::uint64_t published{};
  std::size_t released{};
  bool stopped{};
  bool timed_out{};
  std::vector<TickSelection> selections;
};

template <typename Payload>
class ScheduledTensorSource final : public Operator<> {
 public:
  explicit ScheduledTensorSource(std::shared_ptr<TickFreshnessSchedule> schedule)
      : schedule_(std::move(schedule)) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kByteType, .rank = 1U},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)},
            .storage = TensorOutputStorage::kRuntimePool});
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns, .start_at = TimePoint{0, kClockId}});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override {
    const auto value = schedule_->next_publication();
    if (!value) {
      return {};
    }
    constexpr std::array<std::int64_t, 1U> kShape{sizeof(std::uint64_t)};
    auto loan = output.allocate_tensor(TensorLoanRequest{.shape = kShape, .dtype = kByteType});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto committed = loan->write_host_and_commit(
        [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          std::memcpy(writer.data(), &*value, sizeof(*value));
          return {};
        });
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    const auto emitted = [&]() -> expected<void, Error> {
      if constexpr (std::is_same_v<Payload, Tensor>) {
        return output.emit(std::move(*loan));
      } else {
        schema::ImageT image;
        image.height = 1;
        image.width = sizeof(std::uint64_t);
        image.encoding = schema::ImageEncoding_CUSTOM;
        return output.emit_tensor(std::move(*loan), image);
      }
    }();
    if (emitted) {
      schedule_->complete_publication(*value);
    }
    return emitted;
  }

  Output<Payload> output;

 private:
  std::shared_ptr<TickFreshnessSchedule> schedule_;
};

template <typename Payload>
class HeldTensorConsumer final : public Operator<> {
 public:
  explicit HeldTensorConsumer(std::shared_ptr<TickFreshnessSchedule> schedule)
      : schedule_(std::move(schedule)) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kByteType, .rank = 1U}});
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms, .start_at = TimePoint{10'000'000, kClockId}})
        .input(input, DrainAll{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext& context) override {
    auto batch = input.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    TickSelection selection{.tick_ns = context.activation_time().timestamp_ns,
                            .batch_size = batch->size()};
    for (const auto& sample : *batch) {
      const Tensor& tensor = [&]() -> const Tensor& {
        if constexpr (std::is_same_v<Payload, Tensor>) {
          return sample.data;
        } else {
          return *sample.data.data;
        }
      }();
      std::memcpy(&selection.value, tensor.data(), sizeof(selection.value));
    }
    schedule_->hold_consumer(selection);
    return {};
  }

  Input<Payload> input;

 private:
  std::shared_ptr<TickFreshnessSchedule> schedule_;
};

template <typename Payload>
void check_run_session_tick_freshness() {
  const auto clock = std::make_shared<ManualClock>(0);
  const auto schedule = std::make_shared<TickFreshnessSchedule>();
  Graph graph{"tensor-tick-freshness"};
  graph.set_default_clock(graph.bind_shared_clock("clock", clock));
  const auto source = graph.op<ScheduledTensorSource<Payload>>("source", schedule);
  const auto consumer = graph.op<HeldTensorConsumer<Payload>>("consumer", schedule);
  graph.add_flow(source->output,
                 consumer->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("consumer").add(consumer);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool stepped = [&] {
    if (!schedule->publish(*clock, 1ms, 1U)) {
      return false;
    }
    clock->advance_to(10'000'000);
    if (!schedule->wait_for_consumer(1U)) {
      return false;
    }
    // The first activation pins the typed slot. Overflow staging on both sides of the next tick.
    if (!schedule->publish(*clock, 15ms, 2U) || !schedule->publish(*clock, 21ms, 3U) ||
        !schedule->publish(*clock, 22ms, 4U) || !schedule->publish(*clock, 23ms, 5U)) {
      return false;
    }
    schedule->release_consumer(1U);
    if (!schedule->wait_for_consumer(2U)) {
      return false;
    }
    // RunSession must advance its fence after freezing the delayed 20 ms activation, before
    // dispatching it. No park separates these two delayed ticks.
    if (!schedule->publish(*clock, 25ms, 6U) || !schedule->publish(*clock, 31ms, 7U) ||
        !schedule->publish(*clock, 32ms, 8U) || !schedule->publish(*clock, 33ms, 9U)) {
      return false;
    }
    schedule->release_consumer(2U);
    return schedule->wait_for_consumer(3U);
  }();
  // Always release the compute barriers, including on a failed handshake, before joining.
  session.request_stop();
  schedule->stop();
  session.wait();
  ASSERT_TRUE(stepped);
  EXPECT_FALSE(schedule->timed_out);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(schedule->selections.size(), 3U);
  constexpr std::array<std::uint64_t, 3U> kExpected{1U, 2U, 6U};
  for (std::size_t index = 0U; index < kExpected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(schedule->selections[index].tick_ns, (index + 1U) * 10'000'000);
    EXPECT_EQ(schedule->selections[index].batch_size, 1U);
    EXPECT_EQ(schedule->selections[index].value, kExpected[index]);
  }
}

// REQ: CHAN-131
// REQ: RUNTIME-089
TEST(TensorTickFreshness, RunSessionPreservesTensorAcrossDelayedTicks) {
  check_run_session_tick_freshness<Tensor>();
}

// REQ: CHAN-131
// REQ: RUNTIME-089
TEST(TensorTickFreshness, RunSessionPreservesImageAcrossDelayedTicks) {
  check_run_session_tick_freshness<schema::ImageT>();
}

}  // namespace
}  // namespace holoscan
