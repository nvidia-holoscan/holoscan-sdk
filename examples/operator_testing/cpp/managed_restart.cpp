/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

namespace {
using namespace holoscan;
using namespace holoscan::testing;
using namespace std::chrono_literals;

struct RestartObservation {
  std::mutex mutex;
  std::condition_variable changed;
  std::optional<LifecycleRestartReceipt> receipt;
};

class RestartableScale final : public Operator<> {
 public:
  RestartableScale(std::shared_ptr<RestartObservation> observation, bool self)
      : observation_(std::move(observation)), self_(self) {}
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  void start() override { ++factor_; }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample)
      return make_unexpected(sample.error());
    if (self_ && sample->data == 0) {
      const auto request = context.request_restart({.from_stage = LifecycleStage::kArm});
      if (!request)
        return make_unexpected(request.error());
      const std::lock_guard lock(observation_->mutex);
      observation_->receipt = *request;
      observation_->changed.notify_all();
      return {};
    }
    return out.emit(sample->data * factor_);
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;

 private:
  std::shared_ptr<RestartObservation> observation_;
  bool self_;
  std::int32_t factor_{1};
};

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error{message};
}

void release_next(RunningFixture& running) {
  const auto posted = running.release_frame(5s);
  if (!posted)
    throw RuntimeError{posted.error()};
}

void exercise_restart(bool self) {
  auto observation = std::make_shared<RestartObservation>();
  auto h = create_operator_test<RestartableScale>(observation, self);
  h.push_frame(frame_at(0ms, input("in", std::int32_t{self ? 0 : 21})));
  h.push_frame(frame_at(1ms, input("in", std::int32_t{5})));
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& view) {
    view.graph()
        .lifecycle()
        .allow_self_restart(view.tested_operator())
        .allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions =
        LifecycleTransactionOptions{.max_pending_requests = 4U,
                                    .max_pending_transactions = 4U,
                                    .max_receipts = 8U,
                                    .max_fenced_operators = 16U,
                                    .max_candidate_generations = 8U,
                                    .max_events = 64U,
                                    .max_audit_records = 32U,
                                    .max_quiesce_records = 64U,
                                    .max_rollback_records = 64U};
  });
  auto running = h.run_async({.frames = FrameRelease::kManual});
  const auto target = running.operator_ref();
  require(target.has_value(), "current-run identity unavailable");
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kArm};
  auto control =
      running.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  require(control.has_value(), "restricted lifecycle handle unavailable");

  release_next(running);
  LifecycleRestartReceipt receipt;
  if (self) {
    // This control input submits a restart without emitting an old-generation publication
    // that the transaction could discard before the collector runs.
    std::unique_lock lock(observation->mutex);
    require(
        observation->changed.wait_for(lock, 5s, [&] { return observation->receipt.has_value(); }),
        "self request was not submitted");
    const auto submitted = observation->receipt;
    if (!submitted)
      throw std::runtime_error{"self request receipt unavailable"};
    receipt = *submitted;
  } else {
    require(running.wait_for_output("out", 1U, 5s), "first output not observed");
    const auto accepted = control->request_restart(*target, {.from_stage = LifecycleStage::kArm});
    require(accepted.has_value(), "restart request rejected");
    receipt = *accepted;
  }
  // Acceptance only queues a native transaction. Await its terminal result before phase two.
  const auto terminal = control->await(receipt, 5s);
  require(terminal && terminal->state == LifecycleTransactionState::kPublished &&
              terminal->effective_stage == LifecycleStage::kArm,
          "restart did not publish at Arm");
  const auto after_restart = running.operator_ref();
  require(after_restart && *after_restart == *target, "restart changed the native run identity");
  release_next(running);
  require(running.wait_for_output("out", self ? 1U : 2U, 5s), "resumed output not observed");
  const auto result = running.finish();
  if (!result.ok())
    throw std::runtime_error{result.describe_failure()};
  require(result.values<std::int32_t>("out") ==
              (self ? std::vector<std::int32_t>{15} : std::vector<std::int32_t>{42, 15}),
          "restart must re-enter Start on the existing native instance");
}
}  // namespace

int main() {
  exercise_restart(false);
  exercise_restart(true);
  std::cout << "managed operator restart examples complete\n";
}
