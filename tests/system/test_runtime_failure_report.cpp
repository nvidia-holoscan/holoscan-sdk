/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace holoscan;

enum class Mode {
  kStart,
  kCompute,
  kComputeError,
  kAdvanced,
  kAdvancedSuccess,
  kCleanup,
  kSuccess,
  kEmbedded,
  kWorker,
  kAsync
};

class Source final : public Operator<> {
 public:
  explicit Source(Mode mode) : mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }
  void start() override { std::cerr << "source started\n"; }
  expected<void, Error> compute(ExecutionContext&) override {
    return output.emit(std::uint32_t{1U});
  }
  void stop() override {
    std::cerr << "source stopped\n";
    if (mode_ == Mode::kCleanup) {
      throw RuntimeError{ErrorCode::kFailure, "secondary source cleanup failed"};
    }
  }
  Output<std::uint32_t> output;

 private:
  Mode mode_;
};

class Display final : public Operator<> {
 public:
  explicit Display(Mode mode) : mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    if (mode_ == Mode::kAdvanced || mode_ == Mode::kAdvancedSuccess) {
      spec.lifecycle().stage(LifecycleStage::kConfigure, &Display::configure);
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }
  LifecycleStatus configure(LifecycleContext& context) noexcept {
    if (mode_ == Mode::kAdvancedSuccess) {
      context.report_error(ErrorCode::kFailure, "ignored successful stage detail");
      return LifecycleStatus::kOk;
    }
    context.report_error(ErrorCode::kInvalidArgument, "RGB unsupported; requires RGBA");
    context.report_error(ErrorCode::kFailure, "must not replace first reported error");
    return LifecycleStatus::kFatalFailure;
  }
  void start() override {
    if (mode_ != Mode::kCompute && mode_ != Mode::kComputeError && mode_ != Mode::kAdvanced &&
        mode_ != Mode::kSuccess && mode_ != Mode::kAdvancedSuccess) {
      throw RuntimeError{ErrorCode::kInvalidArgument, "RGB unsupported; requires RGBA"};
    }
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (mode_ == Mode::kCompute) {
      throw RuntimeError{ErrorCode::kInvalidArgument, "RGB unsupported; requires RGBA"};
    }
    if (mode_ == Mode::kComputeError) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "RGB unsupported; requires RGBA"});
    }
    context.request_completion();
    return {};
  }
  Input<std::uint32_t> input;

 private:
  Mode mode_;
};

void populate(Graph& graph, Mode mode) {
  auto source = graph.op<Source>("a_source", mode);
  auto display = graph.op<Display>("z_display", mode);
  graph.add_flow(source->output, display->input);
  if (mode == Mode::kWorker) {
    graph.partition("worker").add(source).add(display);
  }
}

class Pipeline final : public Graph {
 public:
  Pipeline() : Graph("pipeline") {}
  void compose() override {
    set_default_clock(add_clock<RealtimeClock>("clock"));
    populate(*this, Mode::kEmbedded);
  }
};

ExecutionPlan make_plan(Mode mode) {
  Graph graph{"failure-report"};
  if (mode == Mode::kEmbedded) {
    graph.embed<Pipeline>("pipeline");
  } else {
    populate(graph, mode);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  if (mode == Mode::kWorker) {
    options.deployment.bind_thread_placement("worker",
                                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  }
  return compile(graph, options);
}

}  // namespace

// REQ: RUNTIME-054
// REQ: RUNTIME-060
// REQ: EXPECTED-033
// REQ: EXPECTED-032
// REQ: LIFECYCLE-013
// REQ: LIFECYCLE-014
int main(int argc, char** argv) {
  const std::string_view name = argc > 1 ? argv[1] : "start";
  const std::string_view presentation = argc > 2 ? argv[2] : "default";
  if (presentation == "on") {
    set_automatic_diagnostic_logging_enabled(true);
  } else if (presentation == "off" || presentation == "explicit") {
    set_automatic_diagnostic_logging_enabled(false);
  }
  const Mode mode = name == "compute"            ? Mode::kCompute
                    : name == "compute-error"    ? Mode::kComputeError
                    : name == "advanced"         ? Mode::kAdvanced
                    : name == "advanced-success" ? Mode::kAdvancedSuccess
                    : name == "cleanup"          ? Mode::kCleanup
                    : name == "success"          ? Mode::kSuccess
                    : name == "embedded"         ? Mode::kEmbedded
                    : name == "worker"           ? Mode::kWorker
                    : name == "async"            ? Mode::kAsync
                                                 : Mode::kStart;
  if (presentation == "explicit") {
    const auto plan = make_plan(mode);
    auto session = run_async(plan);
    session.wait();
    log_diagnostics(session.diagnostics(), plan.graph());
    return session.termination() == RunTermination::kFailed ? 1 : 0;
  }
  if (mode == Mode::kAsync) {
    auto session = run_async(make_plan(mode));
    session.wait();
    std::cout << "Session diagnostic: "
              << format_diagnostics(session.diagnostics(), ExecutionGraphView{}) << '\n';
    return session.termination() == RunTermination::kFailed ? 1 : 0;
  }
  const int result = run(make_plan(mode), RunSignalPolicy::kCallerManaged);
  return name == "invalid-exit" ? 2 : result;
}
