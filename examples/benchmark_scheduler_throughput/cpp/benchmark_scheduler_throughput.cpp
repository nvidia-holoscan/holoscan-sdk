/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>

// Increments a counter on each invocation.
class CountOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CountOp)

  CountOp() = default;

  void start() override {
    // Use wall clock instead of Holoscan's internal clock in order to
    // separately measurement from implementation.
    start_time_ = std::chrono::steady_clock::now();
  }

  void stop() override { end_time_ = std::chrono::steady_clock::now(); }

  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    count_++;
  };

  int64_t count() const { return count_; }
  std::chrono::steady_clock::time_point start_time() const { return start_time_; }
  std::chrono::steady_clock::time_point end_time() const { return end_time_; }

 private:
  int64_t count_ = 0;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;
};

// Benchmarks scheduler throughput by counting operations over a period of time.
class BenchmarkSchedulerThroughputApp : public holoscan::Application {
 public:
  struct Options {
    // Number of worker threads used by the scheduler.
    int num_threads = 1;

    // Number of operators.
    int num_operators = 1;

    // Operations to execute per operator.
    int num_operations = 100000;

    // Whether EventBasedScheduler workers may steal ready jobs from other worker queues.
    bool enable_queue_stealing = false;

    // Whether EventBasedScheduler workers use the postcheck fastpath after executeEntity().
    bool enable_postcheck_fastpath = false;
  };

  struct Results {
    int total_operations;
    float throughput_hz;
  };

  void set_options(const Options& options) { options_ = options; }

  void compose() override {
    for (int i = 0; i < options_.num_operators; i++) {
      count_ops_.push_back(make_operator<CountOp>(
          fmt::format("count_{}", i),
          make_condition<holoscan::CountCondition>(options_.num_operations)));
      add_operator(count_ops_.back());
    }

    scheduler(make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler",
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(options_.num_threads)),
        holoscan::Arg("enable_queue_stealing", options_.enable_queue_stealing),
        holoscan::Arg("enable_worker_postcheck_fastpath", options_.enable_postcheck_fastpath)));
  }

  Results results() {
    // Aggregate statistics from all operator flows.
    int total_operations = 0;
    std::chrono::steady_clock::time_point min_start_time = count_ops_[0]->start_time();
    std::chrono::steady_clock::time_point max_end_time = count_ops_[0]->end_time();
    for (const std::shared_ptr<CountOp>& count_op : count_ops_) {
      total_operations += count_op->count();
      min_start_time = std::min(min_start_time, count_op->start_time());
      max_end_time = std::max(max_end_time, count_op->end_time());
    }

    // Calculate and return the results.
    float duration_s =
        std::chrono::duration_cast<std::chrono::duration<float>>(max_end_time - min_start_time)
            .count();
    Results results;
    results.total_operations = total_operations;
    results.throughput_hz = total_operations / duration_s;
    return results;
  }

 private:
  Options options_;

  std::vector<std::shared_ptr<CountOp>> count_ops_;
};

namespace {

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name
            << " [--enable_queue_stealing] [--enable_postcheck_fastpath]" << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkSchedulerThroughputApp::Options scheduler_options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--enable_queue_stealing") {
      scheduler_options.enable_queue_stealing = true;
    } else if (arg == "--enable_postcheck_fastpath") {
      scheduler_options.enable_postcheck_fastpath = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  auto make_trial_options = [&](int num_threads,
                                int num_operators) -> BenchmarkSchedulerThroughputApp::Options {
    BenchmarkSchedulerThroughputApp::Options options = scheduler_options;
    options.num_threads = num_threads;
    options.num_operators = num_operators;
    return options;
  };

  // Construct trial options to benchmark: {num_threads, num_operators}.
  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {
      make_trial_options(1, 1),
      make_trial_options(2, 2),
      make_trial_options(4, 4),
      make_trial_options(8, 8),
      make_trial_options(16, 16),
      make_trial_options(2, 8),
  };
  std::vector<BenchmarkSchedulerThroughputApp::Results> trial_results;

  for (BenchmarkSchedulerThroughputApp::Options& options : trial_options) {
    auto app = holoscan::make_application<BenchmarkSchedulerThroughputApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nScheduler Throughput Benchmark Results:" << '\n';
  std::cout << fmt::format("Scheduler options: queue_stealing={}, postcheck_fastpath={}",
                           scheduler_options.enable_queue_stealing ? "on" : "off",
                           scheduler_options.enable_postcheck_fastpath ? "on" : "off")
            << '\n';
  std::cout << fmt::format("\n| {:>5} | {:>7} | {:>9} | {:>10} | {:>12} |",
                           "Trial",
                           "Threads",
                           "Operators",
                           "Total Ops",
                           "Ops/s")
            << '\n';
  std::cout << "|-------|---------|-----------|------------|--------------|" << '\n';

  for (size_t i = 0; i < trial_options.size(); i++) {
    BenchmarkSchedulerThroughputApp::Options& options = trial_options[i];
    BenchmarkSchedulerThroughputApp::Results& results = trial_results[i];

    std::cout << fmt::format("| {:>5} | {:>7} | {:>9} | {:>10} | {:>12.1f} |",
                             i,
                             options.num_threads,
                             options.num_operators,
                             results.total_operations,
                             results.throughput_hz)
              << '\n';
  }

  return 0;
}
