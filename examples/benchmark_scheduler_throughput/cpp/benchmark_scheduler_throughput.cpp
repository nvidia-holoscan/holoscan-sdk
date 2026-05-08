/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

  void start() {
    // Use wall clock instead of Holoscan's internal clock in order to
    // separately measurement from implementation.
    start_time_ = std::chrono::steady_clock::now();
  }

  void stop() { end_time_ = std::chrono::steady_clock::now(); }

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
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(options_.num_threads))));
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

int main() {
  // Construct trial options to benchmark: {num_threads, num_operators}.
  std::vector<BenchmarkSchedulerThroughputApp::Options> trial_options = {
      BenchmarkSchedulerThroughputApp::Options{1, 1},
      BenchmarkSchedulerThroughputApp::Options{2, 2},
      BenchmarkSchedulerThroughputApp::Options{4, 4},
      BenchmarkSchedulerThroughputApp::Options{8, 8},
      BenchmarkSchedulerThroughputApp::Options{16, 16},
      BenchmarkSchedulerThroughputApp::Options{2, 8},
  };
  std::vector<BenchmarkSchedulerThroughputApp::Results> trial_results;

  for (BenchmarkSchedulerThroughputApp::Options& options : trial_options) {
    auto app = holoscan::make_application<BenchmarkSchedulerThroughputApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nScheduler Throughput Benchmark Results:" << '\n';
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
