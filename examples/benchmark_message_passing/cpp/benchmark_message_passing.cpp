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

#include "fmt/format.h"
#include "holoscan/holoscan.hpp"

// Sends messages containing the time the message was created.
class MessageSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MessageSourceOp)

  MessageSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<std::chrono::steady_clock::time_point>>("out");
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& output,
               holoscan::ExecutionContext&) override {
    auto message =
        std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());
    output.emit(message, "out");
  }
};

// Receives messages and tracks statistics.
class MessageSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MessageSinkOp)

  MessageSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::shared_ptr<std::chrono::steady_clock::time_point>>("in");
  }

  void compute(holoscan::InputContext& input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto start_time =
        *(input.receive<std::shared_ptr<std::chrono::steady_clock::time_point>>("in").value());

    end_time_ = std::chrono::steady_clock::now();

    messages_received_++;

    std::chrono::duration message_latency = end_time_ - start_time;
    total_message_latency_ += message_latency;
    if (message_latency > max_message_latency_) {
      max_message_latency_ = message_latency;
    }

    if (!start_time_.has_value()) {
      start_time_ = start_time;
    }
  }

  int messages_received() const { return messages_received_; }
  std::chrono::steady_clock::time_point start_time() const { return *start_time_; }
  std::chrono::steady_clock::time_point end_time() const { return end_time_; }
  std::chrono::steady_clock::duration total_message_latency() const {
    return total_message_latency_;
  }
  std::chrono::steady_clock::duration max_message_latency() const { return max_message_latency_; }

 private:
  int messages_received_ = 0;
  std::optional<std::chrono::steady_clock::time_point> start_time_;
  std::chrono::steady_clock::time_point end_time_;
  std::chrono::steady_clock::duration total_message_latency_ =
      std::chrono::steady_clock::duration::zero();
  std::chrono::steady_clock::duration max_message_latency_ =
      std::chrono::steady_clock::duration::zero();
};

// Benchmarks message latency and throughput by passing messages between source
// and sink operators.
class BenchmarkMessagePassingApp : public holoscan::Application {
 public:
  struct Options {
    // Number of worker threads used by the scheduler.
    int num_threads = 1;

    // Number of source -> sink operator flows.
    int num_flows = 1;

    // Messages per flow.
    int num_messages = 100000;
  };

  struct Results {
    int total_messages;
    float message_throughput_hz;
    float average_message_latency_us;
    float max_message_latency_us;
  };

  void set_options(const Options& options) { options_ = options; }

  void compose() override {
    for (int i = 0; i < options_.num_flows; i++) {
      sources_.push_back(make_operator<MessageSourceOp>(
          fmt::format("source_{}", i),
          make_condition<holoscan::CountCondition>(options_.num_messages)));
      sinks_.push_back(make_operator<MessageSinkOp>(fmt::format("sink_{}", i)));

      add_flow(sources_.back(), sinks_.back(), {{"out", "in"}});
    }

    scheduler(make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler",
        holoscan::Arg("worker_thread_number", static_cast<int64_t>(options_.num_threads))));
  }

  Results results() {
    // Aggregate statistics from all operator flows.
    int total_messages = 0;
    std::chrono::steady_clock::time_point min_start_time = sinks_[0]->start_time();
    std::chrono::steady_clock::time_point max_end_time = sinks_[0]->end_time();
    float total_message_latency_us = 0;
    float max_message_latency_us = 0;
    for (const std::shared_ptr<MessageSinkOp>& sink : sinks_) {
      total_messages += sink->messages_received();
      min_start_time = std::min(min_start_time, sink->start_time());
      max_end_time = std::max(max_end_time, sink->end_time());
      total_message_latency_us +=
          std::chrono::duration_cast<std::chrono::duration<float, std::micro>>(
              sink->total_message_latency())
              .count();
      max_message_latency_us =
          std::max(max_message_latency_us,
                   std::chrono::duration_cast<std::chrono::duration<float, std::micro>>(
                       sink->max_message_latency())
                       .count());
    }

    // Calculate and return the results.
    float duration_s =
        std::chrono::duration_cast<std::chrono::duration<float>>(max_end_time - min_start_time)
            .count();
    Results results;
    results.total_messages = total_messages;
    results.message_throughput_hz = total_messages / duration_s;
    results.average_message_latency_us = total_message_latency_us / total_messages;
    results.max_message_latency_us = max_message_latency_us;
    return results;
  }

 private:
  Options options_;

  std::vector<std::shared_ptr<MessageSourceOp>> sources_;
  std::vector<std::shared_ptr<MessageSinkOp>> sinks_;
};

int main() {
  // Construct trial options to benchmark: {num_threads, num_flows}.
  std::vector<BenchmarkMessagePassingApp::Options> trial_options = {
      BenchmarkMessagePassingApp::Options{1, 1}, BenchmarkMessagePassingApp::Options{2, 2},
      BenchmarkMessagePassingApp::Options{4, 4}, BenchmarkMessagePassingApp::Options{8, 8},
      BenchmarkMessagePassingApp::Options{2, 8},
  };
  std::vector<BenchmarkMessagePassingApp::Results> trial_results;

  for (BenchmarkMessagePassingApp::Options& options : trial_options) {
    auto app = holoscan::make_application<BenchmarkMessagePassingApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nMessage Passing Benchmark Results:" << '\n';
  std::cout << fmt::format("\n| {:>5} | {:>7} | {:>5} | {:>10} | {:>12} | {:>10} | {:>10} |",
                           "Trial",
                           "Threads",
                           "Flows",
                           "Total Msgs",
                           "Msgs/s",
                           "Avg μs",
                           "Max μs")
            << '\n';
  std::cout << "|-------|---------|-------|------------|--------------|------------|------------|"
            << '\n';

  for (size_t i = 0; i < trial_options.size(); i++) {
    BenchmarkMessagePassingApp::Options& options = trial_options[i];
    BenchmarkMessagePassingApp::Results& results = trial_results[i];

    std::cout << fmt::format(
                     "| {:>5} | {:>7} | {:>5} | {:>10} | {:>12.1f} | {:>10.3f} | {:>10.3f} |",
                     i,
                     options.num_threads,
                     options.num_flows,
                     results.total_messages,
                     results.message_throughput_hz,
                     results.average_message_latency_us,
                     results.max_message_latency_us)
              << '\n';
  }

  return 0;
}
