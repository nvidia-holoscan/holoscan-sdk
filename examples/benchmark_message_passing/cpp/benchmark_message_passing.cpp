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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/logger/logger.hpp>

// Sends messages containing the time the message was created.
class MessageSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MessageSourceOp)

  MessageSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<std::chrono::steady_clock::time_point>>("out");
    spec.param(add_timestamp_component_,
               "add_timestamp_component",
               "Add timestamp component",
               "Whether to attach a GXF Timestamp component to each emitted message.",
               false);
    spec.param(add_cuda_stream_component_,
               "add_cuda_stream_component",
               "Add CUDA stream component",
               "Whether to attach a CudaStreamId component to each emitted message.",
               false);
    spec.param(add_metadata_component_,
               "add_metadata_component",
               "Add metadata component",
               "Whether to attach a MetadataDictionary component to each emitted message.",
               false);
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override {
    if (add_metadata_component_.get() && !is_metadata_enabled()) {
      const std::string error_message =
          "Metadata component was requested, but metadata is disabled for this operator or "
          "application.";
      HOLOSCAN_LOG_ERROR(error_message);
      throw std::runtime_error(error_message);
    }

    auto message =
        std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());

    if (add_metadata_component_.get()) {
      metadata()->set("count", messages_emitted_);
    }

    if (add_cuda_stream_component_.get()) {
      if (!cuda_stream_.has_value()) {
        const std::string stream_name = fmt::format("{}_stream", name());
        auto maybe_stream = context.allocate_cuda_stream(stream_name);
        if (!maybe_stream) {
          throw std::runtime_error(
              fmt::format("Failed to allocate CUDA stream: {}", maybe_stream.error().what()));
        }
        cuda_stream_ = maybe_stream.value();
      }
      output.set_cuda_stream(cuda_stream_.value(), "out");
    }

    if (add_timestamp_component_.get()) {
      auto scheduler = fragment_->scheduler();
      auto clock = scheduler->clock();
      auto timestamp = clock->timestamp();
      output.emit(message, "out", timestamp);
    } else {
      output.emit(message, "out");
    }

    ++messages_emitted_;
  }

 private:
  holoscan::Parameter<bool> add_timestamp_component_;
  holoscan::Parameter<bool> add_cuda_stream_component_;
  holoscan::Parameter<bool> add_metadata_component_;
  std::optional<cudaStream_t> cuda_stream_;
  int messages_emitted_ = 0;
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
  std::chrono::steady_clock::time_point start_time() const {
    if (!start_time_.has_value()) {
      throw std::logic_error("MessageSinkOp::start_time() called before receiving a message");
    }
    return start_time_.value();
  }
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

    // Whether to attach a GXF Timestamp component to each emitted message.
    bool add_timestamp_component = false;

    // Whether to attach a CudaStreamId component to each emitted message.
    bool add_cuda_stream_component = false;

    // Whether to attach a MetadataDictionary component to each emitted message.
    bool add_metadata_component = false;
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
          make_condition<holoscan::CountCondition>(options_.num_messages),
          holoscan::Arg("add_timestamp_component", options_.add_timestamp_component),
          holoscan::Arg("add_cuda_stream_component", options_.add_cuda_stream_component),
          holoscan::Arg("add_metadata_component", options_.add_metadata_component)));
      sinks_.push_back(make_operator<MessageSinkOp>(fmt::format("sink_{}", i)));

      add_flow(sources_.back(), sinks_.back(), {{"out", "in"}});
    }

    if (options_.num_threads == 0) {
      scheduler(make_scheduler<holoscan::GreedyScheduler>("scheduler"));
    } else {
      scheduler(make_scheduler<holoscan::EventBasedScheduler>(
          "scheduler",
          holoscan::Arg("worker_thread_number", static_cast<int64_t>(options_.num_threads)),
          holoscan::Arg("enable_queue_stealing", true),
          holoscan::Arg("enable_worker_postcheck_fastpath", true)));
    }
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
    results.message_throughput_hz =
        (duration_s > 0.0f) ? static_cast<float>(total_messages) / duration_s : 0.0f;
    results.average_message_latency_us =
        (total_messages > 0) ? total_message_latency_us / total_messages : 0.0f;
    results.max_message_latency_us = max_message_latency_us;
    return results;
  }

 private:
  Options options_;

  std::vector<std::shared_ptr<MessageSourceOp>> sources_;
  std::vector<std::shared_ptr<MessageSinkOp>> sinks_;
};

namespace {

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name
            << " [--add-timestamp] [--add-cuda-stream] [--add-metadata] [--threads N]" << '\n';
  std::cout << "  --threads 0 uses GreedyScheduler; values > 0 use EventBasedScheduler." << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkMessagePassingApp::Options component_options;
  std::optional<int> thread_override;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--add-timestamp") {
      component_options.add_timestamp_component = true;
    } else if (arg == "--add-cuda-stream") {
      component_options.add_cuda_stream_component = true;
    } else if (arg == "--add-metadata") {
      component_options.add_metadata_component = true;
    } else if (arg == "--threads") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --threads" << '\n';
        print_usage(argv[0]);
        return 1;
      }
      thread_override = std::stoi(argv[++i]);
      if (thread_override.value() < 0) {
        std::cerr << "--threads must be greater than or equal to 0" << '\n';
        print_usage(argv[0]);
        return 1;
      }
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
                                int num_flows) -> BenchmarkMessagePassingApp::Options {
    BenchmarkMessagePassingApp::Options options = component_options;
    options.num_threads = num_threads;
    options.num_flows = num_flows;
    return options;
  };

  // Construct trial options to benchmark: {num_threads, num_flows}.
  const std::vector<std::pair<int, int>> trial_matrix = {
      {0, 1},
      {0, 8},
      {1, 1},
      {2, 2},
      {4, 4},
      {8, 8},
      {2, 8},
  };
  std::vector<BenchmarkMessagePassingApp::Options> trial_options;
  trial_options.reserve(trial_matrix.size());
  for (const auto& [num_threads, num_flows] : trial_matrix) {
    const int trial_num_threads = thread_override.value_or(num_threads);
    auto options = make_trial_options(trial_num_threads, num_flows);
    const auto duplicate =
        std::find_if(trial_options.begin(),
                     trial_options.end(),
                     [&](const BenchmarkMessagePassingApp::Options& existing_options) {
                       return existing_options.num_threads == options.num_threads &&
                              existing_options.num_flows == options.num_flows;
                     });
    if (duplicate == trial_options.end()) {
      trial_options.push_back(std::move(options));
    }
  }
  std::vector<BenchmarkMessagePassingApp::Results> trial_results;

  for (BenchmarkMessagePassingApp::Options& options : trial_options) {
    auto app = holoscan::make_application<BenchmarkMessagePassingApp>();
    app->set_options(options);
    app->run();
    trial_results.push_back(app->results());
  }

  std::cout << "\nMessage Passing Benchmark Results:" << '\n';
  std::cout << fmt::format("Extra entity components: timestamp={}, cuda_stream={}, metadata={}",
                           component_options.add_timestamp_component ? "on" : "off",
                           component_options.add_cuda_stream_component ? "on" : "off",
                           component_options.add_metadata_component ? "on" : "off")
            << '\n';
  if (thread_override.has_value()) {
    std::cout << fmt::format(
                     "Scheduler override: {}",
                     thread_override.value() == 0
                         ? "GreedyScheduler (--threads 0)"
                         : fmt::format("EventBasedScheduler ({} threads)", thread_override.value()))
              << '\n';
  }
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
