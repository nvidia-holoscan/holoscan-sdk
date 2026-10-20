/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Thin driver for the InferenceOp YAML-config examples (see configs/*.yaml and README.md).
//
// Loads an InferenceOpParams aggregate from the given YAML file via
// InferenceOpParams::from_yaml(), builds a one-InferenceOp Graph with one synthetic
// device-Tensor source per configured input tensor and one counting sink per configured output
// tensor, runs `ticks` end-to-end activations, and prints per-tick wall-clock time plus an
// aggregate summary.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
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
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;
using holoscan::ops::InferenceOp;
using holoscan::ops::InferenceOpParams;

constexpr std::size_t kDefaultTicks = 10U;
// Feed as fast as possible so the delta between consecutive sink ticks reflects the true
// inference latency (feed-bounded when the period is larger than inference time).
constexpr std::chrono::nanoseconds kFeedPeriod = 100us;
constexpr std::chrono::seconds kRunTimeout = 30s;

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};

[[nodiscard]] holoscan::expected<void, holoscan::Error> emit_zero_device_tensor(
    holoscan::Output<holoscan::Tensor>& output, const std::vector<int64_t>& shape,
    holoscan::ExecutionContext& context) {
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported});
  }
  auto loan =
      output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kFloat32});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (cudaMemsetAsync(writer->data(), 0, writer->byte_size(), stream) != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan),
                     holoscan::EmitOptions{.capture_time = context.activation_time()});
}

// TensorFeeder: emits one zero-filled device Tensor per configured input tensor name, in lockstep,
// once per clock tick, so InferenceOp always sees a complete set of inputs for one activation.
// Stops emitting (without erroring) once `ticks` activations have been produced.
class TensorFeeder final : public holoscan::Operator<> {
 public:
  TensorFeeder(std::vector<std::string> tensor_names, std::vector<int64_t> shape, std::size_t ticks)
      : tensor_names_(std::move(tensor_names)), shape_(std::move(shape)), ticks_(ticks) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const std::uint32_t rank = static_cast<std::uint32_t>(shape_.size());
    std::size_t elements = 1U;
    for (const int64_t dim : shape_) {
      elements *= static_cast<std::size_t>(dim);
    }
    const std::size_t element_bytes = (kFloat32.bits * kFloat32.lanes + 7U) / 8U;
    const std::size_t byte_span = elements * element_bytes;
    for (const auto& name : tensor_names_) {
      outputs_.emplace_back(name);
      spec.output(outputs_.back(), name)
          .produces_tensor(holoscan::TensorOutputSpec{
              .representation =
                  holoscan::TensorRepresentation{
                      .memory_kind = holoscan::MemoryKind::kCudaDevice,
                  },
              .bounds = holoscan::TensorBounds{.max_rank = rank, .max_byte_span = byte_span},
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
          });
    }
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFeedPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= ticks_) {
      return {};
    }
    for (auto& output : outputs_) {
      auto result = emit_zero_device_tensor(output, shape_, context);
      if (!result) {
        return holoscan::make_unexpected(std::move(result).error());
      }
    }
    ++sent_;
    return {};
  }

  /// @brief Look up the dynamically-declared output port for one of the fed tensor names.
  [[nodiscard]] holoscan::Output<holoscan::Tensor>* output_port(const std::string& name) const {
    auto it = std::find(tensor_names_.begin(), tensor_names_.end(), name);
    if (it == tensor_names_.end()) {
      return nullptr;
    }
    return const_cast<holoscan::Output<holoscan::Tensor>*>(
        &outputs_[static_cast<std::size_t>(it - tensor_names_.begin())]);
  }

 private:
  std::vector<std::string> tensor_names_;
  std::vector<int64_t> shape_;
  std::size_t ticks_{};
  std::size_t sent_{};
  std::deque<holoscan::Output<holoscan::Tensor>> outputs_;
};

// RunObservation: records one wall-clock timestamp per activation of the designated "primary"
// output tensor (the first output, in sorted tensor-name order); other outputs are drained
// silently. main() waits on this to know when `ticks` end-to-end activations have completed.
struct RunObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    tick_times.clear();
  }

  static void record_tick() {
    {
      const std::lock_guard lock(mutex);
      tick_times.push_back(std::chrono::steady_clock::now());
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_ticks(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, kRunTimeout, [expected] { return tick_times.size() >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::chrono::steady_clock::time_point> tick_times;
};

// TensorSink: drains one InferenceOp output port. The primary sink (is_primary=true) also times
// each activation via RunObservation.
class TensorSink final : public holoscan::Operator<> {
 public:
  explicit TensorSink(bool is_primary) : is_primary_(is_primary) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (is_primary_) {
      RunObservation::record_tick();
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;

 private:
  bool is_primary_{};
};

std::vector<std::string> unique_sorted_values(
    const std::map<std::string, std::vector<std::string>>& model_to_tensors) {
  std::set<std::string> unique;
  for (const auto& [_, names] : model_to_tensors) {
    for (const auto& name : names) {
      unique.insert(name);
    }
  }
  return {unique.begin(), unique.end()};
}

// Mirrors InferenceOp::setup()'s produced/consumed split (see inference.cpp): a tensor name that
// is both an output_map value (for one model) and an input_map value (for another model) is a
// "chained" tensor -- an internal handoff between models that InferenceOp does not expose as an
// external Input<Tensor>/Output<Tensor> port. This driver must not create a TensorFeeder/TensorSink
// for such a name: InferenceOp declares no matching port for it, so `infer->input_port(name)`/
// `infer->output_port(name)` would return nullptr for configs/03_multi_model_chained.yaml's
// "tensor_ab".
std::vector<std::string> external_names(
    const std::map<std::string, std::vector<std::string>>& own_map,
    const std::map<std::string, std::vector<std::string>>& other_map) {
  std::set<std::string> other_values;
  for (const auto& [_, names] : other_map) {
    other_values.insert(names.begin(), names.end());
  }
  std::vector<std::string> result;
  for (const auto& name : unique_sorted_values(own_map)) {
    if (other_values.find(name) == other_values.end()) {
      result.push_back(name);
    }
  }
  return result;
}

struct CliArgs {
  std::string config_path;
  std::size_t ticks = kDefaultTicks;
};

[[nodiscard]] bool parse_args(int argc, char** argv, CliArgs& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      out.config_path = argv[++i];
    } else if (arg == "--ticks" && i + 1 < argc) {
      // Validate before assigning: reject non-numeric input (std::stoul would otherwise
      // throw and escape parse_args) and reject 0 (the summary path in main() reads
      // RunObservation::tick_times.back() after the run, which would trip on an empty vec).
      const std::string ticks_str = argv[++i];
      std::size_t consumed = 0;
      unsigned long value = 0;
      try {
        value = std::stoul(ticks_str, &consumed);
      } catch (const std::exception&) {
        std::cerr << "invalid --ticks value: '" << ticks_str << "' (expected positive integer)\n";
        return false;
      }
      if (consumed != ticks_str.size() || value == 0) {
        std::cerr << "invalid --ticks value: '" << ticks_str << "' (expected positive integer)\n";
        return false;
      }
      out.ticks = static_cast<std::size_t>(value);
    } else {
      std::cerr << "unrecognized argument: " << arg << '\n';
      return false;
    }
  }
  return !out.config_path.empty();
}

}  // namespace

// Keep the example's config, graph execution, and printed timing summary in walkthrough order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(int argc, char** argv) {
  CliArgs args;
  if (!parse_args(argc, argv, args)) {
    std::cerr << "usage: holoinfer_example --config <path.yaml> [--ticks N]\n";
    return 1;
  }

  RunObservation::reset();

  InferenceOpParams params;
  try {
    params = InferenceOpParams::from_yaml(args.config_path);
  } catch (const std::exception& exception) {
    std::cerr << "failed to load '" << args.config_path << "': " << exception.what() << '\n';
    return 1;
  }

  const std::vector<int64_t> shape = params.dynamic_input_dims
                                         ? std::vector<int64_t>{1, 256, 256}
                                         : std::vector<int64_t>{1, 3, 256, 256};

  // External (non-chained) input/output tensor names only -- see external_names() above.
  const std::vector<std::string> input_names = external_names(params.input_map, params.output_map);
  const std::vector<std::string> output_names = external_names(params.output_map, params.input_map);
  if (input_names.empty() || output_names.empty()) {
    std::cerr << "config '" << args.config_path << "' declares no input_map/output_map tensors\n";
    return 1;
  }

  try {
    holoscan::Graph graph{"holoinfer-example"};

    const auto infer = graph.op<InferenceOp>("infer", params);

    const auto feeder = graph.op<TensorFeeder>("feeder", input_names, shape, args.ticks);
    for (const auto& name : input_names) {
      auto* target_port = infer->input_port(name);
      if (target_port == nullptr) {
        std::cerr << "InferenceOp declared no input port named '" << name << "'\n";
        return 1;
      }
      graph.add_flow(
          *feeder->output_port(name), *target_port, holoscan::ConnectionOptions{.queue_depth = 4U});
    }

    bool primary_selected = false;
    for (const auto& name : output_names) {
      auto* source_port = infer->output_port(name);
      if (source_port == nullptr) {
        std::cerr << "InferenceOp declared no output port named '" << name << "'\n";
        return 1;
      }
      const bool is_primary = !primary_selected;
      primary_selected = true;
      const auto sink = graph.op<TensorSink>("sink_" + name, is_primary);
      graph.add_flow(*source_port, sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
    }

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    for (const auto& name : input_names) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "feeder",
          .output_port = name,
          .device = holoscan::DeviceId{0},
      });
    }

    for (const auto& name : output_names) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "infer",
          .output_port = name,
          .device = holoscan::DeviceId{0},
      });
    }

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (const char* debug = std::getenv("HOLOINFER_EXAMPLE_DEBUG_PLAN"); debug != nullptr) {
      std::cerr << plan.json() << '\n';
    }
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      return 1;
    }

    const auto wall_start = std::chrono::steady_clock::now();
    holoscan::RunSession session = holoscan::run_async(plan);
    const bool complete = RunObservation::wait_for_ticks(args.ticks);
    session.request_stop();
    session.wait();

    if (!complete) {
      std::cerr << "run did not complete " << args.ticks << " ticks within the timeout\n";
      for (const auto& diag : session.diagnostics()) {
        std::cerr << "  diagnostic: id=" << diag.id.value << " token=" << diag.token
                  << " vertex=" << diag.source.vertex.value << " port=" << diag.source.port.value()
                  << " required=" << diag.required << " provided=" << diag.provided << '\n';
      }
      return 2;
    }

    const std::string config_name = std::filesystem::path(args.config_path).filename().string();
    std::cout << "Config: " << config_name << '\n';

    std::vector<std::chrono::steady_clock::time_point> ticks;
    {
      const std::lock_guard<std::mutex> lock(RunObservation::mutex);
      ticks = RunObservation::tick_times;
    }
    // Compute per-tick deltas (skip first as warmup).
    std::vector<double> deltas_us;
    deltas_us.reserve(ticks.size());
    for (std::size_t i = 1; i < ticks.size(); ++i) {
      deltas_us.push_back(
          std::chrono::duration<double, std::micro>(ticks[i] - ticks[i - 1]).count());
    }
    // Drop first 5% as warm-up.
    const std::size_t warmup = deltas_us.size() / 20;
    if (deltas_us.size() > warmup) {
      deltas_us.erase(
          deltas_us.begin(),
          deltas_us.begin() + static_cast<std::vector<double>::difference_type>(warmup));
    }
    auto sorted = deltas_us;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
      if (sorted.empty())
        return 0.0;
      auto k = std::min<std::size_t>(
          sorted.size() - 1,
          static_cast<std::size_t>(p / 100.0 * static_cast<double>(sorted.size())));
      return sorted[k];
    };
    double sum = 0.0;
    for (const double d : deltas_us)
      sum += d;
    const double mean = deltas_us.empty() ? 0.0 : sum / static_cast<double>(deltas_us.size());
    double sq = 0.0;
    for (const double d : deltas_us)
      sq += (d - mean) * (d - mean);
    const double stddev =
        deltas_us.size() > 1 ? std::sqrt(sq / static_cast<double>(deltas_us.size() - 1)) : 0.0;
    const double total_ms =
        std::chrono::duration<double, std::milli>(ticks.back() - wall_start).count();
    std::cout << "Summary: " << ticks.size() << " ticks in " << total_ms << " ms\n";
    std::cout << "Per-tick delta (µs) — samples=" << deltas_us.size() << " (post-warmup)\n";
    std::cout << "  min=" << pct(0) << " p50=" << pct(50) << " mean=" << mean << " p90=" << pct(90)
              << " p95=" << pct(95) << " p99=" << pct(99) << " max=" << pct(100)
              << " stddev=" << stddev << "\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "holoinfer_example: error running config '" << args.config_path
              << "': " << exception.what() << '\n';
    return 1;
  }
}
