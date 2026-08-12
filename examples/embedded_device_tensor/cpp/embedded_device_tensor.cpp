/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A reusable device-Tensor producer embedded behind one exported Graph port:
//
//   flat:     source --(Tensor)--> sink
//   embedded: [device-stage/source -> exported frame] --(Tensor)--> sink
//
// The two forms compile to the same one-edge, one-pool execution shape. The
// embedded plan is compiled once and starts two fresh RunSessions.

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
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

constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr std::size_t kFrameBytes = 128U;
constexpr std::size_t kFramesPerRun = 4U;
constexpr std::array<std::int64_t, 1U> kFrameShape{static_cast<std::int64_t>(kFrameBytes)};
constexpr holoscan::TensorPortLayout kDeviceFrame{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kByteDtype,
    .rank = 1U,
};
constexpr int kNoCudaDeviceReturnCode = 77;

struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    frames = 0U;
    readiness_contract = true;
  }

  static void record(bool ready) {
    {
      const std::lock_guard lock(mutex);
      ++frames;
      readiness_contract = readiness_contract && ready;
    }
    changed.notify_all();
  }

  [[nodiscard]] static bool wait_for_run() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 10s, [] { return frames >= kFramesPerRun; });
  }

  [[nodiscard]] static bool valid() {
    const std::lock_guard lock(mutex);
    return frames == kFramesPerRun && readiness_contract;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable changed;
  static inline std::size_t frames{};
  static inline bool readiness_contract{true};
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> fail(
    holoscan::ErrorCode code = holoscan::ErrorCode::kFailure) {
  return holoscan::make_unexpected(holoscan::Error{code});
}

class DeviceSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(kDeviceFrame)
        .tensor_allocation(
            holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = kFrameBytes});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 2ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sequence_ >= kFramesPerRun) {
      return {};
    }

    cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }

    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = kFrameShape, .dtype = kByteDtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }

    const int fill = static_cast<int>(++sequence_);
    if (writer->byte_size() != kFrameBytes ||
        cudaMemsetAsync(writer->data(), fill, kFrameBytes, stream) != cudaSuccess) {
      return fail();
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return frame.emit(std::move(*loan),
                      holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::size_t sequence_{};
};

class DeviceSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame").queue_depth(kFramesPerRun).expects_tensor(kDeviceFrame);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (context.cuda_stream().get() == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    auto sample = frame.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = sample->data;
    const DLDevice device = tensor.device();
    if (tensor.nbytes() != static_cast<std::int64_t>(kFrameBytes) || tensor.data() == nullptr ||
        device.device_type != kDLCUDA) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    Observation::record(tensor.has_readiness_stream_contract());
    return {};
  }

  holoscan::Input<holoscan::Tensor> frame;
};

class DeviceStageGraph final : public holoscan::Graph {
 public:
  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

  holoscan::Output<holoscan::Tensor> frame{"frame"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto source = op<DeviceSource>("source");
    export_output(frame, source->frame);
  }

 private:
  holoscan::ClockKey clock_import_;
};

void bind_device(holoscan::CompileOptions& options, std::string_view source_path) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = source_path,
      .output_port = "frame",
      .device = holoscan::DeviceId{0},
  });
}

[[nodiscard]] holoscan::ExecutionPlan compile_embedded() {
  holoscan::Graph graph{"embedded-device-tensor"};
  const auto stage = graph.embed<DeviceStageGraph>("device-stage");
  const auto sink = graph.op<DeviceSink>("sink");
  graph.add_flow(
      stage->frame, sink->frame, holoscan::ConnectionOptions{.queue_depth = kFramesPerRun});

  const holoscan::ClockKey runtime_clock =
      graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.bind_clock(stage, stage->clock_import(), runtime_clock);

  holoscan::CompileOptions options;
  bind_device(options, "/device-stage/source");
  return holoscan::compile(graph, std::move(options));
}

[[nodiscard]] holoscan::ExecutionPlan compile_flat() {
  holoscan::Graph graph{"flat-device-tensor"};
  const auto source = graph.op<DeviceSource>("source");
  const auto sink = graph.op<DeviceSink>("sink");
  graph.add_flow(
      source->frame, sink->frame, holoscan::ConnectionOptions{.queue_depth = kFramesPerRun});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions options;
  bind_device(options, "source");
  return holoscan::compile(graph, std::move(options));
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::cerr << "  diagnostic token=" << diagnostic.token
              << " severity=" << static_cast<int>(diagnostic.severity)
              << " required=" << diagnostic.required << " provided=" << diagnostic.provided << '\n';
  }
}

[[nodiscard]] const holoscan::PortPlanRecord* find_port(const holoscan::ExecutionPlan& plan,
                                                        const holoscan::PortId id) noexcept {
  for (const holoscan::PortPlanRecord& port : plan.ports()) {
    if (port.id == id) {
      return &port;
    }
  }
  return nullptr;
}

[[nodiscard]] bool vertex_is_named(const holoscan::ExecutionPlan& plan, const holoscan::VertexId id,
                                   const std::string_view name) noexcept {
  for (const holoscan::VertexRecord& vertex : plan.graph().vertices()) {
    if (vertex.id == id) {
      return vertex.kind == holoscan::VertexKind::kOperator && vertex.name == name;
    }
  }
  return false;
}

[[nodiscard]] bool equivalent_interface_shape(const holoscan::ExecutionPlan& flat,
                                              const holoscan::ExecutionPlan& embedded) noexcept {
  if (!flat.interface_port_bindings().empty() || embedded.interface_port_bindings().size() != 1U ||
      flat.payload_edges().size() != 1U || embedded.payload_edges().size() != 1U) {
    return false;
  }
  const holoscan::PayloadEdgePlanRecord& flat_edge = flat.payload_edges().front();
  const holoscan::PayloadEdgePlanRecord& embedded_edge = embedded.payload_edges().front();
  const holoscan::InterfacePortBindingPlanRecord& binding =
      embedded.interface_port_bindings().front();
  const holoscan::PortPlanRecord* const flat_source = find_port(flat, flat_edge.source);
  const holoscan::PortPlanRecord* const flat_target = find_port(flat, flat_edge.target);
  const holoscan::PortPlanRecord* const public_source = find_port(embedded, binding.public_port);
  const holoscan::PortPlanRecord* const internal_source =
      find_port(embedded, binding.internal_port);
  const holoscan::PortPlanRecord* const embedded_target = find_port(embedded, embedded_edge.target);
  if (flat_source == nullptr || flat_target == nullptr || public_source == nullptr ||
      internal_source == nullptr || embedded_target == nullptr) {
    return false;
  }
  const bool same_edge_contract =
      flat_edge.queue_policy == embedded_edge.queue_policy &&
      flat_edge.queue_depth == embedded_edge.queue_depth &&
      flat_edge.max_inflight_samples == embedded_edge.max_inflight_samples &&
      flat_edge.producer_max_emits_per_compute == embedded_edge.producer_max_emits_per_compute &&
      flat_edge.fanout == embedded_edge.fanout &&
      flat_edge.cache_capacity == embedded_edge.cache_capacity &&
      flat_edge.locality == embedded_edge.locality &&
      flat_edge.disposition == embedded_edge.disposition &&
      flat_edge.reason == embedded_edge.reason;
  return same_edge_contract && binding.internal_port == embedded_edge.source &&
         binding.direction == holoscan::PortDirection::kOutput &&
         public_source->owner_kind == holoscan::PortOwnerKind::kGraphInterface &&
         internal_source->owner_kind == holoscan::PortOwnerKind::kOperator &&
         public_source->direction == holoscan::PortDirection::kOutput &&
         internal_source->direction == holoscan::PortDirection::kOutput &&
         flat_source->payload_identity.kind == internal_source->payload_identity.kind &&
         flat_source->payload_identity.payload_kind ==
             internal_source->payload_identity.payload_kind &&
         public_source->payload_identity.kind == internal_source->payload_identity.kind &&
         public_source->payload_identity.payload_kind ==
             internal_source->payload_identity.payload_kind &&
         vertex_is_named(flat, flat_source->owner, "source") &&
         vertex_is_named(flat, flat_target->owner, "sink") &&
         vertex_is_named(embedded, internal_source->owner, "source") &&
         vertex_is_named(embedded, embedded_target->owner, "sink");
}

[[nodiscard]] bool equivalent_tensor_resource_shape(
    const holoscan::ExecutionPlan& flat, const holoscan::ExecutionPlan& embedded) noexcept {
  if (flat.tensor_device_resources().size() != 1U ||
      embedded.tensor_device_resources().size() != 1U || flat.payload_edges().size() != 1U ||
      embedded.payload_edges().size() != 1U) {
    return false;
  }
  const holoscan::TensorDeviceResourcePlanRecord& flat_resource =
      flat.tensor_device_resources().front();
  const holoscan::TensorDeviceResourcePlanRecord& embedded_resource =
      embedded.tensor_device_resources().front();
  return flat_resource.fused_linear_edges == embedded_resource.fused_linear_edges &&
         flat_resource.fused_linear_edges.size() == 1U &&
         flat_resource.fused_linear_edges.front() == flat.payload_edges().front().edge &&
         embedded_resource.fused_linear_edges.front() == embedded.payload_edges().front().edge &&
         flat_resource.device.value == 0 && embedded_resource.device.value == 0 &&
         flat_resource.pool_count == embedded_resource.pool_count &&
         flat_resource.slot_capacity == embedded_resource.slot_capacity;
}

[[nodiscard]] bool equivalent_shapes(const holoscan::ExecutionPlan& flat,
                                     const holoscan::ExecutionPlan& embedded) {
  const bool comparable = flat.ok() && embedded.ok() && flat.payload_edges().size() == 1U &&
                          embedded.payload_edges().size() == 1U &&
                          flat.payload_edges().front().queue_depth == kFramesPerRun &&
                          embedded.payload_edges().front().queue_depth == kFramesPerRun;
  const bool same_interface = equivalent_interface_shape(flat, embedded);
  const bool same_tensor_resource = equivalent_tensor_resource_shape(flat, embedded);
  const bool equivalent = comparable && same_interface && same_tensor_resource;
  if (!equivalent) {
    std::cerr << "  flat: edges=" << flat.payload_edges().size()
              << " device_resources=" << flat.tensor_device_resources().size() << '\n';
    std::cerr << "  embedded: edges=" << embedded.payload_edges().size()
              << " device_resources=" << embedded.tensor_device_resources().size() << '\n';
    std::cerr << "  contract comparison: interface=" << same_interface
              << " tensor_resource=" << same_tensor_resource << '\n';
  }
  return equivalent;
}

[[nodiscard]] bool run_replay(const holoscan::ExecutionPlan& plan, std::size_t replay) {
  Observation::reset();
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = Observation::wait_for_run();
  session.request_stop();
  session.wait();
  if (!complete || !Observation::valid() ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "replay " << replay << " failed: complete=" << complete
              << " termination=" << static_cast<int>(session.termination()) << '\n';
    print_diagnostics(session);
    return false;
  }
  std::cout << "[embedded_device_tensor] replay=" << replay << " frames=4 termination=stopped\n";
  return true;
}

struct Arguments {
  std::filesystem::path artifact_root;
};

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& result) {
  if (argc == 1) {
    return true;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--emit-artifacts") {
    result.artifact_root = argv[2];
    return !result.artifact_root.empty();
  }
  return false;
}

[[nodiscard]] int require_cuda() {
  int device_count = 0;
  const cudaError_t probe = cudaGetDeviceCount(&device_count);
  if (probe == cudaErrorNoDevice || probe == cudaErrorInsufficientDriver ||
      (probe == cudaSuccess && device_count == 0)) {
    std::cout << "embedded_device_tensor skipped: no CUDA device\n";
    return kNoCudaDeviceReturnCode;
  }
  if (probe != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(probe) << '\n';
    return 1;
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    std::cout << "embedded_device_tensor skipped: no usable CUDA device\n";
    return kNoCudaDeviceReturnCode;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: embedded_device_tensor [--emit-artifacts <root>]\n";
    return 2;
  }

  if (!arguments.artifact_root.empty()) {
    const holoscan::ExecutionPlan artifact_plan = compile_embedded();
    if (!artifact_plan.ok()) {
      std::cerr << "artifact plan rejected:\n" << artifact_plan.json() << '\n';
      return 4;
    }
    auto emitted = holoscan::emit_artifacts(
        artifact_plan,
        holoscan::ArtifactEmitOptions{.root = std::move(arguments.artifact_root),
                                      .deployment_id = "embedded-device-tensor"});
    if (!emitted) {
      std::cerr << "artifact publication failed: "
                << holoscan::error_code_name(emitted.error().code) << '\n';
      return 5;
    }
    std::cout << "artifact-only publication complete: catalog=" << emitted->catalog_path
              << " manifest=" << emitted->manifest_path
              << " changed=" << (emitted->changed ? "yes" : "no") << '\n';
    return 0;
  }

  const int cuda_status = require_cuda();
  if (cuda_status != 0) {
    return cuda_status;
  }

  const holoscan::ExecutionPlan embedded = compile_embedded();
  if (!embedded.ok()) {
    std::cerr << "embedded plan rejected:\n" << embedded.json() << '\n';
    return 4;
  }

  const holoscan::ExecutionPlan flat = compile_flat();
  if (!equivalent_shapes(flat, embedded)) {
    std::cerr << "flat and embedded plans did not retain the expected equivalent physical shape\n";
    if (!flat.ok()) {
      std::cerr << flat.json() << '\n';
    }
    return 6;
  }

  const bool replay_one = run_replay(embedded, 1U);
  const bool replay_two = run_replay(embedded, 2U);
  if (!replay_one || !replay_two) {
    return 7;
  }
  std::cout << "embedded_device_tensor PASS: flat-equivalent reusable plan replayed twice "
               "device=0\n";
  return 0;
}
