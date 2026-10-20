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
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "descriptor_frame_schema_traits.hpp"

namespace {

using namespace std::chrono_literals;

constexpr int kNoCudaDeviceReturnCode = 77;

bool use_device_tensor{};
bool use_fanout{};
bool validate_by_value_contract{};

[[nodiscard]] bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

class DeviceTensorMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  explicit DeviceTensorMemoryReference(void* data) : data_(data) {}

  ~DeviceTensorMemoryReference() override {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }

  [[nodiscard]] void* data() const noexcept { return data_; }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t) override {
    return {};
  }

  bool set_deallocation_stream(cudaStream_t) override { return true; }

 private:
  void* data_{};
};

struct Observation {
  static void reset(std::size_t expected) {
    const std::lock_guard lock(mutex);
    source_pointer = nullptr;
    expected_deliveries = expected;
    deliveries = 0U;
    done = false;
    valid = true;
    error.reset();
  }

  static void record_source(const void* pointer) {
    const std::lock_guard lock(mutex);
    source_pointer = pointer;
  }

  static void record(const holoscan::Sample<app3p::DescriptorFrameT>& sample, cudaStream_t stream) {
    const std::lock_guard lock(mutex);
    const holoscan::Tensor* tensor = sample.data.tensor.get();
    std::array<std::uint8_t, 8U> copied_bytes{};
    const bool bytes_available =
        tensor != nullptr &&
        (!use_device_tensor ||
         (stream != nullptr &&
          tensor->copy_to_host_async(copied_bytes.data(), copied_bytes.size(), stream)
              .has_value() &&
          cudaStreamSynchronize(stream) == cudaSuccess));
    const std::uint8_t* bytes =
        tensor == nullptr
            ? nullptr
            : (use_device_tensor ? copied_bytes.data() : tensor->data_as<std::uint8_t>());
    valid = valid && bytes_available && tensor->data() == source_pointer && tensor->nbytes() == 8 &&
            std::ranges::equal(tensor->shape_span(), kShape) && bytes != nullptr &&
            std::ranges::equal(std::span<const std::uint8_t>{bytes, 8U}, kBytes) &&
            sample.data.sequence == 17U && sample.data.label == "installed descriptor frame" &&
            sample.data.gain == 0.25F && sample.data.samples == kSamples &&
            sample.data.aliases == kAliases && sample.data.details != nullptr &&
            sample.data.details->code == 29U && sample.data.details->note == "generated nested" &&
            sample.data.point != nullptr && sample.data.point->x() == 1.5F &&
            sample.data.point->y() == -2.0F && sample.metadata.frame_id == 23U;
    ++deliveries;
    if (deliveries == expected_deliveries) {
      done = true;
      ready.notify_all();
    }
  }

  static void fail(holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      done = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return done; });
  }

  static constexpr std::array<std::int64_t, 1U> kShape{8};
  static constexpr std::array<std::uint8_t, 8U> kBytes{1U, 3U, 5U, 7U, 9U, 11U, 13U, 15U};
  static inline const std::vector<std::int16_t> kSamples{2, 4, 8, 16};
  static inline const std::vector<std::string> kAliases{"alpha", "beta"};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline const void* source_pointer{};
  static inline std::size_t expected_deliveries{};
  static inline std::size_t deliveries{};
  static inline bool done{};
  static inline bool valid{};
  static inline std::optional<holoscan::ErrorCode> error;
};

// Publishes one generated third-party descriptor schema using a host or CUDA Tensor output loan.
// Representative tests: CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation and
// CHANNEL_SCHEMA.public_descriptor_schema_cuda[_fanout]_validation.
class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = use_device_tensor ? holoscan::MemoryKind::kCudaDevice
                                                     : holoscan::MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
            .bounds = holoscan::tensor_bounds(8U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    if (validate_by_value_contract) {
      return emit_by_value_contract_probe();
    }
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = Observation::kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return fail(loan.error().code);
    }
    holoscan::expected<void, holoscan::Error> written;
    if (use_device_tensor) {
      auto* const stream = context.cuda_stream().get();
      written = loan->write_and_commit(
          context.cuda_stream(),
          [stream](const holoscan::TensorOutputWriteGuard& writer)
              -> holoscan::expected<void, holoscan::Error> {
            Observation::record_source(writer.data());
            if (cudaMemcpyAsync(writer.data(),
                                Observation::kBytes.data(),
                                Observation::kBytes.size(),
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
              return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
            }
            return {};
          });
    } else {
      written = loan->write_host_and_commit([](const holoscan::TensorOutputWriteGuard& writer) {
        Observation::record_source(writer.data());
        std::ranges::copy(Observation::kBytes, writer.data_as<std::uint8_t>());
      });
    }
    if (!written) {
      return fail(written.error().code);
    }
    auto frame = make_frame();
    auto emitted =
        output.emit_tensor(std::move(*loan), frame, holoscan::EmitOptions{.frame_id = 23U});
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  holoscan::Output<app3p::DescriptorFrameT> output;

 private:
  [[nodiscard]] static app3p::DescriptorFrameT make_frame() {
    app3p::DescriptorFrameT frame{};
    frame.sequence = 17U;
    frame.label = "installed descriptor frame";
    frame.gain = 0.25F;
    frame.samples = Observation::kSamples;
    frame.aliases = Observation::kAliases;
    frame.details = std::make_shared<app3p::FrameDetailsT>();
    frame.details->code = 29U;
    frame.details->note = "generated nested";
    frame.point = std::make_shared<app3p::FramePoint>(1.5F, -2.0F);
    return frame;
  }

  [[nodiscard]] holoscan::expected<std::shared_ptr<holoscan::Tensor>, holoscan::Error>
  make_device_tensor(std::initializer_list<std::int64_t> shape, DLDataType dtype) {
    if (external_storage_ == nullptr) {
      void* storage = nullptr;
      if (cudaMalloc(&storage, 16U) != cudaSuccess ||
          cudaMemcpy(storage,
                     Observation::kBytes.data(),
                     Observation::kBytes.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaStreamSynchronize(nullptr) != cudaSuccess) {
        if (storage != nullptr) {
          static_cast<void>(cudaFree(storage));
        }
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
      }
      external_storage_ = std::make_shared<DeviceTensorMemoryReference>(storage);
      Observation::record_source(storage);
    }
    auto tensor = std::make_shared<holoscan::DLManagedTensorContext>();
    tensor->memory_reference = external_storage_;
    tensor->dl_shape.assign(shape);
    tensor->tensor.dl_tensor.data = external_storage_->data();
    tensor->tensor.dl_tensor.device = DLDevice{.device_type = kDLCUDA, .device_id = 0};
    tensor->tensor.dl_tensor.ndim = static_cast<std::int32_t>(tensor->dl_shape.size());
    tensor->tensor.dl_tensor.dtype = dtype;
    tensor->tensor.dl_tensor.shape = tensor->dl_shape.data();
    tensor->tensor.dl_tensor.strides = nullptr;
    tensor->tensor.dl_tensor.byte_offset = 0U;
    return std::make_shared<holoscan::Tensor>(std::move(tensor));
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> expect_contract_rejection(
      std::initializer_list<std::int64_t> shape, DLDataType dtype) {
    auto tensor = make_device_tensor(shape, dtype);
    if (!tensor) {
      return holoscan::make_unexpected(std::move(tensor).error());
    }
    auto frame = make_frame();
    frame.tensor = std::move(*tensor);
    auto rejected = output.emit(holoscan::Sample<app3p::DescriptorFrameT>{
        std::move(frame), holoscan::SampleMetadata{.frame_id = 23U}});
    if (rejected || rejected.error().code != holoscan::ErrorCode::kInvalidArgument) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_by_value_contract_probe() {
    // Key setup: exercise dtype, rank, and maximum-span drift through the ordinary schema-value
    // emission API before publishing one matching externally owned CUDA Tensor.
    auto wrong_dtype = expect_contract_rejection({2}, DLDataType{kDLFloat, 32U, 1U});
    auto wrong_rank = expect_contract_rejection({2, 4}, DLDataType{kDLUInt, 8U, 1U});
    auto wrong_span = expect_contract_rejection({9}, DLDataType{kDLUInt, 8U, 1U});
    if (!wrong_dtype || !wrong_rank || !wrong_span) {
      return fail(holoscan::ErrorCode::kFailure);
    }
    auto tensor = make_device_tensor({8}, DLDataType{kDLUInt, 8U, 1U});
    if (!tensor) {
      return fail(tensor.error().code);
    }
    auto frame = make_frame();
    frame.tensor = std::move(*tensor);
    // Key assertion: the matching by-value sample is accepted and reaches every HoloMQ sink.
    auto emitted = output.emit(holoscan::Sample<app3p::DescriptorFrameT>{
        std::move(frame), holoscan::SampleMetadata{.frame_id = 23U}});
    return emitted ? holoscan::expected<void, holoscan::Error>{} : fail(emitted.error().code);
  }

  [[nodiscard]] static holoscan::expected<void, holoscan::Error> fail(holoscan::ErrorCode code) {
    Observation::fail(code);
    return holoscan::make_unexpected(holoscan::Error{code});
  }

  bool emitted_{};
  std::shared_ptr<DeviceTensorMemoryReference> external_storage_;
};

// Receives and validates the generated metadata, Tensor bytes, and retained source-pool identity.
// Representative tests: CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation and
// CHANNEL_SCHEMA.public_descriptor_schema_cuda[_fanout]_validation.
class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = use_device_tensor ? holoscan::MemoryKind::kCudaDevice
                                                     : holoscan::MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      Observation::fail(sample.error().code);
      return holoscan::make_unexpected(std::move(sample).error());
    }
    Observation::record(*sample, context.cuda_stream().get());
    return {};
  }

  holoscan::Input<app3p::DescriptorFrameT> input;
};

[[nodiscard]] bool validate_generated_adapter() {
  static_assert(holoscan::detail::HasDescriptorTensorSchemaNativeTraits<app3p::DescriptorFrameT>);
  static_assert(!holoscan::detail::HasGeneratedSchemaNativeTraits<app3p::DescriptorFrameT>);
  static_assert(holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<app3p::DescriptorFrameT>);

  const auto package =
      holoscan::validate_schema_package(holoscan::schema_package<app3p::DescriptorFrame>());
  if (!package) {
    return false;
  }
  const holoscan::detail::RuntimePortTypeOps* ops =
      holoscan::detail::runtime_port_type_ops<app3p::DescriptorFrameT>();
  if (ops == nullptr ||
      ops->codec_family != holoscan::detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema ||
      ops->payload_kind != holoscan::PayloadKind::kDescriptorBacked ||
      ops->descriptor_tensor == nullptr || ops->descriptor_tensor_count == nullptr ||
      ops->descriptor_tensor_at == nullptr || ops->encode_descriptor_tensor_sample == nullptr ||
      ops->descriptor_tensor_view_metadata_at == nullptr ||
      ops->assign_descriptor_tensor_sample == nullptr ||
      ops->assign_descriptor_tensors_sample == nullptr) {
    return false;
  }
  alignas(holoscan::Sample<app3p::DescriptorFrameT>)
      std::array<std::byte, sizeof(holoscan::Sample<app3p::DescriptorFrameT>)>
          sample_storage{};
  auto prepared = ops->prepare_sample_storage(sample_storage.data());
  if (!prepared) {
    return false;
  }
  auto* prepared_sample = std::launder(
      reinterpret_cast<holoscan::Sample<app3p::DescriptorFrameT>*>(sample_storage.data()));
  if (prepared_sample->data.tensor == nullptr || prepared_sample->data.details != nullptr ||
      prepared_sample->data.point != nullptr || !prepared_sample->data.label.empty() ||
      !prepared_sample->data.samples.empty() || !prepared_sample->data.aliases.empty() ||
      ops->descriptor_tensor_count(&prepared_sample->data) != 1U ||
      ops->descriptor_tensor(&prepared_sample->data) !=
          ops->descriptor_tensor_at(&prepared_sample->data, 0U) ||
      ops->descriptor_tensor_at(&prepared_sample->data, 1U) != nullptr) {
    ops->destroy_sample(sample_storage.data());
    return false;
  }
  prepared_sample->data.details = std::make_shared<app3p::FrameDetailsT>();
  prepared_sample->data.point = std::make_shared<app3p::FramePoint>();
  const auto prepared_details = prepared_sample->data.details;
  const auto prepared_point = prepared_sample->data.point;
  auto nested_prepared =
      app3p::holoscan_generated_DescriptorFrame_package::detail::GeneratedDescriptorCodec{}.prepare(
          prepared_sample->data);
  if (!nested_prepared || prepared_sample->data.details != prepared_details ||
      prepared_sample->data.point != prepared_point || !prepared_details->note.empty()) {
    ops->destroy_sample(sample_storage.data());
    return false;
  }
  prepared_sample->data.sequence = 99U;
  prepared_sample->data.label = "stale";
  prepared_sample->data.samples = {7};
  prepared_sample->data.aliases = {"stale"};
  prepared_details->code = 99U;
  prepared_details->note = "stale";
  *prepared_point = app3p::FramePoint{9.0F, 9.0F};
  ops->reset_sample_storage(sample_storage.data());
  const bool reset =
      prepared_sample->data.sequence == 0U && prepared_sample->data.label.empty() &&
      prepared_sample->data.samples.empty() && prepared_sample->data.aliases.empty() &&
      prepared_sample->data.details == prepared_details && prepared_details->code == 0U &&
      prepared_details->note.empty() && prepared_sample->data.point == prepared_point &&
      prepared_point->x() == 0.0F && prepared_point->y() == 0.0F;
  ops->destroy_sample(sample_storage.data());
  return reset;
}

[[nodiscard]] bool has_expected_plan_shape(const holoscan::ExecutionPlan& plan) {
  return std::ranges::any_of(plan.payload_edges(), [](const auto& edge) {
    return edge.payload_identity.payload_kind == holoscan::PayloadKind::kDescriptorBacked;
  });
}

}  // namespace

// REQ: CHAN-120
// REQ: OPERATOR-035
//   source(Source) --[descriptor-backed DescriptorFrameT]--> sink(Sink)
int main(int argc, char** argv) {
  const std::string_view mode = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
  if (argc > 2 || (!mode.empty() && mode != "device" && mode != "device_fanout" &&
                   mode != "device_fanout_by_value")) {
    return 4;
  }
  use_device_tensor = !mode.empty();
  use_fanout = mode == "device_fanout" || mode == "device_fanout_by_value";
  validate_by_value_contract = mode == "device_fanout_by_value";
  if (use_device_tensor && !cuda_available()) {
    std::cout << "SKIP: public descriptor-schema validation requires CUDA device 0\n";
    return kNoCudaDeviceReturnCode;
  }
  if (!validate_generated_adapter()) {
    return 1;
  }

  Observation::reset(use_fanout ? 2U : 1U);
  const std::string graph_name = mode.empty()
                                     ? "public-descriptor-schema-validation-host"
                                     : "public-descriptor-schema-validation-" + std::string{mode};
  holoscan::Graph graph{graph_name};
  const auto source = graph.op<Source>("source");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  if (use_fanout) {
    const auto second_sink = graph.op<Sink>("second-sink");
    graph.add_flow(
        source->output, second_sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  holoscan::CompileOptions options;
  if (use_device_tensor) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = "output", .device = holoscan::DeviceId{0}});
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, options);
  if (!plan.ok() || !has_expected_plan_shape(plan)) {
    std::cerr << plan.json() << '\n';
    return 2;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = Observation::wait();
  session.request_stop();
  session.wait();

  const std::lock_guard lock(Observation::mutex);
  return observed && Observation::valid && !Observation::error.has_value() &&
                 session.termination() == holoscan::RunTermination::kStopped &&
                 session.diagnostics().empty()
             ? 0
             : 3;
}
