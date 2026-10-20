/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/memory_storage_type.hpp>
#include <holoscan/memory/tensor_storage.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_inline_execution_test_support.hpp"

namespace holoscan {

// REQ: RUNTIME-040
// REQ: RUNTIME-041

namespace {

using namespace std::chrono_literals;

using namespace test::inline_execution;

class ScopedCUDADeviceRestore final {
 public:
  explicit ScopedCUDADeviceRestore(std::int32_t device) noexcept : device_(device) {}

  ScopedCUDADeviceRestore(const ScopedCUDADeviceRestore&) = delete;
  ScopedCUDADeviceRestore& operator=(const ScopedCUDADeviceRestore&) = delete;

  ~ScopedCUDADeviceRestore() {
    if (device_ >= 0) {
      static_cast<void>(cudaSetDevice(device_));
    }
  }

 private:
  std::int32_t device_{-1};
};

// Publishes one runtime-pooled raw Tensor in the requested memory kind.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineTensorSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineTensorSource<Kind>> {
 public:
  explicit InlineTensorSource(std::int32_t ambient_device_after_emit = -1)
      : InlinePartitionLifecycle<InlineTensorSource<Kind>>(1U),
        ambient_device_after_emit_(ambient_device_after_emit) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::array<std::int64_t, 1U> kShape{1};
    constexpr std::uint8_t kValue = 0x5AU;
    InlinePartitionProbeState::record_compute(this->operator_id_, kValue);
    if (emitted_) {
      return {};
    }

    auto* const callback_stream = context.cuda_stream().get();
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    void* pointer = nullptr;
    {
      auto loan = output.allocate_tensor(TensorLoanRequest{
          .shape = kShape,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
      });
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }

      if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
        if (callback_stream == nullptr || cudaGetDevice(&callback_device) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        auto writer = loan->write(context.cuda_stream());
        if (!writer) {
          return make_unexpected(std::move(writer.error()));
        }
        pointer = writer->data();
        if (cudaMemsetAsync(pointer, kValue, 1U, callback_stream) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        auto committed = std::move(*writer).commit();
        if (!committed) {
          return make_unexpected(std::move(committed.error()));
        }
      } else {
        auto writer = loan->write_host();
        if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
          return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
        }
        pointer = writer->data();
        *writer->data_as<std::uint8_t>() = kValue;
        auto committed = std::move(*writer).commit();
        if (!committed) {
          return make_unexpected(std::move(committed.error()));
        }
      }

      InlineTensorProbeState::record_source(pointer, callback_stream, callback_device);
      auto published = output.emit(std::move(*loan));
      if (!published) {
        return make_unexpected(std::move(published.error()));
      }
    }
    emitted_ = true;

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (ambient_device_after_emit_ >= 0 &&
          cudaSetDevice(ambient_device_after_emit_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    return {};
  }

  Output<Tensor> output;

 private:
  std::int32_t ambient_device_after_emit_{-1};
  bool emitted_{};
};

// Reads a raw Tensor, captures backing/device evidence, and emits a POD result.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineTensorRelay final : public Operator<>,
                                public InlinePartitionLifecycle<InlineTensorRelay<Kind>> {
 public:
  explicit InlineTensorRelay(std::uint32_t operator_id = 2U, bool fail_compute = false,
                             bool retain_tensor = false,
                             std::int32_t ambient_device_before_failure = -1)
      : InlinePartitionLifecycle<InlineTensorRelay<Kind>>(operator_id,
                                                          ambient_device_before_failure >= 0),
        fail_compute_(fail_compute),
        retain_tensor_(retain_tensor),
        ambient_device_before_failure_(ambient_device_before_failure) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }

    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    auto* const callback_stream = context.cuda_stream().get();
    auto* const readiness_stream = detail::TensorAccess::readiness_stream(sample->data);
    std::uint8_t value{};
    auto copied = sample->data.copy_to_host(&value, sizeof(value), context.cuda_stream());
    if (!copied) {
      return make_unexpected(std::move(copied.error()));
    }

    InlineTensorProbeState::record_relay(
        sample->data.data(), callback_stream, readiness_stream, callback_device, value);
    if (retain_tensor_) {
      InlineTensorProbeState::retain_tensor(sample->data);
    }
    InlinePartitionProbeState::record_compute(this->operator_id_, value);
    if (fail_compute_) {
      if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
        if (ambient_device_before_failure_ >= 0 &&
            cudaSetDevice(ambient_device_before_failure_) != cudaSuccess) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
      }
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_TENSOR_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<Tensor> input;
  Output<InlinePartitionPod> output;

 private:
  bool fail_compute_{};
  bool retain_tensor_{};
  std::int32_t ambient_device_before_failure_{-1};
};

constexpr std::array<std::int64_t, 1U> kCausalTensorShape{1};

constexpr TensorRepresentation kCausalTensorRepresentation{
    .memory_kind = MemoryKind::kHost,
    .dtype = DLDataType{kDLUInt, 32U, 1U},
    .rank = 1U,
    .strides = TensorStridePolicy::kContiguousOnly,
};

// Publishes numbered raw-Tensor frames from an effectively continuous clock.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorSource final : public Operator<> {
 public:
  explicit CausalTensorSource(std::size_t frame_count) : frame_count_(frame_count) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .bounds = tensor_bounds(kCausalTensorShape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ == frame_count_) {
      return {};
    }
    auto loan = output.allocate_tensor(kCausalTensorShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto committed = loan->write_host_and_commit(
        [frame = static_cast<std::uint32_t>(next_frame_)](const TensorOutputWriteGuard& writer) {
          *writer.data_as<std::uint32_t>() = frame;
        });
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(
        1U, static_cast<std::uint32_t>(next_frame_), context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<Tensor> output;

 private:
  std::size_t frame_count_{};
  std::size_t next_frame_{};
};

// Forwards each raw Tensor without replacing its backing and records the causal middle stage.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPassThrough final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    auto synchronized = sample->data.synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data_as<std::uint32_t>();
    auto emitted = output.emit(std::move(sample->data));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(2U, frame, context.tag_time());
    return {};
  }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Consumes numbered raw Tensors and signals completion after recording the terminal stage.
// Representative test: RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    auto synchronized = sample->data.synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data_as<std::uint32_t>();
    InlinePartitionProbeState::record_compute(3U, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    return {};
  }

  Input<Tensor> input;
};

// Publishes numbered one-Tensor schema samples from an effectively continuous clock.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketSource final : public Operator<> {
 public:
  explicit CausalTensorPacketSource(std::size_t frame_count) : frame_count_(frame_count) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .bounds = tensor_bounds(kCausalTensorShape),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ == frame_count_) {
      return {};
    }
    auto loan = output.allocate_tensor(kCausalTensorShape);
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    auto committed = loan->write_host_and_commit(
        [frame = static_cast<std::uint32_t>(next_frame_)](const TensorOutputWriteGuard& writer) {
          *writer.data_as<std::uint32_t>() = frame;
        });
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    schema::TensorPacketT descriptor{};
    descriptor.sequence = next_frame_;
    descriptor.source = "causal-tensor-packet";
    auto emitted =
        output.emit_tensor(std::move(*loan), descriptor, EmitOptions{.frame_id = next_frame_});
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(
        1U, static_cast<std::uint32_t>(next_frame_), context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<schema::TensorPacketT> output;

 private:
  std::size_t frame_count_{};
  std::size_t next_frame_{};
};

// Forwards an owning TensorPacket sample while preserving its Tensor and semantic metadata.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketPassThrough final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = kCausalTensorRepresentation,
            .storage = TensorOutputStorage::kPassThrough,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto synchronized = sample->data.data->synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data->data_as<std::uint32_t>();
    if (sample->data.sequence != frame || sample->metadata.frame_id != frame) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(2U, frame, context.tag_time());
    return {};
  }

  Input<schema::TensorPacketT> input;
  Output<schema::TensorPacketT> output;
};

// Validates forwarded TensorPacket fields and records the terminal causal stage.
// Representative test: TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick.
class CausalTensorPacketSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCausalTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto synchronized = sample->data.data->synchronize_for_host();
    if (!synchronized) {
      return make_unexpected(std::move(synchronized.error()));
    }
    const std::uint32_t frame = *sample->data.data->data_as<std::uint32_t>();
    if (sample->data.sequence != frame || sample->metadata.frame_id != frame) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    InlinePartitionProbeState::record_compute(3U, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    return {};
  }

  Input<schema::TensorPacketT> input;
};

// Publishes a descriptor-backed Image with a runtime-pooled Tensor payload.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineImageSource final : public Operator<>,
                                public InlinePartitionLifecycle<InlineImageSource<Kind>> {
 public:
  explicit InlineImageSource(std::int32_t ambient_device_after_emit = -1,
                             bool declare_allocation = true)
      : InlinePartitionLifecycle<InlineImageSource<Kind>>(1U),
        ambient_device_after_emit_(ambient_device_after_emit),
        declare_allocation_(declare_allocation) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    TensorOutputSpec contract{
        .representation = {.memory_kind = Kind, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U},
        .bounds = tensor_bounds(8U),
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                .max_staging_byte_span = 8U,
            },
    };
    if (declare_allocation_) {
      contract.storage = TensorOutputStorage::kRuntimePool;
    }
    spec.output(output, "output").max_emits_per_compute(1U).produces_tensor(contract);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::array<std::int64_t, 1U> kShape{8};
    constexpr std::uint8_t kValue = 0x5AU;
    InlinePartitionProbeState::record_compute(this->operator_id_, kValue);
    if (emitted_) {
      return {};
    }

    auto* const callback_stream = context.cuda_stream().get();
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    void* pointer = nullptr;
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (callback_stream == nullptr || cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      pointer = writer->data();
      if (cudaMemsetAsync(pointer, kValue, 8U, callback_stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    } else {
      auto writer = loan->write_host();
      if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
        return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
      }
      pointer = writer->data();
      std::ranges::fill(std::span<std::uint8_t>{writer->data_as<std::uint8_t>(), 8U}, kValue);
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    }

    InlineTensorProbeState::record_source(pointer, callback_stream, callback_device);
    schema::ImageT descriptor{};
    descriptor.header = std::make_shared<schema::HeaderT>();
    descriptor.header->stamp = std::make_shared<schema::TimeT>();
    descriptor.header->stamp->time_ns = 777;
    descriptor.header->stamp->acqtime_ns = 888;
    descriptor.header->frame_id = "inline-image";
    descriptor.height = 1;
    descriptor.width = 8;
    descriptor.encoding = schema::ImageEncoding_CUSTOM;
    auto published = output.emit_tensor(
        std::move(*loan),
        descriptor,
        EmitOptions{
            .capture_time = TimePoint{.timestamp_ns = 1234, .clock_id = ClockId{1U}},
            .frame_id = 9U,
            .trace = TraceContext::root(0xCAFEU),
        });
    if (!published) {
      return make_unexpected(std::move(published.error()));
    }
    emitted_ = true;

    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (ambient_device_after_emit_ >= 0 &&
          cudaSetDevice(ambient_device_after_emit_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    return {};
  }

  Output<schema::ImageT> output;

 private:
  std::int32_t ambient_device_after_emit_{-1};
  bool declare_allocation_{true};
  bool emitted_{};
};

// Validates an Image descriptor and Tensor payload before emitting a POD result.
// Representative test: InlinePayloadParity.HostPreservesBackingMetadataAndControllerOwnership.
template <MemoryKind Kind>
class InlineImageRelay final : public Operator<>,
                               public InlinePartitionLifecycle<InlineImageRelay<Kind>> {
 public:
  explicit InlineImageRelay(std::uint32_t operator_id = 2U, bool fail_compute = false,
                            bool retain_tensor = false)
      : InlinePartitionLifecycle<InlineImageRelay<Kind>>(operator_id),
        fail_compute_(fail_compute),
        retain_tensor_(retain_tensor) {}

  void setup(OperatorSpec& spec) override {
    this->setup_lifecycle(spec);
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    if (!sample->data.data) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_TENSOR_MISSING"});
    }

    const Tensor& tensor = *sample->data.data;
    // Device-backed template specializations populate this through cudaGetDevice().
    std::int32_t callback_device = -1;  // NOLINT(misc-const-correctness)
    if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
      if (cudaGetDevice(&callback_device) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    std::array<std::uint8_t, 8U> bytes{};
    auto copied = tensor.copy_to_host(bytes.data(), bytes.size(), context.cuda_stream());
    if (!copied) {
      return make_unexpected(std::move(copied.error()));
    }

    constexpr ClockId kExpectedClockId{1U};
    const bool descriptor_valid =
        tensor.nbytes() == bytes.size() &&
        std::ranges::all_of(bytes, [](std::uint8_t value) { return value == 0x5AU; }) &&
        sample->data.height == 1 && sample->data.width == 8 &&
        sample->data.encoding == schema::ImageEncoding_CUSTOM && sample->data.header &&
        sample->data.header->stamp && sample->data.header->stamp->time_ns == 777 &&
        sample->data.header->stamp->acqtime_ns == 888 &&
        sample->data.header->frame_id == "inline-image" &&
        sample->metadata.capture_timestamp_ns == 1234 &&
        sample->metadata.source_clock_id == kExpectedClockId &&
        sample->metadata.trace_id == 0xCAFEU && sample->metadata.frame_id == 9U &&
        sample->metadata.sequence == 1U && sample->metadata.byte_size == bytes.size() &&
        sample->metadata.payload_kind == PayloadKind::kDescriptorBacked;
    if (!descriptor_valid) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_METADATA_MISMATCH"});
    }

    auto* const callback_stream = context.cuda_stream().get();
    InlineTensorProbeState::record_relay(tensor.data(),
                                         callback_stream,
                                         detail::TensorAccess::readiness_stream(tensor),
                                         callback_device,
                                         bytes.front());
    InlinePartitionProbeState::record_compute(this->operator_id_, bytes.front());
    InlinePartitionProbeState::record_schema("image-ok");
    if (retain_tensor_) {
      InlineTensorProbeState::retain_tensor(tensor);
    }
    if (fail_compute_) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_IMAGE_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = bytes.front()});
  }

  Input<schema::ImageT> input;
  Output<InlinePartitionPod> output;

 private:
  bool fail_compute_{};
  bool retain_tensor_{};
};

// Converts one POD fork branch into a Tensor with an independently selected placement.
// Representative test: MixedTensorPlacementKeepsTheOneRootForkOrdinary.
template <MemoryKind Kind>
class InlineTensorBranch final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = Kind,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
            .bounds = tensor_bounds(1U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    constexpr std::array<std::int64_t, 1U> kShape{1};
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return make_unexpected(std::move(loan.error()));
    }
    if constexpr (Kind == MemoryKind::kCudaDevice) {
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer.error()));
      }
      if (cudaMemsetAsync(writer->data(),
                          static_cast<int>(sample->data.value),
                          1U,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    } else {
      auto writer = loan->write_host();
      if (!writer || writer->data_as<std::uint8_t>() == nullptr) {
        return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer.error()));
      }
      *writer->data_as<std::uint8_t>() = static_cast<std::uint8_t>(sample->data.value);
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed.error()));
      }
    }
    return output.emit(std::move(*loan));
  }

  Input<InlinePartitionPod> input;
  Output<Tensor> output;
};

enum class TensorPlanRoute : std::uint8_t {
  kInline,
  kHoloMQ,
  kFanout,
};

template <MemoryKind Kind>
[[nodiscard]] ExecutionPlan make_tensor_chain_plan(
    QueuePolicy policy, TensorPlanRoute route = TensorPlanRoute::kInline,
    DeviceId device = DeviceId{}, std::int32_t ambient_device_after_emit = -1,
    bool fail_relay = false, bool retain_tensor = false,
    std::int32_t ambient_device_before_failure = -1) {
  // [source] -- Tensor --> [relay] -- POD --> [sink]
  //     \---- optional fanout --> [second relay] --> [second sink]
  Graph graph{"inline-tensor-chain"};
  auto source = graph.op<InlineTensorSource<Kind>>("source", ambient_device_after_emit);
  auto relay = graph.op<InlineTensorRelay<Kind>>(
      "relay", 2U, fail_relay, retain_tensor, ambient_device_before_failure);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(policy));
  graph.add_flow(relay->output, sink->input, queue_options(policy));
  if (route == TensorPlanRoute::kFanout) {
    auto second_relay = graph.op<InlineTensorRelay<Kind>>("second-relay", 4U);
    auto second_sink = graph.op<InlinePodSink>("second-sink", 5U);
    graph.add_flow(source->output, second_relay->input, queue_options(policy));
    graph.add_flow(second_relay->output, second_sink->input, queue_options(policy));
  }
  if (route == TensorPlanRoute::kInline) {
    graph.partition("inline-tensor-complete-cover").include_all();
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = device,
    });
  }
  return route == TensorPlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

template <MemoryKind Kind>
[[nodiscard]] ExecutionPlan make_image_chain_plan(
    QueuePolicy policy, TensorPlanRoute route = TensorPlanRoute::kInline,
    DeviceId device = DeviceId{}, std::int32_t ambient_device_after_emit = -1,
    bool fail_relay = false, bool retain_tensor = false, bool declare_allocation = true) {
  // [source] -- Image --> [relay] -- POD --> [sink]
  //     \---- optional fanout --> [second relay] --> [second sink]
  Graph graph{"inline-image-chain"};
  auto source =
      graph.op<InlineImageSource<Kind>>("source", ambient_device_after_emit, declare_allocation);
  auto relay = graph.op<InlineImageRelay<Kind>>("relay", 2U, fail_relay, retain_tensor);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(policy));
  graph.add_flow(relay->output, sink->input, queue_options(policy));
  if (route == TensorPlanRoute::kFanout) {
    auto second_relay = graph.op<InlineImageRelay<Kind>>("second-relay", 4U);
    auto second_sink = graph.op<InlinePodSink>("second-sink", 5U);
    graph.add_flow(source->output, second_relay->input, queue_options(policy));
    graph.add_flow(second_relay->output, second_sink->input, queue_options(policy));
  }
  if (route == TensorPlanRoute::kInline) {
    graph.partition("inline-image-complete-cover").include_all();
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (Kind == MemoryKind::kCudaDevice || Kind == MemoryKind::kCudaManaged) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = device,
    });
  }
  return route == TensorPlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

struct InlinePayloadPlanOptions {
  QueuePolicy policy{QueuePolicy::kBlocking};
  TensorPlanRoute route{TensorPlanRoute::kInline};
  DeviceId device{};
  std::int32_t ambient_device_after_emit{-1};
  bool fail_relay{false};
  bool retain_tensor{false};
};

// Exercises direct raw-Tensor inline execution through the common parity scenarios.
struct RawTensorCase {
  static constexpr std::string_view name{"RawTensor"};

  template <MemoryKind Kind>
  [[nodiscard]] static ExecutionPlan make_plan(const InlinePayloadPlanOptions& options) {
    return make_tensor_chain_plan<Kind>(options.policy,
                                        options.route,
                                        options.device,
                                        options.ambient_device_after_emit,
                                        options.fail_relay,
                                        options.retain_tensor);
  }

  static void expect_payload_evidence(const InlinePartitionRunEvidence& evidence) {
    EXPECT_FALSE(evidence.schema_value.has_value());
  }
};

// Exercises descriptor-backed Image inline execution through the common parity scenarios.
struct DescriptorImageCase {
  static constexpr std::string_view name{"DescriptorImage"};

  template <MemoryKind Kind>
  [[nodiscard]] static ExecutionPlan make_plan(const InlinePayloadPlanOptions& options) {
    return make_image_chain_plan<Kind>(options.policy,
                                       options.route,
                                       options.device,
                                       options.ambient_device_after_emit,
                                       options.fail_relay,
                                       options.retain_tensor);
  }

  static void expect_payload_evidence(const InlinePartitionRunEvidence& evidence) {
    EXPECT_EQ(evidence.schema_value, std::optional<std::string>{"image-ok"});
  }
};

template <MemoryKind LeftKind, MemoryKind RightKind>
[[nodiscard]] ExecutionPlan make_mixed_tensor_fork_plan(DeviceId left_device = DeviceId{},
                                                        DeviceId right_device = DeviceId{}) {
  // [source] -> [left branch]  -- Tensor --> [left relay]  -> [left sink]
  //     \----> [right branch] -- Tensor --> [right relay] -> [right sink]
  Graph graph{"mixed-inline-tensor-fork"};
  auto source = graph.op<InlineForkSource>("source");
  auto left_branch = graph.op<InlineTensorBranch<LeftKind>>("left-branch");
  auto left_relay = graph.op<InlineTensorRelay<LeftKind>>("left-relay", 4U);
  auto left_sink = graph.op<InlinePodSink>("left-sink", 6U);
  auto right_branch = graph.op<InlineTensorBranch<RightKind>>("right-branch");
  auto right_relay = graph.op<InlineTensorRelay<RightKind>>("right-relay", 5U);
  auto right_sink = graph.op<InlinePodSink>("right-sink", 7U);
  graph.add_flow(source->first, left_branch->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(left_branch->output, left_relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(left_relay->output, left_sink->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->second, right_branch->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(right_branch->output, right_relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(right_relay->output, right_sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));

  CompileOptions options;
  options.profile = Profile::kStandard;
  if constexpr (LeftKind == MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "left-branch",
        .output_port = "output",
        .device = left_device,
    });
  }
  if constexpr (RightKind == MemoryKind::kCudaDevice) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "right-branch",
        .output_port = "output",
        .device = right_device,
    });
  }
  return compile(graph, std::move(options));
}

class InlineTensorContinuationTest : public testing::TestWithParam<QueuePolicy> {};

// REQ: CHAN-128
// REQ: RUNTIME-041
TEST_P(InlineTensorContinuationTest, RawTensorPassThroughContinuesCausalFrameBeforeNextClockTick) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();

  // Key setup: place a 1 ns source, pass-through relay, and sink in one direct raw-Tensor partition
  // with capacity-one edges so later root frames continuously compete with channel readiness.
  // [source] -- Tensor --> [relay] -- Tensor --> [sink]
  Graph graph{"inline-causal-tensor-chain"};
  auto source = graph.op<CausalTensorSource>("source", kFrameCount);
  auto relay = graph.op<CausalTensorPassThrough>("relay");
  auto sink = graph.op<CausalTensorSink>("sink");
  graph.add_flow(source->output,
                 relay->input,
                 ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.add_flow(
      relay->output, sink->input, ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.partition("inline-causal-tensor-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-causal-tensor-clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: two receipts per frame preserve source->relay->sink order at successive
  // microsteps before the next source frame, without a causal-frontier fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 3U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 3U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 2U);
    for (std::size_t stage = 0U; stage < 3U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

// REQ: RUNTIME-041
TEST_P(InlineTensorContinuationTest,
       TensorPacketPassThroughContinuesCausalFrameBeforeNextClockTick) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();

  // Key setup: forward owning one-Tensor schema samples through a direct three-Operator partition
  // while a 1 ns source keeps later root frames ready.
  // [source] -- TensorPacket --> [relay] -- TensorPacket --> [sink]
  Graph graph{"inline-causal-tensor-packet-chain"};
  auto source = graph.op<CausalTensorPacketSource>("source", kFrameCount);
  auto relay = graph.op<CausalTensorPacketPassThrough>("relay");
  auto sink = graph.op<CausalTensorPacketSink>("sink");
  graph.add_flow(source->output,
                 relay->input,
                 ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.add_flow(
      relay->output, sink->input, ConnectionOptions{.queue_policy = GetParam(), .queue_depth = 1U});
  graph.partition("inline-causal-tensor-packet-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-causal-tensor-packet-clock"));

  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: schema publication produces one receipt per edge, so each TensorPacket frame
  // drains at successive microsteps before the next root frame starts.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 3U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 3U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 2U);
    for (std::size_t stage = 0U; stage < 3U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            2U * kFrameCount);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

INSTANTIATE_TEST_SUITE_P(QueuePolicies, InlineTensorContinuationTest,
                         testing::Values(QueuePolicy::kBlocking, QueuePolicy::kDropOldest),
                         [](const testing::TestParamInfo<QueuePolicy>& info) {
                           return info.param == QueuePolicy::kBlocking ? "Blocking" : "DropOldest";
                         });

void expect_tensor_chain_observation(const InlineTensorObservation& observation,
                                     MemoryKind memory_kind,
                                     std::optional<std::int32_t> device = std::nullopt) {
  ASSERT_TRUE(observation.source_recorded);
  ASSERT_TRUE(observation.relay_recorded);
  EXPECT_EQ(observation.memory_kind, memory_kind);
  EXPECT_NE(observation.source_pointer, nullptr);
  EXPECT_EQ(observation.relay_pointer, observation.source_pointer);
  EXPECT_EQ(observation.value, 0x5AU);
  EXPECT_EQ(observation.relay_stream, observation.source_stream);
  EXPECT_EQ(observation.readiness_stream, observation.source_stream);
  if (device) {
    EXPECT_NE(observation.source_stream, nullptr);
    EXPECT_EQ(observation.source_device, *device);
    EXPECT_EQ(observation.relay_device, *device);
  }
}

template <typename PayloadCase>
class InlinePayloadParity : public testing::Test {};

using InlinePayloadCases = testing::Types<RawTensorCase, DescriptorImageCase>;

struct InlinePayloadCaseNames {
  template <typename PayloadCase>
  static std::string GetName(int) {
    return std::string{PayloadCase::name};
  }
};

TYPED_TEST_SUITE(InlinePayloadParity, InlinePayloadCases, InlinePayloadCaseNames);

TYPED_TEST(InlinePayloadParity, HostPreservesBackingMetadataAndControllerOwnership) {
  // Key setup: run the host raw-Tensor or descriptor-Image chain under both queue policies.
  for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(policy));
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kHost);
    const ExecutionPlan compiled =
        TypeParam::template make_plan<MemoryKind::kHost>({.policy = policy});
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
    // Key assertion: payload identity and metadata survive while all callbacks stay inline.
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
    TypeParam::expect_payload_evidence(evidence);
    expect_tensor_chain_observation(InlineTensorProbeState::observation(), MemoryKind::kHost);
    expect_inline_thread_evidence(evidence, 3U, 3U);
  }
}

TYPED_TEST(InlinePayloadParity, CudaReentersPlacedDeviceAndStreamForEveryCallback) {
  // Key setup: run the CUDA payload chain on each tested device after changing caller affinity.
  std::int32_t device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};

  const std::int32_t tested_device_count = std::min(device_count, 2);
  for (std::int32_t device = 0; device < tested_device_count; ++device) {
    for (const QueuePolicy policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
      SCOPED_TRACE(device);
      SCOPED_TRACE(static_cast<std::uint32_t>(policy));
      const std::int32_t ambient_device = device_count > 1 ? (device + 1) % device_count : device;
      ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
      InlinePartitionProbeState::reset();
      InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
      const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kCudaDevice>({
          .policy = policy,
          .device = DeviceId{device},
          .ambient_device_after_emit = ambient_device,
      });
      ASSERT_TRUE(compiled.ok()) << compiled.json();
      const auto state = detail::ExecutionPlanAccess::retain(compiled);
      ASSERT_TRUE(state);
      ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
      EXPECT_EQ(compiled.cuda_compute_stream_policy(),
                CudaComputeStreamPolicy::kPerExecutionPartition);
      ASSERT_EQ(compiled.cuda_compute_streams().size(), 1U);
      ASSERT_TRUE(compiled.cuda_compute_streams().front().partition.has_value());
      ASSERT_FALSE(compiled.operator_cuda_stream_bindings().empty());
      EXPECT_TRUE(
          std::ranges::all_of(compiled.operator_cuda_stream_bindings(), [&](const auto& binding) {
            return binding.stream_id == compiled.cuda_compute_streams().front().stream_id &&
                   binding.partition == *compiled.cuda_compute_streams().front().partition;
          }));

      const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
      // Key assertion: each callback re-enters the placed device and shares the source stream.
      ASSERT_TRUE(evidence.observed);
      EXPECT_EQ(evidence.termination, RunTermination::kStopped);
      EXPECT_EQ(evidence.diagnostic_count, 0U);
      EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
      TypeParam::expect_payload_evidence(evidence);
      expect_tensor_chain_observation(
          InlineTensorProbeState::observation(), MemoryKind::kCudaDevice, device);
      expect_inline_thread_evidence(evidence, 3U, 3U);
      std::int32_t caller_device = -1;
      ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
      EXPECT_EQ(caller_device, ambient_device);
    }
  }
}

TYPED_TEST(InlinePayloadParity, RetainedHandleKeepsPoolSlotUntilRelease) {
  // Key setup: retain the host Tensor at the relay while observing its runtime output pool.
  InlinePartitionProbeState::reset();
  InlineTensorProbeState::reset(MemoryKind::kHost);
  const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kHost>({
      .retain_tensor = true,
  });
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
  const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  ExecutionPlan run_plan =
      detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
  run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
  run_plan = detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

  std::optional<RunSession> session{run_async(run_plan)};
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_sinks(1U));
  // Key assertion: the slot remains claimed until handle release, then retires without leaks.
  ASSERT_TRUE(InlineTensorProbeState::has_retained_tensor());
  const auto retained = tensor_control->snapshot(0U);
  EXPECT_EQ(retained.active_slot_count, 1U);
  EXPECT_NE(retained.active_claim_mask, 0U);

  InlineTensorProbeState::release_retained_tensor();
  session->request_stop();
  session->wait();
  const InlinePartitionRunEvidence evidence =
      capture_run_evidence(*session, true, counters, support_control);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0x5AU}));
  TypeParam::expect_payload_evidence(evidence);
  expect_inline_thread_evidence(evidence, 3U, 3U);

  session.reset();
  const auto released = tensor_control->snapshot(0U);
  EXPECT_TRUE(released.terminal);
  EXPECT_EQ(released.active_slot_count, 0U);
  EXPECT_EQ(released.active_claim_mask, 0U);
  EXPECT_EQ(released.invariant_failure_count, 0U);
}

enum class RuntimeTensorMismatchMode {
  kPropagate,
  kHandle,
  kFirstRejectedPort,
  kUnrelatedErrorNextActivation,
  kUnrelatedErrorSameActivation,
};

class RuntimeTensorMismatchSource final : public Operator<> {
 public:
  explicit RuntimeTensorMismatchSource(RuntimeTensorMismatchMode mode) : mode_(mode) {}
  void setup(OperatorSpec& spec) override {
    const auto declare_output = [&](Output<Tensor>& port, const char* name) {
      spec.output(port, name)
          .max_emits_per_compute(1U)
          .produces_tensor(TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .dtype = DLDataType{kDLUInt, 8U, 1U},
                                 .rank = 1U},
              .bounds = tensor_bounds(8U),
              .storage = TensorOutputStorage::kRuntimePool,
          });
    };
    declare_output(output, "output");
    if (mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort) {
      declare_output(first_rejected, "z_first_rejected");
    }
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      if (mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "unrelated failure in next activation"});
      }
      context.request_completion();
      return {};
    }
    emitted_ = true;
    constexpr std::array<std::int64_t, 1U> shape{8};
    auto storage = std::make_unique<std::array<std::uint8_t, 8U>>();
    auto* const storage_owner = storage.get();
    auto tensor = make_tensor(shape,
                              DLDataType{kDLUInt, 8U, 1U},
                              std::nullopt,
                              MemoryStorageType::kSystem,
                              storage->data(),
                              [storage_owner](void*) -> expected<void, Error> {
                                delete storage_owner;
                                deleted.fetch_add(1U, std::memory_order_relaxed);
                                return {};
                              });
    if (!tensor) {
      return make_unexpected(std::move(tensor.error()));
    }
    EXPECT_EQ(storage.release(), storage_owner);
    Sample<Tensor> sample{std::move(*tensor), {}};
    expected<void, Error> first;
    if (mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort) {
      first = first_rejected.emit(sample);
      if (!first) {
        rejected.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    auto emitted = output.emit(std::move(sample));
    if (!emitted) {
      rejected.fetch_add(1U, std::memory_order_relaxed);
    }
    if (mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "unrelated failure in same activation"});
    }
    if (mode_ == RuntimeTensorMismatchMode::kHandle ||
        mode_ == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
      return {};
    }
    return mode_ == RuntimeTensorMismatchMode::kFirstRejectedPort ? std::move(first)
                                                                  : std::move(emitted);
  }
  Output<Tensor> output;
  Output<Tensor> first_rejected;
  static inline std::atomic_uint32_t deleted{};
  static inline std::atomic_uint32_t rejected{};

 private:
  RuntimeTensorMismatchMode mode_;
  bool emitted_{};
};

class RuntimeTensorMismatchSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U},
        });
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    received.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }
  Input<Tensor> input;
  static inline std::atomic_uint32_t received{};
};

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeInlinePartitionExecution, RuntimeTensorMismatchRetainsPortMemoryAndCleanup) {
  std::int32_t device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  for (const auto mode : {RuntimeTensorMismatchMode::kPropagate,
                          RuntimeTensorMismatchMode::kHandle,
                          RuntimeTensorMismatchMode::kFirstRejectedPort,
                          RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation,
                          RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation}) {
    SCOPED_TRACE(static_cast<int>(mode));
    const bool two_ports = mode == RuntimeTensorMismatchMode::kFirstRejectedPort;
    RuntimeTensorMismatchSource::deleted.store(0U, std::memory_order_relaxed);
    RuntimeTensorMismatchSource::rejected.store(0U, std::memory_order_relaxed);
    RuntimeTensorMismatchSink::received.store(0U, std::memory_order_relaxed);
    Graph graph{"runtime-tensor-mismatch"};
    const auto source = graph.op<RuntimeTensorMismatchSource>("source", mode);
    const auto sink = graph.op<RuntimeTensorMismatchSink>("sink");
    graph.add_flow(source->output, sink->input);
    if (two_ports) {
      const auto second_sink = graph.op<RuntimeTensorMismatchSink>("second_sink");
      graph.add_flow(source->first_rejected, second_sink->input);
    }
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    CompileOptions options;
    options.deployment.bind_tensor_output_device({"source", "output", DeviceId{0}});
    if (two_ports) {
      options.deployment.bind_tensor_output_device({"source", "z_first_rejected", DeviceId{0}});
    }
    const auto compiled = compile(graph, options);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    const auto plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
    std::optional<RunSession> session{run_async(plan)};
    const bool terminal = wait_for_terminal(*session);
    if (!terminal) {
      session->request_stop();
    }
    session->wait();
    ASSERT_TRUE(terminal) << "Run did not terminate within 5 seconds";
    EXPECT_EQ(RuntimeTensorMismatchSource::rejected.load(std::memory_order_relaxed),
              two_ports ? 2U : 1U);
    EXPECT_EQ(RuntimeTensorMismatchSource::deleted.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(RuntimeTensorMismatchSink::received.load(std::memory_order_relaxed), 0U);
    if (mode == RuntimeTensorMismatchMode::kHandle) {
      EXPECT_EQ(session->termination(), RunTermination::kRequestedCompleted);
      EXPECT_TRUE(session->diagnostics().empty());
    } else {
      EXPECT_EQ(session->termination(), RunTermination::kFailed);
      const auto diagnostics = session->diagnostics();
      const auto found = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
        return row.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
      });
      ASSERT_NE(found, diagnostics.end());
      const auto& failure_detail = found->detail;
      if (!failure_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      EXPECT_EQ(failure_detail->operation, DiagnosticOperation::kCompute);
      EXPECT_EQ(failure_detail->operator_path, "source");
      EXPECT_TRUE(failure_detail->causes.empty());
      if (mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation ||
          mode == RuntimeTensorMismatchMode::kUnrelatedErrorSameActivation) {
        EXPECT_EQ(failure_detail->error.code, ErrorCode::kNotSupported);
        EXPECT_EQ(failure_detail->error.message,
                  mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation
                      ? "unrelated failure in next activation"
                      : "unrelated failure in same activation");
      } else {
        EXPECT_EQ(failure_detail->error.code, ErrorCode::kInvalidArgument);
      }
      if (mode == RuntimeTensorMismatchMode::kUnrelatedErrorNextActivation) {
        EXPECT_FALSE(found->source.port);
        EXPECT_TRUE(failure_detail->port_name.empty());
        EXPECT_TRUE(failure_detail->mismatches.empty());
      } else {
        const std::string expected_port = two_ports ? "z_first_rejected" : "output";
        EXPECT_EQ(failure_detail->port_name, expected_port);
        EXPECT_TRUE(found->source.port);
        const auto& mismatches = failure_detail->mismatches;
        ASSERT_EQ(mismatches.size(), 1U);
        EXPECT_EQ(mismatches.front().field, "memory_kind");
        EXPECT_EQ(mismatches.front().provided, "host");
        EXPECT_EQ(mismatches.front().required, "cuda_device");
        const auto formatted = format_diagnostics(diagnostics, ExecutionGraphView{});
        EXPECT_NE(formatted.find(expected_port), std::string::npos);
        EXPECT_NE(formatted.find("host"), std::string::npos);
        EXPECT_NE(formatted.find("cuda_device"), std::string::npos);
      }
    }
    session.reset();
    for (std::size_t pool_index = 0U; pool_index < (two_ports ? 2U : 1U); ++pool_index) {
      const auto pool = control->snapshot(pool_index);
      EXPECT_TRUE(pool.terminal);
      EXPECT_EQ(pool.active_slot_count, 0U);
      EXPECT_EQ(pool.active_claim_mask, 0U);
      EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
      EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
      EXPECT_EQ(pool.invariant_failure_count, 0U);
    }
  }
}

TEST(RuntimeInlinePartitionExecution,
     RawCudaTensorFailureCleansEveryDeviceWithoutLeakingCallerDevice) {
  // Key setup: fail a placed CUDA Tensor relay after deliberately changing the ambient device.
  std::int32_t device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};

  const std::int32_t tested_device_count = std::min(device_count, 2);
  for (std::int32_t device = 0; device < tested_device_count; ++device) {
    SCOPED_TRACE(device);
    const std::int32_t ambient_device = device_count > 1 ? (device + 1) % device_count : device;
    ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
    const ExecutionPlan compiled =
        make_tensor_chain_plan<MemoryKind::kCudaDevice>(QueuePolicy::kBlocking,
                                                        TensorPlanRoute::kInline,
                                                        DeviceId{device},
                                                        ambient_device,
                                                        true,
                                                        false,
                                                        ambient_device);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
    const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    ExecutionPlan run_plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
    run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
    run_plan =
        detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

    std::optional<RunSession> session{run_async(run_plan)};
    session->wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(*session, true, counters, support_control);
    // Key assertion: failure releases all CUDA resources and restores the caller's device.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session->diagnostics().size(), 1U);
    EXPECT_EQ(session->diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    expect_tensor_chain_observation(
        InlineTensorProbeState::observation(), MemoryKind::kCudaDevice, device);
    EXPECT_TRUE(InlineTensorProbeState::observation().stop_device_recorded);
    EXPECT_GE(InlineTensorProbeState::observation().stop_device, 0);
    EXPECT_LT(InlineTensorProbeState::observation().stop_device, device_count);
    expect_inline_thread_evidence(evidence, 2U, 3U);

    session.reset();
    const auto pool = tensor_control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
    EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
    std::int32_t caller_device = -1;
    ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
    EXPECT_EQ(caller_device, ambient_device);
  }
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeInlinePartitionExecution, CudaRestoreFailureIsSeparateFromCallbackCause) {
  std::int32_t device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  std::int32_t original_device = -1;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const ScopedCUDADeviceRestore restore_device{original_device};
  const std::int32_t ambient_device = device_count > 1 ? 1 : 0;
  for (const bool fail_callback : {true, false}) {
    SCOPED_TRACE(fail_callback ? "callback and restore" : "restore only");
    ASSERT_EQ(cudaSetDevice(ambient_device), cudaSuccess);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kCudaDevice);
    const auto compiled = make_tensor_chain_plan<MemoryKind::kCudaDevice>(QueuePolicy::kBlocking,
                                                                          TensorPlanRoute::kInline,
                                                                          DeviceId{0},
                                                                          ambient_device,
                                                                          fail_callback,
                                                                          false,
                                                                          ambient_device);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto fault = std::make_shared<detail::RuntimeCudaExecutionTestControl>(
        detail::RuntimeCudaExecutionFailurePoint::kComputeDeviceRestore, 1U);
    const auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    auto plan = detail::RuntimeTestPlanAccess::with_cuda_execution_test_control(compiled, fault);
    plan = detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control);
    std::optional<RunSession> session{run_async(plan)};
    const bool terminal = wait_for_terminal(*session);
    if (!terminal) {
      session->request_stop();
    }
    session->wait();
    ASSERT_TRUE(terminal) << "Run did not terminate within 5 seconds";
    EXPECT_GE(fault->visits(), 2U);
    EXPECT_EQ(session->termination(), RunTermination::kFailed);
    const auto diagnostics = session->diagnostics();
    ASSERT_EQ(diagnostics.size(), fail_callback ? 2U : 1U);
    const auto& restoration = diagnostics.back();
    EXPECT_EQ(restoration.token, "RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED");
    const auto& restoration_detail = restoration.detail;
    if (!restoration_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_EQ(restoration_detail->operation, DiagnosticOperation::kRuntime);
    EXPECT_EQ(restoration_detail->operator_path, "relay");
    EXPECT_EQ(restoration_detail->error.code, ErrorCode::kNotSupported);
    EXPECT_TRUE(restoration_detail->causes.empty());
    if (fail_callback) {
      const auto& callback = diagnostics.front();
      EXPECT_EQ(callback.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
      const auto& callback_detail = callback.detail;
      if (!callback_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      EXPECT_EQ(callback_detail->operation, DiagnosticOperation::kCompute);
      EXPECT_EQ(callback_detail->error.message, "INLINE_TENSOR_CALLBACK_FAILURE");
      EXPECT_TRUE(callback_detail->causes.empty());
      EXPECT_EQ(restoration_detail->role, DiagnosticFailureRole::kTeardown);
      EXPECT_EQ(restoration_detail->primary_diagnostic_index, 0U);
    } else {
      EXPECT_EQ(restoration_detail->role, DiagnosticFailureRole::kPrimary);
      EXPECT_FALSE(restoration_detail->primary_diagnostic_index);
    }
    session.reset();
    const auto pool = control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.cuda_streams_created, pool.cuda_streams_destroyed);
    EXPECT_EQ(pool.cuda_events_created, pool.cuda_events_destroyed);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
    std::int32_t caller_device = -1;
    ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
    EXPECT_EQ(caller_device, ambient_device);
  }
}

TEST(RuntimeInlinePartitionExecution, MixedTensorPlacementKeepsTheOneRootForkOrdinary) {
  // Key setup: compile one-root forks whose branches require mixed memory or CUDA devices.
  const ExecutionPlan host_cuda =
      make_mixed_tensor_fork_plan<MemoryKind::kHost, MemoryKind::kCudaDevice>(DeviceId{},
                                                                              DeviceId{0});
  ASSERT_TRUE(host_cuda.ok()) << host_cuda.json();
  const auto host_cuda_state = detail::ExecutionPlanAccess::retain(host_cuda);
  ASSERT_TRUE(host_cuda_state);
  // Key assertion: neither placement-incompatible fork is admitted as inline execution.
  EXPECT_FALSE(host_cuda_state->inline_partition_execution.has_value());

  const ExecutionPlan multi_device =
      make_mixed_tensor_fork_plan<MemoryKind::kCudaDevice, MemoryKind::kCudaDevice>(DeviceId{0},
                                                                                    DeviceId{1});
  ASSERT_TRUE(multi_device.ok()) << multi_device.json();
  const auto multi_device_state = detail::ExecutionPlanAccess::retain(multi_device);
  ASSERT_TRUE(multi_device_state);
  EXPECT_FALSE(multi_device_state->inline_partition_execution.has_value());
}

TYPED_TEST(InlinePayloadParity, IneligibleTransportAndMemoryRemainOrdinary) {
  // Key setup: compile equivalent payload chains over HoloMQ, fanout, and ineligible memory kinds.
  const ExecutionPlan holomq = TypeParam::template make_plan<MemoryKind::kHost>({
      .route = TensorPlanRoute::kHoloMQ,
  });
  ASSERT_TRUE(holomq.ok()) << holomq.json();
  const auto holomq_state = detail::ExecutionPlanAccess::retain(holomq);
  ASSERT_TRUE(holomq_state);
  // Key assertion: every ineligible route remains outside the inline execution fast path.
  EXPECT_FALSE(holomq_state->inline_partition_execution.has_value());

  const ExecutionPlan fanout = TypeParam::template make_plan<MemoryKind::kHost>({
      .route = TensorPlanRoute::kFanout,
  });
  ASSERT_TRUE(fanout.ok()) << fanout.json();
  const auto fanout_state = detail::ExecutionPlanAccess::retain(fanout);
  ASSERT_TRUE(fanout_state);
  EXPECT_FALSE(fanout_state->inline_partition_execution.has_value());

  for (const MemoryKind memory_kind : {MemoryKind::kPinnedHost, MemoryKind::kCudaManaged}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(memory_kind));
    const ExecutionPlan plan = memory_kind == MemoryKind::kPinnedHost
                                   ? TypeParam::template make_plan<MemoryKind::kPinnedHost>({})
                                   : TypeParam::template make_plan<MemoryKind::kCudaManaged>({
                                         .device = DeviceId{0},
                                     });
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_TRUE(state);
    EXPECT_FALSE(state->inline_partition_execution.has_value());
  }
}

TYPED_TEST(InlinePayloadParity, FailureReleasesPoolAndReplaysCleanly) {
  // Key setup: reuse a host payload plan whose relay fails, running it twice against fresh pools.
  const ExecutionPlan compiled = TypeParam::template make_plan<MemoryKind::kHost>({
      .fail_relay = true,
  });
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    InlineTensorProbeState::reset(MemoryKind::kHost);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto support_control = std::make_shared<InlinePartitionControllerProbe>();
    const auto tensor_control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
    ExecutionPlan run_plan =
        detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, tensor_control);
    run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(run_plan, counters);
    run_plan =
        detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, support_control);

    std::optional<RunSession> session{run_async(run_plan)};
    session->wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(*session, true, counters, support_control);
    // Key assertion: each run replays the typed failure and retires every active pool claim.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session->diagnostics().size(), 1U);
    EXPECT_EQ(session->diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    EXPECT_TRUE(evidence.sink_values.empty());
    TypeParam::expect_payload_evidence(evidence);
    expect_tensor_chain_observation(InlineTensorProbeState::observation(), MemoryKind::kHost);
    expect_inline_thread_evidence(evidence, 2U, 3U);

    session.reset();
    const auto pool = tensor_control->snapshot(0U);
    EXPECT_TRUE(pool.terminal);
    EXPECT_EQ(pool.active_slot_count, 0U);
    EXPECT_EQ(pool.active_claim_mask, 0U);
    EXPECT_EQ(pool.invariant_failure_count, 0U);
  }
}

TEST(RuntimeInlinePartitionExecution, DescriptorImageMissingAllocationRemainsOrdinary) {
  // Key setup: compile a descriptor-Image chain without runtime Tensor allocation evidence.
  const ExecutionPlan no_allocation = make_image_chain_plan<MemoryKind::kHost>(
      QueuePolicy::kBlocking, TensorPlanRoute::kInline, DeviceId{}, -1, false, false, false);
  ASSERT_TRUE(no_allocation.ok()) << no_allocation.json();
  const auto no_allocation_state = detail::ExecutionPlanAccess::retain(no_allocation);
  ASSERT_TRUE(no_allocation_state);
  // Key assertion: missing allocation proof prevents admission to inline execution.
  EXPECT_FALSE(no_allocation_state->inline_partition_execution.has_value());
}

}  // namespace
}  // namespace holoscan
