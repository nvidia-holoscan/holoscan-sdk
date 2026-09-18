/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/tensor_storage.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/execution_plan_state.hpp"
#include "image_generated.h"
#include "tensor_packet_generated.h"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] bool cuda_device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename Native>
struct DescriptorOutputScenario;

// REQ: OPERATOR-031
// REQ: OPERATOR-035
// REQ: CHAN-107
// REQ: CHAN-122
// REQ: TENSOR-SCHEMA-005
// REQ: TENSOR-SCHEMA-006
template <>
struct DescriptorOutputScenario<schema::ImageT> {
  static constexpr std::string_view name{"Image"};
  static constexpr std::array<std::uint8_t, 8U> bytes{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};

  [[nodiscard]] static schema::ImageT descriptor() {
    schema::ImageT value{};
    value.height = 1;
    value.width = 8;
    value.encoding = schema::ImageEncoding_CUSTOM;
    return value;
  }

  [[nodiscard]] static EmitOptions emit_options() {
    return EmitOptions{
        .capture_time = TimePoint{.timestamp_ns = 1234, .clock_id = ClockId{1U}},
        .frame_id = 9U,
        .trace = TraceContext::root(0xCAFEU),
    };
  }

  [[nodiscard]] static bool matches(const Sample<schema::ImageT>& sample) {
    return sample.data.height == 1 && sample.data.width == 8 &&
           sample.data.encoding == schema::ImageEncoding_CUSTOM &&
           sample.metadata.capture_timestamp_ns == 1234 && sample.metadata.trace_id == 0xCAFEU &&
           sample.metadata.frame_id == 9U && sample.metadata.sequence == 1U;
  }
};

// REQ: CHAN-120
// REQ: OPERATOR-035
template <>
struct DescriptorOutputScenario<schema::TensorPacketT> {
  static constexpr std::string_view name{"TensorPacket"};
  static constexpr std::array<std::uint8_t, 8U> bytes{7U, 6U, 5U, 4U, 3U, 2U, 1U, 0U};

  [[nodiscard]] static schema::TensorPacketT descriptor() {
    schema::TensorPacketT value{};
    value.sequence = 42U;
    value.source = "third-party-camera";
    value.scale = 0.5;
    return value;
  }

  [[nodiscard]] static EmitOptions emit_options() {
    return EmitOptions{
        .capture_time = TimePoint{.timestamp_ns = 4321, .clock_id = ClockId{1U}},
        .frame_id = 11U,
        .trace = TraceContext::root(0xBEEFU),
    };
  }

  [[nodiscard]] static bool matches(const Sample<schema::TensorPacketT>& sample) {
    return sample.data.sequence == 42U && sample.data.source == "third-party-camera" &&
           sample.data.scale == 0.5 && sample.metadata.capture_timestamp_ns == 4321 &&
           sample.metadata.trace_id == 0xBEEFU && sample.metadata.frame_id == 11U &&
           sample.metadata.sequence == 1U;
  }
};

// Captures provider pointer identity and validates descriptor, Tensor, and Sample metadata.
template <typename Native>
struct RuntimeDescriptorOutputProbeState {
  static void reset(const std::uint32_t expected_observations = 1U) {
    const std::lock_guard lock(mutex);
    source_pointer = nullptr;
    observed = false;
    valid = true;
    observations = 0U;
    expected = expected_observations;
    error.reset();
  }

  static void record_source(const void* pointer) {
    const std::lock_guard lock(mutex);
    source_pointer = pointer;
  }

  static void record_sink(const Sample<Native>& sample) {
    const std::lock_guard lock(mutex);
    const Tensor* const tensor = sample.data.data.get();
    const std::uint8_t* const bytes = tensor == nullptr ? nullptr : tensor->data_as<std::uint8_t>();
    valid = valid && tensor != nullptr && tensor->data() == source_pointer &&
            tensor->nbytes() == 8 && tensor->ndim() == 1 &&
            std::ranges::equal(tensor->shape_span(), kShape) && tensor->dtype().code == kDLUInt &&
            tensor->dtype().bits == 8U && tensor->dtype().lanes == 1U && bytes != nullptr &&
            std::ranges::equal(std::span<const std::uint8_t>{bytes, 8U},
                               DescriptorOutputScenario<Native>::bytes) &&
            DescriptorOutputScenario<Native>::matches(sample);
    ++observations;
    observed = observations >= expected;
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      observed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed; });
  }

  static constexpr std::array<std::int64_t, 1U> kShape{8};
  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline const void* source_pointer{};
  static inline bool observed{};
  static inline bool valid{};
  static inline std::uint32_t observations{};
  static inline std::uint32_t expected{1U};
  static inline std::optional<ErrorCode> error;
};

// Allocates a descriptor Tensor from the output's finite provider-backed pool.
// Used by DescriptorOutputScenarioTest.PreservesProviderBackingAndMetadata.
template <typename Native, MemoryKind Storage = MemoryKind::kHost, bool DerivedCapacity = false>
class RuntimeDescriptorOutputSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    // The non-derived specialization adds literal bounds below.
    // NOLINTNEXTLINE(misc-const-correctness)
    TensorOutputSpec tensor_spec{
        .representation =
            TensorRepresentation{
                .memory_kind = Storage,
                .dtype = DLDataType{kDLUInt, 8U, 1U},
                .rank = 1U,
            },
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
            },
    };
    auto& builder = spec.output(output, "output").max_emits_per_compute(1U);
    if constexpr (DerivedCapacity) {
      constexpr TensorFieldSelector<Native> kData{
          DescriptorTensorSchemaNativeTraits<Native>::tensor_field_ids[0U], 0U};
      builder.tensor_field(kData).produces_tensor(tensor_spec, tensor_byte_span(8U));
    } else {
      tensor_spec.bounds = tensor_bounds(8U);
      builder.produces_tensor(tensor_spec);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;

    auto loan = output.allocate_tensor(RuntimeDescriptorOutputProbeState<Native>::kShape);
    if (!loan) {
      return fail(loan.error().code);
    }
    const auto written = loan->write_host_and_commit([](const TensorOutputWriteGuard& writer) {
      RuntimeDescriptorOutputProbeState<Native>::record_source(writer.data());
      std::ranges::copy(DescriptorOutputScenario<Native>::bytes, writer.data_as<std::uint8_t>());
    });
    if (!written) {
      return fail(written.error().code);
    }
    const auto emitted = output.emit_tensor(std::move(*loan),
                                            DescriptorOutputScenario<Native>::descriptor(),
                                            DescriptorOutputScenario<Native>::emit_options());
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  Output<Native> output;

 private:
  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDescriptorOutputProbeState<Native>::fail(code);
    return make_unexpected(Error{code});
  }

  bool emitted_{};
};

// Verifies that descriptor metadata and the provider Tensor arrive without a data copy.
// Used by DescriptorOutputScenarioTest.PreservesProviderBackingAndMetadata.
template <typename Native, MemoryKind Storage = MemoryKind::kHost>
class RuntimeDescriptorOutputSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = Storage,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      RuntimeDescriptorOutputProbeState<Native>::fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    RuntimeDescriptorOutputProbeState<Native>::record_sink(*sample);
    return {};
  }

  Input<Native> input;
};

struct RuntimeDescriptorStreamBindingProbe {
  static void reset(std::uint32_t expected_mask) {
    const std::lock_guard lock(mutex_);
    expected_mask_ = expected_mask;
    observed_mask_ = 0U;
  }

  static void record(std::uint32_t index) {
    {
      const std::lock_guard lock(mutex_);
      observed_mask_ |= 1U << index;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, 5s, [] { return (observed_mask_ & expected_mask_) == expected_mask_; });
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::uint32_t expected_mask_{};
  static inline std::uint32_t observed_mask_{};
};

constexpr std::uint32_t kDescriptorRelayProbeIndex = 2U;

// Publishes one host or pooled-device descriptor Tensor and records its completion.
// Used by AllocationFreeInputDoesNotConflictWithCudaOutput.
template <std::uint32_t Index, bool Pooled>
class RuntimeDescriptorStreamBindingSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& declaration = spec.output(output, "output").max_emits_per_compute(1U);
    if constexpr (Pooled) {
      declaration.produces_tensor(TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = MemoryKind::kCudaDevice,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                  .rank = 1U,
              },
          .bounds = tensor_bounds(8U),
          .storage = TensorOutputStorage::kRuntimePool,
          .transfer =
              TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
      });
    } else {
      declaration.produces_tensor(TensorOutputSpec{
          .representation =
              holoscan::TensorRepresentation{
                  .memory_kind = MemoryKind::kHost,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
                  .rank = 1U,
              },
          .bounds = tensor_bounds(8U),
          .transfer =
              TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
      });
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;

    schema::ImageT descriptor{};
    descriptor.height = 1;
    descriptor.width = 8;
    descriptor.encoding = schema::ImageEncoding_CUSTOM;
    if constexpr (Pooled) {
      constexpr std::array<std::int64_t, 1U> kShape{8};
      auto loan = output.allocate_tensor(TensorLoanRequest{
          .shape = kShape,
          .dtype = DLDataType{kDLUInt, 8U, 1U},
      });
      if (!loan) {
        return make_unexpected(std::move(loan).error());
      }
      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        return make_unexpected(std::move(writer).error());
      }
      if (cudaMemsetAsync(
              writer->data(), static_cast<int>(Index + 1U), 8U, context.cuda_stream().get()) !=
          cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      auto committed = std::move(*writer).commit();
      if (!committed) {
        return make_unexpected(std::move(committed).error());
      }
      auto emitted = output.emit_tensor(std::move(*loan), descriptor);
      if (!emitted) {
        return make_unexpected(std::move(emitted).error());
      }
    } else {
      auto* storage = new (std::nothrow) std::uint8_t[8U]{};
      if (storage == nullptr) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      constexpr std::array<std::int64_t, 1U> kShape{8};
      auto tensor = make_tensor(kShape,
                                DLDataType{kDLUInt, 8U, 1U},
                                std::nullopt,
                                MemoryStorageType::kSystem,
                                storage,
                                [](void* pointer) -> expected<void, Error> {
                                  delete[] static_cast<std::uint8_t*>(pointer);
                                  return {};
                                });
      if (!tensor) {
        delete[] storage;
        return make_unexpected(std::move(tensor).error());
      }
      descriptor.data = std::make_shared<Tensor>(std::move(*tensor));
      auto emitted = output.emit(std::move(descriptor));
      if (!emitted) {
        return make_unexpected(std::move(emitted).error());
      }
    }
    RuntimeDescriptorStreamBindingProbe::record(Index);
    return {};
  }

  Output<schema::ImageT> output;

 private:
  bool emitted_{};
};

// Joins descriptor inputs onto an independently pooled CUDA Tensor output.
// Used by NonLinearCudaJoinUsesServiceDeviceWithoutFusion.
class RuntimeDescriptorCudaOutputRelay final : public Operator<> {
 public:
  RuntimeDescriptorCudaOutputRelay(bool join, MemoryKind input_memory_kind)
      : join_(join), input_memory_kind_(input_memory_kind) {}

  void setup(OperatorSpec& spec) override {
    const TensorRepresentation input_layout{
        .memory_kind = input_memory_kind_,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    spec.input(first, "first")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = input_layout});
    if (join_) {
      spec.input(second, "second")
          .queue_depth(1U)
          .expects_tensor(TensorInputSpec{.representation = input_layout});
    }
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (join_) {
      result.trigger(OnAnyInput{});
    } else {
      result.trigger(OnEach{first});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute([[maybe_unused]] ExecutionContext& context) override {
    auto received = first.receive();
    if (!received && join_) {
      received = second.receive();
    }
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    RuntimeDescriptorStreamBindingProbe::record(kDescriptorRelayProbeIndex);
    return {};
  }

  Input<schema::ImageT> first;
  Input<schema::ImageT> second;
  Output<Tensor> output;

 private:
  bool join_{};
  MemoryKind input_memory_kind_{MemoryKind::kUnknown};
};

// Thread-safe probe for the fanout stream test. Tracks two application CUDA
// streams: blocked_stream is held via a host callback to force the source into
// pool exhaustion while quick_stream completes immediately. Detects premature
// slot reuse by comparing subsequent allocation pointers against the first.

// Graph topology:
//   descriptor source --[queue_depth=1, direct retained]--> descriptor sink
//                     `--[when Fanout=true]---------------> second descriptor sink
// The allocation declaration materializes one provider-owned pool for either generated schema.
template <typename Native, MemoryKind Storage = MemoryKind::kHost, bool Fanout = false,
          bool DerivedCapacity = false>
[[nodiscard]] ExecutionPlan make_runtime_descriptor_output_plan() {
  Graph graph{Storage == MemoryKind::kCudaManaged ? "runtime-descriptor-output-managed"
                                                  : "runtime-descriptor-output-host"};
  auto source =
      graph.op<RuntimeDescriptorOutputSourceOperator<Native, Storage, DerivedCapacity>>("source");
  auto sink = graph.op<RuntimeDescriptorOutputSinkOperator<Native, Storage>>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  if constexpr (Fanout) {
    auto second_sink =
        graph.op<RuntimeDescriptorOutputSinkOperator<Native, Storage>>("second-sink");
    graph.add_flow(source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  if constexpr (Storage == MemoryKind::kCudaManaged) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = DeviceId{0},
    });
  }
  return compile(graph, std::move(options));
}

template <typename Native>
class DescriptorOutputScenarioTest : public testing::Test {};

using DescriptorOutputNatives = testing::Types<schema::ImageT, schema::TensorPacketT>;

struct DescriptorOutputScenarioNames {
  template <typename Native>
  static std::string GetName(int) {
    return std::string{DescriptorOutputScenario<Native>::name};
  }
};

TYPED_TEST_SUITE(DescriptorOutputScenarioTest, DescriptorOutputNatives,
                 DescriptorOutputScenarioNames);

TYPED_TEST(DescriptorOutputScenarioTest, PreservesProviderBackingAndMetadata) {
  using Probe = RuntimeDescriptorOutputProbeState<TypeParam>;
  Probe::reset();
  const ExecutionPlan plan = make_runtime_descriptor_output_plan<TypeParam>();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_rank, 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 8U);
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().payload_identity.payload_kind,
            PayloadKind::kDescriptorBacked);

  RunSession session = run_async(plan);
  const bool observed = Probe::wait();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(Probe::mutex);
  EXPECT_FALSE(Probe::error.has_value());
  EXPECT_TRUE(Probe::valid);
}

// REQ: COMPILE-067
TEST(RunSessionDescriptorOutput, DerivedFieldCapacityPreservesBackingAndRetirementPath) {
  using Probe = RuntimeDescriptorOutputProbeState<schema::ImageT>;
  Probe::reset();
  // Key setup: compile a singular descriptor field whose pool bytes come from cold derivation.
  const ExecutionPlan plan =
      make_runtime_descriptor_output_plan<schema::ImageT, MemoryKind::kHost, false, true>();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_byte_span_derivations().size(), 1U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().bounds.max_byte_span, 8U);

  RunSession session = run_async(plan);
  const bool observed = Probe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: Runtime sees only the ordinary pool plan and preserves the exact provider
  // pointer through publication, delivery, and normal session retirement.
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(Probe::mutex);
  EXPECT_FALSE(Probe::error.has_value());
  EXPECT_TRUE(Probe::valid);
}

// A process-local pinned descriptor pool is host-synchronized and has no CUDA device obligation.
// Its HoloMQ fanout retains that device-free service contract while sharing the provider backing.
// REQ: CHAN-107
// REQ: TENSOR-SCHEMA-005
TEST(RunSessionDescriptorOutput, PinnedHostHoloMQFanoutNeedsNoDeviceBinding) {
  if (cudaFree(nullptr) != cudaSuccess) {
    GTEST_SKIP() << "CUDA pinned allocation is unavailable";
  }
  using Probe = RuntimeDescriptorOutputProbeState<schema::ImageT>;
  Probe::reset(2U);
  // Key setup: compile a singular pinned-host Image descriptor output with two subscribers so
  // process-local delivery uses HoloMQ without assigning a private CUDA device to the pool.
  const ExecutionPlan plan =
      make_runtime_descriptor_output_plan<schema::ImageT, MemoryKind::kPinnedHost, true>();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_FALSE(plan.tensor_allocations().front().device.valid());
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto service =
      std::ranges::find_if(state->process_local_channel_services, [](const auto& row) {
        return row.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      });
  ASSERT_NE(service, state->process_local_channel_services.end());
  EXPECT_EQ(service->delivery_kind, detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_EQ(service->tensor_memory_kind, MemoryKind::kPinnedHost);
  EXPECT_FALSE(service->tensor_device.valid());

  RunSession session = run_async(plan);
  const bool observed = Probe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: both HoloMQ subscribers receive the same pinned backing without a runtime
  // port-binding diagnostic or a fabricated CUDA device requirement.
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(Probe::mutex);
  EXPECT_FALSE(Probe::error.has_value());
  EXPECT_TRUE(Probe::valid);
}

// A process-local singular descriptor retains its managed-memory layout through direct delivery.
// REQ: COMPILE-049
// REQ: TENSOR-SCHEMA-005
TEST(RunSessionDescriptorOutput, ManagedTensorPreservesPassThroughLayout) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device 0 is unavailable";
  }
  using Probe = RuntimeDescriptorOutputProbeState<schema::ImageT>;
  Probe::reset();
  // Key setup: allocate the descriptor Tensor from a managed-memory output pool and deliver it
  // through the process-local direct-retained path.
  const ExecutionPlan plan =
      make_runtime_descriptor_output_plan<schema::ImageT, MemoryKind::kCudaManaged>();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  EXPECT_EQ(plan.tensor_allocations().front().memory_kind, MemoryKind::kCudaManaged);
  EXPECT_EQ(plan.tensor_allocations().front().device, DeviceId{0});
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto service =
      std::ranges::find_if(state->process_local_channel_services, [](const auto& row) {
        return row.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      });
  ASSERT_NE(service, state->process_local_channel_services.end());
  EXPECT_EQ(service->tensor_memory_kind, MemoryKind::kCudaManaged);
  EXPECT_EQ(service->tensor_device, DeviceId{0});

  RunSession session = run_async(plan);
  const bool observed = Probe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: the managed Tensor reaches the sink without a layout-contract rejection or
  // replacement of its provider-owned backing.
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::lock_guard lock(Probe::mutex);
  EXPECT_FALSE(Probe::error.has_value());
  EXPECT_TRUE(Probe::valid);
}

// An allocation-free descriptor input does not contribute a device obligation. An unrelated
// CUDA output on the same Operator remains sufficient to select the worker stream.
// Graph topology:
//   descriptor source --[host Image]--> relay --[CUDA Tensor]--> (unconnected output)
// REQ: COMPILE-049
// REQ: TENSOR-SCHEMA-005
TEST(RunSessionDescriptorOutput, AllocationFreeInputDoesNotConflictWithCudaOutput) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device 0 is unavailable";
  }
  RuntimeDescriptorStreamBindingProbe::reset((1U << 0U) | (1U << kDescriptorRelayProbeIndex));
  Graph graph{"runtime-descriptor-allocation-free-input"};
  auto source = graph.op<RuntimeDescriptorStreamBindingSource<0U, false>>("source");
  auto relay = graph.op<RuntimeDescriptorCudaOutputRelay>("relay", false, MemoryKind::kHost);
  // Key setup: the relay has an unrelated CUDA allocation but receives an allocation-free
  // descriptor, so only the declared output contributes a device obligation.
  graph.add_flow(source->output, relay->first, ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  CompileOptions options;
  options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "relay",
      .output_port = "output",
      .device = DeviceId{0},
  });

  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  const auto service =
      std::ranges::find_if(state->process_local_channel_services, [](const auto& row) {
        return row.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      });
  ASSERT_NE(service, state->process_local_channel_services.end());
  EXPECT_EQ(service->delivery_kind, detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
  EXPECT_TRUE(service->tensor_pool_id.empty());
  EXPECT_FALSE(service->tensor_device.valid());

  RunSession session = run_async(plan);
  const bool observed = RuntimeDescriptorStreamBindingProbe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: the allocation-free descriptor leaves the CUDA worker stream unambiguous.
  EXPECT_TRUE(observed) << (session.diagnostics().empty() ? "no source callback"
                                                          : session.diagnostics().front().token);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// Two same-device descriptor inputs intentionally lack strict-linear fusion evidence. Their
// compiler-owned service contracts still select the relay's process/device stream, while normal
// readiness fences remain enabled.
// Graph topology:
//   first source  --[CUDA Image]--+
//                                +--> relay --[CUDA Tensor]--> (unconnected output)
//   second source --[CUDA Image]--+
// REQ: COMPILE-049
// REQ: TENSOR-SCHEMA-005
TEST(RunSessionDescriptorOutput, NonLinearCudaJoinUsesServiceDeviceWithoutFusion) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device 0 is unavailable";
  }
  RuntimeDescriptorStreamBindingProbe::reset((1U << 0U) | (1U << 1U) |
                                             (1U << kDescriptorRelayProbeIndex));
  Graph graph{"runtime-descriptor-cuda-join"};
  auto first = graph.op<RuntimeDescriptorStreamBindingSource<0U, true>>("first-source");
  auto second = graph.op<RuntimeDescriptorStreamBindingSource<1U, true>>("second-source");
  auto relay = graph.op<RuntimeDescriptorCudaOutputRelay>("relay", true, MemoryKind::kCudaDevice);
  // Key setup: the join consumes two CUDA descriptor services and owns an unrelated CUDA
  // allocation, while the two incoming edges intentionally cannot be strict-linear fusions.
  graph.add_flow(first->output, relay->first, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second->output, relay->second, ConnectionOptions{.queue_depth = 1U});
  graph.partition("runtime-descriptor-cuda-join-first").add(first);
  graph.partition("runtime-descriptor-cuda-join-second").add(second);
  graph.partition("runtime-descriptor-cuda-join-relay").add(relay);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  CompileOptions options;
  for (const std::string_view operator_path : {"first-source", "second-source", "relay"}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = std::string{operator_path},
        .output_port = "output",
        .device = DeviceId{0},
    });
  }

  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  std::vector<EdgeId> descriptor_edges;
  for (const auto& subscriber : state->process_local_channel_subscribers) {
    const auto service = std::ranges::find_if(
        state->process_local_channel_services,
        [&](const auto& row) { return row.service_id == subscriber.service_id; });
    if (service != state->process_local_channel_services.end() &&
        service->payload_identity.payload_kind == PayloadKind::kDescriptorBacked) {
      EXPECT_EQ(service->delivery_kind, detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
      EXPECT_FALSE(service->tensor_pool_id.empty());
      EXPECT_EQ(service->tensor_memory_kind, MemoryKind::kCudaDevice);
      EXPECT_EQ(service->tensor_device, DeviceId{0});
      descriptor_edges.push_back(subscriber.edge);
    }
  }
  ASSERT_EQ(descriptor_edges.size(), 2U);
  ASSERT_EQ(state->tensor_device_resources.size(), 1U);
  ASSERT_EQ(state->cuda_compute_streams.size(), 1U);
  for (const EdgeId edge : descriptor_edges) {
    EXPECT_EQ(std::ranges::find(state->cuda_compute_streams.front().fused_linear_edges, edge),
              state->cuda_compute_streams.front().fused_linear_edges.end());
  }

  RunSession session = run_async(plan);
  const bool observed = RuntimeDescriptorStreamBindingProbe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: both non-fused services select the common CUDA worker stream.
  EXPECT_TRUE(observed) << (session.diagnostics().empty() ? "not all source callbacks ran"
                                                          : session.diagnostics().front().token);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

}  // namespace
}  // namespace holoscan
