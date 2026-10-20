/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_channel_binding.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "support/host_unique_test_name.hpp"

#include "runtime_channel_binding_test_support.hpp"

// These tests drive a compiled pool-backed producer into the largest set of Tensor backings each
// delivery path can hold at once, then take one more producer loan. The pool's own low-water mark
// of free slots gives the live set it actually reached, so each test checks that the compiled
// slot count covers the worst case the runtime can construct, not only an analytical sum.
//
// The harness materializes the producer's pools and the Channel session the way RunSession does,
// but drives activations, the pump, and selections by hand, so every interleaving is exact. A
// cross-process plan gets one session per process group, as its two workers would, both in this
// test process; the consumer imports the producer's slots through HoloMQ. A mixed-locality plan
// adds a "remote" sink in the second process group beside the direct "sink" of the first.

namespace holoscan::detail {
namespace {

using namespace std::chrono_literals;
using runtime_channel_binding_test::wait_for_event;

constexpr std::uint32_t kDepth = 2U;
constexpr DLDataType kU64{kDLUInt, 64U, 1U};
constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr std::uint64_t kImageBytes = 8U;
constexpr std::array<std::int64_t, 1U> kImageShape{static_cast<std::int64_t>(kImageBytes)};

class BudgetTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kU64, .rank = 1U},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }
  // The harness drives activations by hand; the clock only makes the source a valid root.
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

enum class Selector : std::uint8_t {
  kDrainAll,  ///< One activation selects every pending sample; the cache equals the depth.
  kOnEach,    ///< One sample per activation and no cache.
};

template <Selector Kind>
class BudgetTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(kDepth)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kU64, .rank = 1U}});
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    if constexpr (Kind == Selector::kDrainAll) {
      result.trigger(OnAnyInput{}).input(input, DrainAll{});
    } else {
      result.trigger(OnEach{input});
    }
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// An ImageT source whose one Tensor field has a runtime pool. A pool loan is exported as is; a
// foreign Tensor is copied into the publisher's descriptor staging pool across processes.
class BudgetImageSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kU8, .rank = 1U},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = kImageBytes},
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer =
                TensorTransferPolicy{
                    .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                    .max_staging_byte_span = kImageBytes,
                },
        });
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output;
};

template <Selector Kind>
class BudgetImageSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(kDepth)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .dtype = kU8, .rank = 1U}});
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    if constexpr (Kind == Selector::kDrainAll) {
      result.trigger(OnAnyInput{}).input(input, DrainAll{});
    } else {
      result.trigger(OnEach{input});
    }
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
};

enum class Payload : std::uint8_t {
  kTensor,         ///< Raw Tensor loans from the output's pool.
  kForeignTensor,  ///< Raw Tensor in caller-owned host memory.
  kPooledImage,    ///< ImageT whose Tensor is a loan from the field pool.
  kForeignImage,   ///< ImageT whose Tensor is caller-owned host memory.
};

enum class Delivery : std::uint8_t {
  kDirect,         ///< Process-local direct retention.
  kHoloMQ,         ///< Process-local HoloMQ transport.
  kCrossProcess,   ///< Source and sink in two process groups of one application.
  kMixedLocality,  ///< Source and sink direct in one process group, and a remote sink in another.
};

// Adds a "source" Operator connected to a "sink" Operator.
template <typename Source, typename Sink>
void connect(Graph& graph, const ConnectionOptions& connection) {
  graph.add_flow(graph.op<Source>("source")->output, graph.op<Sink>("sink")->input, connection);
}

// Adds a "source" Operator connected to a "sink" and a "remote" Operator.
template <typename Source, typename Sink>
void connect_mixed_locality(Graph& graph, const ConnectionOptions& connection) {
  const auto source = graph.op<Source>("source");
  graph.add_flow(source->output, graph.op<Sink>("sink")->input, connection);
  graph.add_flow(source->output, graph.op<Sink>("remote")->input, connection);
}

// Places "source" and "sink" in two process groups.
[[nodiscard]] ProcessPlacementBinding split_source_sink_placement() {
  constexpr std::array source{GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array sink{GraphObjectSelector{.graph_path = "sink"}};
  ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         ProcessPlacement{.members = source, .launch_group = "producer-process"})
      .bind_process_role("consumer",
                         ProcessPlacement{.members = sink, .launch_group = "consumer-process"});
  return placement;
}

// Places "source" and "sink" in one process group and "remote" in another.
[[nodiscard]] ProcessPlacementBinding mixed_locality_placement() {
  constexpr std::array local{GraphObjectSelector{.graph_path = "source"},
                             GraphObjectSelector{.graph_path = "sink"}};
  constexpr std::array remote{GraphObjectSelector{.graph_path = "remote"}};
  ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         ProcessPlacement{.members = local, .launch_group = "producer-process"})
      .bind_process_role("consumer",
                         ProcessPlacement{.members = remote, .launch_group = "consumer-process"});
  return placement;
}

[[nodiscard]] expected<std::vector<RuntimeOperatorInstance>, Error> reconstruct_operators(
    const ExecutionPlanState& plan) {
  std::vector<RuntimeOperatorInstance> operators;
  operators.reserve(plan.executable_factories.operators.size());
  for (const auto& record : plan.executable_factories.operators) {
    if (record.factory == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "operator factory missing"});
    }
    auto created = record.factory->create(nullptr);
    if (!created || created.value() == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "operator reconstruction failed"});
    }
    auto replay = std::make_unique<OperatorSpec>();
    created.value()->setup(*replay);
    const auto partition = std::find_if(
        plan.execution_partitions.begin(),
        plan.execution_partitions.end(),
        [&](const auto& candidate) {
          return std::find(candidate.members.begin(), candidate.members.end(), record.vertex) !=
                 candidate.members.end();
        });
    if (partition == plan.execution_partitions.end()) {
      return make_unexpected(Error{ErrorCode::kFailure, "operator partition missing"});
    }
    operators.push_back(RuntimeOperatorInstance{
        .vertex = record.vertex,
        .operator_id = record.operator_id,
        .partition = partition->id,
        .execution_lane = partition->execution_lane,
        .op = std::move(created).value(),
        .replay_spec = std::move(replay),
        .tags = std::make_unique<TagAllocator>(),
    });
  }
  // Each instance retains its TagAllocator in the vector moved into expected; the analyzer
  // does not follow this aggregate ownership transfer and reports a leak at the closing brace.
  return operators;
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

// One compiled source -> sink plan with its producer pool, Channel sessions, and sink readiness.
// `session` serves the sink. It also serves the source unless the plan places them in two process
// groups, when `producer_session` serves the source. A mixed-locality plan's `remote_session`
// serves its remote sink.
struct PoolHarness {
  std::shared_ptr<RuntimeTensorOutputTestControl> control =
      std::make_shared<RuntimeTensorOutputTestControl>();
  std::shared_ptr<const ExecutionPlanState> state;
  std::vector<RuntimeOperatorInstance> operators;
  std::vector<RuntimeOperatorInstance> consumer_operators;
  std::unique_ptr<RuntimeCudaExecutionResources> cuda;
  std::unique_ptr<RuntimeCudaExecutionResources> consumer_cuda;
  std::unique_ptr<RuntimeTensorResources> tensors;
  std::vector<RuntimeChannelPortReplayRecord> replay;
  std::vector<RuntimeChannelPortReplayRecord> consumer_replay;
  std::unique_ptr<RuntimeChannelSession> session;
  std::unique_ptr<RuntimeChannelSession> producer_session;
  std::unique_ptr<RuntimeChannelSession> remote_session;
  std::unique_ptr<PortBinding> sink_binding;
  std::unique_ptr<PortBinding> remote_binding;
  const RuntimeOperatorInstance* source{};
  const RuntimeOperatorInstance* sink{};
  const RuntimeOperatorInstance* remote{};
  PortId source_port{};
  PortId sink_input{};
  PortId remote_input{};
  std::uint64_t source_activation{};
  std::vector<std::optional<TensorOutputLoan>> held_loans;
  Delivery delivery_{Delivery::kHoloMQ};
  // A mixed-locality plan compiled under the HoloMQ fanout control.
  bool holomq_fanout{};
  Payload payload_{Payload::kTensor};

  Selector selector_{Selector::kDrainAll};

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void build(const Delivery delivery, const QueuePolicy policy,
             const Selector selector = Selector::kDrainAll,
             const Payload payload = Payload::kTensor) {
    delivery_ = delivery;
    selector_ = selector;
    payload_ = payload;
    // A cross-process service is named from the plan, so parallel tests need distinct graphs.
    Graph graph{test::make_host_unique_test_name("tensor-pool-budget")};
    const auto connection = ConnectionOptions{.queue_policy = policy, .queue_depth = kDepth};
    const bool tensor_payload = payload == Payload::kTensor || payload == Payload::kForeignTensor;
    if (delivery == Delivery::kMixedLocality) {
      if (tensor_payload) {
        connect_mixed_locality<BudgetTensorSource, BudgetTensorSink<Selector::kDrainAll>>(
            graph, connection);
      } else {
        connect_mixed_locality<BudgetImageSource, BudgetImageSink<Selector::kDrainAll>>(graph,
                                                                                        connection);
      }
    } else if (tensor_payload) {
      if (selector == Selector::kDrainAll) {
        connect<BudgetTensorSource, BudgetTensorSink<Selector::kDrainAll>>(graph, connection);
      } else {
        connect<BudgetTensorSource, BudgetTensorSink<Selector::kOnEach>>(graph, connection);
      }
    } else if (selector == Selector::kDrainAll) {
      connect<BudgetImageSource, BudgetImageSink<Selector::kDrainAll>>(graph, connection);
    } else {
      connect<BudgetImageSource, BudgetImageSink<Selector::kOnEach>>(graph, connection);
    }
    graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
    const ExecutionPlan compiled = [&] {
      switch (delivery) {
        case Delivery::kHoloMQ:
          return compile_with_process_local_holomq(graph, CompileOptions{});
        case Delivery::kCrossProcess:
          return compile_with_process_placement(
              graph, CompileOptions{}, split_source_sink_placement());
        case Delivery::kMixedLocality:
          return compile_with_runtime_evidence_and_process_placement(
              graph,
              CompileOptions{},
              RuntimeCompileEvidenceView{
                  .package_capabilities = runtime_package_capability_view(),
                  .force_holomq_process_local_fanout = holomq_fanout,
              },
              mixed_locality_placement());
        case Delivery::kDirect:
          break;
      }
      return compile(graph);
    }();
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    ExecutionPlan plan = RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
    if (delivery == Delivery::kHoloMQ) {
      // The budget covers HoloMQ's own worst case, a full queue behind an endpoint head, which
      // only a plan without publication credits reaches (restart authority or the control).
      auto fanout_control = std::make_shared<RuntimeFanoutTestControl>();
      fanout_control->disable_publication_credits = true;
      plan = RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(fanout_control));
    }
    state = ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->tensor_allocations.size(), 1U);
    ProcessGroupId process_group = state->vertex_process_placements.empty()
                                       ? ProcessGroupId{}
                                       : state->vertex_process_placements.front().process_group;
    ProcessGroupId consumer_group = process_group;
    const bool two_groups =
        delivery == Delivery::kCrossProcess || delivery == Delivery::kMixedLocality;
    if (two_groups) {
      ASSERT_EQ(state->process_local_channel_services.size(),
                delivery == Delivery::kMixedLocality ? 1U : 0U);
      ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
      process_group = state->cross_process_channel_services.front().source_process_group;
      consumer_group = state->cross_process_channel_services.front().target_process_group;
      ASSERT_NE(process_group, consumer_group);
    }
    if (delivery != Delivery::kCrossProcess) {
      ASSERT_EQ(state->process_local_channel_services.size(), 1U);
      ASSERT_EQ(state->process_local_channel_services.front().delivery_kind,
                delivery == Delivery::kHoloMQ || holomq_fanout
                    ? ProcessLocalChannelDeliveryKind::kHoloMQ
                    : ProcessLocalChannelDeliveryKind::kDirectRetained);
    }
    control->prepare(state->tensor_allocations.size());

    // Each process group materializes only its own Operators, as its worker would.
    auto reconstructed = reconstruct_operators(*state);
    ASSERT_TRUE(reconstructed) << runtime_error_message(reconstructed.error());
    for (RuntimeOperatorInstance& instance : *reconstructed) {
      const bool consumer =
          two_groups && !worker_owns_vertex(*state, process_group, instance.vertex);
      (consumer ? consumer_operators : operators).push_back(std::move(instance));
    }
    auto created_cuda = RuntimeCudaExecutionResources::create(
        *state, operators, RunGenerationId{1U}, process_group);
    ASSERT_TRUE(created_cuda) << runtime_error_message(created_cuda.error());
    cuda = std::move(*created_cuda);
    auto created_tensors = RuntimeTensorResources::create(
        *state, operators, RunGenerationId{1U}, process_group, *cuda);
    ASSERT_TRUE(created_tensors) << runtime_error_message(created_tensors.error());
    tensors = std::move(*created_tensors);
    if (two_groups) {
      auto created_consumer_cuda = RuntimeCudaExecutionResources::create(
          *state, consumer_operators, RunGenerationId{1U}, consumer_group);
      ASSERT_TRUE(created_consumer_cuda) << runtime_error_message(created_consumer_cuda.error());
      consumer_cuda = std::move(*created_consumer_cuda);
    }

    // The consumer group's input is the remote sink's in a mixed-locality plan.
    const bool remote_consumer = delivery == Delivery::kMixedLocality;
    const auto add_replay = [&](std::vector<RuntimeOperatorInstance>& instances,
                                std::vector<RuntimeChannelPortReplayRecord>& records,
                                const bool remote_inputs) {
      for (RuntimeOperatorInstance& instance : instances) {
        for (std::size_t index = 0U;
             index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
             ++index) {
          const auto port = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
          const auto planned =
              std::ranges::find_if(state->ports, [&](const PortPlanRecord& record) {
                return record.owner == instance.vertex &&
                       record.owner_kind == PortOwnerKind::kOperator &&
                       record.direction == port.direction && record.local_name == port.name;
              });
          ASSERT_NE(planned, state->ports.end());
          records.push_back(RuntimeChannelPortReplayRecord{
              .port = planned->id,
              .owner = instance.vertex,
              .direction = port.direction,
              .declaration = const_cast<void*>(port.declaration),
              .type_ops = port.runtime_type_ops,
              .tensor_failure = &instance.tensor_failure,
          });
          if (port.direction == PortDirection::kOutput) {
            source = &instance;
            source_port = planned->id;
          } else if (remote_inputs) {
            remote = &instance;
            remote_input = planned->id;
          } else {
            sink = &instance;
            sink_input = planned->id;
          }
        }
      }
    };
    ASSERT_NO_FATAL_FAILURE(add_replay(operators, replay, false));
    ASSERT_NO_FATAL_FAILURE(add_replay(consumer_operators, consumer_replay, remote_consumer));
    ASSERT_NE(source, nullptr);
    ASSERT_NE(sink, nullptr);
    const RuntimePortBindingRequest binding_request{.plan_generation = state->plan_generation,
                                                    .run_generation = 1U,
                                                    .kind = RuntimeEventKind::kInputArrival,
                                                    .source = make_source_id(1U),
                                                    .source_clock = ClockId{1U},
                                                    .reference_clock = ClockId{1U},
                                                    .capacity = 8U};
    if (delivery == Delivery::kMixedLocality) {
      // The subscriber exists before the publisher, as when the consumer worker starts first.
      ASSERT_NE(remote, nullptr);
      auto consumer_session = RuntimeChannelSession::create(
          *state, consumer_replay, consumer_group, {}, consumer_cuda.get());
      ASSERT_TRUE(consumer_session) << runtime_error_message(consumer_session.error());
      remote_session = std::move(*consumer_session);
      auto binding = remote_session->create_readiness_binding(remote->vertex, binding_request);
      ASSERT_TRUE(binding) << runtime_error_message(binding.error());
      remote_binding = std::move(*binding);
      ASSERT_TRUE(remote_session->refresh_connections());
    }
    if (delivery == Delivery::kCrossProcess) {
      // The subscriber exists before the publisher, as when the consumer worker starts first.
      auto consumer_session = RuntimeChannelSession::create(
          *state, consumer_replay, consumer_group, {}, consumer_cuda.get());
      ASSERT_TRUE(consumer_session) << runtime_error_message(consumer_session.error());
      session = std::move(*consumer_session);
      auto created_producer =
          RuntimeChannelSession::create(*state, replay, process_group, {}, cuda.get());
      ASSERT_TRUE(created_producer) << runtime_error_message(created_producer.error());
      producer_session = std::move(*created_producer);
    } else {
      auto created_session =
          RuntimeChannelSession::create(*state, replay, process_group, {}, cuda.get());
      ASSERT_TRUE(created_session) << runtime_error_message(created_session.error());
      session = std::move(*created_session);
    }
    auto binding = session->create_readiness_binding(sink->vertex, binding_request);
    ASSERT_TRUE(binding) << runtime_error_message(binding.error());
    sink_binding = std::move(*binding);
    ASSERT_TRUE(session->refresh_connections());
    if (producer_session) {
      ASSERT_TRUE(producer_session->refresh_connections());
    }
  }

  [[nodiscard]] RuntimeChannelSession& source_session() const {
    return producer_session ? *producer_session : *session;
  }

  [[nodiscard]] Output<Tensor>& source_output() const {
    return static_cast<BudgetTensorSource&>(*source->op).output;
  }

  [[nodiscard]] Output<schema::ImageT>& image_output() const {
    return static_cast<BudgetImageSource&>(*source->op).output;
  }

  // Loan one backing from the source's pool. Call inside a compute activation.
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate() const {
    if (payload_ == Payload::kTensor || payload_ == Payload::kForeignTensor) {
      constexpr std::array<std::int64_t, 1U> kShape{1};
      return source_output().allocate_tensor(TensorLoanRequest{.shape = kShape, .dtype = kU64});
    }
    return image_output().allocate_tensor(TensorLoanRequest{.shape = kImageShape, .dtype = kU8});
  }

  // Take one producer loan inside a compute activation and keep it unpublished.
  [[nodiscard]] expected<void, Error> hold_loan() {
    const Tag tag{.sequence = ++source_activation, .generation = 1U};
    const RuntimeCallbackContextScope scope{
        RuntimeCallbackIdentity{.context = RuntimeCallbackContext::kCompute,
                                .operator_id = source->operator_id,
                                .tag = tag}};
    auto loan = allocate();
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    held_loans.emplace_back(std::move(*loan));
    return {};
  }

  // Emit one sample holding `value` from inside the source's compute activation.
  [[nodiscard]] expected<void, Error> emit_value(const std::uint64_t value) const {
    if (payload_ == Payload::kForeignTensor) {
      auto* storage = new (std::nothrow) std::uint64_t{value};
      if (storage == nullptr) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      constexpr std::array<std::int64_t, 1U> kShape{1};
      auto tensor = make_tensor(kShape,
                                kU64,
                                std::nullopt,
                                MemoryStorageType::kSystem,
                                storage,
                                [](void* pointer) -> expected<void, Error> {
                                  delete static_cast<std::uint64_t*>(pointer);
                                  return {};
                                });
      if (!tensor) {
        delete storage;
        return make_unexpected(std::move(tensor).error());
      }
      return source_output().emit(std::move(*tensor));
    }
    if (payload_ == Payload::kForeignImage) {
      auto* storage = new (std::nothrow) std::uint8_t[kImageBytes]{};
      if (storage == nullptr) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      std::fill_n(storage, kImageBytes, static_cast<std::uint8_t>(value));
      auto tensor = make_tensor(kImageShape,
                                kU8,
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
      schema::ImageT image{};
      image.height = 1;
      image.width = static_cast<std::int32_t>(kImageBytes);
      image.encoding = schema::ImageEncoding_CUSTOM;
      image.data = std::make_shared<Tensor>(std::move(*tensor));
      auto emitted = image_output().emit(std::move(image));
      return emitted ? expected<void, Error>{} : make_unexpected(std::move(emitted).error());
    }
    auto loan = allocate();
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer || writer->data() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer).error());
    }
    if (payload_ == Payload::kTensor) {
      *writer->data_as<std::uint64_t>() = value;
    } else {
      std::fill_n(static_cast<std::uint8_t*>(writer->data()),
                  kImageBytes,
                  static_cast<std::uint8_t>(value));
    }
    if (auto committed = std::move(*writer).commit(); !committed) {
      return make_unexpected(std::move(committed).error());
    }
    if (payload_ == Payload::kTensor) {
      auto emitted = source_output().emit(std::move(*loan));
      return emitted ? expected<void, Error>{} : make_unexpected(std::move(emitted).error());
    }
    schema::ImageT image{};
    image.height = 1;
    image.width = static_cast<std::int32_t>(kImageBytes);
    image.encoding = schema::ImageEncoding_CUSTOM;
    auto emitted = image_output().emit_tensor(std::move(*loan), image);
    return emitted ? expected<void, Error>{} : make_unexpected(std::move(emitted).error());
  }

  // Publish one sample holding `value` and return what emit() returned. A transported publication
  // is pumped into the sink when `pump` is set; a direct one is stored inside emit().
  [[nodiscard]] expected<void, Error> publish(const std::uint64_t value, const bool pump) {
    const std::uint64_t sequence = ++source_activation;
    const Tag tag{.sequence = sequence, .generation = 1U};
    if (auto prepared = source_session().prepare_activation(source->vertex, sequence, {});
        !prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    expected<void, Error> emitted;
    {
      const RuntimeCallbackContextScope scope{
          RuntimeCallbackIdentity{.context = RuntimeCallbackContext::kCompute,
                                  .operator_id = source->operator_id,
                                  .tag = tag}};
      emitted = emit_value(value);
    }
    if (auto retired = source_session().retire_activation(source->vertex, sequence, true);
        !retired) {
      return make_unexpected(std::move(retired).error());
    }
    if (!emitted) {
      return emitted;
    }
    if (pump && delivery_ == Delivery::kMixedLocality) {
      if (holomq_fanout) {
        if (auto received = RuntimeChannelSessionTestAccess::receive_one(*session, sink_input);
            !received) {
          return make_unexpected(std::move(received).error());
        }
      }
      return RuntimeChannelSessionTestAccess::receive_one(*remote_session, remote_input);
    }
    if (pump && delivery_ != Delivery::kDirect) {
      if (auto received = RuntimeChannelSessionTestAccess::receive_one(*session, sink_input);
          !received) {
        return make_unexpected(std::move(received).error());
      }
    }
    return {};
  }

  // Select the given readiness events in one sink activation and open Tensor access, which keeps
  // every selected typed slot claimed until the activation retires.
  [[nodiscard]] expected<void, Error> pin(std::span<const TemporalSampleRecord> records,
                                          const bool remote_sink = false) {
    RuntimeChannelSession& target = remote_sink ? *remote_session : *session;
    const RuntimeOperatorInstance& owner = remote_sink ? *remote : *sink;
    const std::array selection{RuntimeChannelActivationSelection{
        .port = remote_sink ? remote_input : sink_input,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .samples = records,
        .batch = selector_ == Selector::kDrainAll,
        .destructive = true,
    }};
    if (auto prepared = target.prepare_activation(owner.vertex, 1U, selection); !prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    return target.prepare_tensor_access(owner.vertex, 1U, nullptr);
  }

  // Wait for `count` sink readiness events and return their temporal records in order.
  [[nodiscard]] expected<std::vector<TemporalSampleRecord>, Error> take_events(
      const std::size_t count, const bool remote_sink = false) {
    RuntimeChannelSession& target = remote_sink ? *remote_session : *session;
    std::vector<TemporalSampleRecord> records;
    for (std::size_t index = 0U; index < count; ++index) {
      auto event = wait_for_event(target, remote_sink ? *remote_binding : *sink_binding);
      if (!event) {
        return make_unexpected(std::move(event).error());
      }
      auto record = target.temporal_sample(event->port, event->source_sequence);
      if (!record) {
        return make_unexpected(std::move(record).error());
      }
      records.push_back(*record);
    }
    return records;
  }

  [[nodiscard]] const TensorAllocationPlanRecord& allocation() const {
    return state->tensor_allocations.front();
  }

  // The most pool slots this run has held at once.
  [[nodiscard]] std::uint32_t peak_live_slots() const {
    const auto snapshot = control->snapshot(0U);
    return snapshot.physical_slot_count - snapshot.metrics.minimum_physical_free_slot_depth;
  }

  [[nodiscard]] std::string budget() const {
    std::ostringstream text;
    text << "slot_count=" << allocation().slot_count << " contributors:";
    for (const auto& contributor : state->pool_capacity_contributors) {
      if (contributor.pool_id == allocation().pool_id) {
        text << ' ' << contributor.contributor_id << '=' << contributor.maximum_distinct_slots;
      }
    }
    for (const auto& dependency : state->pool_capacity_dependencies) {
      if (dependency.pool_id == allocation().pool_id) {
        text << " dep(origin=" << static_cast<int>(dependency.origin)
             << ",source=" << static_cast<int>(dependency.source) << ')';
      }
    }
    return text.str();
  }

  // Retire one sink activation that pinned `records`, which promotes staged survivors.
  [[nodiscard]] expected<void, Error> release(std::span<const TemporalSampleRecord> records,
                                              const bool remote_sink = false) {
    RuntimeChannelSession& target = remote_sink ? *remote_session : *session;
    PortBinding& binding = remote_sink ? *remote_binding : *sink_binding;
    for (const TemporalSampleRecord& record : records) {
      if (auto retired = binding.retire(record.source_sequence); !retired) {
        return make_unexpected(std::move(retired).error());
      }
    }
    if (auto retired = target.retire_activation((remote_sink ? *remote : *sink).vertex, 1U, true);
        !retired) {
      return make_unexpected(std::move(retired).error());
    }
    return target.retire_samples(remote_sink ? remote_input : sink_input, records);
  }

  void shutdown() {
    held_loans.clear();
    for (RuntimeChannelSession* active :
         {producer_session.get(), session.get(), remote_session.get()}) {
      if (active != nullptr) {
        active->cancel();
        EXPECT_TRUE(active->join());
      }
    }
  }
};

// Pins `kDepth` typed slots through one sink activation, then leaves `staged` arrivals in the
// subscriber and `queued` publications in the transport before one more producer loan.
void expect_worst_case_fits(PoolHarness& harness, const std::uint32_t staged,
                            const std::uint32_t queued, const std::uint32_t expected_live) {
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto pinned = harness.take_events(kDepth);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  const auto pinned_slots = harness.pin(*pinned);
  ASSERT_TRUE(pinned_slots) << runtime_error_message(pinned_slots.error());
  for (std::uint32_t index = 0U; index < staged; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  for (std::uint32_t index = 0U; index < queued; ++index) {
    ASSERT_TRUE(harness.publish(++value, false));
  }

  // Key assertion: the extra loan succeeds and the pool really held the predicted live set.
  const auto loan = harness.hold_loan();
  EXPECT_TRUE(loan) << (loan ? std::string{} : runtime_error_message(loan.error())) << "; "
                    << harness.budget();
  EXPECT_EQ(harness.peak_live_slots(), expected_live) << harness.budget();
  // Retirement capacity stays free for slots awaiting release.
  EXPECT_LE(expected_live + 1U, harness.allocation().slot_count) << harness.budget();
}

// A process-local HoloMQ drop-oldest subscriber pins kDepth typed Tensors, stages kDepth newer
// arrivals, and leaves kDepth more queued in the transport while the producer takes its next
// loan: 3 * kDepth + 1 live backings (CHAN-161).
// REQ: CHAN-161
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, HoloMQDropOldestStagingReachesItsWorstCase) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kHoloMQ, QueuePolicy::kDropOldest));
  ASSERT_NO_FATAL_FAILURE(expect_worst_case_fits(harness, kDepth, kDepth, 3U * kDepth + 1U));
  harness.shutdown();
}

// A drop-oldest subscriber in another process group pins kDepth typed imports, stages kDepth newer
// ones, and leaves kDepth more queued in the transport while the producer takes its next loan.
// Every import keeps its exporter slot leased until the consumer drops it, so the exporter's pool
// reaches the same 3 * kDepth + 1 live backings as a process-local HoloMQ subscriber (CHAN-163).
// REQ: CHAN-163
// REQ: TENSOR-STORAGE-010
TEST(TensorPoolBudget, CrossProcessDropOldestStagingReachesItsWorstCase) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kCrossProcess, QueuePolicy::kDropOldest));
  ASSERT_NO_FATAL_FAILURE(expect_worst_case_fits(harness, kDepth, kDepth, 3U * kDepth + 1U));
  harness.shutdown();
}

// While the consumer pins its typed slots, the pump keeps draining imports into staging, and each
// one past the staged depth replaces the oldest and releases its exporter slot. Retiring the
// pinned activation promotes the newest imports. Without staging, the first arrival behind the
// pinned slots would stay at the endpoint head and be the next sample delivered (CHAN-163).
// REQ: CHAN-163
TEST(TensorPoolBudget, CrossProcessDropOldestPromotesTheNewestImports) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kCrossProcess, QueuePolicy::kDropOldest));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto pinned = harness.take_events(kDepth);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  ASSERT_TRUE(harness.pin(*pinned));

  // Key setup: one more import arrives than the staging depth holds.
  for (std::uint32_t index = 0U; index <= kDepth; ++index) {
    const auto published = harness.publish(++value, true);
    ASSERT_TRUE(published) << runtime_error_message(published.error());
  }
  const auto head =
      RuntimeChannelSessionTestAccess::staged_data_sequence(*harness.session, harness.sink_input);
  ASSERT_TRUE(head) << runtime_error_message(head.error());
  EXPECT_FALSE(head->has_value());
  const auto pending =
      RuntimeChannelSessionTestAccess::input_reset_state(*harness.session, harness.sink_input);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_EQ(pending->pending_direct_ingress_samples, kDepth);

  // Key assertion: the promoted survivors are the newest kDepth publications.
  const auto released = harness.release(*pinned);
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  auto promoted = harness.take_events(kDepth);
  ASSERT_TRUE(promoted) << runtime_error_message(promoted.error());
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    EXPECT_EQ((*promoted)[index].source_sequence, value - kDepth + 1U + index);
  }
  harness.shutdown();
}

// A drop-oldest ImageT subscriber in another process group stages behind its pinned typed slots
// as a raw Tensor one does. The image's Tensor is a loan from the producer's field pool, exported
// as is, so that pool reaches the same 3 * kDepth + 1 live backings with the next loan (CHAN-164).
// REQ: CHAN-164
// REQ: TENSOR-STORAGE-010
TEST(TensorPoolBudget, CrossProcessDropOldestImageStagingReachesItsWorstCase) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kCrossProcess,
                                        QueuePolicy::kDropOldest,
                                        Selector::kDrainAll,
                                        Payload::kPooledImage));
  ASSERT_NO_FATAL_FAILURE(expect_worst_case_fits(harness, kDepth, kDepth, 3U * kDepth + 1U));
  harness.shutdown();
}

// An ImageT whose Tensor the producer does not own is copied into the publisher's staging pool, and
// the consumer imports every copy into a receive slot. While the consumer pins kDepth typed images
// it stages one more than its depth, so a staged import is replaced while its successor is
// imported, and kDepth more wait in the transport. One more copy must still find a staging slot,
// and retiring the pinned activation promotes the newest staged images (CHAN-164).
// REQ: CHAN-164
TEST(TensorPoolBudget, CrossProcessDropOldestImageCopiesFitTheirStagingPool) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kCrossProcess,
                                        QueuePolicy::kDropOldest,
                                        Selector::kDrainAll,
                                        Payload::kForeignImage));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto pinned = harness.take_events(kDepth);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  ASSERT_TRUE(harness.pin(*pinned));
  for (std::uint32_t index = 0U; index <= kDepth; ++index) {
    const auto staged = harness.publish(++value, true);
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
  }
  const std::uint64_t newest_staged = value;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    const auto queued = harness.publish(++value, false);
    ASSERT_TRUE(queued) << runtime_error_message(queued.error());
  }

  // Key assertion: the publisher still has a staging slot for one more copy.
  const auto extra = harness.publish(++value, false);
  EXPECT_TRUE(extra) << runtime_error_message(extra.error());
  const auto head =
      RuntimeChannelSessionTestAccess::staged_data_sequence(*harness.session, harness.sink_input);
  ASSERT_TRUE(head) << runtime_error_message(head.error());
  EXPECT_FALSE(head->has_value());

  // Key assertion: retirement promotes the newest staged images.
  const auto released = harness.release(*pinned);
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  auto promoted = harness.take_events(kDepth);
  ASSERT_TRUE(promoted) << runtime_error_message(promoted.error());
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    EXPECT_EQ((*promoted)[index].source_sequence, newest_staged - kDepth + 1U + index);
  }
  harness.shutdown();
}

// A blocking HoloMQ subscriber pins kDepth typed Tensors; the next arrival waits at the endpoint
// head and kDepth more wait in the transport: 2 * kDepth + 2 live backings.
// REQ: CHAN-128
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, HoloMQBlockingEndpointHeadReachesItsWorstCase) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kHoloMQ, QueuePolicy::kBlocking));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto pinned = harness.take_events(kDepth);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  const auto pinned_slots = harness.pin(*pinned);
  ASSERT_TRUE(pinned_slots) << runtime_error_message(pinned_slots.error());
  const auto head = harness.publish(++value, true);
  ASSERT_FALSE(head);
  EXPECT_EQ(head.error().code, ErrorCode::kBackpressured);
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, false));
  }
  const auto loan = harness.hold_loan();
  EXPECT_TRUE(loan) << (loan ? std::string{} : runtime_error_message(loan.error())) << "; "
                    << harness.budget();
  EXPECT_EQ(harness.peak_live_slots(), 2U * kDepth + 2U) << harness.budget();
  EXPECT_LE(2U * kDepth + 3U, harness.allocation().slot_count) << harness.budget();
  harness.shutdown();
}

// A direct drop-oldest subscriber pins kDepth typed Tensors and keeps kDepth newer arrivals in its
// unread ingress while the producer takes its next loan: 2 * kDepth + 1 live backings (CHAN-118).
// REQ: CHAN-128
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, DirectDropOldestSplitIngressReachesItsWorstCase) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kDirect, QueuePolicy::kDropOldest));
  ASSERT_NO_FATAL_FAILURE(expect_worst_case_fits(harness, 0U, kDepth, 2U * kDepth + 1U));
  harness.shutdown();
}

// A clock-fenced DrainAll input keeps one additional ingress survivor while its typed slots
// remain pinned. Raw Tensor and descriptor fields must leave producer and retirement slots free.
// REQ: CHAN-131
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, DirectTickFencedDrainAllKeepsItsExtraSurvivorAndProducerLoan) {
  for (const auto payload : {Payload::kTensor, Payload::kPooledImage}) {
    PoolHarness harness;
    ASSERT_NO_FATAL_FAILURE(
        harness.build(Delivery::kDirect, QueuePolicy::kDropOldest, Selector::kDrainAll, payload));
    auto enabled = harness.session->enable_tick_reconciled_staging(harness.sink_input);
    ASSERT_TRUE(enabled);
    ASSERT_TRUE(*enabled);
    ASSERT_NO_FATAL_FAILURE(expect_worst_case_fits(harness, 0U, kDepth + 1U, 2U * kDepth + 2U));
    harness.shutdown();
  }
}

// Check the cold admission boundary independently of any conservative cache overlap in the
// compiler's pool. Both the exact materialized live set and its one-slot-short neighbor matter.
// REQ: CHAN-131
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, DirectTickFencedDrainAllRejectsOneSlotBelowItsObligation) {
  for (const auto payload : {Payload::kTensor, Payload::kPooledImage}) {
    PoolHarness harness;
    ASSERT_NO_FATAL_FAILURE(
        harness.build(Delivery::kDirect, QueuePolicy::kDropOldest, Selector::kDrainAll, payload));
    harness.shutdown();
    harness.sink_binding.reset();
    harness.session.reset();

    auto exact = std::make_shared<ExecutionPlanState>(*harness.state);
    auto& allocation = exact->tensor_allocations.front();
    ASSERT_EQ(exact->process_local_channel_subscribers.size(), 1U);
    const auto& subscriber = exact->process_local_channel_subscribers.front();
    const auto typed = std::max(subscriber.requested_queue_depth, subscriber.cache_capacity);
    // Typed slots, unread ingress, refill reservation, and the protected tick survivor.
    std::uint32_t required = typed + subscriber.requested_queue_depth + 2U;
    for (const auto& contributor : exact->pool_capacity_contributors) {
      if (contributor.pool_id == allocation.pool_id &&
          (contributor.contributor_id.ends_with(".producer") ||
           contributor.contributor_id.ends_with(".history") ||
           contributor.contributor_id.ends_with(".retirement"))) {
        required += contributor.maximum_distinct_slots;
      }
    }
    ASSERT_GE(allocation.slot_count, required);
    // Leave the compiler's recipe projection untouched: this check must use the subscriber
    // slots actually materialized, independently from that conservative projection.
    allocation.slot_count = required;
    auto accepted =
        RuntimeChannelSession::create(*exact, harness.replay, {}, {}, harness.cuda.get());
    ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
    accepted->reset();

    --allocation.slot_count;
    auto rejected =
        RuntimeChannelSession::create(*exact, harness.replay, {}, {}, harness.cuda.get());
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kResourceExhausted);
    EXPECT_TRUE(rejected.error().message.starts_with("PROFILE_MATERIALIZATION_FAILED"))
        << rejected.error().message;
  }
}

// Without a cache, a blocking HoloMQ subscriber holds one selected sample, one more typed sample,
// one endpoint head, and a full transport queue: 2 * kDepth + 1 live backings with the producer's
// next loan. The endpoint head is the one slot no depth sizes, so it must not use up the slot kept
// for retirement.
// REQ: CHAN-128
// REQ: TENSOR-STORAGE-010
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, HoloMQBlockingWithoutACacheKeepsRetirementFree) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(
      harness.build(Delivery::kHoloMQ, QueuePolicy::kBlocking, Selector::kOnEach));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto ready = harness.take_events(1U);
  ASSERT_TRUE(ready) << runtime_error_message(ready.error());
  const auto pinned = harness.pin(*ready);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  const auto head = harness.publish(++value, true);
  ASSERT_FALSE(head);
  EXPECT_EQ(head.error().code, ErrorCode::kBackpressured);
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, false));
  }
  const auto loan = harness.hold_loan();
  EXPECT_TRUE(loan) << (loan ? std::string{} : runtime_error_message(loan.error())) << "; "
                    << harness.budget();
  constexpr std::uint32_t kLive = 2U * kDepth + 2U;
  EXPECT_EQ(harness.peak_live_slots(), kLive) << harness.budget();
  EXPECT_LE(kLive + 1U, harness.allocation().slot_count) << harness.budget();
  harness.shutdown();
}

// The Channel session recomputes, from its materialized subscribers, how many backings each local
// Tensor pool must hold. A plan whose pool is one slot short is rejected before any Operator runs,
// even when its contributors still sum to the slot count. Without a cache the compiled pool holds
// exactly that obligation, so one slot fewer is short.
// REQ: CHAN-161
// REQ: TENSOR-STORAGE-011
// REQ: TENSOR-STORAGE-016
TEST(TensorPoolBudget, MaterializationRejectsAPoolBelowItsDeliveryObligation) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(
      harness.build(Delivery::kHoloMQ, QueuePolicy::kDropOldest, Selector::kOnEach));
  harness.shutdown();
  harness.sink_binding.reset();
  harness.session.reset();

  auto understated = std::make_shared<ExecutionPlanState>(*harness.state);
  auto& allocation = understated->tensor_allocations.front();
  const auto contributor = std::ranges::find_if(
      understated->pool_capacity_contributors, [&](const PoolCapacityContributorRecord& record) {
        return record.contributor_id == allocation.pool_id + ".transport-retention";
      });
  ASSERT_NE(contributor, understated->pool_capacity_contributors.end()) << harness.budget();
  ASSERT_GT(contributor->maximum_distinct_slots, 1U);
  --contributor->maximum_distinct_slots;
  --allocation.slot_count;

  // Key assertion: the session refuses the understated pool instead of running into exhaustion.
  auto session =
      RuntimeChannelSession::create(*understated, harness.replay, {}, {}, harness.cuda.get());
  ASSERT_FALSE(session);
  EXPECT_EQ(session.error().code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(session.error().message.starts_with("PROFILE_MATERIALIZATION_FAILED"))
      << session.error().message;
}

// A drop-oldest raw Tensor family with its sink in the producer's process group and a remote sink
// in another keeps the local sink direct (CHAN-168), and the two hold disjoint publications at
// once: the local sink pins the first kDepth and the remote sink the next kDepth, the remote sink
// stages kDepth more, and kDepth more wait in the local unread ingress and the transport queue.
// The pool covers those 4 * kDepth backings and the producer's next loan, with retirement free.
// REQ: CHAN-168
// REQ: TENSOR-STORAGE-010
TEST(TensorPoolBudget, MixedLocalityDropOldestHoldingsFitThePool) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kMixedLocality, QueuePolicy::kDropOldest));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto local_pinned = harness.take_events(kDepth);
  ASSERT_TRUE(local_pinned) << runtime_error_message(local_pinned.error());
  ASSERT_TRUE(harness.pin(*local_pinned));
  auto remote_first = harness.take_events(kDepth, true);
  ASSERT_TRUE(remote_first) << runtime_error_message(remote_first.error());
  ASSERT_TRUE(harness.pin(*remote_first, true));
  ASSERT_TRUE(harness.release(*remote_first, true));

  // Key setup: the remote sink pins the next kDepth while the local sink stages them.
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto remote_pinned = harness.take_events(kDepth, true);
  ASSERT_TRUE(remote_pinned) << runtime_error_message(remote_pinned.error());
  ASSERT_TRUE(harness.pin(*remote_pinned, true));
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, false));
  }

  // Key assertion: the extra loan succeeds and the pool really held the predicted live set.
  const auto loan = harness.hold_loan();
  EXPECT_TRUE(loan) << (loan ? std::string{} : runtime_error_message(loan.error())) << "; "
                    << harness.budget();
  EXPECT_EQ(harness.peak_live_slots(), 4U * kDepth + 1U) << harness.budget();
  EXPECT_LE(4U * kDepth + 2U, harness.allocation().slot_count) << harness.budget();
  harness.shutdown();
}

// A foreign Tensor cannot cross processes, and the family checks that before its local stores: the
// publication fails with kInvalidArgument, the local sink gains nothing, and the sequence is kept,
// so the next pool-backed publication reaches both sinks at sequence one.
// REQ: CHAN-168
TEST(TensorPoolBudget, MixedLocalityRejectsAForeignTensorBeforeAnyStore) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kMixedLocality, QueuePolicy::kDropOldest));
  harness.payload_ = Payload::kForeignTensor;
  const auto rejected = harness.publish(1U, false);
  harness.payload_ = Payload::kTensor;

  // Key assertion: nothing was stored and the sequence was not consumed.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
  const auto local =
      RuntimeChannelSessionTestAccess::input_reset_state(*harness.session, harness.sink_input);
  ASSERT_TRUE(local) << runtime_error_message(local.error());
  EXPECT_EQ(local->occupied_samples, 0U);
  EXPECT_EQ(local->last_producer_sequence, 0U);
  const auto next =
      RuntimeChannelSessionTestAccess::output_next_sequence(*harness.session, harness.source_port);
  ASSERT_TRUE(next) << runtime_error_message(next.error());
  EXPECT_EQ(*next, 1U);
  ASSERT_TRUE(harness.publish(2U, true));
  auto local_events = harness.take_events(1U);
  ASSERT_TRUE(local_events) << runtime_error_message(local_events.error());
  EXPECT_EQ(local_events->front().source_sequence, 1U);
  auto remote_events = harness.take_events(1U, true);
  ASSERT_TRUE(remote_events) << runtime_error_message(remote_events.error());
  EXPECT_EQ(remote_events->front().source_sequence, 1U);
  harness.shutdown();
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// A failure while preparing the transported leg is pre-submit: nothing is stored and the sequence
// is kept. A failure after the leg accepted the publication is indeterminate, as CHAN-165 has it:
// the local sink already holds the publication, its sequence is consumed, and the session keeps
// delivering.
// REQ: CHAN-168
TEST(TensorPoolBudget, MixedLocalityLegFailureIsPreSubmitBeforeTheStoresAndIndeterminateAfter) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kMixedLocality, QueuePolicy::kDropOldest));
  const auto local_state = [&] {
    return RuntimeChannelSessionTestAccess::input_reset_state(*harness.session, harness.sink_input);
  };
  const auto next_sequence = [&] {
    return RuntimeChannelSessionTestAccess::output_next_sequence(*harness.session,
                                                                 harness.source_port);
  };
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::throw_tensor_fanout_after_accepts(
      *harness.session, harness.source_port, 0U));
  const auto before = harness.publish(1U, false);
  ASSERT_FALSE(before);
  EXPECT_EQ(before.error().code, ErrorCode::kFailure);
  ASSERT_TRUE(local_state());
  EXPECT_EQ(local_state()->last_producer_sequence, 0U);
  ASSERT_TRUE(next_sequence());
  EXPECT_EQ(*next_sequence(), 1U);

  // Key setup: the leg throws after it accepted the publication.
  ASSERT_TRUE(RuntimeChannelSessionTestAccess::throw_tensor_fanout_after_accepts(
      *harness.session, harness.source_port, 1U));
  const auto after = harness.publish(2U, true);

  // Key assertion: the local store stands, the sequence is consumed, and delivery continues.
  ASSERT_FALSE(after);
  EXPECT_EQ(after.error().code, ErrorCode::kPublicationIndeterminate);
  ASSERT_TRUE(local_state());
  EXPECT_EQ(local_state()->last_producer_sequence, 1U);
  ASSERT_TRUE(next_sequence());
  EXPECT_EQ(*next_sequence(), 2U);
  ASSERT_TRUE(
      RuntimeChannelSessionTestAccess::receive_one(*harness.remote_session, harness.remote_input));
  ASSERT_TRUE(harness.publish(3U, true));
  EXPECT_EQ(local_state()->last_producer_sequence, 2U);
  auto remote_events = harness.take_events(2U, true);
  ASSERT_TRUE(remote_events) << runtime_error_message(remote_events.error());
  EXPECT_EQ(remote_events->back().source_sequence, 2U);
  harness.shutdown();
}
#endif

// Terminal close stores end of stream into the local sink at the next sequence and closes the
// cross-process leg at the same sequence, and the output consumes it once, for a raw Tensor and an
// ImageT family alike.
// REQ: CHAN-168
// REQ: CHAN-169
TEST(TensorPoolBudget, MixedLocalityClosesTheLocalSinkAndTheLegAtOneSequence) {
  for (const Payload payload : {Payload::kTensor, Payload::kPooledImage}) {
    SCOPED_TRACE(payload == Payload::kTensor ? "raw Tensor" : "ImageT");
    PoolHarness harness;
    ASSERT_NO_FATAL_FAILURE(harness.build(
        Delivery::kMixedLocality, QueuePolicy::kDropOldest, Selector::kDrainAll, payload));
    ASSERT_TRUE(harness.publish(1U, true));
    ASSERT_TRUE(harness.session->close_owner_outputs(harness.source->vertex));
    expected<RuntimeChannelInputResetState, Error> remote =
        make_unexpected(Error{ErrorCode::kTimeout});
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      remote = RuntimeChannelSessionTestAccess::input_reset_state(*harness.remote_session,
                                                                  harness.remote_input);
      if (!remote || remote->end_of_stream_received) {
        break;
      }
      static_cast<void>(RuntimeChannelSessionTestAccess::receive_one(*harness.remote_session,
                                                                     harness.remote_input));
      std::this_thread::sleep_for(1ms);
    }

    // Key assertion: both sinks read end of stream at sequence two, and the output is at three.
    ASSERT_TRUE(remote) << runtime_error_message(remote.error());
    EXPECT_TRUE(remote->end_of_stream_received);
    EXPECT_EQ(remote->end_of_stream_transport_sequence, 2U);
    const auto local =
        RuntimeChannelSessionTestAccess::input_reset_state(*harness.session, harness.sink_input);
    ASSERT_TRUE(local) << runtime_error_message(local.error());
    EXPECT_TRUE(local->end_of_stream_received);
    EXPECT_EQ(local->end_of_stream_transport_sequence, 2U);
    const auto next = RuntimeChannelSessionTestAccess::output_next_sequence(*harness.session,
                                                                            harness.source_port);
    ASSERT_TRUE(next) << runtime_error_message(next.error());
    EXPECT_EQ(*next, 3U);
    harness.shutdown();
  }
}

// A drop-oldest ImageT family with its sink in the producer's process group and a remote sink in
// another keeps the local sink direct (CHAN-169), as the HoloMQ fanout control does not. An image
// whose Tensor is a field-pool loan is exported as is; a foreign one is copied into the publisher's
// staging pool for the remote sink. One publication reaches both sinks at sequence one.
// REQ: CHAN-169
TEST(TensorPoolBudget, MixedLocalityImageFamilyKeepsItsLocalSinkDirect) {
  for (const bool control : {false, true}) {
    for (const Payload payload : {Payload::kPooledImage, Payload::kForeignImage}) {
      SCOPED_TRACE(::testing::Message()
                   << (control ? "holomq fanout control" : "mixed locality")
                   << (payload == Payload::kForeignImage ? ", foreign" : ", pooled"));
      PoolHarness harness;
      harness.holomq_fanout = control;
      ASSERT_NO_FATAL_FAILURE(harness.build(
          Delivery::kMixedLocality, QueuePolicy::kDropOldest, Selector::kDrainAll, payload));
      const auto published = harness.publish(1U, true);
      ASSERT_TRUE(published) << runtime_error_message(published.error());

      // Key assertion: both sinks hold the publication at sequence one.
      auto local = harness.take_events(1U);
      ASSERT_TRUE(local) << runtime_error_message(local.error());
      EXPECT_EQ(local->front().source_sequence, 1U);
      auto remote = harness.take_events(1U, true);
      ASSERT_TRUE(remote) << runtime_error_message(remote.error());
      EXPECT_EQ(remote->front().source_sequence, 1U);
      harness.shutdown();
    }
  }
}

// Beside a direct local sink, foreign images still fit the publisher's staging pool: while the
// remote sink pins kDepth typed images it stages one more than its depth and kDepth more wait in
// the transport, and one more copy still finds a staging slot. The local sink keeps only the
// newest kDepth, so it holds no copy, and retiring the remote activation promotes the newest
// staged images (CHAN-164, CHAN-169).
// REQ: CHAN-169
TEST(TensorPoolBudget, MixedLocalityForeignImageCopiesFitTheirStagingPool) {
  PoolHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.build(Delivery::kMixedLocality,
                                        QueuePolicy::kDropOldest,
                                        Selector::kDrainAll,
                                        Payload::kForeignImage));
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    ASSERT_TRUE(harness.publish(++value, true));
  }
  auto pinned = harness.take_events(kDepth, true);
  ASSERT_TRUE(pinned) << runtime_error_message(pinned.error());
  ASSERT_TRUE(harness.pin(*pinned, true));
  for (std::uint32_t index = 0U; index <= kDepth; ++index) {
    const auto staged = harness.publish(++value, true);
    ASSERT_TRUE(staged) << runtime_error_message(staged.error());
  }
  const std::uint64_t newest_staged = value;
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    const auto queued = harness.publish(++value, false);
    ASSERT_TRUE(queued) << runtime_error_message(queued.error());
  }

  // Key assertion: the publisher still has a staging slot for one more copy, and the local sink
  // holds the newest publication.
  const auto extra = harness.publish(++value, false);
  EXPECT_TRUE(extra) << runtime_error_message(extra.error());
  const auto local =
      RuntimeChannelSessionTestAccess::input_reset_state(*harness.session, harness.sink_input);
  ASSERT_TRUE(local) << runtime_error_message(local.error());
  EXPECT_EQ(local->last_producer_sequence, value);

  // Key assertion: retirement promotes the newest staged images at the remote sink.
  const auto released = harness.release(*pinned, true);
  ASSERT_TRUE(released) << runtime_error_message(released.error());
  auto promoted = harness.take_events(kDepth, true);
  ASSERT_TRUE(promoted) << runtime_error_message(promoted.error());
  for (std::uint32_t index = 0U; index < kDepth; ++index) {
    EXPECT_EQ((*promoted)[index].source_sequence, newest_staged - kDepth + 1U + index);
  }
  harness.shutdown();
}

}  // namespace
}  // namespace holoscan::detail
