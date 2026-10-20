/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>

#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "run_session_tensor_test_support.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"
#include "support/host_unique_test_name.hpp"

namespace holoscan::test::run_session_tensor {
namespace {

using namespace std::chrono_literals;

// Three-phase source: emits kFirstValue, then waits for sink confirmation,
// allocates a stale write guard and issues a self-restart request, then on
// re-entry emits kSecondValue and confirms the pre-restart guard is rejected.
// Used by: DirectTensorPoolDeepRestartPreservesCommittedOutputAndRejectsStaleGuard
class RuntimeDirectTensorRestartSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(2U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle().stage(LifecycleStage::kAllocate,
                           &RuntimeDirectTensorRestartSourceOperator::allocate_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (phase_ == 0U) {
      RuntimeDirectTensorRestartProbeState::record_source_stream(context.cuda_stream().get(),
                                                                 false);
      auto first = emit_value(kFirstValue, context.cuda_stream());
      if (!first) {
        return first;
      }
      phase_ = 1U;
      return {};
    }
    if (phase_ == 1U) {
      // Restart only after the sink has retained the committed predecessor
      // sample. Queued old-generation ingress is intentionally discarded by
      // the Channel restart fence, while an already delivered Tensor must keep
      // its backing claim alive across pool replacement.
      if (!RuntimeDirectTensorRestartProbeState::first_sample_observed()) {
        return {};
      }
      auto stale = output.allocate_tensor(request());
      if (!stale) {
        return fail(stale.error().code);
      }
      stale_loan_.emplace(std::move(*stale));
      auto stale_writer = stale_loan_->write(context.cuda_stream());
      if (!stale_writer) {
        return fail(stale_writer.error().code);
      }
      stale_writer_.emplace(std::move(*stale_writer));
      auto receipt = context.request_restart(LifecycleRestartOptions{
          .from_stage = RuntimeDirectTensorRestartProbeState::requested_restart_stage()});
      if (!receipt) {
        return fail(receipt.error().code);
      }
      restart_accepted_ = true;
      phase_ = 2U;
      return {};
    }
    if (source_completed_) {
      return {};
    }
    if (!stale_writer_) {
      return fail(ErrorCode::kFailure);
    }
    RuntimeDirectTensorRestartProbeState::record_source_stream(context.cuda_stream().get(), true);

    auto stale_commit = std::move(*stale_writer_).commit();
    const bool stale_rejected =
        !stale_commit && (stale_commit.error().code == ErrorCode::kClosed ||
                          stale_commit.error().code == ErrorCode::kInvalidArgument);
    stale_writer_.reset();
    stale_loan_.reset();

    auto replacement = output.allocate_tensor(request());
    if (!replacement) {
      return fail(replacement.error().code);
    }
    auto replacement_writer = replacement->write(context.cuda_stream());
    if (!replacement_writer) {
      return fail(replacement_writer.error().code);
    }
    *replacement_writer->data_as<std::uint64_t>() = kSecondValue;
    auto committed = std::move(*replacement_writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*replacement));
    if (!emitted) {
      return fail(emitted.error().code);
    }
    source_completed_ = true;
    RuntimeDirectTensorRestartProbeState::record_source_complete(restart_accepted_, stale_rejected);
    return {};
  }

  LifecycleStatus allocate_stage(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      const std::lock_guard lock(RuntimeDirectTensorRestartProbeState::mutex);
      ++RuntimeDirectTensorRestartProbeState::restart_allocate_count;
    }
    return LifecycleStatus::kOk;
  }

  Output<Tensor> output;

 private:
  [[nodiscard]] static TensorLoanRequest request() noexcept {
    return TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    };
  }

  [[nodiscard]] expected<void, Error> emit_value(std::uint64_t value, ::cuda::stream_ref stream) {
    auto loan = output.allocate_tensor(request());
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write(stream);
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return fail(writer ? ErrorCode::kFailure : writer.error().code);
    }
    *writer->data_as<std::uint64_t>() = value;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan));
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted.error()))};
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    RuntimeDirectTensorRestartProbeState::fail(code);
    return make_unexpected(Error{code});
  }

  static constexpr std::uint64_t kFirstValue = 11U;
  static constexpr std::uint64_t kSecondValue = 22U;
  std::optional<TensorOutputLoan> stale_loan_;
  std::optional<TensorOutputWriteGuard> stale_writer_;
  std::uint32_t phase_{};
  bool restart_accepted_{};
  bool source_completed_{};
};

// Receives Tensor outputs across a restart boundary, records each value, and
// optionally synchronises and retains the most recent Tensor on an external
// stream to probe cross-session pointer safety on teardown.
// Used by: DirectTensorPoolDeepRestart* and related tests.
class RuntimeDirectTensorRestartSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaManaged,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
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
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(sample ? Error{ErrorCode::kFailure} : std::move(sample.error()));
    }
    const auto consumer_stream = RuntimeDirectTensorRestartProbeState::selected_consumer_stream();
    // This fixture writes managed memory synchronously on the CPU. The escaped-readiness
    // regression uses runtime admission alone to preserve the unrecorded lazy event.
    auto ready = RuntimeDirectTensorRestartProbeState::skip_host_wait
                     ? expected<void, Error>{}
                     : (consumer_stream != nullptr
                            ? sample->data.synchronize_for_consumer_stream(consumer_stream)
                            : sample->data.synchronize_for_host());
    if (!ready) {
      return make_unexpected(std::move(ready.error()));
    }
    if (consumer_stream != nullptr && (cudaStreamSynchronize(consumer_stream) != cudaSuccess ||
                                       !sample->data.set_deallocation_stream(consumer_stream))) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    RuntimeDirectTensorRestartProbeState::record_sink(sample->data,
                                                      *sample->data.data_as<std::uint64_t>());
    return {};
  }

  Input<Tensor> input;
};

}  // namespace

[[nodiscard]] std::string unique_runtime_graph_name(const std::string_view base) {
  return test::make_host_unique_test_name(base);
}

[[nodiscard]] bool wait_for_runtime_phase_count(
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const detail::RuntimePhaseCounter counter, const std::uint64_t minimum) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (counters->snapshot()[detail::RuntimePhaseCounters::index(counter)] >= minimum) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return counters->snapshot()[detail::RuntimePhaseCounters::index(counter)] >= minimum;
}

[[nodiscard]] ExecutionPlan make_runtime_direct_tensor_restart_plan(
    const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control,
    const bool add_catalog_export, const std::string_view graph_name) {
  Graph graph{std::string{graph_name}};
  auto source = graph.op<RuntimeDirectTensorRestartSourceOperator>("source");
  auto sink = graph.op<RuntimeDirectTensorRestartSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  if (add_catalog_export) {
    static std::atomic<std::uint64_t> endpoint_ordinal{};
    const std::string endpoint_name =
        "runtime-direct-tensor-mixed-export-" +
        std::to_string(endpoint_ordinal.fetch_add(1U, std::memory_order_relaxed));
    TopicOptions options;
    options.endpoint_name = endpoint_name;
    options.bounds.subscriber =
        SubscriberEndpointBounds{.queue_depth = 2U, .max_borrowed_samples = 2U};
    options.tensor = TensorBoundaryContract{
        .representation =
            holoscan::TensorRepresentation{
                .memory_kind = MemoryKind::kHost,
                .dtype = DLDataType{kDLUInt, 64U, 1U},
                .rank = 0U,
                .strides = TensorStridePolicy::kContiguousOnly,
            },
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
    auto exported = graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
    graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 2U});
  }
  graph.lifecycle().allow_self_restart(source);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 8U,
      .max_candidate_generations = 8U,
      .max_events = 64U,
      .max_audit_records = 16U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 32U,
  };
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto source_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator && vertex.name == "source";
      });
  const auto output_port =
      source_vertex == snapshot->vertices.end()
          ? snapshot->ports.end()
          : std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [&](const auto& port) {
              return port.owner == source_vertex->id && port.direction == PortDirection::kOutput &&
                     port.name == "output";
            });
  if (output_port == snapshot->ports.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "runtime restart Tensor output is unresolved"}};
  }
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output_port->id.value},
                                        .device = DeviceId{0}},
  };
  const ExecutionPlan plan = detail::compile_with_runtime_evidence(
      graph,
      std::move(options),
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      });
  return control ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control)
                 : plan;
}

}  // namespace holoscan::test::run_session_tensor
