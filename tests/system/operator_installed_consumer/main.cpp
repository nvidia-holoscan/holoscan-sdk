/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/time/manual_clock.hpp>

int runtime_header_smoke();
int parameter_compute_header_smoke();

namespace {

using namespace std::chrono_literals;

struct InstalledParams {
  std::uint32_t threshold{4};
};

// REQ: PARAM-010
HOLOSCAN_PARAM_SCHEMA(
    InstalledParams, holoscan::param_schema("nvidia.holoscan.test.InstalledParams", 1),
    holoscan::field<&InstalledParams::threshold>("threshold").runtime_mutable().range(1U, 8U));

static_assert(holoscan::param_schema_availability<InstalledParams>() ==
              holoscan::ParamSchemaAvailability::kDeclared);
static_assert(holoscan::initial_param_generation<InstalledParams>() == 0);

template <typename Runtime>
concept HasLegacyChannelInputFactory = requires(Runtime& runtime) {
  runtime.template create_input<std::uint32_t>(std::string_view{});
};

template <typename Runtime>
concept HasStringTensorPublisherFactory = requires(Runtime& runtime) {
  runtime.template create_publisher<holoscan::Tensor>(std::string_view{});
};

template <typename Output, typename Sample>
concept HasSampleEmit = requires(Output& output, Sample sample) { output.emit(std::move(sample)); };

template <typename Output, typename Sample>
concept HasEmitSampleVerb =
    requires(Output& output, Sample sample) { output.emit_sample(std::move(sample)); };

template <typename Output, typename Sample>
concept HasSampleEmitOptions = requires(Output& output, Sample sample) {
  output.emit(std::move(sample), holoscan::EmitOptions{});
};

template <typename T>
concept HasGraphCompileMember = requires(T& value) { value.compile(); };

template <typename T>
concept HasGraphRunMember = requires(T& value) { value.run(); };

template <typename T>
concept HasGraphRunAsyncMember = requires(T& value) { value.run_async(); };

template <typename T>
concept HasGraphStartMember = requires { &T::start; } || requires(T& value) { value.start(); } ||
                              requires(T& value) { value.start(0); };

template <typename T>
concept HasGraphStopMember = requires { &T::stop; } || requires(T& value) { value.stop(); } ||
                             requires(T& value) { value.stop(0); };

template <typename T>
concept HasGraphRestartMember = requires { &T::restart; } || requires(T& value) {
  value.restart();
} || requires(T& value) { value.restart(0); };

template <typename T>
concept HasGraphRequestRestartMember = requires { &T::request_restart; } || requires(T& value) {
  value.request_restart();
} || requires(T& value) { value.request_restart(0); };

template <typename T>
concept HasGraphSchedulerMember =
    requires { &T::scheduler; } || requires(T& value) { value.scheduler(); };

template <typename T>
concept HasGraphPartitionMember = requires(T& value) {
  {
    value.partition(std::string_view{"installed-partition"})
  } -> std::same_as<holoscan::PartitionHandle>;
};

template <typename T>
concept HasOperatorThreadRole =
    requires(T& value) { value.thread_role(std::string_view{"legacy-role"}); };

template <typename T>
concept HasBindThreadRole = requires(T& value) {
  value.bind_thread_role(std::string_view{"legacy-role"}, holoscan::ThreadPlacement{});
};

template <typename T>
concept HasThreadRoles = requires(const T& value) { value.thread_roles(); };

template <typename T>
concept HasThreadRoleBindings = requires(const T& value) { value.thread_role_bindings(); };

template <typename T>
concept HasThreadRoleUses = requires(const T& value) { value.thread_role_uses(); };

template <typename T>
concept HasThreadRolePlacements = requires(const T& value) { value.thread_role_placements(); };

template <typename T>
concept HasThreadPlacements = requires(const T& value) { value.thread_placements(); };

template <typename T>
concept HasPartitionPlacements = requires(const T& value) { value.partition_placements(); };

template <typename T>
concept HasGraphViewPartitions = requires(const T& value) { value.partitions(); };

template <typename Region, typename Handle>
concept HasRegionContains =
    requires(Region& region, const Handle& handle) { region.contains(handle); };

template <typename Region, typename Handle>
concept HasRegionAdd = requires(Region& region, const Handle& handle) { region.add(handle); };

// REQ: CHANNEL-API-001
// REQ: OPERATOR-013
static_assert(!HasLegacyChannelInputFactory<holoscan::ChannelRuntime>);
static_assert(!HasStringTensorPublisherFactory<holoscan::ChannelRuntime>);
// REQ: OPERATOR-014
static_assert(HasSampleEmit<holoscan::Output<std::uint32_t>, holoscan::Sample<std::uint32_t>>);
static_assert(!HasEmitSampleVerb<holoscan::Output<std::uint32_t>, holoscan::Sample<std::uint32_t>>);
static_assert(
    !HasSampleEmitOptions<holoscan::Output<std::uint32_t>, holoscan::Sample<std::uint32_t>>);
static_assert(
    HasSampleEmit<holoscan::Output<holoscan::Tensor>, holoscan::Sample<holoscan::Tensor>>);
static_assert(
    !HasEmitSampleVerb<holoscan::Output<holoscan::Tensor>, holoscan::Sample<holoscan::Tensor>>);
// REQ: GRAPH-001
// REQ: COMPILE-058
static_assert(!HasGraphCompileMember<holoscan::Graph>);
static_assert(!HasGraphRunMember<holoscan::Graph>);
static_assert(!HasGraphRunAsyncMember<holoscan::Graph>);
static_assert(!HasGraphStartMember<holoscan::Graph>);
static_assert(!HasGraphStopMember<holoscan::Graph>);
static_assert(!HasGraphRestartMember<holoscan::Graph>);
static_assert(!HasGraphRequestRestartMember<holoscan::Graph>);
static_assert(!HasGraphSchedulerMember<holoscan::Graph>);
// REQ: COMPILE-059
static_assert(HasGraphPartitionMember<holoscan::Graph>);
static_assert(!HasOperatorThreadRole<holoscan::OperatorSpec>);
static_assert(!HasBindThreadRole<holoscan::DeploymentBinding>);
static_assert(!HasThreadRoles<holoscan::DeploymentBinding>);
static_assert(!HasThreadRoleBindings<holoscan::ExecutionPlan>);
static_assert(!HasThreadRoleUses<holoscan::ExecutionPlan>);
static_assert(!HasThreadRolePlacements<holoscan::ExecutionPlan>);
static_assert(HasThreadPlacements<holoscan::DeploymentBinding>);
static_assert(HasPartitionPlacements<holoscan::ExecutionPlan>);
static_assert(HasGraphViewPartitions<holoscan::GraphView>);

class InstalledOperator final : public holoscan::Operator<> {
 public:
  holoscan::Input<std::uint32_t> input_{"input"};
  holoscan::Output<std::uint32_t> output_{"output"};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_).max_emits_per_compute();
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    static_cast<void>(context.params<holoscan::NoParams>());
    return {};
  }
};

using InstalledOperatorHandle = holoscan::OperatorHandle<InstalledOperator>;
static_assert(!HasRegionContains<holoscan::RegionHandle, InstalledOperatorHandle>);
static_assert(HasRegionAdd<holoscan::RegionHandle, InstalledOperatorHandle>);

// REQ: GRAPH-021
class InstalledReusableGraph final : public holoscan::Graph {
 public:
  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    auto worker = op<InstalledOperator>("worker");
    export_input(input, worker->input_);
    export_output(output, worker->output_);
  }
};

// REQ: GRAPH-002
// REQ: GRAPH-006
// REQ: GRAPH-009
// REQ: GRAPH-011
class InstalledGraph final : public holoscan::Graph {
 protected:
  void compose() override {
    auto source = op<InstalledOperator>("source");
    auto target = op<InstalledOperator>("target");
    auto reusable = embed<InstalledReusableGraph>("reusable");
    if (!reusable.graph_valid() || !reusable) {
      throw holoscan::RuntimeError(
          holoscan::Error{holoscan::ErrorCode::kFailure, "SUBGRAPH_HANDLE_INVALID"});
    }
    const auto reusable_path = reusable.path();
    if (!reusable_path || reusable_path->segments().size() != 1U ||
        reusable_path->segments().front() != "reusable") {
      throw holoscan::RuntimeError(
          holoscan::Error{holoscan::ErrorCode::kFailure, "SUBGRAPH_HANDLE_PATH_INVALID"});
    }
    (void)add_flow(source->output_, reusable->input);
    (void)add_flow(reusable->output, target->input_);
    (void)import_topic<std::uint32_t>("holomq://installed-import");
    (void)export_topic<std::uint32_t>("holomq://installed-export");
  }
};

struct InstalledRunObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    count.store(0U, std::memory_order_relaxed);
  }

  static void record() {
    {
      const std::lock_guard lock(mutex);
      count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_one() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> count{};
};

class InstalledRunSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    return output.emit(++next_);
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

class InstalledRunSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      [[maybe_unused]] holoscan::ExecutionContext&) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    InstalledRunObservation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

bool has_nonzero_byte(const auto& bytes) {
  return std::any_of(
      bytes.begin(), bytes.end(), [](std::byte byte) { return byte != std::byte{}; });
}

}  // namespace

int main() {
  // Taking these addresses forces the installed consumer to compile and link
  // the out-of-line loan classes plus the runtime/session entry points. On
  // MSVC this also verifies that Core import annotations are effective.
  const auto loan_byte_size = &holoscan::TensorOutputLoan::byte_size;
  const auto write_commit = &holoscan::TensorOutputWriteGuard::commit;
  const auto manual_bind = &holoscan::ChannelRuntime::bind_manual_channel;
  const auto manual_emit = &holoscan::ManualTensorPublisher::emit;
  const auto manual_emit_sample = &holoscan::ManualTensorPublisher::emit_sample;
  const auto manual_update = &holoscan::ManualTensorPublisher::update_connections;
  const auto manual_connected = &holoscan::ManualTensorPublisher::has_subscribers;
  const auto manual_factory = &holoscan::make_manual_tensor_publisher;
  const auto run_sync = static_cast<int (*)(const holoscan::ExecutionPlan&)>(&holoscan::run);
  const auto run_async =
      static_cast<holoscan::RunSession (*)(const holoscan::ExecutionPlan&)>(&holoscan::run_async);
  const auto session_wait = &holoscan::RunSession::wait;
  const auto session_stop = &holoscan::RunSession::request_stop;
  const auto session_running = &holoscan::RunSession::is_running;
  const auto session_termination = &holoscan::RunSession::termination;
  const auto session_diagnostics = &holoscan::RunSession::diagnostics;
  const auto session_operator_ref = &holoscan::RunSession::operator_ref;
  if (loan_byte_size == nullptr || write_commit == nullptr || manual_bind == nullptr ||
      manual_emit == nullptr || manual_emit_sample == nullptr || manual_update == nullptr ||
      manual_connected == nullptr || manual_factory == nullptr || run_sync == nullptr ||
      run_async == nullptr || session_wait == nullptr || session_stop == nullptr ||
      session_running == nullptr || session_termination == nullptr ||
      session_diagnostics == nullptr || session_operator_ref == nullptr) {
    return 3;
  }
  if (runtime_header_smoke() != 0) {
    return 4;
  }
  if (parameter_compute_header_smoke() != 0) {
    return 10;
  }

  // The default plan must reject synchronously through the installed free API.
  // This also forces both shared and static installed-consumer lanes to link
  // the new RunSession entry point rather than merely finding its headers.
  try {
    (void)holoscan::run_async(holoscan::ExecutionPlan{});
    return 9;
  } catch (const holoscan::CompileError&) {
  }

  InstalledOperator op;
  const holoscan::ParamSchemaDescriptor* installed_params =
      holoscan::param_schema_descriptor<InstalledParams>();
  if (installed_params == nullptr || installed_params->fields.size() != 1U ||
      installed_params->fields[0].canonical_name != "threshold") {
    return 5;
  }
  holoscan::OperatorSpec spec;
  op.setup(spec);

  InstalledGraph graph;
  const holoscan::GraphView view = graph.view();
  if (view.graphs().size() != 2U || view.vertices().size() != 7U || view.edges().size() != 2U ||
      view.interface_bindings().size() != 2U || view.external_topics().size() != 2U ||
      !view.partitions().empty() || view.identity().version != view.generation() ||
      !has_nonzero_byte(view.identity().graph.value) ||
      !has_nonzero_byte(view.identity().hash.value)) {
    return 6;
  }

  // REQ: GRAPH-022
  const auto embedded = std::find_if(
      view.graphs().begin(), view.graphs().end(), [](const holoscan::GraphRecord& record) {
        return record.path && record.path->segments().size() == 1U &&
               record.path->segments().front() == "reusable";
      });
  const auto root = std::find_if(
      view.graphs().begin(), view.graphs().end(), [](const holoscan::GraphRecord& record) {
        return !record.path && !record.parent;
      });
  if (embedded == view.graphs().end() || root == view.graphs().end() || !embedded->parent ||
      *embedded->parent != root->id) {
    return 7;
  }

  InstalledRunObservation::reset();
  holoscan::Graph running_graph{"installed-package-run"};
  const auto source = running_graph.op<InstalledRunSource>("source");
  const auto sink = running_graph.op<InstalledRunSink>("sink");
  running_graph.add_flow(
      source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  running_graph.partition("installed-runner").add(source).add(sink);
  running_graph.region("installed-region").add(source).add(sink);
  running_graph.set_default_clock(
      running_graph.add_clock<holoscan::ManualClock>("manual-clock", std::int64_t{0}));
  const holoscan::GraphView running_view = running_graph.view();
  if (running_view.partitions().size() != 1U ||
      running_view.partitions().front().name != "installed-runner" ||
      running_view.partitions().front().members.size() != 2U) {
    return 13;
  }
  holoscan::CompileOptions running_options;
  running_options.deployment.bind_thread_placement("installed-runner", holoscan::ThreadPlacement{});
  if (running_options.deployment.thread_placements().size() != 1U) {
    return 14;
  }
  const holoscan::ExecutionPlan running_plan =
      holoscan::compile(running_graph, std::move(running_options));
  if (!running_plan.ok()) {
    return 11;
  }
  if (running_plan.execution_partitions().size() != 1U ||
      running_plan.partition_placements().size() != 1U ||
      running_plan.partition_placements().front().authored_name != "installed-runner") {
    return 15;
  }
  holoscan::RunSession running_session = holoscan::run_async(running_plan);
  const bool observed = InstalledRunObservation::wait_for_one();
  running_session.request_stop();
  running_session.wait();
  if (!observed || running_session.termination() != holoscan::RunTermination::kStopped ||
      !running_session.diagnostics().empty()) {
    return 12;
  }

  const holoscan::Sample<std::uint32_t> sample{7, holoscan::SampleMetadata{}};
  if (sample.data != 7 || !sample.metadata.valid() || sample.metadata.capture_time().valid()) {
    return 1;
  }
  const auto unbound_emit = op.output_.emit(sample);
  if (unbound_emit || unbound_emit.error().code != holoscan::ErrorCode::kNotReady) {
    return 4;
  }

  const holoscan::NotificationSender sender;
  const auto result = sender.post();
  if (result || result.error().code != holoscan::ErrorCode::kInvalidArgument) {
    return 2;
  }
  return 0;
}
