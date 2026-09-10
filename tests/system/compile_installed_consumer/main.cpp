/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/temporal_plan.hpp>
#include <holoscan/time/manual_clock.hpp>

namespace {

class InstalledCompileOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Output<std::uint32_t> output;
};

class InstalledClockImportGraph final : public holoscan::Graph {
 public:
  [[nodiscard]] holoscan::ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    (void)op<InstalledCompileOperator>("source");
  }

 private:
  holoscan::ClockKey clock_import_;
};

// Provides one manual execution partition for the installed Green Context authoring and plan-
// record check in main().
class InstalledGreenContextGraph final : public holoscan::Graph {
 protected:
  void compose() override {
    const auto source = op<InstalledCompileOperator>("source");
    partition("installed-partition").add(source);
    const holoscan::ClockKey clock =
        add_clock<holoscan::ManualClock>("green-context-clock", std::int64_t{0});
    set_default_clock(clock);
  }
};

template <typename T>
concept HasCompileMember = requires(T& value) { value.compile(); };

template <typename T>
concept HasPartitionSelector = requires(T value) { value.partition_algorithm; };

template <typename T>
concept ExposesBackendHandleMember = requires(T value) { value.handle; };

template <typename T>
concept ExposesPrivateClockHealthRecord = requires(T value) {
  value.publisher_operation_sequence;
  value.report_sequence;
  value.producer_epoch;
};

template <typename T>
concept HasArrowOperator = requires(const T& value) { value.operator->(); };

template <typename T>
concept ExposesTensorLoanTransportMember =
    requires(T value) { value.pool_id; } || requires(T value) { value.slot_id; } ||
    requires(T value) { value.endpoint_generation; } || requires(T value) { value.cuda_event; } ||
    requires(T value) { value.buffer_handle; } || requires(T value) { value.control; };

template <typename Publisher>
concept HasBareTensorEmit =
    requires(Publisher& publisher, holoscan::Tensor tensor) { publisher.emit(std::move(tensor)); };

template <typename Publisher>
concept HasTensorSampleEmit =
    requires(Publisher& publisher, holoscan::Sample<holoscan::Tensor> sample) {
      publisher.emit_sample(std::move(sample));
    };

template <typename Publisher>
concept HasExternalTensorLoan =
    requires(Publisher& publisher, holoscan::TensorLoanRequest request) {
      {
        publisher.loan_tensor(request)
      } -> std::same_as<holoscan::expected<holoscan::TensorPublishLoan, holoscan::Error>>;
    };

template <typename Publisher>
concept HasExternalTensorCopy =
    requires(Publisher& publisher, const holoscan::Tensor& tensor, ::cuda::stream_ref stream) {
      {
        publisher.emit_copy(tensor, stream)
      } -> std::same_as<holoscan::expected<std::uint32_t, holoscan::Error>>;
    };

template <typename Publisher>
concept HasTensorMemoryOptions =
    requires(const Publisher& publisher) { publisher.tensor_memory_options(); };

template <typename Publisher>
concept HasTensorAllocator = requires(Publisher& publisher) { publisher.allocator(); };

template <typename Publisher>
concept HasTensorFactory = requires(Publisher& publisher, holoscan::TensorLoanRequest request) {
  publisher.make_tensor(request);
};

template <typename Publisher>
concept HasClockHealthReport = requires(Publisher& publisher, holoscan::ClockHealthReport report) {
  publisher.report_clock_health(report);
};

template <typename Publisher>
concept HasCaptureTimeFactory =
    requires(const Publisher& publisher) { publisher.make_capture_time(1); };

template <typename Runtime>
concept HasStringTensorPublisherFactory = requires(Runtime& runtime) {
  runtime.template create_publisher<holoscan::Tensor>(std::string_view{"tensor"});
};

template <typename Runtime>
concept HasCatalogTensorPublisherFactory =
    requires(Runtime& runtime, const holoscan::ChannelEndpointCatalog& catalog) {
      runtime.template create_publisher<holoscan::Tensor>(catalog, std::string_view{"tensor"});
    };

template <typename Runtime>
concept HasEndpointTensorPublisherFactory =
    requires(Runtime& runtime, holoscan::ChannelEndpointCatalog::EndpointView endpoint) {
      runtime.template create_publisher<holoscan::Tensor>(endpoint);
    };

static_assert(!HasCompileMember<holoscan::Graph>);
static_assert(!HasPartitionSelector<holoscan::CompileOptions>);
static_assert(!ExposesBackendHandleMember<holoscan::BufferHandlePayloadReference>);
// Q61-T01/Q61-T14 freeze the public reason ABI while keeping ordered transport records private.
static_assert(!ExposesPrivateClockHealthRecord<holoscan::ClockHealthReport>);
static_assert(std::same_as<std::underlying_type_t<holoscan::ClockHealthReason>, std::uint8_t>);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kNone) == 0U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kSynchronizationLockLost) ==
              1U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kPhaseErrorLimitExceeded) ==
              2U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kRateErrorLimitExceeded) ==
              3U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kCalibrationExpired) == 4U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kClockStalled) == 5U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kTimestampRegression) == 6U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kTimestampSourceFault) == 7U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kMonitorFault) == 8U);
static_assert(static_cast<std::uint8_t>(holoscan::ClockHealthReason::kUnspecified) == 9U);
static_assert(holoscan::BufferHandlePayloadReference::kAbiVersion == 1U);
static_assert(holoscan::BufferHandlePayloadReference::kPayloadCapacity == 256U);
static_assert(std::is_standard_layout_v<holoscan::BufferHandlePayloadReference>);
static_assert(std::is_trivially_copyable_v<holoscan::BufferHandlePayloadReference>);
// TA-T01/TA-T02/TA-T43/TA-T45/TA-T46: prove the installed external-loan traits, opacity,
// borrowed surface, constrained catalog factory, and distinct manual compatibility domain.
static_assert(std::is_move_constructible_v<holoscan::TensorPublishLoan>);
static_assert(!std::is_copy_constructible_v<holoscan::TensorPublishLoan>);
static_assert(!std::is_copy_assignable_v<holoscan::TensorPublishLoan>);
static_assert(!std::is_move_assignable_v<holoscan::TensorPublishLoan>);
static_assert(std::is_nothrow_destructible_v<holoscan::TensorPublishLoan>);
static_assert(std::is_move_constructible_v<holoscan::TensorLoanWriteGuard>);
static_assert(!std::is_copy_constructible_v<holoscan::TensorLoanWriteGuard>);
static_assert(!std::is_copy_assignable_v<holoscan::TensorLoanWriteGuard>);
static_assert(!std::is_move_assignable_v<holoscan::TensorLoanWriteGuard>);
static_assert(std::is_nothrow_destructible_v<holoscan::TensorLoanWriteGuard>);
static_assert(!std::is_convertible_v<holoscan::TensorPublishLoan, holoscan::Tensor>);
static_assert(!std::is_convertible_v<holoscan::Tensor, holoscan::TensorPublishLoan>);
static_assert(!std::is_convertible_v<holoscan::TensorPublishLoan, holoscan::TensorOutputLoan>);
static_assert(!std::is_convertible_v<holoscan::TensorOutputLoan, holoscan::TensorPublishLoan>);
static_assert(!ExposesBackendHandleMember<holoscan::TensorPublishLoan>);
static_assert(!ExposesTensorLoanTransportMember<holoscan::TensorPublishLoan>);
static_assert(!ExposesTensorLoanTransportMember<holoscan::TensorLoanWriteGuard>);
static_assert(!ExposesTensorLoanTransportMember<holoscan::TensorOutputLoan>);
static_assert(!ExposesTensorLoanTransportMember<holoscan::TensorOutputWriteGuard>);
static_assert(std::is_move_constructible_v<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!std::is_copy_constructible_v<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!std::is_move_assignable_v<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!std::is_default_constructible_v<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(HasExternalTensorLoan<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(HasExternalTensorCopy<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasBareTensorEmit<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasTensorSampleEmit<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasTensorMemoryOptions<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasTensorAllocator<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasTensorFactory<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(HasBareTensorEmit<holoscan::ManualTensorPublisher>);
static_assert(HasTensorSampleEmit<holoscan::ManualTensorPublisher>);
static_assert(!HasExternalTensorLoan<holoscan::ManualTensorPublisher>);
static_assert(!HasExternalTensorCopy<holoscan::ManualTensorPublisher>);
static_assert(!HasClockHealthReport<holoscan::ManualTensorPublisher>);
static_assert(!HasCaptureTimeFactory<holoscan::ManualTensorPublisher>);
static_assert(!HasStringTensorPublisherFactory<holoscan::ChannelRuntime>);
static_assert(HasCatalogTensorPublisherFactory<holoscan::ChannelRuntime>);
static_assert(HasEndpointTensorPublisherFactory<holoscan::ChannelRuntime>);
static_assert(std::same_as<decltype(std::declval<holoscan::TensorPublishLoan&>().write(
                               std::declval<::cuda::stream_ref>())),
                           holoscan::expected<holoscan::TensorLoanWriteGuard, holoscan::Error>>);
static_assert(std::same_as<decltype(std::declval<holoscan::TensorPublishLoan&>().write_host()),
                           holoscan::expected<holoscan::TensorLoanWriteGuard, holoscan::Error>>);
static_assert(std::same_as<decltype(std::declval<holoscan::TensorPublishLoan&&>().cancel()),
                           holoscan::expected<void, holoscan::Error>>);
static_assert(std::same_as<decltype(std::declval<holoscan::TensorLoanWriteGuard&&>().commit()),
                           holoscan::expected<void, holoscan::Error>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>().temporal_triggers()),
                 std::span<const holoscan::TemporalTriggerPlanRecord>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>().input_selections()),
                 std::span<const holoscan::InputSelectionPlanRecord>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>().predicate_callables()),
                 std::span<const holoscan::PredicateCallablePlanRecord>>);
static_assert(std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>().partitions()),
                           holoscan::ExecutionPartitionsView>);
static_assert(!noexcept(std::declval<const holoscan::ExecutionPlan&>().partitions()));
static_assert(!std::default_initializable<holoscan::ExecutionPartitionView>);
static_assert(!std::default_initializable<holoscan::ExecutionPartitionsView>);
static_assert(std::forward_iterator<holoscan::ExecutionPartitionsView::const_iterator>);
static_assert(
    std::same_as<typename std::iterator_traits<
                     holoscan::ExecutionPartitionsView::const_iterator>::iterator_category,
                 std::input_iterator_tag>);
static_assert(std::ranges::forward_range<holoscan::ExecutionPartitionsView>);
static_assert(!std::ranges::borrowed_range<holoscan::ExecutionPartitionsView>);
static_assert(!HasArrowOperator<holoscan::ExecutionPartitionsView::const_iterator>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionGraphView&>().clock_imports()),
                 std::span<const holoscan::ClockImportAuthoringRecord>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionGraphView&>().clock_bindings()),
                 std::span<const holoscan::EmbeddedClockBindingAuthoringRecord>>);
static_assert(
    std::same_as<
        decltype(holoscan::query_cuda_green_context_capability(std::declval<holoscan::DeviceId>())),
        holoscan::expected<holoscan::CudaGreenContextCapability, holoscan::Error>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::DeploymentBinding&>().cuda_green_contexts()),
                 std::span<const holoscan::PartitionCudaGreenContextBinding>>);
static_assert(std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>()
                                        .cuda_green_context_requests()),
                           std::span<const holoscan::CudaGreenContextRequestPlanRecord>>);
static_assert(std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>()
                                        .cuda_green_context_assignments()),
                           std::span<const holoscan::CudaGreenContextAssignmentPlanRecord>>);
static_assert(std::same_as<
              decltype(std::declval<const holoscan::ExecutionPlan&>().cuda_green_context_pools()),
              std::span<const holoscan::CudaGreenContextPoolPlanRecord>>);
static_assert(std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>()
                                        .cuda_green_context_resources()),
                           std::span<const holoscan::CudaGreenContextResourcePlanRecord>>);
static_assert(
    std::same_as<decltype(std::declval<const holoscan::ExecutionPlan&>().cuda_compute_streams()),
                 std::span<const holoscan::CudaComputeStreamPlanRecord>>);

}  // namespace

// REQ: COMPILE-009
// REQ: COMPILE-025
// REQ: CHAN-069
// REQ: GRAPH-026
// REQ: COMPILE-053
// REQ: COMPILE-068
int main() {
  // The installed consumer checks canonical public evidence rather than depending on the private
  // health-control encoding used to transport the corresponding enum values.
  constexpr std::string_view kClockHealthReasonSchema =
      "\"clock_health_reason_schema\":[{\"id\":0,\"token\":\"none\"},"
      "{\"id\":1,\"token\":\"synchronization_lock_lost\"},"
      "{\"id\":2,\"token\":\"phase_error_limit_exceeded\"},"
      "{\"id\":3,\"token\":\"rate_error_limit_exceeded\"},"
      "{\"id\":4,\"token\":\"calibration_expired\"},"
      "{\"id\":5,\"token\":\"clock_stalled\"},"
      "{\"id\":6,\"token\":\"timestamp_regression\"},"
      "{\"id\":7,\"token\":\"timestamp_source_fault\"},"
      "{\"id\":8,\"token\":\"monitor_fault\"},"
      "{\"id\":9,\"token\":\"unspecified\"}]";

  // Taking addresses ensures the installed package exports the declaration-only HSDK5-013 facade
  // symbols even though catalog construction remains intentionally unavailable until HSDK5-014.
  const auto loan_tensor = &holoscan::ChannelPublisher<holoscan::Tensor>::loan_tensor;
  const auto emit_loan = &holoscan::ChannelPublisher<holoscan::Tensor>::emit;
  const auto emit_captured = &holoscan::ChannelPublisher<holoscan::Tensor>::emit_captured;
  const auto emit_sample = &holoscan::ChannelPublisher<holoscan::Tensor>::emit_sample;
  const auto emit_copy = &holoscan::ChannelPublisher<holoscan::Tensor>::emit_copy;
  const auto make_capture_time = &holoscan::ChannelPublisher<holoscan::Tensor>::make_capture_time;
  const auto report_clock_health =
      &holoscan::ChannelPublisher<holoscan::Tensor>::report_clock_health;
  const auto update_connections = &holoscan::ChannelPublisher<holoscan::Tensor>::update_connections;
  const auto has_subscribers = &holoscan::ChannelPublisher<holoscan::Tensor>::has_subscribers;
  const auto write = &holoscan::TensorPublishLoan::write;
  const auto write_host = &holoscan::TensorPublishLoan::write_host;
  const auto cancel = &holoscan::TensorPublishLoan::cancel;
  const auto data = &holoscan::TensorLoanWriteGuard::data;
  const auto byte_size = &holoscan::TensorLoanWriteGuard::byte_size;
  const auto stream = &holoscan::TensorLoanWriteGuard::stream;
  const auto commit = &holoscan::TensorLoanWriteGuard::commit;
  (void)loan_tensor;
  (void)emit_loan;
  (void)emit_captured;
  (void)emit_sample;
  (void)emit_copy;
  (void)make_capture_time;
  (void)report_clock_health;
  (void)update_connections;
  (void)has_subscribers;
  (void)write;
  (void)write_host;
  (void)cancel;
  (void)data;
  (void)byte_size;
  (void)stream;
  (void)commit;

  holoscan::DeploymentBinding deployment;
  // REQ: MEM-GREENCTX-014
  // REQ: COMPILE-063
  std::string green_context_partition{"installed-partition"};
  deployment.bind_cuda_green_context(
      green_context_partition,
      holoscan::CudaGreenContextPlacement{.device = holoscan::DeviceId{0}, .sm_count = 8U});
  green_context_partition.assign("mutated-after-bind");
  const auto green_context_bindings = deployment.cuda_green_contexts();
  if (green_context_bindings.size() != 1U ||
      green_context_bindings.front().partition_name != "installed-partition" ||
      green_context_bindings.front().placement.device.value != 0 ||
      green_context_bindings.front().placement.sm_count != 8U) {
    return 1;
  }
  const auto capability_query = &holoscan::query_cuda_green_context_capability;
  const auto local_capability = capability_query(holoscan::DeviceId{0});
  if (local_capability) {
    const std::uint32_t structured_sm_count =
        local_capability->max_structured_context_count * local_capability->structured_sm_quantum;
    if (structured_sm_count == 0U) {
      return 1;
    }
    InstalledGreenContextGraph green_context_graph;
    holoscan::CompileOptions green_context_options;
    green_context_options.deployment.bind_cuda_green_context("installed-partition",
                                                             holoscan::CudaGreenContextPlacement{
                                                                 .device = holoscan::DeviceId{0},
                                                                 .sm_count = structured_sm_count,
                                                             });
    const holoscan::ExecutionPlan green_context_plan =
        holoscan::compile(green_context_graph, std::move(green_context_options));
    if (!green_context_plan.ok() || green_context_plan.cuda_green_context_requests().size() != 1U ||
        green_context_plan.cuda_green_context_assignments().size() != 1U ||
        green_context_plan.cuda_green_context_pools().size() != 1U ||
        green_context_plan.cuda_green_context_resources().size() != 1U ||
        green_context_plan.cuda_compute_streams().size() != 1U ||
        green_context_plan.cuda_green_context_pools().front().structured_sm_quantum !=
            local_capability->structured_sm_quantum) {
      return 1;
    }
  }

  const holoscan::TypeKey type_key{};
  const holoscan::TypeKeyHash type_key_hash;
  if (type_key_hash(type_key) == 0U) {
    return 1;
  }

  auto catalog_builder = holoscan::ChannelEndpointCatalog::builder("installed-consumer");
  const auto added =
      catalog_builder.import_topic<std::uint32_t>("sensor", "holomq://installed-consumer/sensor");
  if (!added) {
    return 1;
  }
  auto catalog = std::move(catalog_builder).build();
  if (!catalog || !catalog->endpoint("sensor")) {
    return 1;
  }

  try {
    (void)holoscan::ExecutionPlan{}.partitions();
    return 1;
  } catch (const holoscan::RuntimeError& error) {
    if (error.code() != holoscan::ErrorCode::kInvalidArgument) {
      return 1;
    }
  }

  holoscan::Graph graph{"installed-compile"};
  const holoscan::ClockKey root_clock =
      graph.add_clock<holoscan::ManualClock>("root_clock", std::int64_t{17});
  auto child = graph.embed<InstalledClockImportGraph>("child");
  graph.bind_clock(child, child->clock_import(), root_clock);
  if (graph.view().automatic_partition_remaining()) {
    return 1;
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok() || plan.json().empty() || plan.graph().vertices().size() != 1U ||
      !plan.temporal_triggers().empty() || !plan.input_selections().empty() ||
      !plan.predicate_callables().empty() ||
      !plan.partitioning_plan().accepted_algorithm.has_value() ||
      plan.json().find(kClockHealthReasonSchema) == std::string_view::npos) {
    return 1;
  }
  const holoscan::ExecutionPartitionsView partitions = plan.partitions();
  if (partitions.mode() != holoscan::PartitioningMode::kAutomatic ||
      partitions.authored_count() != 0U || partitions.automatic_count() != 1U ||
      partitions.size() != 1U || partitions.empty()) {
    return 1;
  }
  if (std::distance(partitions.begin(), partitions.end()) != 1) {
    return 1;
  }
  std::vector<holoscan::ExecutionPartitionView> legacy_copied_partitions;
  std::copy(partitions.begin(), partitions.end(), std::back_inserter(legacy_copied_partitions));
  if (legacy_copied_partitions.size() != partitions.size()) {
    return 1;
  }
  const holoscan::ExecutionPartitionView partition = *partitions.begin();
  const auto by_id = partitions.lookup(partition.id());
  const auto by_path = partitions.lookup_for_path("child/source");
  if (!by_id || !by_path || by_id->id() != partition.id() || by_path->id() != partition.id() ||
      partition.operator_paths().size() != 1U ||
      partition.operator_paths().front() != "child/source") {
    return 1;
  }
  const holoscan::ExecutionGraphView compiled_graph = plan.graph();
  if (compiled_graph.clock_imports().size() != 1U || compiled_graph.clock_bindings().size() != 1U) {
    return 1;
  }
  const holoscan::ClockImportAuthoringRecord& imported = compiled_graph.clock_imports().front();
  const holoscan::EmbeddedClockBindingAuthoringRecord& binding =
      compiled_graph.clock_bindings().front();
  if (!imported.graph || imported.key != holoscan::ClockKey::named("timeline") ||
      !imported.is_default || !binding.parent_graph || binding.child_graph != imported.graph ||
      binding.child_clock != imported.key ||
      binding.parent_clock != holoscan::ClockKey::named("root_clock")) {
    return 1;
  }

  holoscan::Graph hybrid_graph{"installed-hybrid-view"};
  auto hybrid_source = hybrid_graph.op<InstalledCompileOperator>("source");
  hybrid_graph.partition("anchor").add(hybrid_source);
  hybrid_graph.auto_partition_remaining();
  if (!hybrid_graph.view().automatic_partition_remaining()) {
    return 1;
  }

  const std::filesystem::path artifact_root =
      std::filesystem::current_path() / "installed-compile-artifacts";
  std::error_code cleanup_error;
  std::filesystem::remove_all(artifact_root, cleanup_error);
  const auto emitted = holoscan::emit_artifacts(
      plan,
      holoscan::ArtifactEmitOptions{.root = artifact_root, .deployment_id = "installed-consumer"});
  const bool artifacts_valid =
      emitted && std::filesystem::exists(emitted->catalog_path) &&
      std::filesystem::exists(emitted->manifest_path) &&
      std::filesystem::exists(emitted->manifest_path.parent_path() / "execution-plan.json") &&
      std::filesystem::exists(emitted->manifest_path.parent_path() / "profile-manifest.json");
  std::filesystem::remove_all(artifact_root, cleanup_error);
  return artifacts_valid ? 0 : 1;
}
