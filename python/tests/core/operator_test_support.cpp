/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/register_graph.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/params/register_operator.hpp>

#include "../../holoscan/timeout.hpp"
#include "pipeline_test_operators.hpp"

namespace holoscan::python::test {

struct PythonChannelClientPod {
  std::uint32_t value{};
};

}  // namespace holoscan::python::test

namespace holoscan {

template <>
struct ChannelTypeIdentity<python::test::PythonChannelClientPod> {
  static constexpr std::string_view name = "holoscan.tests.PythonChannelClientPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<python::test::PythonChannelClientPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &python::test::PythonChannelClientPod::value,
                           offsetof(python::test::PythonChannelClientPod, value)));
};

}  // namespace holoscan

namespace holoscan::python::test {
namespace {

[[nodiscard]] nanobind::tuple tensor_client_catalogs(const std::string& endpoint_name,
                                                     const std::string& memory_kind_name) {
  const MemoryKind memory_kind = [&] {
    if (memory_kind_name == "host") {
      return MemoryKind::kHost;
    }
    if (memory_kind_name == "pinned_host") {
      return MemoryKind::kPinnedHost;
    }
    if (memory_kind_name == "cuda_device") {
      return MemoryKind::kCudaDevice;
    }
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "TEST_TENSOR_MEMORY_KIND_INVALID"}};
  }();
  const std::array<std::int64_t, 2U> max_shape{2, 4};
  const TensorBoundaryContract tensor{
      .representation =
          TensorRepresentation{
              .memory_kind = memory_kind,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .rank = 2U,
              .strides = TensorStridePolicy::kContiguousOnly,
          },
      .bounds = tensor_bounds(max_shape),
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
          },
  };
  const TopicOptions import_options{
      .tensor = tensor,
      .bounds =
          EndpointBounds{
              .publisher = PublisherEndpointBounds{.max_loaned_samples = 1U},
          },
  };
  const TopicOptions export_options{
      .tensor = tensor,
      .bounds =
          EndpointBounds{
              .subscriber = SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U},
          },
  };
  auto import_builder = ChannelEndpointCatalog::builder("python-tensor-client-import");
  auto export_builder = ChannelEndpointCatalog::builder("python-tensor-client-export");
  const std::string uri = "holomq://" + endpoint_name;
  if (auto added = import_builder.import_topic<Tensor>(endpoint_name, uri, import_options);
      !added) {
    throw RuntimeError{std::move(added).error()};
  }
  if (auto added = export_builder.export_topic<Tensor>(endpoint_name, uri, export_options);
      !added) {
    throw RuntimeError{std::move(added).error()};
  }
  auto import_catalog = std::move(import_builder).build();
  if (!import_catalog) {
    throw RuntimeError{std::move(import_catalog).error()};
  }
  auto export_catalog = std::move(export_builder).build();
  if (!export_catalog) {
    throw RuntimeError{std::move(export_catalog).error()};
  }
  return nanobind::make_tuple(import_catalog->json(), export_catalog->json());
}

[[nodiscard]] std::string pod_client_catalog(const std::string& endpoint_name) {
  auto builder = ChannelEndpointCatalog::builder("python-pod-client-import");
  const std::string uri = "holomq://" + endpoint_name;
  if (auto added = builder.import_topic<PythonChannelClientPod>(endpoint_name, uri); !added) {
    throw RuntimeError{std::move(added).error()};
  }
  auto catalog = std::move(builder).build();
  if (!catalog) {
    throw RuntimeError{std::move(catalog).error()};
  }
  return catalog->json();
}

// Provides an incompatible input for test_authoring_rejections_preserve_core_tokens.
class TensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

// Supplies healthy rows for alias-collision and reserved-name discovery tests.
// Representative test: test_operator_aliases_protect_static_and_reserved_module_names.
class DiscoveryOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

struct AttributeCollisionParams {
  std::uint32_t shared{1U};
};

HOLOSCAN_PARAM_SCHEMA(AttributeCollisionParams,
                      param_schema("holoscan.tests.AttributeCollisionParams", 1),
                      field<&AttributeCollisionParams::shared>("shared"));

// Exercises fixed-property and port-before-parameter precedence in generated provider stubs.
class AttributeCollisionOperator final : public Operator<AttributeCollisionParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(name_port, "name");
    spec.output(ports_port, "ports");
    spec.output(shared_port, "shared");
    spec.output(keyword_port, "class");
    spec.output(literal_underscore_port, "class_");
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> name_port;
  Output<std::uint32_t> ports_port;
  Output<std::uint32_t> shared_port;
  Output<std::uint32_t> keyword_port;
  Output<std::uint32_t> literal_underscore_port;
};

// Throws during setup so discovery can prove that one invalid row is isolated.
// Representative test: test_registered_operator_discovery_isolates_failing_rows.
class FailingDiscoveryOperator final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {
    throw RuntimeError{Error{ErrorCode::kFailure, "TEST_OPERATOR_DISCOVERY_FAILED"}};
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct ConfigDetectorParams {
  std::uint32_t threshold{4U};
  float gain{1.0F};
  std::uint32_t window{16U};
};

HOLOSCAN_PARAM_SCHEMA(
    ConfigDetectorParams, param_schema("holoscan.tests.ConfigDetectorParams", 1),
    field<&ConfigDetectorParams::threshold>("threshold").range(1U, 64U).alias("legacy_threshold"),
    field<&ConfigDetectorParams::gain>("gain").range(0.0F, 10.0F),
    field<&ConfigDetectorParams::window>("window").range(1U, 64U));

// Supplies the config-bound sink in test_sparse_config_patch_attaches_to_graph_with_provenance.
class ConfigDetector final : public Operator<ConfigDetectorParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "in"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

enum class ParameterProbeMode : std::uint8_t {
  kDisabled = 0,
  kAutomatic = 2,
  kManual = 7,
};

struct ParameterProbeSensor {
  std::uint32_t channels{1U};
};

HOLOSCAN_PARAM_SCHEMA(ParameterProbeSensor, param_schema("holoscan.tests.ParameterProbeSensor", 1),
                      field<&ParameterProbeSensor::channels>("channels").range(1U, 4U));

}  // namespace
}  // namespace holoscan::python::test

namespace holoscan::config {

template <>
struct ConfigCodec<python::test::ParameterProbeSensor>
    : params::RequiredParamSchemaCodec<python::test::ParameterProbeSensor> {};

}  // namespace holoscan::config

namespace holoscan::python::test {
namespace {

[[nodiscard]] bool parameter_probe_even(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

struct ParameterProbeParams {
  bool enabled{true};
  std::int32_t offset{};
  ParameterProbeMode mode{ParameterProbeMode::kAutomatic};
  double gain{1.0};
  std::string label{"default"};
  std::chrono::nanoseconds timeout{std::chrono::milliseconds{1}};
  std::uint32_t even{2U};
  std::vector<std::int32_t> levels{1};
  std::vector<std::string> labels{"default"};
  std::optional<std::uint32_t> optional_count{7U};
  std::vector<ParameterProbeMode> modes{ParameterProbeMode::kDisabled};
  ParameterProbeSensor sensor{};
  std::optional<ParameterProbeSensor> backup_sensor{};
  std::string secret{"unset"};
  std::string internal_note{"internal"};
  std::string transient_note{"transient"};
  std::filesystem::path output_path{};
};

HOLOSCAN_PARAM_SCHEMA(
    ParameterProbeParams, param_schema("holoscan.tests.ParameterProbeParams", 1),
    field<&ParameterProbeParams::enabled>("enabled"),
    field<&ParameterProbeParams::offset>("offset").range(-4, 4),
    field<&ParameterProbeParams::mode>("mode").enum_values(
        {param_enum_value("disabled", ParameterProbeMode::kDisabled),
         param_enum_value("automatic", ParameterProbeMode::kAutomatic),
         param_enum_value("manual", ParameterProbeMode::kManual)}),
    field<&ParameterProbeParams::gain>("gain").range(0.0, 2.0),
    field<&ParameterProbeParams::label>("label")
        .encoding(ParamStringEncoding::kUtf8)
        .max_utf8_bytes(12U),
    field<&ParameterProbeParams::timeout>("timeout").range(std::chrono::nanoseconds{1},
                                                           std::chrono::seconds{10}),
    field<&ParameterProbeParams::even>("even").validator<parameter_probe_even>(
        "holoscan.tests.parameter_probe.even/v1"),
    field<&ParameterProbeParams::levels>("levels").max_elements(4U),
    field<&ParameterProbeParams::labels>("labels")
        .element_encoding(ParamStringEncoding::kUtf8)
        .max_element_utf8_bytes(8U)
        .max_elements(3U),
    field<&ParameterProbeParams::optional_count>("optional_count"),
    field<&ParameterProbeParams::modes>("modes")
        .enum_values({param_enum_value("disabled", ParameterProbeMode::kDisabled),
                      param_enum_value("automatic", ParameterProbeMode::kAutomatic),
                      param_enum_value("manual", ParameterProbeMode::kManual)})
        .max_elements(4U),
    field<&ParameterProbeParams::sensor>("sensor").nested<ParameterProbeSensor>(),
    field<&ParameterProbeParams::backup_sensor>("backup_sensor").nested<ParameterProbeSensor>(),
    field<&ParameterProbeParams::secret>("secret").sensitive(),
    field<&ParameterProbeParams::internal_note>("internal_note").internal(),
    field<&ParameterProbeParams::transient_note>("transient_note").transient(),
    field<&ParameterProbeParams::output_path>("output_path"));

// Exposes all supported parameter shapes to
// test_kwargs_cover_schema_value_kinds_and_redact_sensitive_values.
class ParameterProbe final : public Operator<ParameterProbeParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

struct RequiredRegistryParams {
  std::uint32_t required_count{7U};
};

HOLOSCAN_PARAM_SCHEMA(RequiredRegistryParams,
                      param_schema("holoscan.tests.RequiredRegistryParams", 1),
                      field<&RequiredRegistryParams::required_count>("required_count")
                          .required_on_load()
                          .range(1U, 64U));

// Keeps a valid native default while requiring every language-facing construction path to supply
// the field explicitly.
class RequiredRegistrySource final : public Operator<RequiredRegistryParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

enum class RuntimeControlMode : std::uint8_t {
  kAutomatic = 0,
  kManual = 1,
};

struct RuntimeControlSensor {
  std::uint32_t channels{1U};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeControlSensor, param_schema("holoscan.tests.RuntimeControlSensor", 1),
                      field<&RuntimeControlSensor::channels>("channels").range(1U, 4U));

struct RuntimeControlParams {
  std::int32_t scalar{1};
  std::chrono::nanoseconds interval{std::chrono::milliseconds{1}};
  std::vector<std::int32_t> runtime_levels{1};
  std::optional<std::uint32_t> runtime_optional_count{};
  RuntimeControlSensor runtime_sensor{};
  std::string runtime_label{"initial"};
  std::vector<std::int32_t> levels{1};
  std::optional<std::uint32_t> optional_count{};
  RuntimeControlMode mode{RuntimeControlMode::kAutomatic};
  std::vector<RuntimeControlMode> modes{RuntimeControlMode::kAutomatic};
  RuntimeControlSensor sensor{};
  std::optional<RuntimeControlSensor> backup_sensor{};
  std::string label{"initial"};
  std::filesystem::path output_path{};
};

HOLOSCAN_PARAM_SCHEMA(RuntimeControlParams, param_schema("holoscan.tests.RuntimeControlParams", 1),
                      field<&RuntimeControlParams::scalar>("scalar")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .range(-10, 10),
                      field<&RuntimeControlParams::interval>("interval")
                          .runtime_mutable()
                          .range(std::chrono::microseconds{1}, std::chrono::seconds{1}),
                      field<&RuntimeControlParams::runtime_levels>("runtime_levels")
                          .runtime_mutable()
                          .max_elements(4U),
                      field<&RuntimeControlParams::runtime_optional_count>("runtime_optional_count")
                          .runtime_mutable(),
                      field<&RuntimeControlParams::runtime_sensor>("runtime_sensor")
                          .runtime_mutable()
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::runtime_label>("runtime_label")
                          .runtime_mutable()
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16U),
                      field<&RuntimeControlParams::levels>("levels")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .max_elements(4U),
                      field<&RuntimeControlParams::optional_count>("optional_count")
                          .lifecycle_mutable(LifecycleStage::kConfigure),
                      field<&RuntimeControlParams::mode>("mode").runtime_mutable().enum_values(
                          {param_enum_value("automatic", RuntimeControlMode::kAutomatic),
                           param_enum_value("manual", RuntimeControlMode::kManual)}),
                      field<&RuntimeControlParams::modes>("modes")
                          .runtime_mutable()
                          .enum_values({param_enum_value("automatic",
                                                         RuntimeControlMode::kAutomatic),
                                        param_enum_value("manual", RuntimeControlMode::kManual)})
                          .max_elements(4U),
                      field<&RuntimeControlParams::sensor>("sensor")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::backup_sensor>("backup_sensor")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .nested<RuntimeControlSensor>(),
                      field<&RuntimeControlParams::label>("label")
                          .lifecycle_mutable(LifecycleStage::kConfigure)
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16U),
                      field<&RuntimeControlParams::output_path>("output_path").runtime_mutable());

// Records fixed summaries of mixed C++ parameter generations without allocating in compute.
// Representative test: test_live_parameter_and_lifecycle_control_of_registered_cpp_operator.
struct RuntimeControlObservation {
  static void reset() noexcept {
    activation_count.store(0U, std::memory_order_relaxed);
    generation.store(0U, std::memory_order_relaxed);
    scalar.store(0, std::memory_order_relaxed);
    interval_ns.store(0, std::memory_order_relaxed);
    level_count.store(0U, std::memory_order_relaxed);
    level_sum.store(0, std::memory_order_relaxed);
    optional_count.store(-1, std::memory_order_relaxed);
    mode.store(0U, std::memory_order_relaxed);
    mode_count.store(0U, std::memory_order_relaxed);
    sensor_channels.store(0U, std::memory_order_relaxed);
    backup_channels.store(-1, std::memory_order_relaxed);
    label_is_live.store(false, std::memory_order_relaxed);
    start_count.store(0U, std::memory_order_relaxed);
  }

  static void record(const RuntimeControlParams& params, ParamGeneration observed_generation) {
    std::int32_t observed_level_sum{};
    for (const std::int32_t level : params.levels) {
      observed_level_sum += level;
    }
    {
      const std::lock_guard lock{mutex};
      scalar.store(params.scalar, std::memory_order_relaxed);
      interval_ns.store(params.interval.count(), std::memory_order_relaxed);
      level_count.store(params.levels.size(), std::memory_order_relaxed);
      level_sum.store(observed_level_sum, std::memory_order_relaxed);
      optional_count.store(
          params.optional_count ? static_cast<std::int64_t>(*params.optional_count) : -1,
          std::memory_order_relaxed);
      mode.store(static_cast<std::uint8_t>(params.mode), std::memory_order_relaxed);
      mode_count.store(params.modes.size(), std::memory_order_relaxed);
      sensor_channels.store(params.sensor.channels, std::memory_order_relaxed);
      backup_channels.store(
          params.backup_sensor ? static_cast<std::int64_t>(params.backup_sensor->channels) : -1,
          std::memory_order_relaxed);
      label_is_live.store(params.label == "live", std::memory_order_relaxed);
      activation_count.fetch_add(1U, std::memory_order_relaxed);
      generation.store(observed_generation, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_generation(ParamGeneration expected,
                                                std::chrono::nanoseconds timeout) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, timeout, [expected] {
      return activation_count.load(std::memory_order_relaxed) != 0U &&
             generation.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint64_t> activation_count{};
  static inline std::atomic<ParamGeneration> generation{};
  static inline std::atomic<std::int32_t> scalar{};
  static inline std::atomic<std::int64_t> interval_ns{};
  static inline std::atomic<std::size_t> level_count{};
  static inline std::atomic<std::int32_t> level_sum{};
  static inline std::atomic<std::int64_t> optional_count{-1};
  static inline std::atomic<std::uint8_t> mode{};
  static inline std::atomic<std::size_t> mode_count{};
  static inline std::atomic<std::uint32_t> sensor_channels{};
  static inline std::atomic<std::int64_t> backup_channels{-1};
  static inline std::atomic_bool label_is_live{};
  static inline std::atomic<std::uint32_t> start_count{};
};

// Provides a clock-driven registered C++ target for Python runtime-control integration tests.
// Representative test: test_live_parameter_and_lifecycle_control_of_registered_cpp_operator.
class RuntimeControlProbe final : public Operator<RuntimeControlParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kStart, &RuntimeControlProbe::on_start);
  }

  [[nodiscard]] Contract contract() const override {
    using namespace std::chrono_literals;
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RuntimeControlObservation::record(context.params<RuntimeControlParams>(),
                                      context.params_generation());
    return {};
  }

  LifecycleStatus on_start(LifecycleContext&) noexcept {
    RuntimeControlObservation::start_count.fetch_add(1U, std::memory_order_relaxed);
    return LifecycleStatus::kOk;
  }
};

struct ManualRowSourceParams {
  std::uint32_t limit{4U};
};

HOLOSCAN_PARAM_SCHEMA(ManualRowSourceParams,
                      param_schema("holoscan.tests.ManualRowSourceParams", 1),
                      field<&ManualRowSourceParams::limit>("limit").range(1U, 1'000'000U));

// Exercises manual registration of a non-default-constructible operator in
// test_manual_rows_accept_python_keyword_parameters.
class ManualRowSource final : public Operator<> {
 public:
  explicit ManualRowSource(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;

 private:
  std::uint32_t limit_{};
};

struct ManualRowSourceThunks final : authoring::ManualOperatorRegistrationThunks<ManualRowSource> {
  static authoring::OpaqueOperatorRef add_to_graph(
      Graph& graph, std::string_view name, const authoring::MaterializedParameters* params) {
    const ManualRowSourceParams defaults{};
    if (params != nullptr &&
        (params->type() == nullptr ||
         params->type()->local_type_token !=
             param_type_descriptor<ManualRowSourceParams>().local_type_token)) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    const auto* values =
        params == nullptr ? &defaults : static_cast<const ManualRowSourceParams*>(params->data());
    return make_operator_ref(graph.op<ManualRowSource>(name, values->limit));
  }

  static void set_params(const authoring::OpaqueOperatorRef&,
                         const authoring::MaterializedParameters&) {
    throw RuntimeError{
        Error{ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }
};

expected<authoring::OpaqueOperatorRef, Error> add_manual_row_from_assignments(
    Graph& graph, std::string_view name,
    std::span<const authoring::ParameterAssignment> assignments) {
  auto materialized =
      authoring::materialize_parameters(param_type_descriptor<ManualRowSourceParams>(),
                                        param_schema_descriptor<ManualRowSourceParams>(),
                                        assignments);
  if (!materialized) {
    return make_unexpected(std::move(materialized).error());
  }
  return ManualRowSourceThunks::invoke_add(
      [&] { return ManualRowSourceThunks::add_to_graph(graph, name, &*materialized); });
}

authoring::RegisteredOperatorType make_manual_row(std::string_view name, bool expose_descriptor) {
  using Thunks = ManualRowSourceThunks;
  return authoring::RegisteredOperatorType{
      .name = name,
      .param_type = &param_type_descriptor<ManualRowSourceParams>(),
      .param_schema =
          expose_descriptor ? param_schema_descriptor<ManualRowSourceParams>() : nullptr,
      .parameter_introspection = expose_descriptor
                                     ? authoring::OperatorParameterIntrospection::kConstructorReplay
                                     : authoring::OperatorParameterIntrospection::kPassThrough,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &add_manual_row_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  };
}

template <std::derived_from<OperatorBase> Op>
authoring::RegisteredOperatorType make_configurable_operator(std::string_view name) {
  return authoring::detail::make_registered_operator<Op>(
      name, &authoring::detail::OperatorConfigBindingThunk<Op>::bind);
}

template <std::derived_from<Graph> G>
void ensure_test_graph_registered(std::string_view name) {
  auto& registry = authoring::authoring_registry();
  if (registry.graph_type(name)) {
    return;
  }
  auto registered = registry.register_graph(authoring::detail::make_registered_graph<G>(name));
  if (!registered) {
    throw RuntimeError{registered.error()};
  }
}

template <std::derived_from<OperatorBase> Op>
void ensure_test_operator_registered(std::string_view name) {
  auto& registry = authoring::authoring_registry();
  if (registry.operator_type(name)) {
    return;
  }
  auto registered =
      registry.register_operator(authoring::detail::make_registered_operator<Op>(name));
  if (!registered) {
    throw RuntimeError{registered.error()};
  }
}

void register_conditional_graph_for_test() {
  ensure_test_graph_registered<ConditionalInterfaceGraph>(
      "holoscan.tests.ConditionalInterfaceGraph");
}

void register_failing_graph_for_test() {
  ensure_test_graph_registered<FailingDiscoveryGraph>("holoscan.tests.FailingDiscoveryGraph");
}

void register_scale_graph_alias_a_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.tests.a.CollisionGraph");
}

void register_scale_graph_alias_b_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.tests.b.CollisionGraph");
}

void register_reserved_graph_alias_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.tests.GraphType");
}

void register_helper_graph_alias_for_test() {
  ensure_test_graph_registered<ScalePairGraph>("holoscan.tests._aliases");
}

void register_failing_operator_for_test() {
  ensure_test_operator_registered<FailingDiscoveryOperator>(
      "holoscan.tests.FailingDiscoveryOperator");
}

void register_operator_aliases_for_test() {
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.tests.a.CollisionOperator");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.tests.b.CollisionOperator");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.tests.OperatorHandle");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.tests.OperatorType");
  ensure_test_operator_registered<DiscoveryOperator>("holoscan.tests.class");
}

void register_runtime_control_probe_for_test() {
  ensure_test_operator_registered<RuntimeControlProbe>("holoscan.tests.RuntimeControlProbe");
}

bool wait_for_runtime_control_generation(std::uint64_t generation, double timeout_seconds) {
  const auto timeout = detail::seconds_to_timeout(timeout_seconds);
  const nanobind::gil_scoped_release release;
  return RuntimeControlObservation::wait_for_generation(generation, timeout);
}

nanobind::dict runtime_control_observation() {
  nanobind::dict result;
  result["activation_count"] =
      RuntimeControlObservation::activation_count.load(std::memory_order_relaxed);
  result["generation"] = RuntimeControlObservation::generation.load(std::memory_order_acquire);
  result["scalar"] = RuntimeControlObservation::scalar.load(std::memory_order_relaxed);
  result["interval_ns"] = RuntimeControlObservation::interval_ns.load(std::memory_order_relaxed);
  result["level_count"] = RuntimeControlObservation::level_count.load(std::memory_order_relaxed);
  result["level_sum"] = RuntimeControlObservation::level_sum.load(std::memory_order_relaxed);
  result["optional_count"] =
      RuntimeControlObservation::optional_count.load(std::memory_order_relaxed);
  result["mode"] = RuntimeControlObservation::mode.load(std::memory_order_relaxed);
  result["mode_count"] = RuntimeControlObservation::mode_count.load(std::memory_order_relaxed);
  result["sensor_channels"] =
      RuntimeControlObservation::sensor_channels.load(std::memory_order_relaxed);
  result["backup_channels"] =
      RuntimeControlObservation::backup_channels.load(std::memory_order_relaxed);
  result["label_is_live"] =
      RuntimeControlObservation::label_is_live.load(std::memory_order_relaxed);
  result["start_count"] = RuntimeControlObservation::start_count.load(std::memory_order_relaxed);
  return result;
}

expected<void, Error> register_test_operator_catalog() {
  try {
    auto& registry = authoring::authoring_registry();
    const auto register_row = [&](authoring::RegisteredOperatorType row) -> expected<void, Error> {
      auto registered = registry.register_operator(std::move(row));
      if (!registered) {
        return make_unexpected(std::move(registered).error());
      }
      return {};
    };
    const auto register_graph = [&](authoring::RegisteredGraphType row) -> expected<void, Error> {
      auto registered = registry.register_graph(std::move(row));
      if (!registered) {
        return make_unexpected(std::move(registered).error());
      }
      return {};
    };

    if (auto result = register_row(authoring::detail::make_registered_operator<CounterSource>(
            "holoscan.tests.CounterSource"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<AttributeCollisionOperator>(
                "holoscan.tests.AttributeCollisionOperator"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<DoubleIt>("holoscan.tests.DoubleIt"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<TopicCameraSource>(
            "holoscan.tests.TopicCameraSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<TopicWindowFusion>(
            "holoscan.tests.TopicWindowFusion"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<Sink>("holoscan.tests.Sink"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<TickSource>("holoscan.tests.TickSource"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<SparseTimestampedSource>(
                "holoscan.tests.SparseTimestampedSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(authoring::detail::make_registered_operator<CompletingSink>(
            "holoscan.tests.CompletingSink"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            authoring::detail::make_registered_operator<TensorSink>("holoscan.tests.TensorSink"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<StructuredFailureSource>(
                "holoscan.tests.StructuredFailureSource"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            make_configurable_operator<ConfigDetector>("holoscan.tests.ConfigDetector"));
        !result) {
      return result;
    }
    if (auto result = register_row(
            make_configurable_operator<ParameterProbe>("holoscan.tests.ParameterProbe"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(authoring::detail::make_registered_operator<RequiredRegistrySource>(
                "holoscan.tests.RequiredRegistrySource"));
        !result) {
      return result;
    }
    if (auto result =
            register_row(make_manual_row("holoscan.tests.ManualRowSourceWithDescriptor", true));
        !result) {
      return result;
    }
    if (auto result =
            register_row(make_manual_row("holoscan.tests.ManualRowSourceWithoutDescriptor", false));
        !result) {
      return result;
    }
    if (auto result = register_graph(authoring::detail::make_registered_graph<ScalePairGraph>(
            "holoscan.tests.ScalePairGraph"));
        !result) {
      return result;
    }
    return register_graph(authoring::detail::make_registered_graph<WrappedScaleGraph>(
        "holoscan.tests.WrappedScaleGraph"));
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "AUTHORING_TEST_CATALOG_REGISTRATION_FAILED"});
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "AUTHORING_TEST_CATALOG_REGISTRATION_FAILED"});
  }
}

// Supplies every supplemental field to test_diagnostic_detail_is_typed_read_only_and_complete.
Diagnostic structured_diagnostic_fixture() {
  return Diagnostic{
      .id = DiagnosticId{19U},
      .severity = DiagnosticSeverity::kError,
      .token = "TEST_STRUCTURED_DIAGNOSTIC",
      .source = DiagnosticSource{.graph_generation = GraphGeneration{7U},
                                 .graph = GraphId{2U},
                                 .vertex = VertexId{3U},
                                 .port = PortId{4U}},
      .cause_id = DiagnosticId{5U},
      .detail =
          DiagnosticDetail{
              .operation = DiagnosticOperation::kSetup,
              .error = Error{ErrorCode::kInvalidArgument, "primary fixture cause"},
              .stage = LifecycleStage::kWarmUp,
              .phase = LifecycleTransitionPhase::kRestartForward,
              .transaction_id = 41U,
              .role = DiagnosticFailureRole::kRollback,
              .operator_path = "source",
              .port_name = "out",
              .related_source = DiagnosticSource{.graph_generation = GraphGeneration{7U},
                                                 .graph = GraphId{2U},
                                                 .vertex = VertexId{6U},
                                                 .port = PortId{8U}},
              .related_operator_path = "sink",
              .related_port_name = "in",
              .mismatches = {DiagnosticMismatch{
                  .field = "dtype", .provided = "float32", .required = "uint8"}},
              .causes = {Error{ErrorCode::kNotFound, "nested fixture cause"}},
              .primary_diagnostic_index = 2U,
              .truncated = true,
          },
  };
}

}  // namespace
}  // namespace holoscan::python::test

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(holoscan_test_support, module) {
  module.doc() = "Private Operator and reusable-Graph support for Holoscan Python tests.";
  auto registered = holoscan::python::test::register_test_operator_catalog();
  if (!registered) {
    throw holoscan::RuntimeError{registered.error()};
  }
  module.def("_reset_pipeline_observation", &holoscan::python::test::PipelineObservation::reset);
  module.def(
      "_wait_for_pipeline_observation",
      [](std::size_t expected, double timeout) {
        const auto duration = holoscan::python::detail::seconds_to_timeout(timeout);
        const nanobind::gil_scoped_release release;
        return holoscan::python::test::PipelineObservation::wait_for(expected, duration);
      },
      nanobind::arg("expected"),
      nanobind::arg("timeout"));
  module.def("_pipeline_observation_count", &holoscan::python::test::PipelineObservation::count);
  module.def("_pipeline_observation_last_value",
             &holoscan::python::test::PipelineObservation::last_value);
  module.def("_structured_diagnostic_fixture", [] {
    return nanobind::cast(holoscan::python::test::structured_diagnostic_fixture());
  });
  module.def("_tensor_client_catalogs",
             &holoscan::python::test::tensor_client_catalogs,
             nanobind::arg("endpoint_name"),
             nanobind::arg("memory_kind"));
  module.def("_pod_client_catalog",
             &holoscan::python::test::pod_client_catalog,
             nanobind::arg("endpoint_name"));
  module.def("_register_conditional_graph_for_test",
             &holoscan::python::test::register_conditional_graph_for_test);
  module.def("_register_failing_graph_for_test",
             &holoscan::python::test::register_failing_graph_for_test);
  module.def("_register_scale_graph_alias_a_for_test",
             &holoscan::python::test::register_scale_graph_alias_a_for_test);
  module.def("_register_scale_graph_alias_b_for_test",
             &holoscan::python::test::register_scale_graph_alias_b_for_test);
  module.def("_register_reserved_graph_alias_for_test",
             &holoscan::python::test::register_reserved_graph_alias_for_test);
  module.def("_register_helper_graph_alias_for_test",
             &holoscan::python::test::register_helper_graph_alias_for_test);
  module.def("_register_failing_operator_for_test",
             &holoscan::python::test::register_failing_operator_for_test);
  module.def("_register_operator_aliases_for_test",
             &holoscan::python::test::register_operator_aliases_for_test);
  module.def("_register_runtime_control_probe_for_test",
             &holoscan::python::test::register_runtime_control_probe_for_test);
  module.def("_reset_runtime_control_observation",
             &holoscan::python::test::RuntimeControlObservation::reset);
  module.def("_wait_for_runtime_control_generation",
             &holoscan::python::test::wait_for_runtime_control_generation,
             nanobind::arg("generation"),
             nanobind::arg("timeout"));
  module.def("_runtime_control_observation", &holoscan::python::test::runtime_control_observation);
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
