/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plan.hpp"

#include <fmt/format.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/rate.hpp>

#include "graph.hpp"
#include "module_name.hpp"
#include "plan_pydoc.hpp"
#include "python_path.hpp"

namespace holoscan {
namespace {

// Owns endpoint schema identity fields that are borrowed by the C++ catalog view.
class PyEndpointSchemaIdentity {
 public:
  explicit PyEndpointSchemaIdentity(SchemaIdentityView identity)
      : canonical_name_(identity.canonical_name()),
        compatibility_epoch_(identity.compatibility_epoch()),
        family_id_(identity.family_id()),
        file_identifier_(identity.file_identifier()) {}

  [[nodiscard]] const std::string& canonical_name() const noexcept { return canonical_name_; }
  [[nodiscard]] std::uint32_t compatibility_epoch() const noexcept { return compatibility_epoch_; }
  [[nodiscard]] nb::bytes family_id() const {
    return nb::bytes(reinterpret_cast<const char*>(family_id_.value.data()),
                     family_id_.value.size());
  }
  [[nodiscard]] nb::bytes file_identifier() const {
    return nb::bytes(file_identifier_.value.data(), file_identifier_.value.size());
  }

 private:
  std::string canonical_name_;
  std::uint32_t compatibility_epoch_{};
  SchemaId family_id_{};
  FlatBufferFileIdentifier file_identifier_{};
};

// Owns the Tensor boundary facts borrowed by an endpoint catalog view.
class PyTensorBoundaryContractView {
 public:
  explicit PyTensorBoundaryContractView(TensorBoundaryContractView view)
      : memory_kind_(view.memory_kind()),
        dtype_(view.dtype()),
        rank_(view.rank()),
        max_rank_(view.max_rank()),
        max_byte_span_(view.max_byte_span()),
        axes_(view.axes().begin(), view.axes().end()),
        stride_policy_(view.stride_policy()),
        foreign_emit_policy_(view.foreign_emit_policy()),
        max_inline_bytes_(view.max_inline_bytes()),
        max_staging_byte_span_(view.max_staging_byte_span()) {}

  [[nodiscard]] std::optional<MemoryKind> memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] std::optional<DLDataType> dtype() const noexcept { return dtype_; }
  [[nodiscard]] std::optional<std::uint32_t> rank() const noexcept { return rank_; }
  [[nodiscard]] std::optional<std::uint32_t> max_rank() const noexcept { return max_rank_; }
  [[nodiscard]] std::optional<std::uint64_t> max_byte_span() const noexcept {
    return max_byte_span_;
  }
  [[nodiscard]] nb::tuple axes() const {
    nb::tuple_builder result(axes_.size());
    for (const TensorAxisBounds& axis : axes_) {
      result.put(nb::make_tuple(axis.max_extent, axis.max_abs_stride_bytes));
    }
    return result.commit();
  }
  [[nodiscard]] TensorStridePolicy stride_policy() const noexcept { return stride_policy_; }
  [[nodiscard]] InterfaceTensorEmitPolicy foreign_emit_policy() const noexcept {
    return foreign_emit_policy_;
  }
  [[nodiscard]] std::optional<std::uint32_t> max_inline_bytes() const noexcept {
    return max_inline_bytes_;
  }
  [[nodiscard]] std::optional<std::uint64_t> max_staging_byte_span() const noexcept {
    return max_staging_byte_span_;
  }

 private:
  std::optional<MemoryKind> memory_kind_;
  std::optional<DLDataType> dtype_;
  std::optional<std::uint32_t> rank_;
  std::optional<std::uint32_t> max_rank_;
  std::optional<std::uint64_t> max_byte_span_;
  std::vector<TensorAxisBounds> axes_;
  TensorStridePolicy stride_policy_{TensorStridePolicy::kContiguousOnly};
  InterfaceTensorEmitPolicy foreign_emit_policy_{InterfaceTensorEmitPolicy::kRequireExportable};
  std::optional<std::uint32_t> max_inline_bytes_;
  std::optional<std::uint64_t> max_staging_byte_span_;
};

// Retains the catalog so endpoint inspection remains valid independently of the source Plan object.
class PyEndpointView {
 public:
  PyEndpointView(ChannelEndpointCatalog catalog, std::string endpoint_name)
      : catalog_(std::move(catalog)), endpoint_name_(std::move(endpoint_name)) {}

  [[nodiscard]] std::string name() const { return std::string{view().name()}; }
  [[nodiscard]] std::string uri() const { return std::string{view().uri()}; }
  [[nodiscard]] std::string service() const { return std::string{view().service()}; }
  [[nodiscard]] TopicDirection direction() const { return view().direction(); }
  [[nodiscard]] EndpointNameOrigin name_origin() const { return view().name_origin(); }
  [[nodiscard]] std::optional<std::string> scalar_identity() const {
    const auto scalar = view().scalar_identity();
    if (!scalar) {
      return std::nullopt;
    }
    switch (*scalar) {
      case ScalarIdentityKind::kBoolean:
        return "boolean";
      case ScalarIdentityKind::kUInt8:
        return "uint8";
      case ScalarIdentityKind::kInt8:
        return "int8";
      case ScalarIdentityKind::kUInt16:
        return "uint16";
      case ScalarIdentityKind::kInt16:
        return "int16";
      case ScalarIdentityKind::kUInt32:
        return "uint32";
      case ScalarIdentityKind::kInt32:
        return "int32";
      case ScalarIdentityKind::kUInt64:
        return "uint64";
      case ScalarIdentityKind::kInt64:
        return "int64";
      case ScalarIdentityKind::kFloat32:
        return "float32";
      case ScalarIdentityKind::kFloat64:
        return "float64";
      case ScalarIdentityKind::kLocalEnum:
        return "local_enum";
    }
    return std::nullopt;
  }
  [[nodiscard]] std::optional<PyEndpointSchemaIdentity> schema_identity() const {
    const auto identity = view().schema_identity();
    return identity ? std::optional{PyEndpointSchemaIdentity{*identity}} : std::nullopt;
  }
  [[nodiscard]] std::optional<std::uint64_t> schema_writer_max_serialized_bytes() const {
    return view().schema_writer_max_serialized_bytes();
  }
  [[nodiscard]] std::optional<std::uint64_t> schema_required_frame_bytes() const {
    return view().schema_required_frame_bytes();
  }
  [[nodiscard]] std::optional<std::uint64_t> schema_selected_inline_bucket_bytes() const {
    return view().schema_selected_inline_bucket_bytes();
  }
  [[nodiscard]] std::optional<std::uint64_t> schema_bucket_usable_serialized_bytes() const {
    return view().schema_bucket_usable_serialized_bytes();
  }
  [[nodiscard]] std::optional<std::uint32_t> schema_transport_profile_tag() const {
    return view().schema_transport_profile_tag();
  }
  [[nodiscard]] TopicAttachPolicy attach_policy() const { return view().attach_policy(); }
  [[nodiscard]] EndpointBounds bounds() const { return view().bounds(); }
  [[nodiscard]] MemoryConversionPolicy memory_conversion() const {
    return view().memory_conversion();
  }
  [[nodiscard]] ClockHealthOptions clock_health() const {
    const auto health = view().clock_health();
    return ClockHealthOptions{.capability = health.capability(),
                              .requirement = health.requirement(),
                              .max_report_age = health.max_report_age()};
  }
  [[nodiscard]] std::optional<std::string> capture_clock() const {
    const auto capture = view().capture_time();
    return capture ? std::optional{std::string{capture->clock().name()}} : std::nullopt;
  }
  [[nodiscard]] bool has_tensor_requirements() const noexcept {
    return view().tensor_requirements().has_value();
  }
  [[nodiscard]] std::optional<PyTensorBoundaryContractView> tensor_requirements() const {
    const auto requirements = view().tensor_requirements();
    return requirements ? std::optional{PyTensorBoundaryContractView{*requirements}} : std::nullopt;
  }
  [[nodiscard]] bool has_pod_identity() const noexcept {
    return view().pod_type_identity().has_value();
  }

 private:
  [[nodiscard]] ChannelEndpointCatalog::EndpointView view() const noexcept {
    auto endpoint = catalog_.endpoint(endpoint_name_);
    return endpoint ? *endpoint : ChannelEndpointCatalog::EndpointView{};
  }

  ChannelEndpointCatalog catalog_;
  std::string endpoint_name_;
};

// Copy immutable Core diagnostics into Python-owned sequence storage, optionally by severity.
std::vector<Diagnostic> copy_diagnostics(
    const ExecutionPlan& plan, std::optional<DiagnosticSeverity> severity = std::nullopt) {
  std::vector<Diagnostic> result;
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    if (!severity || diagnostic.severity == *severity) {
      result.push_back(diagnostic);
    }
  }
  return result;
}

// Delegate whole-report ordering, causal indexes, escaping, and the size bound to Core.
std::string plan_summary(const ExecutionPlan& plan) {
  return format_diagnostics(plan.diagnostics(), plan.graph());
}

std::string plan_name(const ExecutionPlan& plan) {
  const auto graphs = plan.graph().graphs();
  return graphs.empty() ? std::string{} : graphs.front().name;
}

std::string plan_repr(const ExecutionPlan& plan) {
  std::size_t errors{};
  std::size_t warnings{};
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    if (diagnostic.severity == DiagnosticSeverity::kError) {
      ++errors;
    } else {
      ++warnings;
    }
  }
  std::ostringstream output;
  output << "<holoscan.Plan '" << plan_name(plan) << "': " << (plan.ok() ? "ACCEPTED" : "REJECTED")
         << ", " << errors << " error";
  if (errors != 1U) {
    output << 's';
  }
  output << ", " << warnings << " warning";
  if (warnings != 1U) {
    output << 's';
  }
  output << '>';
  return output.str();
}

void save_endpoint_catalog(const ChannelEndpointCatalog& catalog, nb::handle path) {
  const std::filesystem::path destination = python_filesystem_path(path);
  auto saved = [&catalog, &destination]() {
    const nb::gil_scoped_release release;
    return catalog.save_json(destination);
  }();
  if (!saved) {
    throw RuntimeError{saved.error()};
  }
}

ChannelEndpointCatalog load_endpoint_catalog(nb::handle path) {
  const std::filesystem::path source = python_filesystem_path(path);
  auto loaded = [&source]() {
    const nb::gil_scoped_release release;
    return ChannelEndpointCatalog::load_json(source);
  }();
  if (!loaded) {
    throw RuntimeError{std::move(loaded).error()};
  }
  return std::move(*loaded);
}

ChannelEndpointCatalog parse_endpoint_catalog(std::string source) {
  auto parsed = [&source] {
    const nb::gil_scoped_release release;
    return ChannelEndpointCatalog::parse_json(source);
  }();
  if (!parsed) {
    throw RuntimeError{std::move(parsed).error()};
  }
  return std::move(*parsed);
}

PyEndpointView endpoint_view(const ChannelEndpointCatalog& catalog, std::string endpoint_name) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    throw RuntimeError{std::move(endpoint).error()};
  }
  return PyEndpointView{catalog, std::move(endpoint_name)};
}

[[noreturn]] void throw_deployment_value_error(std::string_view token) {
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::string{token}}};
}

std::uint32_t cpu_core_id(nb::handle value) {
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_deployment_value_error("CPU_AFFINITY_ELEMENT_INVALID");
  }
  const unsigned long converted = PyLong_AsUnsignedLong(value.ptr());
  if (PyErr_Occurred() != nullptr || converted > std::numeric_limits<std::uint32_t>::max()) {
    PyErr_Clear();
    throw_deployment_value_error("CPU_AFFINITY_ELEMENT_INVALID");
  }
  return static_cast<std::uint32_t>(converted);
}

// Normalize Python's mixed core/range CPU-affinity syntax into one validated Core value.
CPUAffinity make_cpu_affinity(nb::handle values) {
  if (values.is_none()) {
    return {};
  }
  if (nb::isinstance<nb::str>(values) || nb::isinstance<nb::bytes>(values)) {
    throw_deployment_value_error("CPU_AFFINITY_ITERABLE_REQUIRED");
  }
  std::vector<std::uint32_t> cores;
  std::vector<CPUCoreRange> ranges;
  nb::iterable iterable;
  try {
    iterable = nb::cast<nb::iterable>(values);
  } catch (const nb::cast_error&) {
    throw_deployment_value_error("CPU_AFFINITY_ITERABLE_REQUIRED");
  }
  for (const nb::handle item : iterable) {
    if (nb::isinstance<nb::int_>(item) && !nb::isinstance<nb::bool_>(item)) {
      cores.push_back(cpu_core_id(item));
      continue;
    }
    if (!nb::isinstance<nb::tuple>(item)) {
      throw_deployment_value_error("CPU_AFFINITY_ELEMENT_INVALID");
    }
    const nb::tuple pair = nb::cast<nb::tuple>(item);
    if (pair.size() != 2U) {
      throw_deployment_value_error("CPU_AFFINITY_RANGE_INVALID");
    }
    ranges.push_back(CPUCoreRange{cpu_core_id(pair[0]), cpu_core_id(pair[1])});
  }
  auto affinity = CPUAffinity::from_cores_and_ranges(cores, ranges);
  if (!affinity) {
    throw RuntimeError{affinity.error()};
  }
  return std::move(*affinity);
}

CPUAffinity make_cpu_affinity_from_ranges(nb::handle values) {
  if (values.is_none() || nb::isinstance<nb::str>(values) || nb::isinstance<nb::bytes>(values)) {
    throw_deployment_value_error("CPU_AFFINITY_ITERABLE_REQUIRED");
  }
  nb::iterable iterable;
  try {
    iterable = nb::cast<nb::iterable>(values);
  } catch (const nb::cast_error&) {
    throw_deployment_value_error("CPU_AFFINITY_ITERABLE_REQUIRED");
  }
  std::vector<CPUCoreRange> ranges;
  for (const nb::handle item : iterable) {
    if (!nb::isinstance<nb::tuple>(item)) {
      throw_deployment_value_error("CPU_AFFINITY_RANGE_INVALID");
    }
    const nb::tuple pair = nb::cast<nb::tuple>(item);
    if (pair.size() != 2U) {
      throw_deployment_value_error("CPU_AFFINITY_RANGE_INVALID");
    }
    ranges.push_back(CPUCoreRange{cpu_core_id(pair[0]), cpu_core_id(pair[1])});
  }
  auto affinity = CPUAffinity::from_ranges(ranges);
  if (!affinity) {
    throw RuntimeError{affinity.error()};
  }
  return std::move(*affinity);
}

nb::tuple cpu_affinity_cores(const CPUAffinity& affinity) {
  nb::tuple_builder builder(affinity.cores.size());
  for (const std::uint32_t core : affinity.cores) {
    builder.put(core);
  }
  return builder.commit();
}

ThreadPlacement make_thread_placement(std::optional<CPUAffinity> affinity, SchedulingPolicy policy,
                                      std::optional<int> priority,
                                      std::optional<DeadlineScheduling> deadline,
                                      WakeMode wake_mode, IRQIsolation irq_isolation,
                                      ProbeFailurePolicy probe_failure_policy) {
  return ThreadPlacement{
      .affinity = std::move(affinity),
      .scheduling =
          ThreadScheduling{.policy = policy, .priority = priority, .deadline = std::move(deadline)},
      .wake_mode = wake_mode,
      .irq_isolation = irq_isolation,
      .probe_failure_policy = probe_failure_policy};
}

CudaGreenContextCapability cuda_green_context_capability(DeviceId device) {
  auto result = query_cuda_green_context_capability(device);
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return std::move(*result);
}

std::optional<std::uint32_t> cause_id(const Diagnostic& diagnostic) {
  if (!diagnostic.cause_id) {
    return std::nullopt;
  }
  return diagnostic.cause_id->value;
}

ExecutionPartitionView partition_lookup_named(const ExecutionPartitionsView& partitions,
                                              const std::string& name) {
  auto result = partitions.lookup_named(name);
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return std::move(*result);
}

ExecutionPartitionView partition_for_operator(const ExecutionPartitionsView& partitions,
                                              nb::handle op) {
  return python_operator_partition(partitions, op);
}

nb::object partition_iterator(const ExecutionPartitionsView& partitions) {
  return nb::make_iterator(
      nb::type<ExecutionPartitionsView>(), "iterator", partitions.begin(), partitions.end());
}

CompileOptions make_compile_options(
    DeploymentBinding deployment, Profile profile, bool auto_insert_converters,
    ParameterCompileOptions parameters,
    std::optional<LifecycleTransactionOptions> lifecycle_transactions,
    AutomaticQueueSizingOptions automatic_queue_sizing, CudaExecutionCompileOptions cuda_execution,
    OpaqueObjectPayloadPolicy opaque_object_payloads) {
  CompileOptions options;
  options.deployment = std::move(deployment);
  options.profile = profile;
  options.auto_insert_converters = auto_insert_converters;
  options.parameters = std::move(parameters);
  options.lifecycle_transactions = std::move(lifecycle_transactions);
  options.automatic_queue_sizing = automatic_queue_sizing;
  options.cuda_execution = cuda_execution;
  options.opaque_object_payloads = opaque_object_payloads;
  return options;
}

}  // namespace

void init_plan(nb::module_& module) {
  python::bind_enum_in_module<DiagnosticSeverity>(
      module, "holoscan.diagnostics", "DiagnosticSeverity")
      .value("WARNING", DiagnosticSeverity::kWarning)
      .value("ERROR", DiagnosticSeverity::kError);
  python::bind_enum_in_module<DiagnosticOperation>(
      module, "holoscan.diagnostics", "DiagnosticOperation", doc::plan::kDiagnosticOperation)
      .value("UNKNOWN", DiagnosticOperation::kUnknown)
      .value("COMPOSE", DiagnosticOperation::kCompose)
      .value("CONSTRUCT", DiagnosticOperation::kConstruct)
      .value("SETUP", DiagnosticOperation::kSetup)
      .value("CONTRACT", DiagnosticOperation::kContract)
      .value("LIFECYCLE", DiagnosticOperation::kLifecycle)
      .value("COMPUTE", DiagnosticOperation::kCompute)
      .value("RUNTIME", DiagnosticOperation::kRuntime)
      .value("COMPILE", DiagnosticOperation::kCompile);
  python::bind_enum_in_module<DiagnosticFailureRole>(
      module, "holoscan.diagnostics", "DiagnosticFailureRole", doc::plan::kDiagnosticFailureRole)
      .value("PRIMARY", DiagnosticFailureRole::kPrimary)
      .value("ROLLBACK", DiagnosticFailureRole::kRollback)
      .value("RESTORATION", DiagnosticFailureRole::kRestoration)
      .value("TEARDOWN", DiagnosticFailureRole::kTeardown);
  python::bind_class_in_module<DiagnosticSource>(
      module, "holoscan.diagnostics", "DiagnosticSource", doc::plan::kDiagnosticSource)
      .def_prop_ro("graph_generation",
                   [](const DiagnosticSource& source) { return source.graph_generation; })
      .def_prop_ro("graph", [](const DiagnosticSource& source) { return source.graph.value; })
      .def_prop_ro("vertex", [](const DiagnosticSource& source) { return source.vertex.value; })
      .def_prop_ro("port", [](const DiagnosticSource& source) { return source.port.value(); });
  python::bind_class_in_module<Error>(
      module, "holoscan.diagnostics", "DiagnosticError", doc::plan::kDiagnosticError)
      .def_ro("code", &Error::code)
      .def_ro("message", &Error::message);
  python::bind_class_in_module<DiagnosticMismatch>(
      module, "holoscan.diagnostics", "DiagnosticMismatch", doc::plan::kDiagnosticMismatch)
      .def_ro("field", &DiagnosticMismatch::field)
      .def_ro("provided", &DiagnosticMismatch::provided)
      .def_ro("required", &DiagnosticMismatch::required);
  python::bind_class_in_module<DiagnosticDetail>(
      module, "holoscan.diagnostics", "DiagnosticDetail", doc::plan::kDiagnosticDetail)
      .def_ro("operation", &DiagnosticDetail::operation)
      .def_ro("error", &DiagnosticDetail::error)
      .def_ro("stage", &DiagnosticDetail::stage)
      .def_ro("phase", &DiagnosticDetail::phase)
      .def_ro("transaction_id", &DiagnosticDetail::transaction_id)
      .def_ro("role", &DiagnosticDetail::role)
      .def_ro("operator_path", &DiagnosticDetail::operator_path)
      .def_ro("port_name", &DiagnosticDetail::port_name)
      .def_ro("related_source", &DiagnosticDetail::related_source)
      .def_ro("related_operator_path", &DiagnosticDetail::related_operator_path)
      .def_ro("related_port_name", &DiagnosticDetail::related_port_name)
      .def_ro("mismatches", &DiagnosticDetail::mismatches)
      .def_ro("causes", &DiagnosticDetail::causes)
      .def_ro("primary_diagnostic_index", &DiagnosticDetail::primary_diagnostic_index)
      .def_ro("truncated", &DiagnosticDetail::truncated);
  python::bind_class_in_module<Diagnostic>(
      module, "holoscan.diagnostics", "Diagnostic", doc::plan::kDiagnostic)
      .def_prop_ro("id", [](const Diagnostic& diagnostic) { return diagnostic.id.value; })
      .def_ro("severity", &Diagnostic::severity)
      .def_ro("token", &Diagnostic::token)
      .def_prop_ro("source", [](const Diagnostic& diagnostic) { return diagnostic.source; })
      .def_ro("service", &Diagnostic::service)
      .def_ro("endpoint", &Diagnostic::endpoint)
      .def_ro("json_pointer", &Diagnostic::json_pointer)
      .def_ro("required", &Diagnostic::required)
      .def_ro("provided", &Diagnostic::provided)
      .def_prop_ro("cause_id", &cause_id)
      .def_ro("detail", &Diagnostic::detail)
      .def("__repr__", [](const Diagnostic& diagnostic) {
        return fmt::format("<Diagnostic {}>", diagnostic.token);
      });

  python::bind_enum_in_module<Profile>(module, "holoscan.execution", "Profile")
      .value("STANDARD", Profile::kStandard)
      .value("ROBOTICS", Profile::kRobotics)
      .value("REAL_TIME", Profile::kRealTime)
      .value("REGULATED", Profile::kRegulated)
      .value("RESEARCH", Profile::kResearch);
  python::bind_enum_in_module<CudaComputeStreamPolicy>(
      module, "holoscan.execution", "CudaComputeStreamPolicy")
      .value("PER_EXECUTION_PARTITION", CudaComputeStreamPolicy::kPerExecutionPartition)
      .value("SHARED_PROCESS_DEVICE", CudaComputeStreamPolicy::kSharedProcessDevice);
  python::bind_class_in_module<CudaExecutionCompileOptions>(module,
                                                            "holoscan.execution",
                                                            "CudaExecutionCompileOptions",
                                                            doc::plan::kCudaExecutionCompileOptions)
      .def(nb::init<CudaComputeStreamPolicy>(),
           nb::kw_only(),
           nb::arg("compute_stream_policy") = CudaComputeStreamPolicy::kPerExecutionPartition)
      .def_rw("compute_stream_policy", &CudaExecutionCompileOptions::compute_stream_policy);
  python::bind_class_in_module<DeviceId>(
      module, "holoscan.execution", "DeviceId", doc::plan::kDeviceId)
      .def(nb::init<std::int32_t>(), nb::arg("value"))
      .def_rw("value", &DeviceId::value)
      .def_prop_ro("valid", &DeviceId::valid)
      .def("__repr__", [](DeviceId device) { return fmt::format("<DeviceId {}>", device.value); });
  python::bind_class_in_module<CPUAffinity>(
      module, "holoscan.execution", "CPUAffinity", doc::plan::kCPUAffinity)
      .def(
          "__init__",
          [](CPUAffinity* self, nb::handle values) {
            new (self) CPUAffinity{make_cpu_affinity(values)};
          },
          nb::arg("values").none() = nb::none())
      .def_static("from_ranges", &make_cpu_affinity_from_ranges, nb::arg("ranges"))
      .def_prop_ro("cores", &cpu_affinity_cores)
      .def_prop_ro("empty", &CPUAffinity::empty)
      .def("__repr__", [](const CPUAffinity& affinity) {
        std::ostringstream output;
        output << "<CPUAffinity [";
        for (std::size_t index = 0U; index < affinity.cores.size(); ++index) {
          if (index != 0U) {
            output << ", ";
          }
          output << affinity.cores[index];
        }
        output << "]>";
        return output.str();
      });
  python::bind_enum_in_module<SupportThreadRole>(
      module, "holoscan.execution", "SupportThreadRole", doc::plan::kSupportThreadRole)
      .value("CONTROLLER", SupportThreadRole::kController)
      .value("CHANNEL_PUMP", SupportThreadRole::kChannelPump);
  python::bind_enum_in_module<SchedulingPolicy>(module, "holoscan.execution", "SchedulingPolicy")
      .value("OTHER", SchedulingPolicy::kOther)
      .value("FIFO", SchedulingPolicy::kFIFO)
      .value("ROUND_ROBIN", SchedulingPolicy::kRoundRobin)
      .value("DEADLINE", SchedulingPolicy::kDeadline);
  python::bind_enum_in_module<WakeMode>(module, "holoscan.execution", "WakeMode")
      .value("SLEEPING_WAITSET", WakeMode::kSleepingWaitset)
      .value("BUSY_POLL", WakeMode::kBusyPoll);
  python::bind_enum_in_module<IRQIsolation>(module, "holoscan.execution", "IRQIsolation")
      .value("NOT_REQUESTED", IRQIsolation::kNotRequested)
      .value("PREFER_ISOLATED", IRQIsolation::kPreferIsolated)
      .value("REQUIRE_ISOLATED", IRQIsolation::kRequireIsolated);
  python::bind_enum_in_module<ProbeFailurePolicy>(module, "holoscan.topics", "ProbeFailurePolicy")
      .value("WARN_AND_RECORD_FALLBACK", ProbeFailurePolicy::kWarnAndRecordFallback)
      .value("FAIL_ADMISSION", ProbeFailurePolicy::kFailAdmission);
  python::bind_enum_in_module<OpaqueObjectPayloadPolicy>(
      module, "holoscan", "OpaqueObjectPayloadPolicy")
      .value("ALLOW", OpaqueObjectPayloadPolicy::kAllow)
      .value("WARN", OpaqueObjectPayloadPolicy::kWarn)
      .value("REJECT", OpaqueObjectPayloadPolicy::kReject);
  python::bind_class_in_module<DeadlineScheduling>(
      module, "holoscan.execution", "DeadlineScheduling", doc::plan::kDeadlineScheduling)
      .def(
          "__init__",
          [](DeadlineScheduling* self,
             std::uint64_t runtime_ns,
             std::uint64_t deadline_ns,
             std::uint64_t period_ns) {
            new (self) DeadlineScheduling{runtime_ns, deadline_ns, period_ns};
          },
          nb::kw_only(),
          nb::arg("runtime_ns") = 0U,
          nb::arg("deadline_ns") = 0U,
          nb::arg("period_ns") = 0U)
      .def_rw("runtime_ns", &DeadlineScheduling::runtime_ns)
      .def_rw("deadline_ns", &DeadlineScheduling::deadline_ns)
      .def_rw("period_ns", &DeadlineScheduling::period_ns);
  python::bind_class_in_module<ThreadScheduling>(
      module, "holoscan.execution", "ThreadScheduling", doc::plan::kThreadScheduling)
      .def(
          "__init__",
          [](ThreadScheduling* self,
             SchedulingPolicy policy,
             std::optional<int>
                 priority,
             std::optional<DeadlineScheduling>
                 deadline) { new (self) ThreadScheduling{policy, priority, std::move(deadline)}; },
          nb::kw_only(),
          nb::arg("policy") = SchedulingPolicy::kOther,
          nb::arg("priority").none() = nb::none(),
          nb::arg("deadline").none() = nb::none())
      .def_rw("policy", &ThreadScheduling::policy)
      .def_rw("priority", &ThreadScheduling::priority)
      .def_rw("deadline", &ThreadScheduling::deadline);
  python::bind_class_in_module<ThreadPlacement>(
      module, "holoscan.execution", "ThreadPlacement", doc::plan::kThreadPlacement)
      .def(
          "__init__",
          [](ThreadPlacement* self,
             std::optional<CPUAffinity>
                 affinity,
             SchedulingPolicy policy,
             std::optional<int>
                 priority,
             std::optional<DeadlineScheduling>
                 deadline,
             WakeMode wake_mode,
             IRQIsolation irq_isolation,
             ProbeFailurePolicy probe_failure_policy) {
            new (self) ThreadPlacement{make_thread_placement(std::move(affinity),
                                                             policy,
                                                             priority,
                                                             std::move(deadline),
                                                             wake_mode,
                                                             irq_isolation,
                                                             probe_failure_policy)};
          },
          nb::kw_only(),
          nb::arg("affinity").none() = nb::none(),
          nb::arg("policy") = SchedulingPolicy::kOther,
          nb::arg("priority").none() = nb::none(),
          nb::arg("deadline").none() = nb::none(),
          nb::arg("wake_mode") = WakeMode::kSleepingWaitset,
          nb::arg("irq_isolation") = IRQIsolation::kNotRequested,
          nb::arg("probe_failure_policy") = ProbeFailurePolicy::kWarnAndRecordFallback)
      .def_rw("affinity", &ThreadPlacement::affinity)
      .def_rw("scheduling", &ThreadPlacement::scheduling)
      .def_prop_rw(
          "policy",
          [](const ThreadPlacement& placement) { return placement.scheduling.policy; },
          [](ThreadPlacement& placement, SchedulingPolicy value) {
            placement.scheduling.policy = value;
          })
      .def_prop_rw(
          "priority",
          [](const ThreadPlacement& placement) { return placement.scheduling.priority; },
          [](ThreadPlacement& placement, std::optional<int> value) {
            placement.scheduling.priority = value;
          })
      .def_prop_rw(
          "deadline",
          [](ThreadPlacement& placement) -> DeadlineScheduling* {
            return placement.scheduling.deadline ? &*placement.scheduling.deadline : nullptr;
          },
          [](ThreadPlacement& placement, std::optional<DeadlineScheduling> value) {
            placement.scheduling.deadline = std::move(value);
          },
          nb::rv_policy::reference_internal)
      .def_rw("wake_mode", &ThreadPlacement::wake_mode)
      .def_rw("irq_isolation", &ThreadPlacement::irq_isolation)
      .def_rw("probe_failure_policy", &ThreadPlacement::probe_failure_policy);
  python::bind_class_in_module<CudaGreenContextPlacement>(module,
                                                          "holoscan.execution",
                                                          "CudaGreenContextPlacement",
                                                          doc::plan::kCudaGreenContextPlacement)
      .def(
          "__init__",
          [](CudaGreenContextPlacement* self, DeviceId device, std::uint32_t sm_count) {
            new (self) CudaGreenContextPlacement{device, sm_count};
          },
          nb::kw_only(),
          nb::arg("device"),
          nb::arg("sm_count"))
      .def_rw("device", &CudaGreenContextPlacement::device)
      .def_rw("sm_count", &CudaGreenContextPlacement::sm_count);
  python::bind_class_in_module<CudaGreenContextCapability>(module,
                                                           "holoscan.execution",
                                                           "CudaGreenContextCapability",
                                                           doc::plan::kCudaGreenContextCapability)
      .def_ro("device", &CudaGreenContextCapability::device)
      .def_ro("deployment_device_id", &CudaGreenContextCapability::deployment_device_id)
      .def_ro("total_sm_count", &CudaGreenContextCapability::total_sm_count)
      .def_ro("min_sm_partition_size", &CudaGreenContextCapability::min_sm_partition_size)
      .def_ro("sm_coscheduled_alignment", &CudaGreenContextCapability::sm_coscheduled_alignment)
      .def_ro("structured_sm_quantum", &CudaGreenContextCapability::structured_sm_quantum)
      .def_ro("max_structured_context_count",
              &CudaGreenContextCapability::max_structured_context_count);
  python::bind_functions_in_module(module, "holoscan.execution", [&] {
    module.def("query_cuda_green_context_capability",
               &cuda_green_context_capability,
               nb::arg("device"),
               doc::plan::kQueryCudaGreenContextCapability,
               nb::call_guard<nb::gil_scoped_release>());
  });
  python::bind_class_in_module<DeploymentBinding>(
      module, "holoscan.execution", "DeploymentBinding", doc::plan::kDeploymentBinding)
      .def(nb::init<>())
      .def(
          "bind_thread_placement",
          [](DeploymentBinding& deployment,
             const std::string& name,
             ThreadPlacement placement) -> DeploymentBinding& {
            return deployment.bind_thread_placement(name, std::move(placement));
          },
          nb::arg("name"),
          nb::arg("placement"),
          nb::rv_policy::reference_internal)
      .def(
          "bind_support_thread_affinity",
          [](DeploymentBinding& deployment,
             SupportThreadRole role,
             CPUAffinity affinity) -> DeploymentBinding& {
            return deployment.bind_support_thread_affinity(role, std::move(affinity));
          },
          nb::arg("role"),
          nb::arg("affinity"),
          doc::plan::kBindSupportThreadAffinity,
          nb::rv_policy::reference_internal)
      .def(
          "bind_cuda_green_context",
          [](DeploymentBinding& deployment,
             const std::string& name,
             CudaGreenContextPlacement placement) -> DeploymentBinding& {
            return deployment.bind_cuda_green_context(name, placement);
          },
          nb::arg("name"),
          nb::arg("placement"),
          nb::rv_policy::reference_internal)
      .def(
          "bind_tensor_output_device",
          [](DeploymentBinding& deployment,
             const std::string& operator_path,
             const std::string& output_port,
             DeviceId device) -> DeploymentBinding& {
            return deployment.bind_tensor_output_device(
                TensorOutputDevicePlacement{operator_path, output_port, device});
          },
          nb::kw_only(),
          nb::arg("operator_path"),
          nb::arg("output_port"),
          nb::arg("device"),
          nb::rv_policy::reference_internal);
  python::bind_class_in_module<RateRatio>(
      module, "holoscan.execution", "RateRatio", doc::plan::kRateRatio)
      .def(
          "__init__",
          [](RateRatio* self, std::uint32_t numerator, std::uint32_t denominator) {
            new (self) RateRatio{numerator, denominator};
          },
          nb::arg("numerator") = 1U,
          nb::arg("denominator") = 1U)
      .def_rw("numerator", &RateRatio::numerator)
      .def_rw("denominator", &RateRatio::denominator);
  python::bind_class_in_module<AutomaticQueueSizingOptions>(module,
                                                            "holoscan.execution",
                                                            "AutomaticQueueSizingOptions",
                                                            doc::plan::kAutomaticQueueSizingOptions)
      .def(
          "__init__",
          [](AutomaticQueueSizingOptions* self, RateRatio headroom_ratio) {
            new (self) AutomaticQueueSizingOptions{headroom_ratio};
          },
          nb::kw_only(),
          nb::arg("headroom_ratio") = RateRatio{2U, 1U})
      .def_rw("headroom_ratio", &AutomaticQueueSizingOptions::headroom_ratio);
  python::bind_class_in_module<LifecycleTransactionOptions>(module,
                                                            "holoscan.lifecycle",
                                                            "LifecycleTransactionOptions",
                                                            doc::plan::kLifecycleTransactionOptions)
      .def(
          "__init__",
          [](LifecycleTransactionOptions* self,
             std::uint32_t max_pending_requests,
             std::uint32_t max_pending_transactions,
             std::uint32_t max_receipts,
             std::uint32_t max_fenced_operators,
             std::uint32_t max_candidate_generations,
             std::uint32_t max_events,
             std::uint32_t max_audit_records,
             std::uint32_t max_quiesce_records,
             std::uint32_t max_rollback_records) {
            new (self) LifecycleTransactionOptions{max_pending_requests,
                                                   max_pending_transactions,
                                                   max_receipts,
                                                   max_fenced_operators,
                                                   max_candidate_generations,
                                                   max_events,
                                                   max_audit_records,
                                                   max_quiesce_records,
                                                   max_rollback_records};
          },
          nb::kw_only(),
          nb::arg("max_pending_requests"),
          nb::arg("max_pending_transactions"),
          nb::arg("max_receipts"),
          nb::arg("max_fenced_operators"),
          nb::arg("max_candidate_generations"),
          nb::arg("max_events"),
          nb::arg("max_audit_records"),
          nb::arg("max_quiesce_records"),
          nb::arg("max_rollback_records"))
      .def_rw("max_pending_requests", &LifecycleTransactionOptions::max_pending_requests)
      .def_rw("max_pending_transactions", &LifecycleTransactionOptions::max_pending_transactions)
      .def_rw("max_receipts", &LifecycleTransactionOptions::max_receipts)
      .def_rw("max_fenced_operators", &LifecycleTransactionOptions::max_fenced_operators)
      .def_rw("max_candidate_generations", &LifecycleTransactionOptions::max_candidate_generations)
      .def_rw("max_events", &LifecycleTransactionOptions::max_events)
      .def_rw("max_audit_records", &LifecycleTransactionOptions::max_audit_records)
      .def_rw("max_quiesce_records", &LifecycleTransactionOptions::max_quiesce_records)
      .def_rw("max_rollback_records", &LifecycleTransactionOptions::max_rollback_records);
  nb::enum_<ParamStorageMode>(module, "ParamStorageMode")
      .value("PROFILE_DEFAULT", ParamStorageMode::kProfileDefault)
      .value("DYNAMIC_COLD_PATH", ParamStorageMode::kDynamicColdPath)
      .value("FREEZE_BEFORE_ARM", ParamStorageMode::kFreezeBeforeArm)
      .value("PREALLOCATED_BOUNDED", ParamStorageMode::kPreallocatedBounded)
      .value("INLINE_ONLY", ParamStorageMode::kInlineOnly);
  nb::enum_<ParamRuntimeUpdatePolicy>(module, "ParamRuntimeUpdatePolicy")
      .value("PROFILE_DEFAULT", ParamRuntimeUpdatePolicy::kProfileDefault)
      .value("DISABLED", ParamRuntimeUpdatePolicy::kDisabled)
      .value("ENABLED_IF_ADMITTED", ParamRuntimeUpdatePolicy::kEnabledIfAdmitted)
      .value("REQUIRE_PREALLOCATED", ParamRuntimeUpdatePolicy::kRequirePreallocated);
  nb::enum_<ParamExhaustionPolicy>(module, "ParamExhaustionPolicy")
      .value("REJECT_UPDATE", ParamExhaustionPolicy::kRejectUpdate);
  python::bind_class_in_module<ParameterCompileOptions>(
      module, "holoscan.params", "ParameterCompileOptions")
      .def(nb::init<>())
      .def_rw("storage", &ParameterCompileOptions::storage)
      .def_rw("runtime_updates", &ParameterCompileOptions::runtime_updates)
      .def_rw("generation_slots_per_operator",
              &ParameterCompileOptions::generation_slots_per_operator)
      .def_rw("max_pending_updates_per_operator",
              &ParameterCompileOptions::max_pending_updates_per_operator)
      .def_rw("audit_capacity", &ParameterCompileOptions::audit_capacity)
      .def_rw("max_event_cursors", &ParameterCompileOptions::max_event_cursors)
      .def_rw("max_parameter_arena_bytes", &ParameterCompileOptions::max_parameter_arena_bytes)
      .def_rw("exhaustion", &ParameterCompileOptions::exhaustion)
      .def_rw("require_proven_schema_completeness",
              &ParameterCompileOptions::require_proven_schema_completeness)
      .def_rw("require_explicit_field_policies",
              &ParameterCompileOptions::require_explicit_field_policies);
  python::bind_class_in_module<CompileOptions>(
      module, "holoscan", "CompileOptions", doc::plan::kCompileOptions)
      .def(
          "__init__",
          [](CompileOptions* self,
             DeploymentBinding deployment,
             Profile profile,
             bool auto_insert_converters,
             ParameterCompileOptions parameters,
             std::optional<LifecycleTransactionOptions>
                 lifecycle_transactions,
             AutomaticQueueSizingOptions automatic_queue_sizing,
             CudaExecutionCompileOptions cuda_execution,
             OpaqueObjectPayloadPolicy opaque_object_payloads) {
            new (self) CompileOptions{make_compile_options(std::move(deployment),
                                                           profile,
                                                           auto_insert_converters,
                                                           std::move(parameters),
                                                           std::move(lifecycle_transactions),
                                                           automatic_queue_sizing,
                                                           cuda_execution,
                                                           opaque_object_payloads)};
          },
          nb::kw_only(),
          nb::arg("deployment") = DeploymentBinding{},
          nb::arg("profile") = Profile::kStandard,
          nb::arg("auto_insert_converters") = true,
          nb::arg("parameters") = ParameterCompileOptions{},
          nb::arg("lifecycle_transactions").none() = nb::none(),
          nb::arg("automatic_queue_sizing") = AutomaticQueueSizingOptions{},
          nb::arg("cuda_execution") = CudaExecutionCompileOptions{},
          nb::arg("opaque_object_payloads") = OpaqueObjectPayloadPolicy::kAllow)
      .def_rw("deployment", &CompileOptions::deployment)
      .def_rw("profile", &CompileOptions::profile)
      .def_rw("auto_insert_converters", &CompileOptions::auto_insert_converters)
      .def_rw("parameters", &CompileOptions::parameters)
      .def_prop_rw(
          "lifecycle_transactions",
          [](CompileOptions& options) -> LifecycleTransactionOptions* {
            return options.lifecycle_transactions ? &*options.lifecycle_transactions : nullptr;
          },
          [](CompileOptions& options, std::optional<LifecycleTransactionOptions> value) {
            options.lifecycle_transactions = std::move(value);
          },
          nb::rv_policy::reference_internal)
      .def_rw("automatic_queue_sizing", &CompileOptions::automatic_queue_sizing)
      .def_rw("cuda_execution", &CompileOptions::cuda_execution)
      .def_rw("opaque_object_payloads", &CompileOptions::opaque_object_payloads);

  for (const char* name :
       {"ParamStorageMode", "ParamRuntimeUpdatePolicy", "ParamExhaustionPolicy"}) {
    module.attr(name).attr("__module__") = "holoscan.params";
  }

  auto topic_direction = nb::enum_<TopicDirection>(module, "TopicDirection");
  topic_direction.attr("__module__") = "holoscan.topics";
  topic_direction.value("IMPORT", TopicDirection::kImport).value("EXPORT", TopicDirection::kExport);
  auto endpoint_name_origin = nb::enum_<EndpointNameOrigin>(module, "EndpointNameOrigin");
  endpoint_name_origin.attr("__module__") = "holoscan.topics";
  endpoint_name_origin.value("EXPLICIT", EndpointNameOrigin::kExplicit)
      .value("GENERATED", EndpointNameOrigin::kGenerated);
  auto endpoint_schema_identity =
      nb::class_<PyEndpointSchemaIdentity>(module, "EndpointSchemaIdentity");
  endpoint_schema_identity.attr("__module__") = "holoscan.topics";
  endpoint_schema_identity.def_prop_ro("canonical_name", &PyEndpointSchemaIdentity::canonical_name)
      .def_prop_ro("compatibility_epoch", &PyEndpointSchemaIdentity::compatibility_epoch)
      .def_prop_ro("family_id", &PyEndpointSchemaIdentity::family_id)
      .def_prop_ro("file_identifier", &PyEndpointSchemaIdentity::file_identifier);
  auto tensor_boundary_contract =
      nb::class_<PyTensorBoundaryContractView>(module, "TensorBoundaryContractView");
  tensor_boundary_contract.attr("__module__") = "holoscan.topics";
  tensor_boundary_contract.def_prop_ro("memory_kind", &PyTensorBoundaryContractView::memory_kind)
      .def_prop_ro("dtype", &PyTensorBoundaryContractView::dtype)
      .def_prop_ro("rank", &PyTensorBoundaryContractView::rank)
      .def_prop_ro("max_rank", &PyTensorBoundaryContractView::max_rank)
      .def_prop_ro("max_byte_span", &PyTensorBoundaryContractView::max_byte_span)
      .def_prop_ro("axes", &PyTensorBoundaryContractView::axes)
      .def_prop_ro("stride_policy", &PyTensorBoundaryContractView::stride_policy)
      .def_prop_ro("foreign_emit_policy", &PyTensorBoundaryContractView::foreign_emit_policy)
      .def_prop_ro("max_inline_bytes", &PyTensorBoundaryContractView::max_inline_bytes)
      .def_prop_ro("max_staging_byte_span", &PyTensorBoundaryContractView::max_staging_byte_span);
  auto endpoint_view_type = nb::class_<PyEndpointView>(module, "EndpointView");
  endpoint_view_type.attr("__module__") = "holoscan.topics";
  endpoint_view_type.def_prop_ro("name", &PyEndpointView::name)
      .def_prop_ro("uri", &PyEndpointView::uri)
      .def_prop_ro("service", &PyEndpointView::service)
      .def_prop_ro("direction", &PyEndpointView::direction)
      .def_prop_ro("name_origin", &PyEndpointView::name_origin)
      .def_prop_ro("scalar_identity", &PyEndpointView::scalar_identity)
      .def_prop_ro("schema_identity", &PyEndpointView::schema_identity)
      .def_prop_ro("schema_writer_max_serialized_bytes",
                   &PyEndpointView::schema_writer_max_serialized_bytes)
      .def_prop_ro("schema_required_frame_bytes", &PyEndpointView::schema_required_frame_bytes)
      .def_prop_ro("schema_selected_inline_bucket_bytes",
                   &PyEndpointView::schema_selected_inline_bucket_bytes)
      .def_prop_ro("schema_bucket_usable_serialized_bytes",
                   &PyEndpointView::schema_bucket_usable_serialized_bytes)
      .def_prop_ro("schema_transport_profile_tag", &PyEndpointView::schema_transport_profile_tag)
      .def_prop_ro("attach_policy", &PyEndpointView::attach_policy)
      .def_prop_ro("bounds", &PyEndpointView::bounds)
      .def_prop_ro("memory_conversion", &PyEndpointView::memory_conversion)
      .def_prop_ro("clock_health", &PyEndpointView::clock_health)
      .def_prop_ro("capture_clock", &PyEndpointView::capture_clock)
      .def_prop_ro("has_tensor_requirements", &PyEndpointView::has_tensor_requirements)
      .def_prop_ro("tensor_requirements", &PyEndpointView::tensor_requirements)
      .def_prop_ro("has_pod_identity", &PyEndpointView::has_pod_identity);
  auto compatibility_report =
      nb::class_<ChannelEndpointCompatibilityReport>(module, "EndpointCompatibilityReport");
  compatibility_report.attr("__module__") = "holoscan.topics";
  compatibility_report.def_prop_ro("compatible", &ChannelEndpointCompatibilityReport::compatible)
      .def_prop_ro("diagnostics",
                   [](const ChannelEndpointCompatibilityReport& report) {
                     return std::vector<Diagnostic>{report.diagnostics().begin(),
                                                    report.diagnostics().end()};
                   })
      .def("json", &ChannelEndpointCompatibilityReport::json);

  python::bind_class_in_module<ChannelEndpointCatalog>(
      module, "holoscan.topics", "EndpointCatalog", doc::plan::kEndpointCatalog)
      .def_static("load",
                  &load_endpoint_catalog,
                  nb::arg("path"),
                  nb::sig("def load(path: object) -> holoscan.topics.EndpointCatalog"))
      .def_static("parse",
                  &parse_endpoint_catalog,
                  nb::arg("text"),
                  nb::sig("def parse(text: str) -> holoscan.topics.EndpointCatalog"))
      .def("json", &ChannelEndpointCatalog::json)
      .def("save_json", &save_endpoint_catalog, nb::arg("path"), doc::plan::kSaveEndpointCatalog)
      .def("endpoint", &endpoint_view, nb::arg("name"))
      .def("compatibility_with", &ChannelEndpointCatalog::compatibility_with, nb::arg("candidate"))
      .def_prop_ro(
          "digest",
          [](const ChannelEndpointCatalog& catalog) { return std::string{catalog.digest()}; })
      .def_prop_ro("scope", [](const ChannelEndpointCatalog& catalog) {
        return std::string{catalog.scope()};
      });

  python::bind_enum_in_module<PartitioningMode>(module, "holoscan.execution", "PartitioningMode")
      .value("AUTOMATIC", PartitioningMode::kAutomatic)
      .value("CONSTRAINED_AUTOMATIC", PartitioningMode::kConstrainedAutomatic)
      .value("MANUAL", PartitioningMode::kManual);
  python::bind_enum_in_module<ExecutionPartitionOrigin>(
      module, "holoscan.execution", "ExecutionPartitionOrigin")
      .value("AUTHORED", ExecutionPartitionOrigin::kAuthored)
      .value("AUTOMATIC", ExecutionPartitionOrigin::kAutomatic);
  python::bind_class_in_module<ExecutionPartitionView>(
      module, "holoscan.execution", "ExecutionPartition", doc::plan::kPartition)
      .def_prop_ro("id",
                   [](const ExecutionPartitionView& partition) { return partition.id().value; })
      .def_prop_ro("origin", &ExecutionPartitionView::origin)
      .def_prop_ro("authored_name",
                   [](const ExecutionPartitionView& partition) -> std::optional<std::string> {
                     const auto name = partition.authored_name();
                     return name ? std::optional<std::string>{*name} : std::nullopt;
                   })
      .def_prop_ro("members",
                   [](const ExecutionPartitionView& partition) {
                     std::vector<std::uint64_t> members;
                     for (const VertexId member : partition.members()) {
                       members.push_back(member.value);
                     }
                     return members;
                   })
      .def_prop_ro("operator_paths",
                   [](const ExecutionPartitionView& partition) {
                     return std::vector<std::string>{partition.operator_paths().begin(),
                                                     partition.operator_paths().end()};
                   })
      .def_prop_ro("execution_lane", &ExecutionPartitionView::execution_lane);
  python::bind_class_in_module<ExecutionPartitionsView>(
      module, "holoscan.execution", "ExecutionPartitions", doc::plan::kPartitions)
      .def_prop_ro("mode", &ExecutionPartitionsView::mode)
      .def_prop_ro("authored_count", &ExecutionPartitionsView::authored_count)
      .def_prop_ro("automatic_count", &ExecutionPartitionsView::automatic_count)
      .def("lookup_named", &partition_lookup_named, nb::arg("name"))
      .def("of", &partition_for_operator, nb::arg("operator"))
      .def("__len__", &ExecutionPartitionsView::size)
      .def("__iter__",
           &partition_iterator,
           nb::sig("def __iter__(self) -> "
                   "typing.Iterator[holoscan.execution.ExecutionPartition]"));

  // Exposes the resolved parameter-derived Tensor capacity evidence from the execution plan.
  python::bind_enum_in_module<TensorByteSpanDerivationUpdateDisposition>(
      module, "holoscan.execution", "TensorByteSpanDerivationUpdateDisposition")
      .value("FROZEN_COMPILE_TIME", TensorByteSpanDerivationUpdateDisposition::kFrozenCompileTime)
      .value("RECOMPILE_REQUIRED", TensorByteSpanDerivationUpdateDisposition::kRecompileRequired);
  python::bind_class_in_module<TensorByteSpanDerivationPlanRecord>(
      module,
      "holoscan.execution",
      "TensorByteSpanDerivation",
      doc::plan::kTensorByteSpanDerivation)
      .def_prop_ro(
          "output",
          [](const TensorByteSpanDerivationPlanRecord& value) { return value.output.value(); })
      .def_ro("field_id", &TensorByteSpanDerivationPlanRecord::field_id)
      .def_ro("tensor_index", &TensorByteSpanDerivationPlanRecord::tensor_index)
      .def_ro("resolved_max_byte_span", &TensorByteSpanDerivationPlanRecord::resolved_max_byte_span)
      .def_ro("resolved_max_staging_byte_span",
              &TensorByteSpanDerivationPlanRecord::resolved_max_staging_byte_span)
      .def_ro("update_disposition", &TensorByteSpanDerivationPlanRecord::update_disposition);

  auto memory_kind_update_type =
      python::bind_enum_in_module<TensorMemoryKindDerivationUpdateDisposition>(
          module, "holoscan.execution", "TensorMemoryKindDerivationUpdateDisposition");
  memory_kind_update_type
      .value("FROZEN_COMPILE_TIME", TensorMemoryKindDerivationUpdateDisposition::kFrozenCompileTime)
      .value("RECOMPILE_REQUIRED", TensorMemoryKindDerivationUpdateDisposition::kRecompileRequired);
  python::bind_class_in_module<TensorMemoryKindDerivationPlanRecord>(
      module,
      "holoscan.execution",
      "TensorMemoryKindDerivation",
      doc::plan::kTensorMemoryKindDerivation)
      .def_prop_ro(
          "output",
          [](const TensorMemoryKindDerivationPlanRecord& value) { return value.output.value(); })
      .def_ro("field_id", &TensorMemoryKindDerivationPlanRecord::field_id)
      .def_ro("tensor_index", &TensorMemoryKindDerivationPlanRecord::tensor_index)
      .def_ro("parameter_schema_id", &TensorMemoryKindDerivationPlanRecord::parameter_schema_id)
      .def_ro("parameter_path", &TensorMemoryKindDerivationPlanRecord::parameter_path)
      .def_ro("resolved_memory_kind", &TensorMemoryKindDerivationPlanRecord::resolved_memory_kind)
      .def_ro("update_disposition", &TensorMemoryKindDerivationPlanRecord::update_disposition);

  python::bind_class_in_module<ExecutionPlan>(module, "holoscan", "Plan", doc::plan::kPlan)
      .def_prop_ro("ok", &ExecutionPlan::ok)
      .def_prop_ro("diagnostics", [](const ExecutionPlan& plan) { return copy_diagnostics(plan); })
      .def_prop_ro("errors",
                   [](const ExecutionPlan& plan) {
                     return copy_diagnostics(plan, DiagnosticSeverity::kError);
                   })
      .def_prop_ro("warnings",
                   [](const ExecutionPlan& plan) {
                     return copy_diagnostics(plan, DiagnosticSeverity::kWarning);
                   })
      .def_prop_ro("cuda_compute_stream_policy", &ExecutionPlan::cuda_compute_stream_policy)
      .def_prop_ro("partitions", &ExecutionPlan::partitions)
      .def_prop_ro("endpoint_catalog",
                   [](const ExecutionPlan& plan) { return plan.endpoint_catalog(); })
      .def(
          "find_tensor_byte_span_derivation",
          [](const ExecutionPlan& plan,
             const std::string& operator_path,
             const std::string& output_name,
             std::optional<std::uint16_t>
                 field_id,
             std::optional<std::uint16_t>
                 tensor_index) -> std::optional<TensorByteSpanDerivationPlanRecord> {
            const auto* record = plan.find_tensor_byte_span_derivation(
                operator_path, output_name, field_id, tensor_index);
            return record == nullptr ? std::nullopt
                                     : std::optional<TensorByteSpanDerivationPlanRecord>{*record};
          },
          nb::arg("operator_path"),
          nb::arg("output_name"),
          nb::kw_only(),
          nb::arg("field_id").none() = nb::none(),
          nb::arg("tensor_index").none() = nb::none(),
          doc::plan::kFindTensorByteSpanDerivation)
      .def_prop_ro("tensor_memory_kind_derivations",
                   [](const ExecutionPlan& plan) {
                     const auto records = plan.tensor_memory_kind_derivations();
                     return std::vector<TensorMemoryKindDerivationPlanRecord>{records.begin(),
                                                                              records.end()};
                   })
      .def(
          "find_tensor_memory_kind_derivation",
          [](const ExecutionPlan& plan,
             const std::string& operator_path,
             const std::string& output_name,
             std::optional<std::uint16_t>
                 field_id,
             std::optional<std::uint16_t>
                 tensor_index) -> std::optional<TensorMemoryKindDerivationPlanRecord> {
            const auto* record = plan.find_tensor_memory_kind_derivation(
                operator_path, output_name, field_id, tensor_index);
            return record == nullptr ? std::nullopt
                                     : std::optional<TensorMemoryKindDerivationPlanRecord>{*record};
          },
          nb::arg("operator_path"),
          nb::arg("output_name"),
          nb::kw_only(),
          nb::arg("field_id").none() = nb::none(),
          nb::arg("tensor_index").none() = nb::none(),
          doc::plan::kFindTensorMemoryKindDerivation)
      .def("json", [](const ExecutionPlan& plan) { return std::string{plan.json()}; })
      .def("summary", &plan_summary)
      .def("require_no_errors", &ExecutionPlan::require_no_errors)
      .def("__bool__", &ExecutionPlan::ok)
      .def("__repr__", &plan_repr);

  python::bind_functions_in_module(module, "holoscan", [&] {
    module.def(
        "compile",
        [](nb::handle graph, nb::handle options) {
          prepare_python_graph_for_compile(graph);
          Graph& native_graph = python_graph(graph);
          std::optional<CompileOptions> native_options;
          if (!options.is_none()) {
            native_options = nb::cast<CompileOptions>(options);
          }
          const nb::gil_scoped_release release;
          return native_options ? holoscan::compile(native_graph, std::move(*native_options))
                                : holoscan::compile(native_graph);
        },
        nb::arg("graph"),
        nb::arg("options").none() = nb::none(),
        doc::plan::kCompile);
  });
}

}  // namespace holoscan
