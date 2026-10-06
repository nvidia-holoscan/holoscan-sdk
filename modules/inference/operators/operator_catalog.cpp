// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <holoscan/operators/inference/operator_catalog.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/operators/inference/inference.hpp>

namespace holoscan::ops {
namespace {

using authoring::OpaqueOperatorRef;
using authoring::ParameterAssignment;
using authoring::ParameterValue;
using authoring::ParameterValueKind;

constexpr std::string_view kRegistryName{"holoscan.inference.InferenceOp"};

[[noreturn]] void parameter_error(std::string_view token, std::string_view field = {},
                                  std::string_view detail = {}) {
  std::string message{token};
  if (!field.empty()) {
    message.append(fmt::format(" field_name={}", field));
  }
  if (!detail.empty()) {
    message.append(" ");
    message.append(detail);
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

// Borrows one structural assignment map while tracking the fields consumed by its parser.
class AssignmentView {
 public:
  AssignmentView(std::span<const ParameterAssignment> assignments, std::string path)
      : assignments_(assignments), path_(std::move(path)) {
    for (std::size_t index = 0U; index < assignments_.size(); ++index) {
      const auto duplicate = std::find_if(
          assignments_.begin() + static_cast<std::ptrdiff_t>(index + 1U),
          assignments_.end(),
          [&](const ParameterAssignment& value) { return value.name == assignments_[index].name; });
      if (duplicate != assignments_.end()) {
        parameter_error("INFERENCE_OPERATOR_PARAMETER_DUPLICATE", field_path(duplicate->name));
      }
    }
  }

  [[nodiscard]] const ParameterValue* find(std::string_view name) const {
    if (std::find(accepted_.begin(), accepted_.end(), name) == accepted_.end()) {
      accepted_.push_back(name);
    }
    const auto found = std::find_if(assignments_.begin(),
                                    assignments_.end(),
                                    [&](const auto& item) { return item.name == name; });
    return found == assignments_.end() ? nullptr : &found->value;
  }

  [[nodiscard]] const ParameterValue& require(std::string_view name) const {
    const ParameterValue* value = find(name);
    if (value == nullptr) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_REQUIRED", field_path(name));
    }
    return *value;
  }

  void reject_unknown() const {
    for (const ParameterAssignment& assignment : assignments_) {
      if (std::find(accepted_.begin(), accepted_.end(), assignment.name) == accepted_.end()) {
        std::string accepted;
        for (const std::string_view name : accepted_) {
          if (!accepted.empty()) {
            accepted.append(",");
          }
          accepted.append(name);
        }
        parameter_error("INFERENCE_OPERATOR_PARAMETER_UNKNOWN",
                        field_path(assignment.name),
                        fmt::format("accepted={}", accepted));
      }
    }
  }

  [[nodiscard]] std::string field_path(std::string_view name) const {
    if (path_.empty()) {
      return std::string{name};
    }
    if (name.empty()) {
      return path_;
    }
    return path_ + "." + std::string{name};
  }

  [[nodiscard]] std::size_t size() const noexcept { return assignments_.size(); }

 private:
  std::span<const ParameterAssignment> assignments_;
  std::string path_;
  mutable std::vector<std::string_view> accepted_;
};

[[nodiscard]] AssignmentView nested(const ParameterValue& value, std::string path) {
  if (value.kind != ParameterValueKind::kAssignments) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return AssignmentView{value.nested_assignments(), std::move(path)};
}

[[nodiscard]] std::string string_value(const ParameterValue& value, std::string_view path) {
  if (value.kind != ParameterValueKind::kString) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return std::string{value.string};
}

[[nodiscard]] bool bool_value(const ParameterValue& value, std::string_view path) {
  bool result{};
  if (value.kind != ParameterValueKind::kScalar || !value.scalar.try_as(result)) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return result;
}

template <typename Integer>
[[nodiscard]] Integer integer_value(const ParameterValue& value, std::string_view path) {
  const auto convert = [&]<typename Source>() -> std::optional<Integer> {
    Source source{};
    if (!value.scalar.try_as(source)) {
      return std::nullopt;
    }
    if (!std::in_range<Integer>(source)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      path,
                      fmt::format("value={} accepted_range=[{},{}]",
                                  source,
                                  std::numeric_limits<Integer>::lowest(),
                                  std::numeric_limits<Integer>::max()));
    }
    return static_cast<Integer>(source);
  };
  if (value.kind != ParameterValueKind::kScalar) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  if (const auto result = convert.template operator()<std::int64_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint64_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int32_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint32_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int16_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint16_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int8_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint8_t>())
    return *result;
  parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
}

[[nodiscard]] InferenceDataMap string_map(const ParameterValue& value, std::string path) {
  AssignmentView entries = nested(value, std::move(path));
  InferenceDataMap result;
  for (const ParameterAssignment& entry : value.nested_assignments()) {
    if (entry.name.empty()) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      entries.field_path(entry.name),
                      "accepted=nonempty-key");
    }
    result.emplace(std::string{entry.name},
                   string_value(entry.value, entries.field_path(entry.name)));
  }
  return result;
}

[[nodiscard]] InferenceDataVecMap string_sequence_map(const ParameterValue& value,
                                                      std::string path) {
  AssignmentView entries = nested(value, std::move(path));
  InferenceDataVecMap result;
  for (const ParameterAssignment& entry : value.nested_assignments()) {
    if (entry.name.empty()) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      entries.field_path(entry.name),
                      "accepted=nonempty-key");
    }
    if (entry.value.kind != ParameterValueKind::kSequence) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", entries.field_path(entry.name));
    }
    std::vector<std::string> strings;
    strings.reserve(entry.value.sequence_size);
    for (std::size_t index = 0U; index < entry.value.sequence_size; ++index) {
      strings.push_back(string_value(entry.value.sequence_values()[index],
                                     fmt::format("{}[{}]", entries.field_path(entry.name), index)));
    }
    result.emplace(std::string{entry.name}, std::move(strings));
  }
  return result;
}

template <typename Value, typename Decode>
void assign_if_present(const AssignmentView& values, std::string_view name, Value& destination,
                       Decode&& decode) {
  if (const ParameterValue* value = values.find(name)) {
    destination = std::forward<Decode>(decode)(*value, values.field_path(name));
  }
}

[[nodiscard]] std::shared_ptr<CudaStreamPool> stream_pool(const ParameterValue& value,
                                                          std::string path) {
  AssignmentView fields = nested(value, std::move(path));
  CudaStreamPool::Config config;
  assign_if_present(fields, "device_id", config.device_id, integer_value<std::int32_t>);
  assign_if_present(fields, "stream_flags", config.stream_flags, integer_value<std::uint32_t>);
  assign_if_present(fields, "stream_priority", config.stream_priority, integer_value<std::int32_t>);
  assign_if_present(fields, "reserved_size", config.reserved_size, integer_value<std::uint32_t>);
  assign_if_present(fields, "max_size", config.max_size, integer_value<std::uint32_t>);
  assign_if_present(fields, "nvtx_identifier", config.nvtx_identifier, string_value);
  fields.reject_unknown();
  return std::make_shared<CudaStreamPool>(std::move(config));
}

[[nodiscard]] std::vector<std::int32_t> integer_sequence(const ParameterValue& value,
                                                         std::string_view path) {
  if (value.kind != ParameterValueKind::kSequence) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  std::vector<std::int32_t> result;
  result.reserve(value.sequence_size);
  for (std::size_t index = 0U; index < value.sequence_size; ++index) {
    result.push_back(integer_value<std::int32_t>(value.sequence_values()[index],
                                                 fmt::format("{}[{}]", path, index)));
  }
  return result;
}

[[nodiscard]] std::shared_ptr<CudaGreenContextPool> green_context_pool(const ParameterValue& value,
                                                                       std::string path) {
  AssignmentView fields = nested(value, std::move(path));
  CudaGreenContextPool::Config config;
  assign_if_present(fields, "device_id", config.device_id, integer_value<std::int32_t>);
  assign_if_present(fields, "num_partitions", config.num_partitions, integer_value<std::uint32_t>);
  assign_if_present(fields, "sms_per_partition", config.sms_per_partition, integer_sequence);
  if (const ParameterValue* field = fields.find("default_context")) {
    if (field->kind == ParameterValueKind::kNone) {
      config.default_context.reset();
    } else {
      config.default_context =
          integer_value<std::int32_t>(*field, fields.field_path("default_context"));
    }
  }
  assign_if_present(fields, "nvtx_identifier", config.nvtx_identifier, string_value);
  fields.reject_unknown();
  return std::make_shared<CudaGreenContextPool>(std::move(config));
}

// Translate the facade's direct keyword form into the C++ constructor parameter object.
[[nodiscard]] InferenceOpParams direct_params(const AssignmentView& values) {
  InferenceOpParams params;
  params.model_path_map =
      string_map(values.require("model_path_map"), values.field_path("model_path_map"));
  params.input_map =
      string_sequence_map(values.require("input_map"), values.field_path("input_map"));
  params.output_map =
      string_sequence_map(values.require("output_map"), values.field_path("output_map"));
  assign_if_present(values, "backend", params.backend, string_value);
  assign_if_present(
      values, "backend_map", params.backend_map, [](const ParameterValue& value, std::string path) {
        return string_map(value, std::move(path));
      });
  assign_if_present(values,
                    "trt_opt_profile",
                    params.trt_opt_profile,
                    [](const ParameterValue& value, std::string path) {
                      return string_sequence_map(value, std::move(path));
                    });
  assign_if_present(values,
                    "dla_core_map",
                    params.dla_core_map,
                    [](const ParameterValue& value, std::string path) {
                      return string_map(value, std::move(path));
                    });
  assign_if_present(values, "is_engine_path", params.is_engine_path, bool_value);
  assign_if_present(values, "input_on_cuda", params.input_on_cuda, bool_value);
  assign_if_present(values, "output_on_cuda", params.output_on_cuda, bool_value);
  assign_if_present(values, "transmit_on_cuda", params.transmit_on_cuda, bool_value);
  assign_if_present(values, "dynamic_input_dims", params.dynamic_input_dims, bool_value);
  assign_if_present(values, "output_dynamic_dims", params.output_dynamic_dims, bool_value);
  assign_if_present(values, "enable_fp16", params.enable_fp16, bool_value);
  assign_if_present(values, "enable_cuda_graphs", params.enable_cuda_graphs, bool_value);
  assign_if_present(values, "dla_core", params.dla_core, integer_value<std::int32_t>);
  assign_if_present(values, "dla_gpu_fallback", params.dla_gpu_fallback, bool_value);
  assign_if_present(values, "parallel_inference", params.parallel_inference, bool_value);
  assign_if_present(values, "max_batch_size", params.max_batch_size, integer_value<std::int32_t>);
  assign_if_present(values, "max_memory", params.max_memory, integer_value<std::size_t>);
  if (const ParameterValue* field = values.find("cuda_stream_pool");
      field != nullptr && field->kind != ParameterValueKind::kNone) {
    params.cuda_stream_pool = stream_pool(*field, values.field_path("cuda_stream_pool"));
  }
  if (const ParameterValue* field = values.find("cuda_green_context_pool");
      field != nullptr && field->kind != ParameterValueKind::kNone) {
    params.cuda_green_context_pool =
        green_context_pool(*field, values.field_path("cuda_green_context_pool"));
  }
  return params;
}

void validate_tensor_map(const InferenceDataVecMap& values, std::string_view field,
                         bool require_globally_unique) {
  std::set<std::string, std::less<>> global_names;
  for (const auto& [model, tensor_names] : values) {
    if (tensor_names.empty()) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("{}.{}", field, model),
                      "value=[] accepted=nonempty-sequence");
    }
    std::set<std::string, std::less<>> local_names;
    for (std::size_t index = 0U; index < tensor_names.size(); ++index) {
      const std::string& tensor_name = tensor_names[index];
      const std::string path = fmt::format("{}.{}[{}]", field, model, index);
      if (tensor_name.empty()) {
        parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                        path,
                        "value='' accepted=nonempty-string");
      }
      if (!local_names.emplace(tensor_name).second ||
          (require_globally_unique && !global_names.emplace(tensor_name).second)) {
        parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                        path,
                        fmt::format("value='{}' accepted=unique-tensor-name", tensor_name));
      }
    }
  }
}

void validate_structural_params(const InferenceOpParams& params) {
  if (params.model_path_map.empty()) {
    parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                    "model_path_map",
                    "value={} accepted=nonempty-map");
  }

  for (const auto& [model, model_path] : params.model_path_map) {
    if (model.empty()) {
      parameter_error(
          "INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID", "model_path_map", "accepted=nonempty-key");
    }
    if (model_path.empty()) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("model_path_map.{}", model),
                      "value='' accepted=nonempty-path");
    }
    if (!params.input_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      "input_map",
                      fmt::format("missing_model='{}' accepted=same-model-keys", model));
    }
    if (!params.output_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      "output_map",
                      fmt::format("missing_model='{}' accepted=same-model-keys", model));
    }
  }
  for (const auto& entry : params.input_map) {
    const std::string& model = entry.first;
    if (!params.model_path_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("input_map.{}", model),
                      "accepted=model-key-present-in-model_path_map");
    }
  }
  for (const auto& entry : params.output_map) {
    const std::string& model = entry.first;
    if (!params.model_path_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("output_map.{}", model),
                      "accepted=model-key-present-in-model_path_map");
    }
  }
  for (const auto& entry : params.trt_opt_profile) {
    const std::string& model = entry.first;
    if (!params.model_path_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("trt_opt_profile.{}", model),
                      "accepted=model-key-present-in-model_path_map");
    }
  }
  for (const auto& entry : params.dla_core_map) {
    const std::string& model = entry.first;
    if (!params.model_path_map.contains(model)) {
      parameter_error("INFERENCE_OPERATOR_PARAMETER_VALUE_INVALID",
                      fmt::format("dla_core_map.{}", model),
                      "accepted=model-key-present-in-model_path_map");
    }
  }

  validate_tensor_map(params.input_map, "input_map", false);
  validate_tensor_map(params.output_map, "output_map", true);
}

// Adapts InferenceOp's immutable structural constructor to type-erased registry callbacks.
struct InferenceRegistrationThunks final
    : authoring::ManualOperatorRegistrationThunks<InferenceOp> {
  static OpaqueOperatorRef add(Graph& graph, std::string_view name, InferenceOpParams params) {
    return make_operator_ref(graph.op<InferenceOp>(name, std::move(params)));
  }

  static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
                                        const authoring::MaterializedParameters* params) {
    if (params != nullptr) {
      parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
    }
    return add(graph, name, InferenceOpParams{});
  }

  static void set_params(const OpaqueOperatorRef&, const authoring::MaterializedParameters&) {
    throw RuntimeError{
        Error{ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }
};

// Select and validate the YAML or direct construction mode before mutating the Graph.
expected<OpaqueOperatorRef, Error> add_from_assignments(
    Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments) {
  return InferenceRegistrationThunks::invoke_add([&] {
    if (assignments.empty()) {
      return InferenceRegistrationThunks::add_to_graph(graph, name, nullptr);
    }
    AssignmentView values{assignments, {}};
    if (const ParameterValue* config_path = values.find("config_path")) {
      if (values.size() != 1U) {
        parameter_error("INFERENCE_OPERATOR_CONFIG_PATH_CONFLICT", "config_path");
      }
      InferenceOpParams params;
      const std::string path = string_value(*config_path, "config_path");
      try {
        params = InferenceOpParams::from_yaml(path);
      } catch (const std::exception& exception) {
        parameter_error("INFERENCE_OPERATOR_CONFIG_PATH_INVALID",
                        "config_path",
                        fmt::format("value='{}' reason='{}'", path, exception.what()));
      }
      validate_structural_params(params);
      return InferenceRegistrationThunks::add(graph, name, std::move(params));
    }
    InferenceOpParams params = direct_params(values);
    values.reject_unknown();
    validate_structural_params(params);
    return InferenceRegistrationThunks::add(graph, name, std::move(params));
  });
}

[[nodiscard]] authoring::RegisteredOperatorType make_row() {
  using Thunks = InferenceRegistrationThunks;
  return authoring::RegisteredOperatorType{
      .name = kRegistryName,
      .param_type = &param_type_descriptor<NoParams>(),
      .param_schema = nullptr,
      .parameter_introspection = authoring::OperatorParameterIntrospection::kPassThrough,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &add_from_assignments,
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

[[nodiscard]] const expected<void, Error>& catalog_registration_status() {
  static const expected<void, Error> status = [] {
    return authoring::authoring_registry().register_operator(make_row());
  }();
  return status;
}

[[maybe_unused]] const bool kCatalogRegisteredAtLoad = [] {
  (void)catalog_registration_status();
  return true;
}();

}  // namespace

expected<void, Error> register_inference_operator_catalog() {
  const expected<void, Error>& status = catalog_registration_status();
  if (!status) {
    return make_unexpected(status.error());
  }
  return {};
}

}  // namespace holoscan::ops
