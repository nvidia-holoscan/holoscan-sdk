/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topics.hpp"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>

#include "graph.hpp"
#include "graph_pydoc.hpp"
#include "module_name.hpp"
#include "time.hpp"

namespace holoscan {
namespace {

[[noreturn]] void throw_topic_error(std::string_view token, std::string_view field = {}) {
  std::string message =
      field.empty() ? std::string{token} : fmt::format("{} field_name={}", token, field);
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

std::chrono::nanoseconds python_duration(nb::handle value, std::string_view field) {
  return std::chrono::nanoseconds{
      python_duration_nanoseconds(value, field, "AUTHORING_TOPIC_OPTION_TYPE_MISMATCH")};
}

nb::object optional_python_duration(const std::optional<std::chrono::nanoseconds>& value) {
  return value ? nb::cast(PyDuration{value->count()}) : nb::none();
}

void set_optional_python_duration(std::optional<std::chrono::nanoseconds>& destination,
                                  nb::handle value, std::string_view field) {
  destination.reset();
  if (!value.is_none()) {
    destination = python_duration(value, field);
  }
}

template <typename EnumT>
EnumT enum_option(nb::handle value, std::string_view field) {
  try {
    return nb::cast<EnumT>(value);
  } catch (const nb::cast_error&) {
    throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", field);
  }
}

template <typename T>
T unsigned_option(nb::handle value, std::string_view field) {
  static_assert(std::is_unsigned_v<T>);
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", field);
  }
  const unsigned long long result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() || result > std::numeric_limits<T>::max()) {
    PyErr_Clear();
    throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", field);
  }
  return static_cast<T>(result);
}

}  // namespace

PyTopicOptions::PyTopicOptions(const nb::kwargs& kwargs) {
  const nb::dict values = nb::cast<nb::dict>(kwargs);
  for (const nb::handle item_handle : values.items()) {
    const nb::tuple item = nb::cast<nb::tuple>(item_handle);
    apply(nb::cast<std::string>(item[0]), item[1]);
  }
}

TopicOptions PyTopicOptions::value() const {
  TopicOptions result;
  if (endpoint_name_) {
    result.endpoint_name = *endpoint_name_;
  }
  result.memory_conversion = memory_conversion_;
  result.schema = schema_;
  result.capture_time = capture_time_;
  result.clock_health = clock_health_;
  result.attach = attach_;
  if (tensor_) {
    result.tensor = tensor_->value();
  }
  result.bounds = bounds_;
  return result;
}

const std::optional<std::string>& PyTopicOptions::endpoint_name() const noexcept {
  return endpoint_name_;
}

const std::optional<MemoryConversionPolicy>& PyTopicOptions::memory_conversion() const noexcept {
  return memory_conversion_;
}

std::optional<PyPayloadType> PyTopicOptions::schema() const {
  return schema_payload_ == nullptr ? std::nullopt : std::optional<PyPayloadType>{schema_payload_};
}

std::optional<PyClockKey> PyTopicOptions::capture_time() const {
  return capture_time_ ? std::optional<PyClockKey>{capture_time_->clock} : std::nullopt;
}

ClockHealthOptions* PyTopicOptions::clock_health() noexcept {
  return clock_health_ ? &*clock_health_ : nullptr;
}

TopicAttachPolicy* PyTopicOptions::attach() noexcept {
  return &attach_;
}

EndpointBounds* PyTopicOptions::bounds() noexcept {
  return &bounds_;
}

std::shared_ptr<PyTensorBoundaryContract> PyTopicOptions::tensor() const noexcept {
  return tensor_;
}

nb::object PyTopicOptions::tensor_fields() const {
  if (!has_descriptor_tensor_fields_) {
    return nb::none();
  }
  nb::dict result;
  for (const auto& [name, contract] : descriptor_tensor_fields_) {
    result[nb::str(name.c_str())] = nb::cast(contract);
  }
  return result;
}

void PyTopicOptions::set(std::string_view name, nb::handle value) {
  apply(std::string{name}, value);
}

bool PyTopicOptions::has_descriptor_tensor_fields() const noexcept {
  return has_descriptor_tensor_fields_;
}

std::vector<std::pair<std::string_view, TensorBoundaryContract>>
PyTopicOptions::descriptor_tensor_fields() const {
  std::vector<std::pair<std::string_view, TensorBoundaryContract>> result;
  result.reserve(descriptor_tensor_fields_.size());
  for (const auto& [name, contract] : descriptor_tensor_fields_) {
    result.emplace_back(name, contract->value());
  }
  return result;
}

// Keep the public keyword vocabulary and its reset-before-assign semantics in one dispatch chain;
// extracting per-option helpers would duplicate the uniform error translation without simplifying
// the state transitions.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void PyTopicOptions::apply(const std::string& name, nb::handle value) {
  if (name == "endpoint_name") {
    endpoint_name_.reset();
    if (!value.is_none()) {
      if (!nb::isinstance<nb::str>(value)) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
      endpoint_name_ = nb::cast<std::string>(value);
    }
  } else if (name == "memory_conversion") {
    memory_conversion_.reset();
    if (!value.is_none()) {
      memory_conversion_ = enum_option<MemoryConversionPolicy>(value, name);
    }
  } else if (name == "schema") {
    schema_.reset();
    schema_payload_ = nullptr;
    if (!value.is_none()) {
      if (!nb::isinstance<PyPayloadType>(value)) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
      const authoring::RegisteredPayloadType& payload = payload_row(value);
      if (!payload.identity.schema) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
      schema_ = schema_identity_view(*payload.identity.schema);
      schema_payload_ = &payload;
    }
  } else if (name == "capture_time") {
    capture_time_.reset();
    if (!value.is_none()) {
      if (!nb::isinstance<PyClockKey>(value)) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
      capture_time_ = RequireCaptureTime{nb::cast<const PyClockKey&>(value).value()};
    }
  } else if (name == "clock_health") {
    clock_health_.reset();
    if (!value.is_none()) {
      try {
        clock_health_ = nb::cast<ClockHealthOptions>(value);
      } catch (const nb::cast_error&) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
    }
  } else if (name == "attach") {
    attach_ = {};
    if (!value.is_none()) {
      try {
        attach_ = nb::cast<TopicAttachPolicy>(value);
      } catch (const nb::cast_error&) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
    }
  } else if (name == "bounds") {
    bounds_ = {};
    if (!value.is_none()) {
      try {
        bounds_ = nb::cast<EndpointBounds>(value);
      } catch (const nb::cast_error&) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
    }
  } else if (name == "tensor") {
    std::shared_ptr<PyTensorBoundaryContract> replacement;
    if (!value.is_none()) {
      try {
        replacement =
            std::make_shared<PyTensorBoundaryContract>(nb::cast<PyTensorBoundaryContract>(value));
      } catch (const nb::cast_error&) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
    }
    tensor_ = std::move(replacement);
  } else if (name == "tensor_fields") {
    if (value.is_none()) {
      descriptor_tensor_fields_.clear();
      has_descriptor_tensor_fields_ = false;
      return;
    }
    if (!nb::isinstance<nb::dict>(value)) {
      throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
    }
    std::vector<std::pair<std::string, std::shared_ptr<PyTensorBoundaryContract>>> replacement;
    for (const nb::handle item_handle : nb::cast<nb::dict>(value).items()) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      if (!nb::isinstance<nb::str>(item[0])) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
      try {
        replacement.emplace_back(nb::cast<std::string>(item[0]),
                                 std::make_shared<PyTensorBoundaryContract>(
                                     nb::cast<PyTensorBoundaryContract>(item[1])));
      } catch (const nb::cast_error&) {
        throw_topic_error("AUTHORING_TOPIC_OPTION_TYPE_MISMATCH", name);
      }
    }
    descriptor_tensor_fields_ = std::move(replacement);
    has_descriptor_tensor_fields_ = true;
  } else {
    throw_topic_error("AUTHORING_TOPIC_OPTION_UNKNOWN", name);
  }
}

std::optional<DescriptorTensorFields> descriptor_topic_fields(
    const authoring::RegisteredPayloadType& row, const PyTopicOptions& options) {
  if (!row.descriptor_tensor_fields.empty()) {
    if (!options.has_descriptor_tensor_fields()) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED"}};
    }
    return options.descriptor_tensor_fields();
  }
  if (options.has_descriptor_tensor_fields()) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED"}};
  }
  return std::nullopt;
}

PyImportedTopic::PyImportedTopic(const authoring::OpaqueImportedEndpoint& endpoint,
                                 const authoring::RegisteredPayloadType* payload, std::string uri)
    : port_(endpoint.port()), payload_(payload), uri_(std::move(uri)) {}

nb::object PyImportedTopic::output() const {
  return python_external_topic_port(port_, PortDirection::kOutput, *payload_, uri_);
}

bool PyImportedTopic::graph_valid() const noexcept {
  return static_cast<bool>(port_);
}

std::string PyImportedTopic::repr() const {
  return fmt::format("<ImportedTopic '{}'>", uri_);
}

PyExportedTopic::PyExportedTopic(const authoring::OpaqueExportedEndpoint& endpoint,
                                 const authoring::RegisteredPayloadType* payload, std::string uri)
    : port_(endpoint.port()), payload_(payload), uri_(std::move(uri)) {}

nb::object PyExportedTopic::input() const {
  return python_external_topic_port(port_, PortDirection::kInput, *payload_, uri_);
}

bool PyExportedTopic::graph_valid() const noexcept {
  return static_cast<bool>(port_);
}

std::string PyExportedTopic::repr() const {
  return fmt::format("<ExportedTopic '{}'>", uri_);
}

void init_topics(nb::module_& module) {
  python::bind_enum_in_module<PeerRequirement>(module, "holoscan.topics", "PeerRequirement")
      .value("OPTIONAL", PeerRequirement::kOptional)
      .value("REQUIRED", PeerRequirement::kRequired);
  python::bind_class_in_module<TopicAttachPolicy>(module, "holoscan.topics", "TopicAttachPolicy")
      .def(
          "__init__",
          [](TopicAttachPolicy* self, PeerRequirement peer, nb::handle max_wait) {
            new (self)
                TopicAttachPolicy{peer,
                                  max_wait.is_none() ? std::optional<std::chrono::nanoseconds>{}
                                                     : std::optional<std::chrono::nanoseconds>{
                                                           python_duration(max_wait, "max_wait")}};
          },
          nb::kw_only(),
          nb::arg("peer") = PeerRequirement::kOptional,
          nb::arg("max_wait").none() = nb::none())
      .def_rw("peer", &TopicAttachPolicy::peer)
      .def_prop_rw(
          "max_wait",
          [](const TopicAttachPolicy& policy) { return optional_python_duration(policy.max_wait); },
          [](TopicAttachPolicy& policy, nb::handle value) {
            set_optional_python_duration(policy.max_wait, value, "max_wait");
          },
          nb::for_setter(nb::arg().none()));
  python::bind_enum_in_module<ClockHealthCapability>(
      module, "holoscan.topics", "ClockHealthCapability")
      .value("UNAVAILABLE", ClockHealthCapability::kUnavailable)
      .value("STATUS_REPORTS", ClockHealthCapability::kStatusReports);
  python::bind_enum_in_module<ClockHealthRequirement>(
      module, "holoscan.topics", "ClockHealthRequirement")
      .value("NOT_REQUIRED", ClockHealthRequirement::kNotRequired)
      .value("FRESH_HEALTHY_REPORT", ClockHealthRequirement::kFreshHealthyReport);
  python::bind_class_in_module<ClockHealthOptions>(module, "holoscan.topics", "ClockHealthOptions")
      .def(
          "__init__",
          [](ClockHealthOptions* self,
             ClockHealthCapability capability,
             ClockHealthRequirement requirement,
             nb::handle max_report_age) {
            new (self) ClockHealthOptions{
                capability,
                requirement,
                max_report_age.is_none() ? std::optional<std::chrono::nanoseconds>{}
                                         : std::optional<std::chrono::nanoseconds>{
                                               python_duration(max_report_age, "max_report_age")}};
          },
          nb::kw_only(),
          nb::arg("capability") = ClockHealthCapability::kUnavailable,
          nb::arg("requirement") = ClockHealthRequirement::kNotRequired,
          nb::arg("max_report_age").none() = nb::none())
      .def_rw("capability", &ClockHealthOptions::capability)
      .def_rw("requirement", &ClockHealthOptions::requirement)
      .def_prop_rw(
          "max_report_age",
          [](const ClockHealthOptions& options) {
            return optional_python_duration(options.max_report_age);
          },
          [](ClockHealthOptions& options, nb::handle value) {
            set_optional_python_duration(options.max_report_age, value, "max_report_age");
          },
          nb::for_setter(nb::arg().none()));
  python::bind_class_in_module<PublisherEndpointBounds>(
      module, "holoscan.topics", "PublisherEndpointBounds")
      .def(
          "__init__",
          [](PublisherEndpointBounds* self, nb::handle max_loaned_samples) {
            new (self) PublisherEndpointBounds{
                max_loaned_samples.is_none()
                    ? std::optional<std::uint32_t>{}
                    : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(
                          max_loaned_samples, "max_loaned_samples")}};
          },
          nb::kw_only(),
          nb::arg("max_loaned_samples").none() = nb::none())
      .def_rw("max_loaned_samples", &PublisherEndpointBounds::max_loaned_samples, nb::arg().none());
  python::bind_class_in_module<SubscriberEndpointBounds>(
      module, "holoscan.topics", "SubscriberEndpointBounds")
      .def(
          "__init__",
          [](SubscriberEndpointBounds* self,
             nb::handle queue_depth,
             nb::handle max_borrowed_samples) {
            new (self) SubscriberEndpointBounds{
                queue_depth.is_none() ? std::optional<std::uint32_t>{}
                                      : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(
                                            queue_depth, "queue_depth")},
                max_borrowed_samples.is_none()
                    ? std::optional<std::uint32_t>{}
                    : std::optional<std::uint32_t>{unsigned_option<std::uint32_t>(
                          max_borrowed_samples, "max_borrowed_samples")}};
          },
          nb::kw_only(),
          nb::arg("queue_depth").none() = nb::none(),
          nb::arg("max_borrowed_samples").none() = nb::none())
      .def_rw("queue_depth", &SubscriberEndpointBounds::queue_depth, nb::arg().none())
      .def_rw("max_borrowed_samples",
              &SubscriberEndpointBounds::max_borrowed_samples,
              nb::arg().none());
  python::bind_class_in_module<EndpointBounds>(module, "holoscan.topics", "EndpointBounds")
      .def(
          "__init__",
          [](EndpointBounds* self, nb::handle publisher, nb::handle subscriber) {
            new (self) EndpointBounds{
                publisher.is_none()
                    ? std::optional<PublisherEndpointBounds>{}
                    : std::optional<PublisherEndpointBounds>{nb::cast<PublisherEndpointBounds>(
                          publisher)},
                subscriber.is_none() ? std::optional<SubscriberEndpointBounds>{}
                                     : std::optional<SubscriberEndpointBounds>{
                                           nb::cast<SubscriberEndpointBounds>(subscriber)}};
          },
          nb::kw_only(),
          nb::arg("publisher").none() = nb::none(),
          nb::arg("subscriber").none() = nb::none())
      .def_prop_rw(
          "publisher",
          [](EndpointBounds& bounds) -> PublisherEndpointBounds* {
            return bounds.publisher ? &*bounds.publisher : nullptr;
          },
          [](EndpointBounds& bounds, std::optional<PublisherEndpointBounds> value) {
            bounds.publisher = std::move(value);
          },
          nb::for_setter(nb::arg().none()),
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "subscriber",
          [](EndpointBounds& bounds) -> SubscriberEndpointBounds* {
            return bounds.subscriber ? &*bounds.subscriber : nullptr;
          },
          [](EndpointBounds& bounds, std::optional<SubscriberEndpointBounds> value) {
            bounds.subscriber = std::move(value);
          },
          nb::for_setter(nb::arg().none()),
          nb::rv_policy::reference_internal);
  python::bind_class_in_module<PyTopicOptions>(
      module, "holoscan.topics", "TopicOptions", doc::graph::kTopicOptions)
      .def(
          "__init__",
          [](PyTopicOptions* self, const nb::kwargs& options) {
            new (self) PyTopicOptions{options};
          },
          nb::sig("def __init__(self, *, endpoint_name: str | None = None, "
                  "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None, "
                  "schema: holoscan.payload.PayloadType | None = None, "
                  "capture_time: holoscan.clocks.ClockKey | None = None, "
                  "clock_health: holoscan.topics.ClockHealthOptions | None = None, "
                  "attach: holoscan.topics.TopicAttachPolicy = ..., "
                  "bounds: holoscan.topics.EndpointBounds = ..., "
                  "tensor: holoscan.tensor.TensorBoundaryContract | None = None, "
                  "tensor_fields: dict[str, holoscan.tensor.TensorBoundaryContract] = ...) -> "
                  "None"))
      .def_prop_rw(
          "endpoint_name",
          &PyTopicOptions::endpoint_name,
          [](PyTopicOptions& options, nb::handle value) { options.set("endpoint_name", value); },
          nb::for_setter(nb::arg().none()))
      .def_prop_rw(
          "memory_conversion",
          &PyTopicOptions::memory_conversion,
          [](PyTopicOptions& options, nb::handle value) {
            options.set("memory_conversion", value);
          },
          nb::for_setter(nb::arg().none()))
      .def_prop_rw(
          "schema",
          &PyTopicOptions::schema,
          [](PyTopicOptions& options, nb::handle value) { options.set("schema", value); },
          nb::for_setter(nb::arg().none()))
      .def_prop_rw(
          "capture_time",
          &PyTopicOptions::capture_time,
          [](PyTopicOptions& options, nb::handle value) { options.set("capture_time", value); },
          nb::for_setter(nb::arg().none()))
      .def_prop_rw(
          "clock_health",
          &PyTopicOptions::clock_health,
          [](PyTopicOptions& options, nb::handle value) { options.set("clock_health", value); },
          nb::for_setter(nb::arg().none()),
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "attach",
          &PyTopicOptions::attach,
          [](PyTopicOptions& options, nb::handle value) { options.set("attach", value); },
          nb::for_setter(nb::arg().none()),
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "bounds",
          &PyTopicOptions::bounds,
          [](PyTopicOptions& options, nb::handle value) { options.set("bounds", value); },
          nb::for_setter(nb::arg().none()),
          nb::rv_policy::reference_internal)
      .def_prop_rw(
          "tensor",
          &PyTopicOptions::tensor,
          [](PyTopicOptions& options, nb::handle value) { options.set("tensor", value); },
          nb::for_setter(nb::arg().none()))
      .def_prop_rw(
          "tensor_fields",
          &PyTopicOptions::tensor_fields,
          [](PyTopicOptions& options, nb::handle value) { options.set("tensor_fields", value); },
          nb::for_setter(nb::arg().none()));
  auto imported_topic_type = python::bind_class_in_module<PyImportedTopic>(
      module, "holoscan.topics", "ImportedTopic", doc::graph::kImportedTopic);
  imported_topic_type.def_prop_ro("output", &PyImportedTopic::output)
      .def_prop_ro("graph_valid", &PyImportedTopic::graph_valid)
      .def("__repr__", &PyImportedTopic::repr);
  auto exported_topic_type = python::bind_class_in_module<PyExportedTopic>(
      module, "holoscan.topics", "ExportedTopic", doc::graph::kExportedTopic);
  exported_topic_type.def_prop_ro("input", &PyExportedTopic::input)
      .def_prop_ro("graph_valid", &PyExportedTopic::graph_valid)
      .def("__repr__", &PyExportedTopic::repr);
}

}  // namespace holoscan
