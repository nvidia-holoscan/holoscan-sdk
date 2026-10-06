/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/external_topic.hpp>

#include "clocks.hpp"
#include "payload.hpp"
#include "tensor_authoring.hpp"

namespace holoscan {

namespace nb = nanobind;

// Owns Python topic-option values whose C++ TopicOptions representation contains borrowed views.
class PyTopicOptions {
 public:
  PyTopicOptions() = default;
  explicit PyTopicOptions(const nb::kwargs& kwargs);

  [[nodiscard]] TopicOptions value() const;
  [[nodiscard]] const std::optional<std::string>& endpoint_name() const noexcept;
  [[nodiscard]] const std::optional<MemoryConversionPolicy>& memory_conversion() const noexcept;
  [[nodiscard]] std::optional<PyPayloadType> schema() const;
  [[nodiscard]] std::optional<PyClockKey> capture_time() const;
  [[nodiscard]] ClockHealthOptions* clock_health() noexcept;
  [[nodiscard]] TopicAttachPolicy* attach() noexcept;
  [[nodiscard]] EndpointBounds* bounds() noexcept;
  [[nodiscard]] std::shared_ptr<PyTensorBoundaryContract> tensor() const noexcept;
  [[nodiscard]] nb::object tensor_fields() const;
  void set(std::string_view name, nb::handle value);

  [[nodiscard]] bool has_descriptor_tensor_fields() const noexcept;
  [[nodiscard]] std::vector<std::pair<std::string_view, TensorBoundaryContract>>
  descriptor_tensor_fields() const;

 private:
  void apply(const std::string& name, nb::handle value);

  std::optional<std::string> endpoint_name_;
  std::optional<MemoryConversionPolicy> memory_conversion_;
  const authoring::RegisteredPayloadType* schema_payload_{};
  std::optional<SchemaIdentity> schema_;
  std::optional<RequireCaptureTime> capture_time_;
  std::optional<ClockHealthOptions> clock_health_;
  std::shared_ptr<PyTensorBoundaryContract> tensor_;
  std::vector<std::pair<std::string, std::shared_ptr<PyTensorBoundaryContract>>>
      descriptor_tensor_fields_;
  bool has_descriptor_tensor_fields_{};
  TopicAttachPolicy attach_;
  EndpointBounds bounds_;
};

using DescriptorTensorFields = std::vector<std::pair<std::string_view, TensorBoundaryContract>>;

// Resolves and validates field-qualified Tensor contracts for one descriptor payload row.
[[nodiscard]] std::optional<DescriptorTensorFields> descriptor_topic_fields(
    const authoring::RegisteredPayloadType& row, const PyTopicOptions& options);

// Exposes an imported topic through a weak graph-scoped port without extending Graph lifetime.
class PyImportedTopic {
 public:
  PyImportedTopic(const authoring::OpaqueImportedEndpoint& endpoint,
                  const authoring::RegisteredPayloadType* payload, std::string uri);

  [[nodiscard]] nb::object output() const;
  [[nodiscard]] bool graph_valid() const noexcept;
  [[nodiscard]] std::string repr() const;

 private:
  authoring::OpaquePortRef port_;
  const authoring::RegisteredPayloadType* payload_{};
  std::string uri_;
};

// Exposes an exported topic through a weak graph-scoped port without extending Graph lifetime.
class PyExportedTopic {
 public:
  PyExportedTopic(const authoring::OpaqueExportedEndpoint& endpoint,
                  const authoring::RegisteredPayloadType* payload, std::string uri);

  [[nodiscard]] nb::object input() const;
  [[nodiscard]] bool graph_valid() const noexcept;
  [[nodiscard]] std::string repr() const;

 private:
  authoring::OpaquePortRef port_;
  const authoring::RegisteredPayloadType* payload_{};
  std::string uri_;
};

void init_topics(nb::module_& module);

}  // namespace holoscan
