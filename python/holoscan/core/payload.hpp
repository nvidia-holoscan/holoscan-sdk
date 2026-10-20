/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan {

namespace nb = nanobind;

// Provides a stable Python view of a process-lifetime payload registry row without copying its
// type-erased callbacks.
class PyPayloadType {
 public:
  explicit PyPayloadType(const authoring::RegisteredPayloadType* row) : row_(row) {}

  [[nodiscard]] std::string name() const;
  [[nodiscard]] std::uint32_t epoch() const noexcept;
  [[nodiscard]] std::uint32_t transport_profile_tag() const noexcept;
  [[nodiscard]] std::string repr() const;
  [[nodiscard]] nb::object equals(nb::handle other) const;
  [[nodiscard]] Py_hash_t hash() const;
  [[nodiscard]] const authoring::RegisteredPayloadType& row() const noexcept { return *row_; }

 private:
  const authoring::RegisteredPayloadType* row_{};
};

[[nodiscard]] PyPayloadType payload_type(std::string name);
[[nodiscard]] const authoring::RegisteredPayloadType& python_payload_registration(nb::handle value);
[[nodiscard]] const authoring::RegisteredPayloadType& payload_row(nb::handle value);

[[nodiscard]] ParamScalarValue payload_scalar_from_python(nb::handle value,
                                                          ScalarIdentityKind kind);

[[nodiscard]] nb::object payload_scalar_to_python(ParamScalarValue value);

/// Convert one raw-POD object's native bytes to a dict of its fields: a number per scalar field and
/// bytes for std::byte arrays and a tuple for numeric fixed arrays. @p array_fields is the row's
/// per-field array flags.
[[nodiscard]] nb::dict payload_pod_to_python(const PodIdentityRecord& pod,
                                             std::span<const bool> array_fields,
                                             std::span<const std::byte> native);

/// Convert a mapping with exactly the raw-POD field names, or any buffer holding the exact native
/// object bytes, to native object bytes laid out as @p pod describes. @p array_fields is the row's
/// per-field array flags.
[[nodiscard]] std::vector<std::byte> payload_pod_from_python(nb::handle value,
                                                             const PodIdentityRecord& pod,
                                                             std::span<const bool> array_fields);

void init_payload(nb::module_& module);

}  // namespace holoscan
