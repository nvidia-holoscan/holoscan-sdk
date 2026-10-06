/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <memory>

#include <holoscan/core/port.hpp>

namespace holoscan {

namespace nb = nanobind;

/// @brief Retain one interpreter-owned Python value for process-local typed delivery.
class PyObjectPayload {
 public:
  /// @brief Create an empty payload used only before a Python value is assigned.
  PyObjectPayload() = default;

  /// @brief Retain one Python value while the caller holds the GIL.
  /// @param value Borrowed Python value promoted to shared ownership.
  explicit PyObjectPayload(nb::handle value);

  /// @brief Return the retained Python value while the caller holds the GIL.
  [[nodiscard]] nb::object borrow() const;

  /// @brief Share the retained value without changing its Python reference count.
  [[nodiscard]] std::shared_ptr<void> share() const noexcept;

 private:
  struct PythonDeleter {
    void operator()(PyObject* value) const noexcept;
  };

  [[nodiscard]] static std::shared_ptr<PyObject> retain(nb::handle value);

  std::shared_ptr<PyObject> value_;
};

void init_py_object_payload(nb::module_& module);

template <>
[[nodiscard]] consteval SchemaIdentity schema_identity<PyObjectPayload>() {
  return detail::py_object_payload_schema_identity();
}

namespace detail {

template <>
[[nodiscard]] const RuntimePortTypeOps* runtime_port_type_ops<PyObjectPayload>() noexcept;

}  // namespace detail
}  // namespace holoscan
