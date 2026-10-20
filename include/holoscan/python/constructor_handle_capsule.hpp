/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>

#include <exception>
#include <memory>
#include <string_view>
#include <utility>

#include <holoscan/authoring/constructor_handle.hpp>

namespace holoscan::python {

/// @file
/// @brief Export C++ constructor handles from a provider's Python binding.
///
/// This header has Prototype API stability and requires the Python C API. It uses only
/// limited-API capsule functions, so it works with any binding library (nanobind, pybind11, or
/// hand-written) and does not require the provider to share binding-library internals with
/// Holoscan.
///
/// A provider type participates by exposing a method with this Python signature:
///
/// ```python
/// def __holoscan_constructor_handle__(self, type_name: str) -> object: ...
/// ```
///
/// Holoscan calls it only for an explicitly declared constructor-handle argument, passing the
/// stable name of the interface type that argument expects. The method returns the capsule made by
/// \ref export_constructor_handle, or `NotImplemented` when the object cannot provide that type.
///
/// The exported owner must keep the C++ object valid independently of its Python wrapper. A C++
/// `std::shared_ptr` should therefore own the object from the moment it is created:
///
/// - nanobind: bind the constructor with `nb::new_([](Args... args) { return
///   std::make_shared<T>(args...); })` rather than `nb::init<Args...>()`, derive `T` from
///   `std::enable_shared_from_this<T>`, take `T&` in the method, and export
///   `self.shared_from_this()`. With `nb::init`, the Python wrapper owns the object: nanobind's
///   `std::shared_ptr<T>` conversion then creates an owner that keeps the wrapper alive and needs
///   the interpreter to release it, and `shared_from_this()` throws `std::bad_weak_ptr` until
///   such a conversion has happened at least once.
/// - pybind11: a `std::shared_ptr<T>` holder without a Python trampoline already satisfies this.
///
/// One object may export several interface types. For example, a sensor class can answer requests
/// for both its own type and a shared `CameraSensor` interface by checking \p type_name and
/// passing `std::shared_ptr<CameraSensor>(self.shared_from_this())` for the interface.

/// Capsule name of the version-1 constructor-handle exchange.
inline constexpr const char* kConstructorHandleCapsuleName = "holoscan.constructor_handle.v1";

namespace detail {
inline void destroy_constructor_handle_capsule(PyObject* capsule) noexcept {
  void* box = PyCapsule_GetPointer(capsule, kConstructorHandleCapsuleName);
  if (box == nullptr) {
    PyErr_Clear();
    return;
  }
  authoring::release_constructor_handle_box(box);
}
}  // namespace detail

/// @brief Export one C++ shared owner as a version-1 constructor-handle capsule.
///
/// Each call returns a fresh capsule that owns one independent handle, so the same Python object
/// can be passed to several Operators.
/// @param requested_type Stable type name that Holoscan passed to the protocol method.
/// @param type Typed descriptor binding for \p T.
/// @param owner C++ shared owner; must not depend on the Python object for its lifetime.
/// @return A new reference to the capsule, a new reference to `NotImplemented` when
/// \p requested_type names a different type, or `nullptr` with a Python exception set.
template <typename T>
[[nodiscard]] PyObject* export_constructor_handle(std::string_view requested_type,
                                                  const authoring::ConstructorHandleType<T>& type,
                                                  std::shared_ptr<T> owner) noexcept {
  if (requested_type != type.descriptor().name) {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }
  // Reject a null owner through the Python C API so no C++ diagnostic is built for it.
  if (!owner) {
    PyErr_SetString(PyExc_ValueError, "AUTHORING_CONSTRUCTOR_HANDLE_NULL");
    return nullptr;
  }
  // Every C++ step that can allocate stays inside this boundary; the function is noexcept.
  void* box = nullptr;
  try {
    auto handle = authoring::ConstructorHandle::from_shared(type, std::move(owner));
    if (!handle) {
      PyErr_SetString(PyExc_ValueError, "AUTHORING_CONSTRUCTOR_HANDLE_NULL");
      return nullptr;
    }
    box = authoring::box_constructor_handle(*handle);
  } catch (const std::exception&) {
    PyErr_NoMemory();
    return nullptr;
  }
  PyObject* capsule = PyCapsule_New(
      box, kConstructorHandleCapsuleName, &detail::destroy_constructor_handle_capsule);
  if (capsule == nullptr) {
    authoring::release_constructor_handle_box(box);
    return nullptr;
  }
  if (PyCapsule_SetContext(capsule,
                           const_cast<void*>(authoring::constructor_handle_box_identity())) != 0) {
    Py_DECREF(capsule);
    return nullptr;
  }
  return capsule;
}

}  // namespace holoscan::python
