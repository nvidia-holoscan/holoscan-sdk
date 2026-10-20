/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "callback_ownership.hpp"

#include <nanobind/nanobind.h>

#include <array>
#include <memory>
#include <mutex>
#include <utility>

#include <holoscan/core/errors.hpp>

namespace holoscan {

namespace {

constexpr const char* kCallbackOwnersAttribute = "__holoscan_constructor_callbacks__";

// The Python object that carries one owner set in a Graph's or Plan's `__dict__`. Its traverse and
// clear slots expose the owned callables to Python's cyclic GC.
struct CallbackOwnersObject {
  PythonCallbackOwners owners;
};

int callback_owners_traverse(PyObject* self, visitproc visit, void* arg) {
  // A heap type's instances must visit the type itself.
  // CPython owns the temporary declaration inside Py_VISIT.
  // NOLINTNEXTLINE(misc-const-correctness)
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) {
    return 0;
  }
  return nb::inst_ptr<CallbackOwnersObject>(self)->owners.traverse(visit, arg);
}

int callback_owners_clear(PyObject* self) {
  if (nb::inst_ready(self)) {
    nb::inst_ptr<CallbackOwnersObject>(self)->owners.clear();
  }
  return 0;
}

const std::array<PyType_Slot, 3> callback_owners_slots{{
    {Py_tp_traverse, reinterpret_cast<void*>(callback_owners_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(callback_owners_clear)},
    {0, nullptr},
}};

[[noreturn]] void throw_callback_error(ErrorCode code, const char* token) {
  throw RuntimeError{Error{code, token}};
}

}  // namespace

void PythonCallbackState::operator()() const {
  if (!nb::is_alive()) {
    throw_callback_error(ErrorCode::kNotReady, "CONSTRUCTOR_CALLBACK_INTERPRETER_UNAVAILABLE");
  }
  const nb::gil_scoped_acquire gil;
  if (!gil.is_valid()) {
    throw_callback_error(ErrorCode::kNotReady, "CONSTRUCTOR_CALLBACK_INTERPRETER_UNAVAILABLE");
  }
  nb::object callable;
  {
    const std::scoped_lock lock{mutex_};
    if (owners_ == 0U) {
      throw_callback_error(ErrorCode::kFailure, "CONSTRUCTOR_CALLBACK_RELEASED");
    }
    callable = nb::borrow<nb::object>(nb::handle{callable_});
  }
  try {
    callable();
  } catch (nb::python_error& error) {
    // Report the traceback through sys.unraisablehook while the GIL is held, so no Python
    // exception object escapes into C++ lifecycle handling or teardown.
    error.discard_as_unraisable("Holoscan constructor callback");
    throw_callback_error(ErrorCode::kFailure, "CONSTRUCTOR_CALLBACK_FAILED");
  }
}

void PythonCallbackState::adopt() noexcept {
  const std::scoped_lock lock{mutex_};
  Py_INCREF(callable_);
  ++owners_;
}

void PythonCallbackState::release() noexcept {
  PyObject* released{};
  {
    const std::scoped_lock lock{mutex_};
    released = callable_;
    if (--owners_ == 0U) {
      callable_ = nullptr;
    }
  }
  Py_DECREF(released);
}

int PythonCallbackState::traverse(visitproc visit, void* arg) const noexcept {
  PyObject* callable{};
  {
    const std::scoped_lock lock{mutex_};
    callable = callable_;
  }
  // Each owner holds its own reference, so every owner visits the callable once.
  // CPython owns the temporary declaration inside Py_VISIT.
  // NOLINTNEXTLINE(misc-const-correctness)
  Py_VISIT(callable);
  return 0;
}

PythonCallbackOwners::~PythonCallbackOwners() {
  if (states_.empty() || !nb::is_alive()) {
    return;
  }
  const nb::gil_scoped_acquire gil;
  if (gil.is_valid()) {
    clear();
  }
}

void PythonCallbackOwners::reserve(std::size_t additional) {
  states_.reserve(states_.size() + additional);
}

void PythonCallbackOwners::adopt(const std::shared_ptr<PythonCallbackState>& state) noexcept {
  // Callers reserve capacity first, so this push_back does not allocate.
  state->adopt();
  states_.push_back(state);
}

void PythonCallbackOwners::adopt_all(const PythonCallbackOwners& other) {
  reserve(other.states_.size());
  for (const auto& state : other.states_) {
    adopt(state);
  }
}

void PythonCallbackOwners::clear() noexcept {
  // Release from a detached list: dropping a callable can re-enter this owner set.
  auto states = std::exchange(states_, {});
  for (const auto& state : states) {
    state->release();
  }
}

int PythonCallbackOwners::traverse(visitproc visit, void* arg) const noexcept {
  for (const auto& state : states_) {
    if (const int result = state->traverse(visit, arg); result != 0) {
      return result;
    }
  }
  return 0;
}

PythonCallbackOwners* python_callback_owners(nb::handle front, bool create) {
  const nb::dict attributes = nb::cast<nb::dict>(front.attr("__dict__"));
  if (attributes.contains(kCallbackOwnersAttribute)) {
    return &nb::cast<CallbackOwnersObject&>(attributes[kCallbackOwnersAttribute]).owners;
  }
  if (!create) {
    return nullptr;
  }
  nb::object owners = nb::type<CallbackOwnersObject>()();
  attributes[kCallbackOwnersAttribute] = owners;
  return &nb::cast<CallbackOwnersObject&>(owners).owners;
}

void init_callback_ownership(nb::module_& module) {
  nb::class_<CallbackOwnersObject>(
      module, "_ConstructorCallbackOwners", nb::type_slots(callback_owners_slots.data()))
      .def(nb::init<>());
}

}  // namespace holoscan
