/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>
#include <nanobind/nanobind.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace holoscan {

namespace nb = nanobind;

// Shares one Python callable among every C++ copy of an adapted constructor callback.
//
// The state holds no reference of its own. Each Python object that can keep the callback usable
// (the authoring Graph, a compiled Plan, a running session) adopts it through an owner set and
// contributes one reference, so factory replay and C++ copies never touch Python reference
// counts. Graph and Plan owner sets are visible to Python's cyclic GC, which can therefore
// reclaim a callable that refers back to its Graph. Once the last owner releases the callable,
// invoking the callback fails with CONSTRUCTOR_CALLBACK_RELEASED instead of calling freed memory.
// Representative: test_callback_cycle_through_the_graph_is_collectable.
class PythonCallbackState {
 public:
  // Borrows `callable`; the caller keeps it alive until an owner set adopts this state.
  explicit PythonCallbackState(nb::handle callable) noexcept : callable_(callable.ptr()) {}

  PythonCallbackState(const PythonCallbackState&) = delete;
  PythonCallbackState& operator=(const PythonCallbackState&) = delete;

  // Calls the callable with the GIL held; Python exceptions become CONSTRUCTOR_CALLBACK_FAILED.
  void operator()() const;

 private:
  friend class PythonCallbackOwners;

  // Each requires the GIL. release() drops the reference outside the state lock, because
  // destroying the callable can run arbitrary Python code.
  void adopt() noexcept;
  void release() noexcept;
  int traverse(visitproc visit, void* arg) const noexcept;

  mutable std::mutex mutex_;
  PyObject* callable_{};
  std::size_t owners_{};
};

// Holds one reference to each adopted callback for the Python object that owns this set.
class PythonCallbackOwners {
 public:
  PythonCallbackOwners() = default;
  PythonCallbackOwners(const PythonCallbackOwners&) = delete;
  PythonCallbackOwners& operator=(const PythonCallbackOwners&) = delete;
  // Releases with the GIL when the interpreter is alive; otherwise leaks, as finalization does.
  ~PythonCallbackOwners();

  // Each requires the GIL. reserve() is the only fallible step, so callers can reserve before a
  // committed Graph mutation and adopt afterwards without failing.
  void reserve(std::size_t additional);
  void adopt(const std::shared_ptr<PythonCallbackState>& state) noexcept;
  void adopt_all(const PythonCallbackOwners& other);
  void clear() noexcept;
  int traverse(visitproc visit, void* arg) const noexcept;

  [[nodiscard]] bool empty() const noexcept { return states_.empty(); }

 private:
  std::vector<std::shared_ptr<PythonCallbackState>> states_;
};

// Returns the GC-visible owner set stored on a Graph or Plan object, creating it when `create` is
// true; returns nullptr when it is absent and `create` is false.
PythonCallbackOwners* python_callback_owners(nb::handle front, bool create);

// Registers the GC-tracked owner type; called before Graph and Plan bindings use it.
void init_callback_ownership(nb::module_& module);

}  // namespace holoscan
