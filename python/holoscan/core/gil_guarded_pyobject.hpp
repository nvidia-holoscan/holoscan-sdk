/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_GIL_GUARDED_PYOBJECT_HPP
#define PYHOLOSCAN_CORE_GIL_GUARDED_PYOBJECT_HPP

#include <pybind11/pybind11.h>

#include <holoscan/logger/logger.hpp>

namespace py = pybind11;

namespace holoscan {

/**
 * @brief A wrapper around pybind11::object class that allows to be destroyed
 * with acquiring the GIL.
 *
 * This class is used in PyInputContext::py_receive() and PyOutputContext::py_emit() methods
 * to allow the Python code (decreasing the reference count) to be executed with the GIL acquired.
 *
 * Without this wrapper, the Python code would be executed without the GIL by the GXF execution
 * engine that destroys the Entity object and executes Message::~Message() and
 * pybind11::object::~object(), which would cause a segfault.
 */
class GILGuardedPyObject {
 public:
  GILGuardedPyObject() = delete;
  explicit GILGuardedPyObject(const py::object& obj) : obj_(obj) {}
  explicit GILGuardedPyObject(py::object&& obj) : obj_(obj) {}

  py::object& obj() { return obj_; }

  ~GILGuardedPyObject() {
    // Acquire GIL before destroying the PyObject
    try {
      py::gil_scoped_acquire scope_guard;
      py::handle handle = obj_.release();
      if (handle) {
        handle.dec_ref();
      }
    } catch (py::error_already_set& eas) {
      // Discard any Python error using Python APIs
      // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html#handling-unraisable-exceptions
      try {
        // ignore potential runtime_error from release() call internal to discard_as_unraisable
        eas.discard_as_unraisable(__func__);
      } catch (...) {
      }
    } catch (const std::exception& e) {
      // catch and print info on any C++ exception raised in the destructor
      try {
        HOLOSCAN_LOG_ERROR("error in ~GILGuardedPyObject: {}", e.what());
      } catch (...) {
        // ignore any fmt::format exception thrown by HOLOSCAN_LOG_ERROR
      }
    }
  }

 private:
  py::object obj_;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_GIL_GUARDED_PYOBJECT_HPP */
