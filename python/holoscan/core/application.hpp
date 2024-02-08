/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYBIND11_CORE_APPLICATION_HPP
#define PYBIND11_CORE_APPLICATION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "tensor.hpp"

namespace py = pybind11;

namespace holoscan {

void init_application(py::module_&);

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

class PyApplication : public Application {
 public:
  /* Inherit the constructors */
  using Application::Application;

  /**
   * @brief Return the argv_ as a Python list.
   *
   * This is needed because we want to return a Python list of strings without copying the strings.
   *
   * This returns a Python list of strings, discarding the first element because Python's `sys.argv`
   * doesn't include the Python executable.
   *
   * If the resulting list is empty, we'll return `['']` to match Python's `sys.argv` behavior.
   *
   * @param obj PyApplication object.
   * @return The argv_ as a Python list, discarding the first element.
   */
  py::list py_argv() {
    py::list argv;
    // In Python, `sys.argv` returns `['']` if there are no arguments (i.e., when just `python` is
    // called). We'll do the same here.
    if (argv_.empty()) {
      argv.append(py::cast("", py::return_value_policy::reference));
      return argv;
    }

    for (auto iter = std::next(argv_.begin()); iter != argv_.end(); ++iter) {
      argv.append(py::cast(*iter, py::return_value_policy::reference));
    }

    if (argv.empty()) { argv.append(py::cast("", py::return_value_policy::reference)); }
    return argv;
  }

  /* Trampolines (need one for each virtual function) */
  void add_operator(const std::shared_ptr<Operator>& op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_operator, op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> io_map) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op, io_map);
  }
  void add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                const std::shared_ptr<Fragment>& downstream_frag,
                std::set<std::pair<std::string, std::string>> port_pairs) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_flow, upstream_frag, downstream_frag, port_pairs);
  }
  void compose() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, compose);
  }
  void run() override {
    // Create a deleter for DLManagedTensor objects so that they can be deleted in a separate thread
    // to avoid blocking the GXF runtime mutex.
    LazyDLManagedTensorDeleter deleter;

    // Get the trace and profile functions from sys
    {
      pybind11::gil_scoped_acquire gil;

      auto sys_module = py::module::import("sys");

      // Note that when cProfile is used, the profile_func_ is a cProfile.Profile object, not a
      // function. If the return value of getprofile() is not a function, we need to use the
      // existing c_profilefunc_ and c_profileobj_ instead of calling sys.setprofile() with
      // profile_func_.
      py_profile_func_ = sys_module.attr("getprofile")();
      py_trace_func_ = sys_module.attr("gettrace")();

      auto py_thread_state = _PyThreadState_UncheckedGet();
      c_profilefunc_ = py_thread_state->c_profilefunc;
      c_profileobj_ = py_thread_state->c_profileobj;
      c_tracefunc_ = py_thread_state->c_tracefunc;
      c_traceobj_ = py_thread_state->c_traceobj;

#if PY_VERSION_HEX >= 0x030b0000  // >= Python 3.11.0
      py_last_frame_ = py_thread_state->cframe->current_frame;
#else
      py_last_frame_ = py_thread_state->frame;  // = PyEval_GetFrame();
#endif
    }

    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, run);
  }

 private:
  friend class PyOperator;

  // Fake frame object for the last python frame (where Application.run() was called).
#if PY_VERSION_HEX >= 0x030b0000  // >= Python 3.11.0
  _PyInterpreterFrame* py_last_frame_ = nullptr;
#else
  PyFrameObject* py_last_frame_ = nullptr;
#endif

  // Trace/profile functions
  // - Retain a reference to the Python trace/profile function if available via
  //   sys.settrace/setprofile.
  py::object py_trace_func_;
  py::object py_profile_func_;
  // - Otherwise, use the C trace/profile function and its corresponding argument object.
  Py_tracefunc c_profilefunc_ = nullptr;
  Py_tracefunc c_tracefunc_ = nullptr;
  py::handle c_profileobj_ = nullptr;
  py::handle c_traceobj_ = nullptr;
};

}  // namespace holoscan

#endif /* PYBIND11_CORE_APPLICATION_HPP */
