/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYBIND11_CORE_OPERATOR_HPP
#define PYBIND11_CORE_OPERATOR_HPP

#include <pybind11/pybind11.h>

#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "application.hpp"
#include "execution_context.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"
#include "io_context.hpp"
#include "kwarg_handling.hpp"

namespace py = pybind11;

namespace holoscan {

void init_operator(py::module_&);

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

class PyOperatorSpec : public OperatorSpec {
 public:
  /* Inherit the constructors */
  using OperatorSpec::OperatorSpec;

  // Override the constructor to get the py::object for the Python class
  explicit PyOperatorSpec(Fragment* fragment = nullptr, py::object op = py::none())
      : OperatorSpec(fragment), py_op_(op) {}

  // TOIMPROVE: Should we parse headline and description from kwargs or just
  //            add them to the function signature?
  void py_param(const std::string& name, const py::object& default_value, const ParameterFlag& flag,
                const py::kwargs& kwargs) {
    using std::string_literals::operator""s;

    bool is_receivers = false;
    std::string headline{""s};
    std::string description{""s};
    for (const auto& [name, value] : kwargs) {
      std::string param_name = name.cast<std::string>();
      if (param_name == "headline") {
        headline = value.cast<std::string>();
      } else if (param_name == "description") {
        description = value.cast<std::string>();
      } else if (param_name == "kind") {
        auto kind_val = value.cast<std::string>();
        if (kind_val == "receivers") {
          is_receivers = true;
        } else {
          throw std::runtime_error("unknown kind: '"s + kind_val +
                                   "'. Only `kind='receivers'` is currently supported."s);
        }
      } else {
        throw std::runtime_error("unsupported kwarg: "s + param_name);
      }
    }

    if (is_receivers) {
      // Create receivers object
      py_receivers_params_.emplace_back();

      // Register parameter
      auto& parameter = py_receivers_params_.back();
      param(parameter, name.c_str(), headline.c_str(), description.c_str(), {}, flag);
    } else {
      // Create parameter object
      py_params_.emplace_back(py_op());

      // Register parameter
      auto& parameter = py_params_.back();
      param(parameter, name.c_str(), headline.c_str(), description.c_str(), default_value, flag);
    }
  }

  py::object py_op() const { return py_op_; }

  std::list<Parameter<py::object>>& py_params() { return py_params_; }

  std::list<Parameter<std::vector<IOSpec*>>>& py_receivers() { return py_receivers_params_; }

 private:
  py::object py_op_ = py::none();
  // NOTE: we use std::list instead of std::vector because we register the address of Parameter<T>
  // object to the GXF framework. The address of a std::vector element may change when the vector is
  // resized.
  std::list<Parameter<py::object>> py_params_;
  std::list<Parameter<std::vector<IOSpec*>>> py_receivers_params_;
};

class PyOperator : public Operator {
 public:
  /* Inherit the constructors */
  using Operator::Operator;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyOperator(py::object op, Fragment* fragment, const py::args& args, const py::kwargs& kwargs)
      : Operator() {
    using std::string_literals::operator""s;

    HOLOSCAN_LOG_TRACE("PyOperator::PyOperator()");
    py_op_ = op;
    py_compute_ = py::getattr(op, "compute");        // cache the compute method
    py_initialize_ = py::getattr(op, "initialize");  // cache the initialize method
    py_start_ = py::getattr(op, "start");            // cache the start method
    py_stop_ = py::getattr(op, "stop");              // cache the stop method
    fragment_ = fragment;

    // Store the application object to access the trace/profile functions
    auto app = fragment_->application();
    py_app_ = static_cast<PyApplication*>(app);

    // Parse args
    for (auto& item : args) {
      py::object arg_value = item.cast<py::object>();
      if (py::isinstance<Condition>(arg_value)) {
        this->add_arg(arg_value.cast<std::shared_ptr<Condition>>());
      } else if (py::isinstance<Resource>(arg_value)) {
        this->add_arg(arg_value.cast<std::shared_ptr<Resource>>());
      } else if (py::isinstance<Fragment>(arg_value)) {
        throw std::runtime_error("multiple Fragment objects provided");
      } else if (py::isinstance<ArgList>(arg_value)) {
        this->add_arg(arg_value.cast<ArgList>());
      } else if (py::isinstance<Arg>(arg_value)) {
        this->add_arg(arg_value.cast<Arg>());
      } else {
        this->add_arg(py_object_to_arg(arg_value, ""s));
      }
    }

    // Pars kwargs
    for (const auto& [name, value] : kwargs) {
      std::string kwarg_name = name.cast<std::string>();
      py::object kwarg_value = value.cast<py::object>();
      if (kwarg_name == "name"s) {
        if (py::isinstance<py::str>(kwarg_value)) {
          this->name(kwarg_value.cast<std::string>());
        } else {
          throw std::runtime_error("name kwarg must be a string");
        }
      } else if (kwarg_name == "fragment"s) {
        if (py::isinstance<Fragment>(kwarg_value)) {
          throw std::runtime_error(
              "Cannot add kwarg fragment. Fragment can only be provided positionally");
        } else {
          throw std::runtime_error("fragment kwarg must be a Fragment");
        }
      } else if (py::isinstance<Condition>(kwarg_value)) {
        // Set the condition's name to the kwarg name
        auto cond = kwarg_value.cast<std::shared_ptr<Condition>>();
        cond.get()->name(kwarg_name);
        this->add_arg(cond);
      } else if (py::isinstance<Resource>(kwarg_value)) {
        // Set the resource's name to the kwarg name
        auto resource = kwarg_value.cast<std::shared_ptr<Resource>>();
        resource.get()->name(kwarg_name);
        this->add_arg(resource);
      } else {
        this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
      }
    }

    // Set name if needed
    if (name_ == "") {
      static size_t op_number;
      op_number++;
      this->name("unnamed_operator_" + std::to_string(op_number));
    }
  }

  // Override spec() method
  std::shared_ptr<PyOperatorSpec> py_shared_spec() {
    auto spec_ptr = spec_shared();
    return std::static_pointer_cast<PyOperatorSpec>(spec_ptr);
  }

  /// Thread-local tracing data
  struct TracingThreadLocal {
    bool in_tracing = false;      ///< whether tracing is enabled
    bool is_initialized = false;  ///< whether tracing data is initialized
    bool is_pydevd = false;       ///< whether pydevd is used
    bool is_func_set = false;     ///< whether trace/profile functions are set

    py::object dummy_thread;  ///< dummy thread object for this thread

    py::object pydevd_trace_func;                 ///< pydevd's trace function
    py::object pydevd_set_trace_to_threads_func;  ///< pydevd's set_trace_to_threads function
    py::object pydevd_thread_idents;              ///< thread identifiers for this thread

    // Fake frame object for the last python frame
#if PY_VERSION_HEX >= 0x030b0000  // >= Python 3.11.0
    _PyInterpreterFrame* py_last_frame = nullptr;
#else
    PyFrameObject* py_last_frame = nullptr;
#endif

    Py_tracefunc c_profilefunc = nullptr;
    Py_tracefunc c_tracefunc = nullptr;
    py::handle c_profileobj = nullptr;
    py::handle c_traceobj = nullptr;
  };

  /**
   * @brief Thread-local data guarded by GIL.
   */
  struct GILGuardedThreadLocal {
    GILGuardedThreadLocal() {
      py::gil_scoped_acquire scope_guard;
      data.pydevd_trace_func = py::none();
      data.pydevd_set_trace_to_threads_func = py::none();
      data.pydevd_thread_idents = py::none();
      data.dummy_thread = py::none();
    }
    ~GILGuardedThreadLocal() {
      // Since this destructor is called when exiting the thread, acquiring GIL and
      // decreasing reference count of Python objects in the thread-local data
      // may cause the interpreter to crash. So, we don't acquire GIL or handle
      // reference count of Python objects here.

      // Just release the Python objects.
      data.pydevd_trace_func.release();
      data.pydevd_set_trace_to_threads_func.release();
      data.pydevd_thread_idents.release();
      data.dummy_thread.release();

      data.py_last_frame = nullptr;

      data.c_profilefunc = nullptr;
      data.c_tracefunc = nullptr;
      data.c_profileobj = nullptr;
      data.c_traceobj = nullptr;
    }
    TracingThreadLocal data{};
  };

  /**
   * @brief Get the tracing data object.
   *
   * GIL must be acquired before calling this function.
   *
   * @return The reference to the thread-local tracing data object.
   */
  TracingThreadLocal& get_tracing_data() {
    // Define a thread-local object for storing tracing data.
    // Important: The type of a thread_local variable should be a pointer due to issues
    // with Thread-Local Storage (TLS) when dynamically loading libraries using dlopen().
    // The TLS space is limited to 2048 bytes.
    // For more details, refer to: https://fasterthanli.me/articles/a-dynamic-linker-murder-mystery.
    static thread_local std::unique_ptr<GILGuardedThreadLocal> gil_guarded_thread_local =
        std::make_unique<GILGuardedThreadLocal>();

    py::gil_scoped_acquire scope_guard;

    auto& data = gil_guarded_thread_local->data;

    // Return the cached thread-local data if it is already initialized
    if (data.is_initialized) { return data; }

    try {
      if (data.dummy_thread.is_none()) {
        // Create a dummy thread object for this thread by calling threading.current_thread()
        // so that debugger can recognize this thread as a Python thread.
        auto threading_module = py::module::import("threading");
        auto current_thread_func = py::getattr(threading_module, "current_thread");
        // Create py::object object having the result of current_thread_func()
        data.dummy_thread = current_thread_func();
      }

      // Check if the module name starts with '_pydevd_bundle' which means that it is using
      // PyDevd debugger. If so, then we need to store pydevd-specific data.

      // Get py_trace_func_'s class object using "__class__" attr
      auto trace_module = py_app_->py_trace_func_.attr("__class__").attr("__module__");
      // Check if the module name starts with '_pydevd_bundle' which means that it is using
      // PyDevd debugger. If so, then we need to set the trace function to the current frame.
      auto module_name = trace_module.cast<std::string>();
      if (module_name.find("_pydevd_bundle") != std::string::npos) {
        if (data.pydevd_trace_func.is_none()) {
          // Get the trace function from the debugger
          auto pydevd_module = py::module::import("pydevd");
          auto debugger = py::getattr(pydevd_module, "GetGlobalDebugger")();
          // Get the trace function from the debugger
          data.pydevd_trace_func = py::getattr(debugger, "get_thread_local_trace_func")();
        }

        if (data.pydevd_set_trace_to_threads_func.is_none()) {
          auto pydevd_module = py::module::import("pydevd");

          data.pydevd_set_trace_to_threads_func =
              pydevd_module.attr("pydevd_tracing").attr("set_trace_to_threads");
        }

        if (data.pydevd_thread_idents.is_none()) {
          auto thread_module = py::module::import("_thread");
          auto get_ident_func = py::getattr(thread_module, "get_ident");
          // Create py::list object having the result of get_ident_func()
          auto thread_idents = py::list();
          thread_idents.append(get_ident_func());

          data.pydevd_thread_idents = thread_idents;
        }
      }

      data.is_initialized = true;
      data.in_tracing = (py_app_->c_tracefunc_ != nullptr) || (py_app_->c_profilefunc_ != nullptr);
      data.is_pydevd = (!data.pydevd_trace_func.is_none()) &&
                       (!data.pydevd_set_trace_to_threads_func.is_none()) &&
                       (!data.pydevd_thread_idents.is_none());
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("Exception occurred while initializing tracing data: {}", e.what());
      data.is_initialized = true;
      data.in_tracing = false;  // pretend that tracing is not enabled for this thread
      data.is_pydevd = false;
    }

    return data;
  }

  /**
   * @brief Set the tracing functions to the current thread.
   *
   * GIL must be acquired before calling this function.
   */
  void set_py_tracing() {
    auto& tracing_data = get_tracing_data();

    try {
      // If tracing is not enabled, do nothing and return
      if (!tracing_data.in_tracing) { return; }

      auto py_thread_state = _PyThreadState_UncheckedGet();

      // If tracing_data.is_func_set is false, cache the current trace/profile functions for
      // the current thread.
      if (!tracing_data.is_func_set) {
        auto& py_last_frame = py_app_->py_last_frame_;
        auto& py_profile_func = py_app_->py_profile_func_;
        auto& py_trace_func = py_app_->py_trace_func_;
        auto& c_profilefunc = py_app_->c_profilefunc_;
        auto& c_profileobj = py_app_->c_profileobj_;
        auto& c_tracefunc = py_app_->c_tracefunc_;
        auto& c_traceobj = py_app_->c_traceobj_;

        tracing_data.py_last_frame = py_last_frame;

        // If pydevd is used, call pydevd.pydevd_tracing.set_trace_to_threads() to set
        // the trace function to the current thread.
        if (tracing_data.is_pydevd) {
          tracing_data.pydevd_set_trace_to_threads_func(
              tracing_data.pydevd_trace_func,
              py::arg("thread_idents") = tracing_data.pydevd_thread_idents,
              py::arg("create_dummy_thread") = py::bool_(false));

          tracing_data.c_profilefunc = py_thread_state->c_profilefunc;
          tracing_data.c_profileobj = py_thread_state->c_profileobj;

          tracing_data.c_tracefunc = py_thread_state->c_tracefunc;
          tracing_data.c_traceobj = py_thread_state->c_traceobj;
        } else {
          // If pydevd is not used, call sys.settrace/setprofile() to set
          // the trace/profile function to the current thread.
          auto sys_module = py::module::import("sys");

          // Check if py_profile_func is callable and call it.
          // In case of cProfile.Profile object, it is not callable so should not be called.
          if (!py_profile_func.is_none() && py::isinstance<py::function>(py_profile_func)) {
            sys_module.attr("setprofile")(py_profile_func);
            tracing_data.c_profilefunc = py_thread_state->c_profilefunc;
            tracing_data.c_profileobj = py_thread_state->c_profileobj;
          } else {
            HOLOSCAN_LOG_DEBUG("py_profile_func_ is not callable");
            tracing_data.c_profilefunc = c_profilefunc;
            tracing_data.c_profileobj = c_profileobj;
          }

          // Check if py_trace_func is callable and call it.
          if (!py_trace_func.is_none() && py::isinstance<py::function>(py_trace_func)) {
            sys_module.attr("settrace")(py_trace_func);
            tracing_data.c_tracefunc = py_thread_state->c_tracefunc;
            tracing_data.c_traceobj = py_thread_state->c_traceobj;
          } else {
            HOLOSCAN_LOG_DEBUG("py_trace_func_ is not callable");
            tracing_data.c_tracefunc = c_tracefunc;
            tracing_data.c_traceobj = c_traceobj;
          }
        }
        tracing_data.is_func_set = true;
      }

      // Set the trace/profile functions to the current thread.
      // Depending on the Python version, the way to set the trace/profile functions is different.

      // Set current frame to the last valid Python frame
#if PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
      // https://github.com/python/cpython/blob/c184c6750e40ca4ffa4f62a5d145b892cbd066bc
      //   /Doc/whatsnew/3.11.rst#L2301
      // - tstate->frame is removed.
      py_thread_state->cframe->current_frame = tracing_data.py_last_frame;
#else  // < Python 3.11.0
      py_thread_state->frame = tracing_data.py_last_frame;
#endif

#if PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
      // Recommended way to set the trace/profile functions in Python 3.11
      // (see https://discuss.python.org/t/python-3-11-frame-structure-and-various-changes/17895/19)
      _PyEval_SetProfile(
          py_thread_state, tracing_data.c_profilefunc, tracing_data.c_profileobj.ptr());
      _PyEval_SetTrace(py_thread_state, tracing_data.c_tracefunc, tracing_data.c_traceobj.ptr());
#else  // < Python 3.11.0
      py_thread_state->c_profilefunc = tracing_data.c_profilefunc;
      Py_XINCREF(tracing_data.c_profileobj.ptr());
      Py_XDECREF(py_thread_state->c_profileobj);
      py_thread_state->c_profileobj = tracing_data.c_profileobj.ptr();

      py_thread_state->c_tracefunc = tracing_data.c_tracefunc;
      Py_XINCREF(tracing_data.c_traceobj.ptr());
      Py_XDECREF(py_thread_state->c_traceobj);
      py_thread_state->c_traceobj = tracing_data.c_traceobj.ptr();

#if PY_VERSION_HEX >= 0x030A00B1  // >= Python 3.10.0 b1
      py_thread_state->cframe->use_tracing = 1;
#else                             // < Python 3.10.0 b1
      py_thread_state->use_tracing = 1;
#endif                            // about Python 3.10.0 b1
#endif                            // about Python 3.11.0
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("Exception occurred while setting trace/profile functions: {}", e.what());
      tracing_data.is_initialized = true;
      tracing_data.is_pydevd = false;
      tracing_data.in_tracing = false;  // pretend that tracing is not enabled for this thread
    }
  }

  void initialize() override {
    Operator::initialize();
    try {
      // Get the initialize method of the Python Operator class and call it
      py::gil_scoped_acquire scope_guard;

      set_py_tracing();

      try {
        py_initialize_.operator()();
      } catch (const py::error_already_set& e) { _handle_python_error(e, "initialize"); }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", name(), e.what());
    }
  }

  void start() override {
    try {
      // Get the start method of the Python Operator class and call it
      py::gil_scoped_acquire scope_guard;

      set_py_tracing();

      try {
        py_start_.operator()();
      } catch (const py::error_already_set& e) { _handle_python_error(e, "start"); }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", name(), e.what());
    }
  }

  void stop() override {
    try {
      // Get the stop method of the Python Operator class and call it
      py::gil_scoped_acquire scope_guard;

      set_py_tracing();

      try {
        py_stop_.operator()();
      } catch (const py::error_already_set& e) { _handle_python_error(e, "stop"); }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", name(), e.what());
    }
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto gxf_context = context.context();

    try {
      // Get the compute method of the Python Operator class and call it
      py::gil_scoped_acquire scope_guard;
      auto py_op_input = std::make_shared<PyInputContext>(
          &context, op_input.op(), op_input.inputs(), this->py_op_);
      auto py_op_output = std::make_shared<PyOutputContext>(
          &context, op_output.op(), op_output.outputs(), this->py_op_);
      auto py_context = std::make_shared<PyExecutionContext>(
          gxf_context, py_op_input, py_op_output, this->py_op_);

      set_py_tracing();

      try {
        py_compute_.operator()(py::cast(py_op_input), py::cast(py_op_output), py::cast(py_context));
      } catch (const py::error_already_set& e) { _handle_python_error(e, "compute"); }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception occurred for operator: '{}' - {}", name(), e.what());
    }
  }

 private:
  void _handle_python_error(const py::error_already_set& e, std::string method_name) {
    // Print the Python error to stderr
    auto stderr = py::module::import("sys").attr("stderr");

    py::print(fmt::format("Exception occurred in {} method of operator: '{}'", method_name, name_),
              py::arg("file") = stderr);
    py::module::import("traceback")
        .attr("print_exception")(e.type(), e.value(), e.trace(), py::none(), stderr);
    // Note:: We don't want to throw an exception here, because it will cause the Python
    //        interpreter to exit. Instead, we'll just log the error and continue.
    // throw std::runtime_error(fmt::format("Python error in {} method: {}", method_name,
    // e.what()));
  }

  py::object py_op_ = py::none();
  py::object py_initialize_ = py::none();
  py::object py_start_ = py::none();
  py::object py_stop_ = py::none();
  py::object py_compute_ = py::none();

  /// Python application pointer to access the trace/profile functions
  PyApplication* py_app_ = nullptr;
};

}  // namespace holoscan

#endif /* PYBIND11_CORE_OPERATOR_HPP */
