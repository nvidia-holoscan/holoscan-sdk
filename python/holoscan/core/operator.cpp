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

#include "operator.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "kwarg_handling.hpp"
#include "operator_pydoc.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void init_operator(py::module_& m) {
  py::class_<OperatorSpec, ComponentSpec, std::shared_ptr<OperatorSpec>>(
      m, "OperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::OperatorSpec::doc_OperatorSpec)
      .def("input",
           py::overload_cast<>(&OperatorSpec::input<gxf::Entity>),
           doc::OperatorSpec::doc_input,
           py::return_value_policy::reference_internal)
      .def("input",
           py::overload_cast<std::string>(&OperatorSpec::input<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_input_kwargs,
           py::return_value_policy::reference_internal)
      .def_property_readonly("inputs",
                             &OperatorSpec::inputs,
                             doc::OperatorSpec::doc_inputs,
                             py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<>(&OperatorSpec::output<gxf::Entity>),
           doc::OperatorSpec::doc_output,
           py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<std::string>(&OperatorSpec::output<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_output_kwargs,
           py::return_value_policy::reference_internal)
      .def_property_readonly("outputs",
                             &OperatorSpec::outputs,
                             doc::OperatorSpec::doc_outputs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "description", &OperatorSpec::description, doc::OperatorSpec::doc_description)
      .def(
          "__repr__",
          [](const OperatorSpec& spec) { return spec.description(); },
          R"doc(Return repr(self).)doc");

  // Note: In the case of OperatorSpec, InputContext, OutputContext, ExecutionContext,
  //       there are a separate, independent wrappers for PyOperatorSpec, PyInputContext,
  //       PyOutputContext, PyExecutionContext. These Py* variants are not exposed directly
  //       to end users of the API, but are used internally to enable native operators
  //       defined from python via inheritance from the `Operator` class as defined in
  //       core/__init__.py.

  py::class_<PyOperatorSpec, OperatorSpec, std::shared_ptr<PyOperatorSpec>>(
      m, "PyOperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*, py::object>(),
           "fragment"_a,
           "op"_a,
           doc::OperatorSpec::doc_OperatorSpec)
      .def("param",
           &PyOperatorSpec::py_param,
           "Register parameter",
           "name"_a,
           "default_value"_a = py::none(),
           "flag"_a = ParameterFlag::kNone,
           doc::OperatorSpec::doc_param);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Operator, ComponentBase, PyOperator, std::shared_ptr<Operator>> operator_class(
      m, "Operator", py::dynamic_attr(), doc::Operator::doc_Operator_args_kwargs);

  operator_class
      .def(py::init<py::object, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Operator::doc_Operator_args_kwargs)
      .def_property("name",
                    py::overload_cast<>(&Operator::name, py::const_),
                    (Operator & (Operator::*)(const std::string&)) & Operator::name,
                    doc::Operator::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Operator::fragment), doc::Operator::doc_fragment)
      .def_property("spec",
                    &Operator::spec_shared,
                    py::overload_cast<const std::shared_ptr<OperatorSpec>&>(&Operator::spec),
                    doc::Operator::doc_spec)
      .def_property_readonly("conditions", &Operator::conditions, doc::Operator::doc_conditions)
      .def_property_readonly("resources", &Operator::resources, doc::Operator::doc_resources)
      .def_property_readonly(
          "operator_type", &Operator::operator_type, doc::Operator::doc_operator_type)
      .def(
          "resource",
          [](Operator& op, const py::str& name) -> std::optional<py::object> {
            auto resources = op.resources();
            auto res = resources.find(name);
            if (res == resources.end()) { return py::none(); }
            return py::cast(res->second);
          },
          "name"_a,
          doc::Operator::doc_resource)
      .def("add_arg",
           py::overload_cast<const Arg&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_Arg)
      .def("add_arg",
           py::overload_cast<const ArgList&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_ArgList)
      .def(
          "add_arg",
          [](Operator& op, const py::kwargs& kwargs) {
            return op.add_arg(kwargs_to_arglist(kwargs));
          },
          doc::Operator::doc_add_arg_kwargs)
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Condition>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_condition)
      // Note: to avoid a doc build error, only list Parameters in docstring of the last overload
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Resource>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_resource)
      .def("initialize",
           &Operator::initialize,
           doc::Operator::doc_initialize)                        // note: virtual function
      .def("setup", &Operator::setup, doc::Operator::doc_setup)  // note: virtual function
      .def("start",
           &Operator::start,  // note: virtual function
           doc::Operator::doc_start,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("stop",
           &Operator::stop,  // note: virtual function
           doc::Operator::doc_stop,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("compute",
           &Operator::compute,  // note: virtual function
           doc::Operator::doc_compute,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def_property_readonly("description", &Operator::description, doc::Operator::doc_description)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto op = obj.cast<std::shared_ptr<Operator>>();
            if (op) { return op->description(); }
            return std::string("<Operator: None>");
          },
          R"doc(Return repr(self).)doc");

  py::enum_<Operator::OperatorType>(
      operator_class, "OperatorType", doc::OperatorType::doc_OperatorType)
      .value("NATIVE", Operator::OperatorType::kNative)
      .value("GXF", Operator::OperatorType::kGXF)
      .value("VIRTUAL", Operator::OperatorType::kVirtual);
}

PyOperatorSpec::PyOperatorSpec(Fragment* fragment, py::object op)
    : OperatorSpec(fragment), py_op_(std::move(op)) {}

void PyOperatorSpec::py_param(const std::string& name, const py::object& default_value,
                              const ParameterFlag& flag, const py::kwargs& kwargs) {
  using std::string_literals::operator""s;

  bool is_receivers = false;
  std::string headline{""s};
  std::string description{""s};
  for (const auto& [kw_name, value] : kwargs) {
    std::string param_name = kw_name.cast<std::string>();
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

py::object PyOperatorSpec::py_op() const {
  return py_op_;
}

std::list<Parameter<py::object>>& PyOperatorSpec::py_params() {
  return py_params_;
}

std::list<Parameter<std::vector<IOSpec*>>>& PyOperatorSpec::py_receivers() {
  return py_receivers_params_;
}

// PyOperator

PyOperator::PyOperator(py::object op, Fragment* fragment, const py::args& args,
                       const py::kwargs& kwargs)
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

std::shared_ptr<PyOperatorSpec> PyOperator::py_shared_spec() {
  auto spec_ptr = spec_shared();
  return std::static_pointer_cast<PyOperatorSpec>(spec_ptr);
}

PyOperator::GILGuardedThreadLocal::GILGuardedThreadLocal() {
  py::gil_scoped_acquire scope_guard;
  data.pydevd_trace_func = py::none();
  data.pydevd_set_trace_to_threads_func = py::none();
  data.pydevd_thread_idents = py::none();
  data.dummy_thread = py::none();
}

PyOperator::GILGuardedThreadLocal::~GILGuardedThreadLocal() {
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

PyOperator::TracingThreadLocal& PyOperator::get_tracing_data() {
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
    HOLOSCAN_LOG_WARN("Exception occurredPyOperator:: while initializing tracing data: {}",
                      e.what());
    data.is_initialized = true;
    data.in_tracing = false;  // pretend that tracing is not enabled for this thread
    data.is_pydevd = false;
  }

  return data;
}

void PyOperator::set_py_tracing() {
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
    py_thread_state->cframe->current_frame =
        reinterpret_cast<_PyInterpreterFrame*>(tracing_data.py_last_frame);
#else  // < Python 3.11.0
    py_thread_state->frame = reinterpret_cast<PyFrameObject*>(tracing_data.py_last_frame);
#endif

#if PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
    // Recommended way to set the trace/profile functions in Python 3.11
    // (see
    // https://discuss.python.org/t/python-3-11-frame-structure-and-various-changes/17895/19)
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

void PyOperator::initialize() {
  // Get the initialize method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

  set_py_tracing();

  py_initialize_.operator()();

  // Call the parent class's initialize method after invoking the Python Operator's initialize
  // method.
  Operator::initialize();
}

void PyOperator::start() {
  // Get the start method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

  set_py_tracing();

  py_start_.operator()();
}

void PyOperator::stop() {
  // Get the stop method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

  set_py_tracing();

  py_stop_.operator()();
}

void PyOperator::compute(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context) {
  auto gxf_context = context.context();

  // Get the compute method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;
  auto py_op_input =
      std::make_shared<PyInputContext>(&context, op_input.op(), op_input.inputs(), this->py_op_);
  auto py_op_output = std::make_shared<PyOutputContext>(
      &context, op_output.op(), op_output.outputs(), this->py_op_);
  auto py_context =
      std::make_shared<PyExecutionContext>(gxf_context, py_op_input, py_op_output, this->py_op_);

  set_py_tracing();

  py_compute_.operator()(py::cast(py_op_input), py::cast(py_op_output), py::cast(py_context));
}

}  // namespace holoscan
