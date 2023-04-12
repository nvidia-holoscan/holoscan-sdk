/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYBIND11_CORE_TRAMPOLINES_HPP
#define PYBIND11_CORE_TRAMPOLINES_HPP

#include <pybind11/pybind11.h>

#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../kwarg_handling.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_io_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

namespace py = pybind11;

namespace holoscan {

class PyComponent : public Component {
 public:
  /* Inherit the constructors */
  using Component::Component;

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Component, initialize);
  }
};

class PyCondition : public Condition {
 public:
  /* Inherit the constructors */
  using Condition::Condition;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyCondition(const py::args& args, const py::kwargs& kwargs) : Condition() {
    using std::string_literals::operator""s;

    int n_fragments = 0;
    for (auto& item : args) {
      py::object arg_value = item.cast<py::object>();
      if (py::isinstance<Fragment>(arg_value)) {
        if (n_fragments > 0) { throw std::runtime_error("multiple Fragment objects provided"); }
        fragment_ = arg_value.cast<Fragment*>();
        n_fragments += 1;
      } else {
        this->add_arg(py_object_to_arg(arg_value, ""s));
      }
    }
    for (auto& item : kwargs) {
      std::string kwarg_name = item.first.cast<std::string>();
      py::object kwarg_value = item.second.cast<py::object>();
      if (kwarg_name == "name"s) {
        if (py::isinstance<py::str>(kwarg_value)) {
          name_ = kwarg_value.cast<std::string>();
        } else {
          throw std::runtime_error("name kwarg must be a string");
        }
      } else if (kwarg_name == "fragment"s) {
        if (py::isinstance<Fragment>(kwarg_value)) {
          if (n_fragments > 0) {
            throw std::runtime_error(
                "Cannot add kwarg fragment, when a Fragment was also provided positionally");
          }
          fragment_ = kwarg_value.cast<Fragment*>();
        } else {
          throw std::runtime_error("fragment kwarg must be a Fragment");
        }
      } else {
        this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
      }
    }
  }

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Condition, initialize);
  }
  void setup(ComponentSpec& spec) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Condition, setup, spec);
  }
};

class PyResource : public Resource {
 public:
  /* Inherit the constructors */
  using Resource::Resource;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyResource(const py::args& args, const py::kwargs& kwargs) : Resource() {
    using std::string_literals::operator""s;

    int n_fragments = 0;
    for (auto& item : args) {
      py::object arg_value = item.cast<py::object>();
      if (py::isinstance<Fragment>(arg_value)) {
        if (n_fragments > 0) { throw std::runtime_error("multiple Fragment objects provided"); }
        fragment_ = arg_value.cast<Fragment*>();
        n_fragments += 1;
      } else {
        this->add_arg(py_object_to_arg(arg_value, ""s));
      }
    }
    for (auto& item : kwargs) {
      std::string kwarg_name = item.first.cast<std::string>();
      py::object kwarg_value = item.second.cast<py::object>();
      if (kwarg_name == "name"s) {
        if (py::isinstance<py::str>(kwarg_value)) {
          name_ = kwarg_value.cast<std::string>();
        } else {
          throw std::runtime_error("name kwarg must be a string");
        }
      } else if (kwarg_name == "fragment"s) {
        if (py::isinstance<Fragment>(kwarg_value)) {
          if (n_fragments > 0) {
            throw std::runtime_error(
                "Cannot add kwarg fragment, when a Fragment was also provided positionally");
          }
          fragment_ = kwarg_value.cast<Fragment*>();
        } else {
          throw std::runtime_error("fragment kwarg must be a Fragment");
        }
      } else {
        this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
      }
    }
  }

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Resource, initialize);
  }
  void setup(ComponentSpec& spec) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Resource, setup, spec);
  }
};

class PyOperatorSpec : public OperatorSpec {
 public:
  /* Inherit the constructors */
  using OperatorSpec::OperatorSpec;

  // Override the constructor to get the py::object for the Python class
  explicit PyOperatorSpec(Fragment* fragment = nullptr, py::object op = py::none())
      : OperatorSpec(fragment), py_op_(op) {}

  // TOIMPROVE: Should we parse headline and description from kwargs or just
  //            add them to the function signature?
  void py_param(const std::string& name, const py::object& default_value,
                const py::kwargs& kwargs) {
    using std::string_literals::operator""s;

    bool is_receivers = false;
    std::string headline{""s};
    std::string description{""s};
    for (auto item : kwargs) {
      std::string name = item.first.cast<std::string>();
      if (name == "headline") {
        headline = item.second.cast<std::string>();
      } else if (name == "description") {
        description = item.second.cast<std::string>();
      } else if (name == "kind") {
        auto kind_val = item.second.cast<std::string>();
        if (kind_val == "receivers") {
          is_receivers = true;
        } else {
          throw std::runtime_error("unknown kind: '"s + kind_val +
                                   "'. Only `kind='receivers'` is currently supported."s);
        }
      } else {
        throw std::runtime_error("unsupported kwarg: "s + name);
      }
    }

    if (is_receivers) {
      // Create receivers object
      py_receivers_params_.emplace_back();

      // Register parameter
      auto& parameter = py_receivers_params_.back();
      param(parameter, name.c_str(), headline.c_str(), description.c_str(), {});
    } else {
      // Create parameter object
      py_params_.emplace_back(py_op());

      // Register parameter
      auto& parameter = py_params_.back();
      param(parameter, name.c_str(), headline.c_str(), description.c_str(), default_value);
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
    py::gil_scoped_acquire scope_guard;
    py::handle handle = obj_.release();
    if (handle) { handle.dec_ref(); }
  }

 private:
  py::object obj_;
};

class PyInputContext : public gxf::GXFInputContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFInputContext::GXFInputContext;
  PyInputContext(gxf_context_t context, Operator* op,
                 std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs, py::object py_op)
      : gxf::GXFInputContext::GXFInputContext(context, op, inputs), py_op_(py_op) {}

  py::object py_receive(const std::string& name);

 private:
  py::object py_op_ = py::none();
};

class PyOutputContext : public gxf::GXFOutputContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFOutputContext::GXFOutputContext;

  PyOutputContext(gxf_context_t context, Operator* op,
                  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs,
                  py::object py_op)
      : gxf::GXFOutputContext::GXFOutputContext(context, op, outputs), py_op_(py_op) {}

  void py_emit(py::object& data, const std::string& name = "");

 private:
  py::object py_op_ = py::none();
};

class PyExecutionContext : public gxf::GXFExecutionContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFExecutionContext::GXFExecutionContext;

  PyExecutionContext(gxf_context_t context, std::shared_ptr<PyInputContext>& py_input_context,
                     std::shared_ptr<PyOutputContext>& py_output_context,
                     py::object op = py::none())
      : gxf::GXFExecutionContext(context, py_input_context, py_output_context),
        py_op_(op),
        py_input_context_(py_input_context),
        py_output_context_(py_output_context) {}

  std::shared_ptr<PyInputContext> py_input() const { return py_input_context_; }

  std::shared_ptr<PyOutputContext> py_output() const { return py_output_context_; }

 private:
  py::object py_op_ = py::none();
  std::shared_ptr<PyInputContext> py_input_context_;
  std::shared_ptr<PyOutputContext> py_output_context_;
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
    fragment_ = fragment;

    int n_fragments = 0;
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
    for (auto& item : kwargs) {
      std::string kwarg_name = item.first.cast<std::string>();
      py::object kwarg_value = item.second.cast<py::object>();
      if (kwarg_name == "name"s) {
        if (py::isinstance<py::str>(kwarg_value)) {
          name_ = kwarg_value.cast<std::string>();
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
  }

  // Override spec() method
  std::shared_ptr<PyOperatorSpec> py_shared_spec() {
    auto spec_ptr = spec_shared();
    return std::static_pointer_cast<PyOperatorSpec>(spec_ptr);
  }

  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Operator, initialize);
  }

  void start() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Operator, start);
  }

  void stop() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Operator, stop);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto gxf_context = dynamic_cast<gxf::GXFInputContext&>(op_input).gxf_context();

    auto py_op_input = std::make_shared<PyInputContext>(
        gxf_context, op_input.op(), op_input.inputs(), this->py_op_);
    auto py_op_output = std::make_shared<PyOutputContext>(
        gxf_context, op_output.op(), op_output.outputs(), this->py_op_);
    auto py_context =
        std::make_shared<PyExecutionContext>(gxf_context, py_op_input, py_op_output, this->py_op_);
    {
      // Get the compute method of the Python Operator class and call it
      py::gil_scoped_acquire scope_guard;
      py::object py_compute = py::getattr(py_op_, "compute");
      try {
        py_compute.operator()(py::cast(py_op_input), py::cast(py_op_output), py::cast(py_context));
      } catch (const py::error_already_set& e) {
        // Print the Python error to stderr
        auto stderr = py::module::import("sys").attr("stderr");

        py::print(fmt::format("Exception occurred for operator: '{}'", name_),
                  py::arg("file") = stderr);
        py::module::import("traceback")
            .attr("print_exception")(e.type(), e.value(), e.trace(), py::none(), stderr);

        // Note:: We don't want to throw an exception here, because it will cause the Python
        //        interpreter to exit. Instead, we'll just log the error and continue.
        // throw std::runtime_error(fmt::format("Python error in compute method: {}", e.what()));
      }
    }
  }

 private:
  py::object py_op_ = py::none();
};

class PyExecutor : public Executor {
 public:
  /* Inherit the constructors */
  using Executor::Executor;

  /* Trampolines (need one for each virtual function) */
  void run(Graph& graph) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Executor, run, graph);
  }
};

class PyFragment : public Fragment {
 public:
  /* Inherit the constructors */
  using Fragment::Fragment;

  /* Trampolines (need one for each virtual function) */
  void add_operator(const std::shared_ptr<Operator>& op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_operator, op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> io_map) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, io_map);
  }

  void compose() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, compose);
  }
  void run() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, run);
  }
};

class PyApplication : public Application {
 public:
  /* Inherit the constructors */
  using Application::Application;

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
                const std::shared_ptr<Fragment>& downstream_frag) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_flow, upstream_frag, downstream_frag);
  }
  void add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                const std::shared_ptr<Fragment>& downstream_frag,
                const std::set<std::pair<std::string, std::string>>& port_pairs) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, add_flow, upstream_frag, downstream_frag, port_pairs);
  }
  void compose() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, compose);
  }
  void run() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Application, run);
  }
};

}  // namespace holoscan

#endif /* PYBIND11_CORE_TRAMPOLINES_HPP */
