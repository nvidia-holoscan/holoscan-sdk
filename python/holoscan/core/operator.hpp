/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_OPERATOR_HPP
#define PYHOLOSCAN_CORE_OPERATOR_HPP

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

class PYBIND11_EXPORT PyOperatorSpec : public OperatorSpec {
 public:
  /* Inherit the constructors */
  using OperatorSpec::OperatorSpec;

  // Override the constructor to get the py::object for the Python class
  explicit PyOperatorSpec(Fragment* fragment = nullptr, py::object op = py::none());

  // TOIMPROVE: Should we parse headline and description from kwargs or just
  //            add them to the function signature?
  void py_param(const std::string& name, const py::object& default_value, const ParameterFlag& flag,
                const py::kwargs& kwargs);

  py::object py_op() const;

  std::list<Parameter<py::object>>& py_params();

  std::list<Parameter<std::vector<IOSpec*>>>& py_receivers();

 private:
  py::object py_op_ = py::none();
  // NOTE: we use std::list instead of std::vector because we register the address of Parameter<T>
  // object to the GXF framework. The address of a std::vector element may change when the vector is
  // resized.
  std::list<Parameter<py::object>> py_params_;
};

class PYBIND11_EXPORT PyOperator : public Operator {
 public:
  /* Inherit the constructors */
  using Operator::Operator;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyOperator(const py::object& op, Fragment* fragment, const py::args& args,
             const py::kwargs& kwargs);

  // Override spec() method
  std::shared_ptr<PyOperatorSpec> py_shared_spec();

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
    // Actual type is either _PyInterpreterFrame* (PY_VERSION_HEX >= 0x030b0000) or PyFrameObject*.
    void* py_last_frame = nullptr;

    Py_tracefunc c_profilefunc = nullptr;
    Py_tracefunc c_tracefunc = nullptr;
    py::handle c_profileobj = nullptr;
    py::handle c_traceobj = nullptr;
  };

  /**
   * @brief Thread-local data guarded by GIL.
   */
  struct GILGuardedThreadLocal {
    GILGuardedThreadLocal();
    ~GILGuardedThreadLocal();
    TracingThreadLocal data{};
  };

  /**
   * @brief Get the tracing data object.
   *
   * GIL must be acquired before calling this function.
   *
   * @return The reference to the thread-local tracing data object.
   */
  TracingThreadLocal& get_tracing_data();

  /**
   * @brief Set the tracing functions to the current thread.
   *
   * GIL must be acquired before calling this function.
   */
  void set_py_tracing();

  void initialize() override;

  void start() override;

  void stop() override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  std::shared_ptr<holoscan::ExecutionContext> execution_context() const override;

 private:
  py::object py_op_ = py::none();                               ///> cache the Python operator
  py::object py_initialize_ = py::none();                       ///> cache the initialize method
  py::object py_start_ = py::none();                            ///> cache the start method
  py::object py_stop_ = py::none();                             ///> cache the stop method
  py::object py_compute_ = py::none();                          ///> cache the compute method
  std::shared_ptr<holoscan::PyInputContext> py_op_input_{};     ///> cache the PyInputContext
  std::shared_ptr<holoscan::PyOutputContext> py_op_output_{};   ///> cache the PyOutputContext
  std::shared_ptr<holoscan::PyExecutionContext> py_context_{};  ///> cache the PyExecutionContext

  /// Python application pointer to access the trace/profile functions
  PyApplication* py_app_ = nullptr;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_OPERATOR_HPP */
