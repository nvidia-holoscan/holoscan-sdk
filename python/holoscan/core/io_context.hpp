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

#ifndef PYHOLOSCAN_CORE_IO_CONTEXT_HPP
#define PYHOLOSCAN_CORE_IO_CONTEXT_HPP

#include <pybind11/pybind11.h>

#include <any>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/gxf_io_context.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"

namespace py = pybind11;

namespace holoscan {

void init_io_context(py::module_&);

class PyInputContext : public gxf::GXFInputContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFInputContext::GXFInputContext;
  PyInputContext(ExecutionContext* execution_context, Operator* op,
                 std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs,
                 py::object py_op);

  py::object py_receive(const std::string& name, const std::string& kind);

 private:
  py::object py_op_ = py::none();
};

class PyOutputContext : public gxf::GXFOutputContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFOutputContext::GXFOutputContext;

  PyOutputContext(ExecutionContext* execution_context, Operator* op,
                  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs,
                  py::object py_op);

  void py_emit(py::object& data, const std::string& name, const std::string& emitter_name = "",
               const int64_t acq_timestamp = -1);

 private:
  py::object py_op_ = py::none();
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_IO_CONTEXT_HPP */
