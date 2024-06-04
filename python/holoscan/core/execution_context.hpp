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

#ifndef PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP
#define PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP

#include <pybind11/pybind11.h>

#include <memory>

#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "io_context.hpp"

namespace py = pybind11;

namespace holoscan {

void init_execution_context(py::module_&);

class PyExecutionContext : public gxf::GXFExecutionContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFExecutionContext::GXFExecutionContext;

  PyExecutionContext(gxf_context_t context, std::shared_ptr<PyInputContext>& py_input_context,
                     std::shared_ptr<PyOutputContext>& py_output_context,
                     py::object op = py::none());

  std::shared_ptr<PyInputContext> py_input() const;

  std::shared_ptr<PyOutputContext> py_output() const;

 private:
  py::object py_op_ = py::none();
  std::shared_ptr<PyInputContext> py_input_context_;
  std::shared_ptr<PyOutputContext> py_output_context_;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP */
