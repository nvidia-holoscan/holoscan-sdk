/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_SUBGRAPH_HPP
#define PYHOLOSCAN_CORE_SUBGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/subgraph.hpp"

namespace py = pybind11;

namespace holoscan {

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

class PySubgraph : public Subgraph {
 public:
  /* Inherit the constructors */
  using Subgraph::Subgraph;

  PySubgraph(py::object subgraph, Fragment* fragment, const std::string& name);
  ~PySubgraph() override;

  /* Trampolines (need one for each virtual function) */
  void compose() override;

 private:
  py::object py_subgraph_ = py::none();
  py::object py_compose_ = py::none();
};

void init_subgraph(py::module_&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_SUBGRAPH_HPP */
