/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_SUBGRAPH_HPP
#define PYHOLOSCAN_CORE_SUBGRAPH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/subgraph.hpp>

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

  PySubgraph(py::object subgraph, Fragment* fragment, const std::string& name,
             const std::string& config_file = "");
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
