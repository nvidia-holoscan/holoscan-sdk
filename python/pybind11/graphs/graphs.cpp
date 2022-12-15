/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// #include "holoscan/core/component_spec.hpp"

// #include "holoscan/core/gxf/gxf_resource.hpp"
// #include "holoscan/core/resources/gxf/allocator.hpp"
// #include "holoscan/core/resources/gxf/block_memory_pool.hpp"
// #include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
// #include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>

#include "./graphs_pydoc.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// can only bind these containers of shared pointers if they are made opaque
// Note: compilation will fail if pybind11/stl.h was included above. see:
// https://github.com/RosettaCommons/binder/issues/100#issuecomment-867197266
PYBIND11_MAKE_OPAQUE(std::vector<::holoscan::Graph::NodeType>);
PYBIND11_MAKE_OPAQUE(std::vector<::holoscan::Graph::EdgeDataElementType>);
PYBIND11_MAKE_OPAQUE(std::vector<::holoscan::Graph::EdgeDataType>);

namespace holoscan {

class PyGraph : public Graph {
 public:
  /* Inherit the constructors */
  using Graph::Graph;

  using Graph::EdgeDataElementType;
  using Graph::EdgeDataType;
  using Graph::NodeType;

  /* Trampolines (need one for each virtual function) */
  void add_operator(const NodeType& op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Graph, add_operator, op);
  }
  void add_flow(const NodeType& op_u, const NodeType& op_v, const EdgeDataType& port_map) override {
    PYBIND11_OVERRIDE(void, Graph, add_flow, op_u, op_v, port_map);
  }
  std::optional<EdgeDataType> get_port_map(const NodeType& op_u, const NodeType& op_v) override {
    PYBIND11_OVERRIDE(std::optional<EdgeDataType>, Graph, get_port_map, op_u, op_v);
  }
  bool is_root(const NodeType& op) override { PYBIND11_OVERRIDE(bool, Graph, is_root, op); }
  bool is_leaf(const NodeType& op) override { PYBIND11_OVERRIDE(bool, Graph, is_leaf, op); }
  std::vector<NodeType> get_root_operators() override {
    PYBIND11_OVERRIDE(std::vector<NodeType>, Graph, get_root_operators);
  }
  std::vector<NodeType> get_operators() override {
    PYBIND11_OVERRIDE(std::vector<NodeType>, Graph, get_operators);
  }
  std::vector<NodeType> get_next_operators(const NodeType& op) override {
    PYBIND11_OVERRIDE(std::vector<NodeType>, Graph, get_next_operators, op);
  }
  void context(void* context) override { PYBIND11_OVERRIDE(void, Graph, context, context); }
  void* context() override { PYBIND11_OVERRIDE(void*, Graph, context); }
};

PYBIND11_MODULE(_graphs, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _graphs
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<Graph::NodeType>(m, "NodeType");
  py::class_<Graph::EdgeDataElementType>(m, "EdgeDataElementType");
  py::class_<Graph::EdgeDataType>(m, "EdgeDataType");

  py::class_<Graph, PyGraph, std::shared_ptr<Graph>>(m, "Graph", doc::Graph::doc_Graph);

  py::class_<FlowGraph, Graph, std::shared_ptr<FlowGraph>>(
      m, "FlowGraph", doc::FlowGraph::doc_FlowGraph)
      .def(py::init<>(), doc::FlowGraph::doc_FlowGraph)
      .def("add_operator", &FlowGraph::add_operator, "op"_a, doc::FlowGraph::doc_add_operator)
      .def("add_flow",
           &FlowGraph::add_flow,
           "op_u"_a,
           "op_v"_a,
           "port_map"_a,
           doc::FlowGraph::doc_add_flow)
      .def("get_port_map",
           &FlowGraph::get_port_map,
           "op_u"_a,
           "op_v"_a,
           doc::FlowGraph::doc_get_port_map)
      .def("is_root", &FlowGraph::is_root, "op"_a, doc::FlowGraph::doc_is_root)
      .def("is_leaf", &FlowGraph::is_leaf, "op"_a, doc::FlowGraph::doc_is_leaf)
      .def_property_readonly("get_root_operators",
                             &FlowGraph::get_root_operators,
                             doc::FlowGraph::doc_get_root_operators)
      .def_property_readonly(
          "get_operators", &FlowGraph::get_operators, doc::FlowGraph::doc_get_operators)
      .def_property_readonly("get_next_operators",
                             &FlowGraph::get_next_operators,
                             doc::FlowGraph::doc_get_next_operators)
      .def_property("context",
                    py::overload_cast<>(&FlowGraph::context),
                    py::overload_cast<void*>(&FlowGraph::context),
                    doc::FlowGraph::doc_context);
  // note: add_flow and get_port_map involve containers of opaque objects
}  // PYBIND11_MODULE
}  // namespace holoscan
