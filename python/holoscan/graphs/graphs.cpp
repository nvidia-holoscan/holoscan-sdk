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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./graphs_pydoc.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;  // NOLINT(misc-unused-alias-decls)

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace PYBIND11_NAMESPACE {
namespace detail {

// NOLINTBEGIN(altera-struct-pack-align)
template <typename NodeT>
struct graph_caster {
 public:
  using GraphVectorT = std::vector<NodeT>;

  /**
   * This macro establishes the name 'vector_of_node_type' in
   * function signatures and declares a local variable
   * 'value' of type vector_of_node_type
   */
  PYBIND11_TYPE_CASTER(GraphVectorT, const_name("vector_of_node_type"));

  // get type caster for the individual values stored in the vector
  using value_conv = make_caster<NodeT>;

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into a
   * std::vector<NodeT> instance or return false upon failure. The
   * second argument indicates whether implicit conversions should be applied.
   */
  bool load([[maybe_unused]] handle src, [[maybe_unused]] bool use_implicit) {
    // not implemented
    return false;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert a std::vector<NodeT>
   * instance into a Python object. The second and third arguments are used to indicate the return
   * value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(GraphVectorT src, return_value_policy policy, handle parent) {
    list out(src.size());
    ssize_t index = 0;
    for (auto&& value : src) {
      auto value_ =
          reinterpret_steal<object>(value_conv::cast(std::forward<NodeT>(value), policy, parent));
      if (!value_) {
        return {};
      }
      PyList_SET_ITEM(out.ptr(), index++, value_.release().ptr());  // steals a reference
    }
    return out.release();
  }
};

template <>
class type_caster<std::vector<::holoscan::OperatorGraph::NodeType>>
    : public graph_caster<::holoscan::OperatorGraph::NodeType> {};

template <>
class type_caster<std::vector<::holoscan::FragmentGraph::NodeType>>
    : public graph_caster<::holoscan::FragmentGraph::NodeType> {};
// NOLINTEND(altera-struct-pack-align)

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

namespace holoscan {

template <typename NodeT = std::shared_ptr<Operator>, typename GraphT = OperatorGraph,
          typename EdgeDataElementT =
              std::unordered_map<std::string, std::set<std::string, std::less<>>>>
class PyGraph : public Graph<NodeT, EdgeDataElementT> {
 public:
  using NodeType = NodeT;
  using NodePredicate = std::function<bool(const NodeType&)>;
  using EdgeDataElementType = EdgeDataElementT;
  using EdgeDataType = std::shared_ptr<EdgeDataElementType>;

  /* Inherit the constructors */
  using Graph<NodeT, EdgeDataElementT>::Graph;

  // not implementing trampolines for virtual functions
  // (do not intend to override any of these from Python)
};

using PyOperatorGraph =
    PyGraph<std::shared_ptr<Operator>, OperatorGraph,
            std::unordered_map<std::string, std::set<std::string, std::less<>>>>;

using PyFragmentGraph =
    PyGraph<std::shared_ptr<Fragment>, FragmentGraph,
            std::unordered_map<std::string, std::set<std::string, std::less<>>>>;

PYBIND11_MODULE(_graphs, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Graph Python Bindings
        ----------------------------------
        .. currentmodule:: _graphs
    )pbdoc";

  // NOLINTBEGIN(bugprone-unused-raii)
  py::class_<OperatorGraph::NodeType>(m, "OperatorNodeType");
  py::class_<OperatorGraph::EdgeDataElementType>(m, "OperatorEdgeDataElementType");
  py::class_<OperatorGraph::EdgeDataType>(m, "OperatorEdgeDataType");
  py::class_<OperatorGraph, PyOperatorGraph, std::shared_ptr<OperatorGraph>>(
      m, "OperatorGraph", doc::Graph::doc_Graph);

  py::class_<FragmentGraph::NodeType>(m, "FragmentNodeType");
  // since the edge types are the same, can't redefine them here...
  // py::class_<FragmentGraph::EdgeDataElementType>(m, "FragmentEdgeDataElementType");
  // py::class_<FragmentGraph::EdgeDataType>(m, "FragmentEdgeDataType");
  py::class_<FragmentGraph, PyFragmentGraph, std::shared_ptr<FragmentGraph>>(
      m, "FragmentGraph", doc::Graph::doc_Graph);
  // NOLINTEND(bugprone-unused-raii)

  py::class_<OperatorFlowGraph, OperatorGraph, std::shared_ptr<OperatorFlowGraph>>(
      m, "OperatorFlowGraph", doc::FlowGraph::doc_FlowGraph)
      .def(py::init<>(), doc::FlowGraph::doc_FlowGraph)
      .def("add_node", &OperatorFlowGraph::add_node, "node"_a, doc::FlowGraph::doc_add_node)
      .def(
          "get_port_map",
          [](const OperatorFlowGraph& graph,
             const ::holoscan::OperatorGraph::NodeType& node_u,
             const ::holoscan::OperatorGraph::NodeType& node_v) -> py::dict {
            py::dict port_dict;
            auto port_map_opt = graph.get_port_map(node_u, node_v);
            if (!port_map_opt.has_value()) {
              return port_dict;
            }
            const auto& port_map = port_map_opt.value();
            for (const auto& [key, cpp_set] : *port_map) {
              py::set py_port_set;
              for (const std::string& port_name : cpp_set) {
                py_port_set.add(py::cast(port_name));
              }
              port_dict[py::str(key)] = py_port_set;
            }
            return port_dict;
          },
          "node_u"_a,
          "node_v"_a,
          doc::FlowGraph::doc_get_port_map)
      .def(
          "is_root",
          [](const OperatorFlowGraph& graph, const ::holoscan::OperatorGraph::NodeType& node) {
            return graph.is_root(node);
          },
          "node"_a,
          doc::FlowGraph::doc_is_root)
      .def(
          "is_leaf",
          [](const OperatorFlowGraph& graph, const ::holoscan::OperatorGraph::NodeType& node) {
            return graph.is_leaf(node);
          },
          "node"_a,
          doc::FlowGraph::doc_is_leaf)
      .def(
          "get_root_nodes",
          [](const OperatorFlowGraph& graph) { return graph.get_root_nodes(); },
          doc::FlowGraph::doc_get_root_nodes)
      .def(
          "get_nodes",
          [](const OperatorFlowGraph& graph) { return graph.get_nodes(); },
          doc::FlowGraph::doc_get_nodes)
      .def(
          "get_next_nodes",
          [](const OperatorFlowGraph& graph, const ::holoscan::OperatorGraph::NodeType& node) {
            return graph.get_next_nodes(node);
          },
          "node"_a,
          doc::FlowGraph::doc_get_next_nodes)
      .def(
          "get_previous_nodes",
          [](const OperatorFlowGraph& graph, const ::holoscan::OperatorGraph::NodeType& node) {
            return graph.get_previous_nodes(node);
          },
          "node"_a,
          doc::FlowGraph::doc_get_previous_nodes)
      .def(
          "remove_node",
          [](OperatorFlowGraph& graph, const ::holoscan::OperatorGraph::NodeType& node) {
            graph.remove_node(node);
          },
          "node"_a,
          doc::FlowGraph::doc_remove_node)
      .def_property(
          "context",
          [](const OperatorFlowGraph& graph) { return graph.context(); },
          [](OperatorFlowGraph& graph, void* ctx) { graph.context(ctx); },
          doc::FlowGraph::doc_context)
      .def(
          "get_port_connectivity_maps",
          [](const OperatorFlowGraph& graph) {
            auto result = graph.get_port_connectivity_maps();
            return py::make_tuple(result.first, result.second);
          },
          doc::FlowGraph::doc_get_port_connectivity_maps)
      .def("port_map_description",
           &OperatorFlowGraph::port_map_description,
           doc::FlowGraph::doc_port_map_description);

  py::class_<FragmentFlowGraph, FragmentGraph, std::shared_ptr<FragmentFlowGraph>>(
      m, "FragmentFlowGraph", doc::FlowGraph::doc_FlowGraph)
      .def(py::init<>(), doc::FlowGraph::doc_FlowGraph)
      .def("add_node", &FragmentFlowGraph::add_node, "node"_a, doc::FlowGraph::doc_add_node)
      .def(
          "get_port_map",
          [](const FragmentFlowGraph& graph,
             const ::holoscan::FragmentGraph::NodeType& node_u,
             const ::holoscan::FragmentGraph::NodeType& node_v) -> py::dict {
            py::dict port_dict;
            auto port_map_opt = graph.get_port_map(node_u, node_v);
            if (!port_map_opt.has_value()) {
              return port_dict;
            }
            const auto& port_map = port_map_opt.value();
            for (const auto& [key, cpp_set] : *port_map) {
              py::set py_port_set;
              for (const std::string& port_name : cpp_set) {
                py_port_set.add(py::cast(port_name));
              }
              port_dict[py::str(key)] = py_port_set;
            }
            return port_dict;
          },
          "node_u"_a,
          "node_v"_a,
          doc::FlowGraph::doc_get_port_map)
      .def(
          "is_root",
          [](const FragmentFlowGraph& graph, const ::holoscan::FragmentGraph::NodeType& node) {
            return graph.is_root(node);
          },
          "node"_a,
          doc::FlowGraph::doc_is_root)
      .def(
          "is_leaf",
          [](const FragmentFlowGraph& graph, const ::holoscan::FragmentGraph::NodeType& node) {
            return graph.is_leaf(node);
          },
          "node"_a,
          doc::FlowGraph::doc_is_leaf)
      .def(
          "get_root_nodes",
          [](const FragmentFlowGraph& graph) { return graph.get_root_nodes(); },
          doc::FlowGraph::doc_get_root_nodes)
      .def(
          "get_nodes",
          [](const FragmentFlowGraph& graph) { return graph.get_nodes(); },
          doc::FlowGraph::doc_get_nodes)
      .def(
          "get_next_nodes",
          [](const FragmentFlowGraph& graph, const ::holoscan::FragmentGraph::NodeType& node) {
            return graph.get_next_nodes(node);
          },
          "node"_a,
          doc::FlowGraph::doc_get_next_nodes)
      .def(
          "get_previous_nodes",
          [](const FragmentFlowGraph& graph, const ::holoscan::FragmentGraph::NodeType& node) {
            return graph.get_previous_nodes(node);
          },
          "node"_a,
          doc::FlowGraph::doc_get_previous_nodes)
      .def(
          "remove_node",
          [](FragmentFlowGraph& graph, const ::holoscan::FragmentGraph::NodeType& node) {
            graph.remove_node(node);
          },
          "node"_a,
          doc::FlowGraph::doc_remove_node)
      .def_property(
          "context",
          [](const FragmentFlowGraph& graph) { return graph.context(); },
          [](FragmentFlowGraph& graph, void* ctx) { graph.context(ctx); },
          doc::FlowGraph::doc_context)
      .def(
          "get_port_connectivity_maps",
          [](const FragmentFlowGraph& graph) {
            auto result = graph.get_port_connectivity_maps();
            return py::make_tuple(result.first, result.second);
          },
          doc::FlowGraph::doc_get_port_connectivity_maps)
      .def("port_map_description",
           &FragmentFlowGraph::port_map_description,
           doc::FlowGraph::doc_port_map_description);
}  // PYBIND11_MODULE
}  // namespace holoscan
