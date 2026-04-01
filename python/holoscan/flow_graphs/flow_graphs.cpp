/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./flow_graphs_pydoc.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/flow_graphs/flow_graph_impl.hpp"
#include "holoscan/core/flow_graphs/flow_graph.hpp"
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
class type_caster<std::vector<::holoscan::OperatorFlowGraph::NodeType>>
    : public graph_caster<::holoscan::OperatorFlowGraph::NodeType> {};

template <>
class type_caster<std::vector<::holoscan::FragmentFlowGraph::NodeType>>
    : public graph_caster<::holoscan::FragmentFlowGraph::NodeType> {};
// NOLINTEND(altera-struct-pack-align)

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

namespace holoscan {

template <typename NodeT = std::shared_ptr<Operator>, typename GraphT = OperatorFlowGraph,
          typename EdgeDataElementT =
              std::unordered_map<std::string, std::set<std::string, std::less<>>>>
class PyFlowGraph : public FlowGraph<NodeT, EdgeDataElementT> {
 public:
  using NodeType = NodeT;
  using NodePredicate = std::function<bool(const NodeType&)>;
  using EdgeDataElementType = EdgeDataElementT;
  using EdgeDataType = std::shared_ptr<EdgeDataElementType>;

  /* Inherit the constructors */
  using FlowGraph<NodeT, EdgeDataElementT>::FlowGraph;

  // not implementing trampolines for virtual functions
  // (do not intend to override any of these from Python)
};

using PyOperatorFlowGraph =
    PyFlowGraph<std::shared_ptr<Operator>, OperatorFlowGraph,
                std::unordered_map<std::string, std::set<std::string, std::less<>>>>;

using PyFragmentFlowGraph =
    PyFlowGraph<std::shared_ptr<Fragment>, FragmentFlowGraph,
                std::unordered_map<std::string, std::set<std::string, std::less<>>>>;

PYBIND11_MODULE(_flow_graphs, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Flow Graph Python Bindings
        ----------------------------------
        .. currentmodule:: _flow_graphs
    )pbdoc";

  // NOLINTBEGIN(bugprone-unused-raii)
  py::class_<OperatorFlowGraph::NodeType>(m, "OperatorNodeType");
  py::class_<OperatorFlowGraph::EdgeDataElementType>(m, "OperatorEdgeDataElementType");
  py::class_<OperatorFlowGraph::EdgeDataType>(m, "OperatorEdgeDataType");
  py::class_<OperatorFlowGraph, PyOperatorFlowGraph, std::shared_ptr<OperatorFlowGraph>>(
      m, "OperatorFlowGraph", doc::FlowGraph::doc_FlowGraph);

  py::class_<FragmentFlowGraph::NodeType>(m, "FragmentNodeType");
  // since the edge types are the same, can't redefine them here...
  // py::class_<FragmentFlowGraph::EdgeDataElementType>(m, "FragmentEdgeDataElementType");
  // py::class_<FragmentFlowGraph::EdgeDataType>(m, "FragmentEdgeDataType");
  py::class_<FragmentFlowGraph, PyFragmentFlowGraph, std::shared_ptr<FragmentFlowGraph>>(
      m, "FragmentFlowGraph", doc::FlowGraph::doc_FlowGraph);
  // NOLINTEND(bugprone-unused-raii)

  py::class_<OperatorFlowGraphImpl, OperatorFlowGraph, std::shared_ptr<OperatorFlowGraphImpl>>(
      m, "OperatorFlowGraphImpl", doc::FlowGraphImpl::doc_FlowGraphImpl)
      .def(py::init<>(), doc::FlowGraphImpl::doc_FlowGraphImpl)
      .def("add_node", &OperatorFlowGraphImpl::add_node, "node"_a, doc::FlowGraphImpl::doc_add_node)
      .def(
          "get_port_map",
          [](const OperatorFlowGraphImpl& graph,
             const ::holoscan::OperatorFlowGraph::NodeType& node_u,
             const ::holoscan::OperatorFlowGraph::NodeType& node_v) -> py::dict {
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
          doc::FlowGraphImpl::doc_get_port_map)
      .def(
          "is_root",
          [](const OperatorFlowGraphImpl& graph,
             const ::holoscan::OperatorFlowGraph::NodeType& node) { return graph.is_root(node); },
          "node"_a,
          doc::FlowGraphImpl::doc_is_root)
      .def(
          "is_leaf",
          [](const OperatorFlowGraphImpl& graph,
             const ::holoscan::OperatorFlowGraph::NodeType& node) { return graph.is_leaf(node); },
          "node"_a,
          doc::FlowGraphImpl::doc_is_leaf)
      .def(
          "get_root_nodes",
          [](const OperatorFlowGraphImpl& graph) { return graph.get_root_nodes(); },
          doc::FlowGraphImpl::doc_get_root_nodes)
      .def(
          "get_nodes",
          [](const OperatorFlowGraphImpl& graph) { return graph.get_nodes(); },
          doc::FlowGraphImpl::doc_get_nodes)
      .def(
          "get_next_nodes",
          [](const OperatorFlowGraphImpl& graph,
             const ::holoscan::OperatorFlowGraph::NodeType& node) {
            return graph.get_next_nodes(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_get_next_nodes)
      .def(
          "get_previous_nodes",
          [](const OperatorFlowGraphImpl& graph,
             const ::holoscan::OperatorFlowGraph::NodeType& node) {
            return graph.get_previous_nodes(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_get_previous_nodes)
      .def(
          "remove_node",
          [](OperatorFlowGraphImpl& graph, const ::holoscan::OperatorFlowGraph::NodeType& node) {
            graph.remove_node(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_remove_node)
      .def_property(
          "context",
          [](const OperatorFlowGraphImpl& graph) { return graph.context(); },
          [](OperatorFlowGraphImpl& graph, void* ctx) { graph.context(ctx); },
          doc::FlowGraphImpl::doc_context)
      .def(
          "get_port_connectivity_maps",
          [](const OperatorFlowGraphImpl& graph) {
            auto result = graph.get_port_connectivity_maps();
            return py::make_tuple(result.first, result.second);
          },
          doc::FlowGraphImpl::doc_get_port_connectivity_maps)
      .def("port_map_description",
           &OperatorFlowGraphImpl::port_map_description,
           doc::FlowGraphImpl::doc_port_map_description);

  py::class_<FragmentFlowGraphImpl, FragmentFlowGraph, std::shared_ptr<FragmentFlowGraphImpl>>(
      m, "FragmentFlowGraphImpl", doc::FlowGraphImpl::doc_FlowGraphImpl)
      .def(py::init<>(), doc::FlowGraphImpl::doc_FlowGraphImpl)
      .def("add_node", &FragmentFlowGraphImpl::add_node, "node"_a, doc::FlowGraphImpl::doc_add_node)
      .def(
          "get_port_map",
          [](const FragmentFlowGraphImpl& graph,
             const ::holoscan::FragmentFlowGraph::NodeType& node_u,
             const ::holoscan::FragmentFlowGraph::NodeType& node_v) -> py::dict {
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
          doc::FlowGraphImpl::doc_get_port_map)
      .def(
          "is_root",
          [](const FragmentFlowGraphImpl& graph,
             const ::holoscan::FragmentFlowGraph::NodeType& node) { return graph.is_root(node); },
          "node"_a,
          doc::FlowGraphImpl::doc_is_root)
      .def(
          "is_leaf",
          [](const FragmentFlowGraphImpl& graph,
             const ::holoscan::FragmentFlowGraph::NodeType& node) { return graph.is_leaf(node); },
          "node"_a,
          doc::FlowGraphImpl::doc_is_leaf)
      .def(
          "get_root_nodes",
          [](const FragmentFlowGraphImpl& graph) { return graph.get_root_nodes(); },
          doc::FlowGraphImpl::doc_get_root_nodes)
      .def(
          "get_nodes",
          [](const FragmentFlowGraphImpl& graph) { return graph.get_nodes(); },
          doc::FlowGraphImpl::doc_get_nodes)
      .def(
          "get_next_nodes",
          [](const FragmentFlowGraphImpl& graph,
             const ::holoscan::FragmentFlowGraph::NodeType& node) {
            return graph.get_next_nodes(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_get_next_nodes)
      .def(
          "get_previous_nodes",
          [](const FragmentFlowGraphImpl& graph,
             const ::holoscan::FragmentFlowGraph::NodeType& node) {
            return graph.get_previous_nodes(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_get_previous_nodes)
      .def(
          "remove_node",
          [](FragmentFlowGraphImpl& graph, const ::holoscan::FragmentFlowGraph::NodeType& node) {
            graph.remove_node(node);
          },
          "node"_a,
          doc::FlowGraphImpl::doc_remove_node)
      .def_property(
          "context",
          [](const FragmentFlowGraphImpl& graph) { return graph.context(); },
          [](FragmentFlowGraphImpl& graph, void* ctx) { graph.context(ctx); },
          doc::FlowGraphImpl::doc_context)
      .def(
          "get_port_connectivity_maps",
          [](const FragmentFlowGraphImpl& graph) {
            auto result = graph.get_port_connectivity_maps();
            return py::make_tuple(result.first, result.second);
          },
          doc::FlowGraphImpl::doc_get_port_connectivity_maps)
      .def("port_map_description",
           &FragmentFlowGraphImpl::port_map_description,
           doc::FlowGraphImpl::doc_port_map_description);
}  // PYBIND11_MODULE
}  // namespace holoscan
