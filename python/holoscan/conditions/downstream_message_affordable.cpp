/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/conditions/gxf/downstream_affordable.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/resources/gxf/transmitter.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./downstream_message_affordable_pydoc.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the condition.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the condition's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_condition<ConditionT>
 */

class PyDownstreamMessageAffordableCondition : public DownstreamMessageAffordableCondition {
 public:
  /* Inherit the constructors */
  using DownstreamMessageAffordableCondition::DownstreamMessageAffordableCondition;

  // Define a constructor that fully initializes the object.
  explicit PyDownstreamMessageAffordableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, uint64_t min_size = 1L,
      std::optional<const std::string> transmitter = std::nullopt,
      const std::string& name = condition_default_name_v<DownstreamMessageAffordableCondition>)
      : DownstreamMessageAffordableCondition(Arg{"min_size", min_size}) {
    if (transmitter.has_value()) {
      this->add_arg(Arg("transmitter", transmitter.value()));
    }
    // Note "transmitter" parameter is set automatically from GXFExecutor
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_downstream_message_affordable(py::module_& m) {
  py::class_<DownstreamMessageAffordableCondition,
             PyDownstreamMessageAffordableCondition,
             gxf::GXFCondition,
             std::shared_ptr<DownstreamMessageAffordableCondition>>(
      m,
      "DownstreamMessageAffordableCondition",
      doc::DownstreamMessageAffordableCondition::doc_DownstreamMessageAffordableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    uint64_t,
                    std::optional<const std::string>,
                    const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1L,
           "transmitter"_a = py::none(),
           "name"_a = std::string(condition_default_name_v<DownstreamMessageAffordableCondition>),
           doc::DownstreamMessageAffordableCondition::doc_DownstreamMessageAffordableCondition)
      .def_property(
          "min_size",
          py::overload_cast<>(&DownstreamMessageAffordableCondition::min_size, py::const_),
          py::overload_cast<uint64_t>(&DownstreamMessageAffordableCondition::min_size),
          doc::DownstreamMessageAffordableCondition::doc_min_size)
      .def_property(
          "transmitter",
          py::overload_cast<>(&DownstreamMessageAffordableCondition::transmitter, py::const_),
          py::overload_cast<std::shared_ptr<Transmitter>>(
              &DownstreamMessageAffordableCondition::transmitter),
          doc::DownstreamMessageAffordableCondition::doc_transmitter);
}
}  // namespace holoscan
