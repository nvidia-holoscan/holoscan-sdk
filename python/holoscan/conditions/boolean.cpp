/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <variant>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/conditions/gxf/boolean.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./boolean_pydoc.hpp"

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

class PyBooleanCondition : public BooleanCondition {
 public:
  /* Inherit the constructors */
  using BooleanCondition::BooleanCondition;

  // Define a constructor that fully initializes the object.
  explicit PyBooleanCondition(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                              bool enable_tick = true,
                              const std::string& name = condition_default_name_v<BooleanCondition>)
      : BooleanCondition(Arg{"enable_tick", enable_tick}) {
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_boolean(py::module_& m) {
  py::class_<BooleanCondition,
             PyBooleanCondition,
             gxf::GXFCondition,
             std::shared_ptr<BooleanCondition>>(
      m, "BooleanCondition", doc::BooleanCondition::doc_BooleanCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>, bool, const std::string&>(),
           "fragment"_a,
           "enable_tick"_a = true,
           "name"_a = std::string(condition_default_name_v<BooleanCondition>),
           doc::BooleanCondition::doc_BooleanCondition)
      .def("enable_tick", &BooleanCondition::enable_tick, doc::BooleanCondition::doc_enable_tick)
      .def("disable_tick", &BooleanCondition::disable_tick, doc::BooleanCondition::doc_disable_tick)
      .def("check_tick_enabled",
           &BooleanCondition::check_tick_enabled,
           doc::BooleanCondition::doc_check_tick_enabled);
}  // PYBIND11_MODULE
}  // namespace holoscan
