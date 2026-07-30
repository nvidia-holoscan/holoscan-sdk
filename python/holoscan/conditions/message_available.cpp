/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/conditions/gxf/message_available.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/resources/gxf/receiver.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./message_available_pydoc.hpp"

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

class PyMessageAvailableCondition : public MessageAvailableCondition {
 public:
  /* Inherit the constructors */
  using MessageAvailableCondition::MessageAvailableCondition;

  // Define a constructor that fully initializes the object.
  explicit PyMessageAvailableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, uint64_t min_size = 1UL,
      size_t front_stage_max_size = 1UL, std::optional<const std::string> receiver = std::nullopt,
      const std::string& name = condition_default_name_v<MessageAvailableCondition>)
      : MessageAvailableCondition(
            ArgList{Arg{"min_size", min_size}, Arg{"front_stage_max_size", front_stage_max_size}}) {
    if (receiver.has_value()) {
      this->add_arg(Arg("receiver", receiver.value()));
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_message_available(py::module_& m) {
  py::class_<MessageAvailableCondition,
             PyMessageAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<MessageAvailableCondition>>(
      m, "MessageAvailableCondition", doc::MessageAvailableCondition::doc_MessageAvailableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    uint64_t,
                    size_t,
                    std::optional<const std::string>,
                    const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1UL,
           "front_stage_max_size"_a = 1UL,
           "receiver"_a = py::none(),
           "name"_a = std::string(condition_default_name_v<MessageAvailableCondition>),
           doc::MessageAvailableCondition::doc_MessageAvailableCondition)
      .def_property(
          "receiver",
          py::overload_cast<>(&MessageAvailableCondition::receiver, py::const_),
          py::overload_cast<std::shared_ptr<Receiver>>(&MessageAvailableCondition::receiver),
          doc::MessageAvailableCondition::doc_receiver)
      .def_property("min_size",
                    py::overload_cast<>(&MessageAvailableCondition::min_size, py::const_),
                    py::overload_cast<size_t>(&MessageAvailableCondition::min_size),
                    doc::MessageAvailableCondition::doc_min_size)
      .def_property(
          "front_stage_max_size",
          py::overload_cast<>(&MessageAvailableCondition::front_stage_max_size, py::const_),
          py::overload_cast<size_t>(&MessageAvailableCondition::front_stage_max_size),
          doc::MessageAvailableCondition::doc_front_stage_max_size);
}
}  // namespace holoscan
