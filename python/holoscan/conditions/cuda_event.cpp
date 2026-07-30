/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/core/conditions/gxf/cuda_event.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/resources/gxf/receiver.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./cuda_event_pydoc.hpp"

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

class PyCudaEventCondition : public CudaEventCondition {
 public:
  /* Inherit the constructors */
  using CudaEventCondition::CudaEventCondition;

  // Define a constructor that fully initializes the object.
  explicit PyCudaEventCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      const std::string& event_name = "", std::optional<const std::string> receiver = std::nullopt,
      const std::string& name = condition_default_name_v<CudaEventCondition>)
      : CudaEventCondition(Arg("event_name", event_name)) {
    if (receiver.has_value()) {
      this->add_arg(Arg("receiver", receiver.value()));
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_cuda_event(py::module_& m) {
  py::class_<CudaEventCondition,
             PyCudaEventCondition,
             gxf::GXFCondition,
             std::shared_ptr<CudaEventCondition>>(
      m, "CudaEventCondition", doc::CudaEventCondition::doc_CudaEventCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    const std::string&,
                    std::optional<const std::string>,
                    const std::string&>(),
           "fragment"_a,
           "event_name"_a = ""s,
           "receiver"_a = py::none(),
           "name"_a = std::string(condition_default_name_v<CudaEventCondition>),
           doc::CudaEventCondition::doc_CudaEventCondition)
      .def_property("receiver",
                    py::overload_cast<>(&CudaEventCondition::receiver),
                    py::overload_cast<std::shared_ptr<Receiver>>(&CudaEventCondition::receiver),
                    doc::CudaEventCondition::doc_receiver);
}
}  // namespace holoscan
