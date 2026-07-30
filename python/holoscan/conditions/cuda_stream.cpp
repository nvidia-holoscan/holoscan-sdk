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
#include <type_traits>
#include <variant>
#include <vector>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/conditions/gxf/cuda_stream.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/resources/gxf/receiver.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./cuda_stream_pydoc.hpp"

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

// Type alias for receivers parameter: accepts either a single string or a list of strings
using ReceiversArg = std::variant<std::string, std::vector<std::string>>;

class PyCudaStreamCondition : public CudaStreamCondition {
 public:
  /* Inherit the constructors */
  using CudaStreamCondition::CudaStreamCondition;

  // Define a constructor that fully initializes the object.
  // Supports both legacy 'receiver' (single port) and new 'receivers' (one or more ports) APIs.
  explicit PyCudaStreamCondition(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                 std::optional<std::string> receiver = std::nullopt,
                                 std::optional<ReceiversArg> receivers = std::nullopt,
                                 bool check_all_messages = true,
                                 const std::string& name = "noname_cuda_stream_condition") {
    // Validate that exactly one of 'receiver' or 'receivers' is specified
    if (receiver.has_value() && receivers.has_value()) {
      throw std::runtime_error(
          "CudaStreamCondition: cannot specify both 'receiver' and 'receivers' parameters. "
          "Use 'receiver' for a single port (legacy API) or 'receivers' for one or more ports.");
    }
    if (!receiver.has_value() && !receivers.has_value()) {
      throw std::runtime_error(
          "CudaStreamCondition: must specify either 'receiver' or 'receivers' parameter. "
          "Use 'receiver' for a single port (legacy API) or 'receivers' for one or more ports.");
    }

    if (receiver.has_value()) {
      // Legacy API: single receiver
      this->add_arg(Arg("receiver", receiver.value()));
    } else {
      // New API: receivers (single string or vector)
      std::visit(
          [this](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
              // Single string - wrap in vector for the C++ side
              this->add_arg(Arg("receivers", std::vector<std::string>{arg}));
            } else {
              // Already a vector
              this->add_arg(Arg("receivers", arg));
            }
          },
          receivers.value());
    }
    this->add_arg(Arg("check_all_messages", check_all_messages));
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_cuda_stream(py::module_& m) {
  py::class_<CudaStreamCondition,
             PyCudaStreamCondition,
             Condition,
             std::shared_ptr<CudaStreamCondition>>(
      m, "CudaStreamCondition", doc::CudaStreamCondition::doc_CudaStreamCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::optional<std::string>,
                    std::optional<ReceiversArg>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "receiver"_a = py::none(),
           "receivers"_a = py::none(),
           "check_all_messages"_a = true,
           "name"_a = "noname_cuda_stream_condition"s,
           doc::CudaStreamCondition::doc_CudaStreamCondition)
      .def_property("receiver",
                    py::overload_cast<>(&CudaStreamCondition::receiver),
                    py::overload_cast<std::shared_ptr<Receiver>>(&CudaStreamCondition::receiver),
                    doc::CudaStreamCondition::doc_receiver)
      .def_property("receivers",
                    py::overload_cast<>(&CudaStreamCondition::receivers),
                    py::overload_cast<std::vector<std::shared_ptr<Receiver>>>(
                        &CudaStreamCondition::receivers),
                    doc::CudaStreamCondition::doc_receivers)
      .def_property("check_all_messages",
                    static_cast<bool (CudaStreamCondition::*)() const>(
                        &CudaStreamCondition::check_all_messages),
                    static_cast<void (CudaStreamCondition::*)(bool)>(
                        &CudaStreamCondition::check_all_messages),
                    doc::CudaStreamCondition::doc_check_all_messages);
}
}  // namespace holoscan
