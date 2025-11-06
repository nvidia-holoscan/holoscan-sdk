/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "../core/component_util.hpp"
#include "./memory_available_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/memory_available.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/subgraph.hpp"

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

class PyMemoryAvailableCondition : public MemoryAvailableCondition {
 public:
  /* Inherit the constructors */
  using MemoryAvailableCondition::MemoryAvailableCondition;

  // Define a constructor that fully initializes the object.
  explicit PyMemoryAvailableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<Allocator> allocator, std::optional<uint64_t> min_bytes = std::nullopt,
      std::optional<uint64_t> min_blocks = std::nullopt,
      const std::string& name = "noname_memory_available_condition")
      : MemoryAvailableCondition(Arg{"allocator", allocator}) {
    if (!min_bytes.has_value() && !min_blocks.has_value()) {
      throw pybind11::value_error("Either `min_bytes` or `min_blocks` must be provided.");
    }
    if (min_bytes.has_value() && min_blocks.has_value()) {
      throw pybind11::value_error("Only one of `min_bytes` or `min_blocks` can be set.");
    }
    if (min_bytes.has_value()) {
      this->add_arg(Arg("min_bytes", min_bytes.value()));
    } else if (min_blocks.has_value()) {
      this->add_arg(Arg("min_blocks", min_blocks.value()));
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_memory_available(py::module_& m) {
  py::class_<MemoryAvailableCondition,
             PyMemoryAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<MemoryAvailableCondition>>(
      m, "MemoryAvailableCondition", doc::MemoryAvailableCondition::doc_MemoryAvailableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<Allocator>,
                    std::optional<uint64_t>,
                    std::optional<uint64_t>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "min_bytes"_a = py::none(),
           "min_blocks"_a = py::none(),
           "name"_a = "noname_memory_available_condition"s,
           doc::MemoryAvailableCondition::doc_MemoryAvailableCondition)
      .def_property(
          "allocator",
          py::overload_cast<>(&MemoryAvailableCondition::allocator),
          py::overload_cast<std::shared_ptr<Allocator>>(&MemoryAvailableCondition::allocator),
          doc::MemoryAvailableCondition::doc_allocator);
}
}  // namespace holoscan
