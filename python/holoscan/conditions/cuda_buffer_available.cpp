/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "./cuda_buffer_available_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/cuda_buffer_available.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/receiver.hpp"
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

class PyCudaBufferAvailableCondition : public CudaBufferAvailableCondition {
 public:
  /* Inherit the constructors */
  using CudaBufferAvailableCondition::CudaBufferAvailableCondition;

  // Define a constructor that fully initializes the object.
  explicit PyCudaBufferAvailableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::optional<const std::string> receiver = std::nullopt,
      const std::string& name = "noname_cuda_buffer_available_condition") {
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_cuda_buffer_available(py::module_& m) {
  py::class_<CudaBufferAvailableCondition,
             PyCudaBufferAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<CudaBufferAvailableCondition>>(
      m,
      "CudaBufferAvailableCondition",
      doc::CudaBufferAvailableCondition::doc_CudaBufferAvailableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::optional<const std::string>,
                    const std::string&>(),
           "fragment"_a,
           "receiver"_a = py::none(),
           "name"_a = "noname_cuda_buffer_available_condition"s,
           doc::CudaBufferAvailableCondition::doc_CudaBufferAvailableCondition)
      .def_property(
          "receiver",
          py::overload_cast<>(&CudaBufferAvailableCondition::receiver),
          py::overload_cast<std::shared_ptr<Receiver>>(&CudaBufferAvailableCondition::receiver),
          doc::CudaBufferAvailableCondition::doc_receiver);
}
}  // namespace holoscan
