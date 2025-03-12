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

#include "./cuda_stream_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/cuda_stream.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/receiver.hpp"

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

class PyCudaStreamCondition : public CudaStreamCondition {
 public:
  /* Inherit the constructors */
  using CudaStreamCondition::CudaStreamCondition;

  // Define a constructor that fully initializes the object.
  explicit PyCudaStreamCondition(Fragment* fragment,
                                 std::optional<const std::string> receiver = std::nullopt,
                                 const std::string& name = "noname_cuda_stream_condition") {
    name_ = name;
    fragment_ = fragment;
    if (receiver.has_value()) { this->add_arg(Arg("receiver", receiver.value())); }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_cuda_stream(py::module_& m) {
  py::class_<CudaStreamCondition,
             PyCudaStreamCondition,
             gxf::GXFCondition,
             std::shared_ptr<CudaStreamCondition>>(
      m, "CudaStreamCondition", doc::CudaStreamCondition::doc_CudaStreamCondition)
      .def(py::init<Fragment*, std::optional<const std::string>, const std::string&>(),
           "fragment"_a,
           "receiver"_a = py::none(),
           "name"_a = "noname_cuda_stream_condition"s,
           doc::CudaStreamCondition::doc_CudaStreamCondition)
      .def_property("receiver",
                    py::overload_cast<>(&CudaStreamCondition::receiver),
                    py::overload_cast<std::shared_ptr<Receiver>>(&CudaStreamCondition::receiver),
                    doc::CudaStreamCondition::doc_receiver);
}
}  // namespace holoscan
