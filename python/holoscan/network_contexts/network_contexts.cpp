/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <string>

#include "./network_contexts_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"
#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"
#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the network context.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the network context's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on
 * Fragment::make_network_context<NetworkContextT>
 */

class PyUcxContext : public UcxContext {
 public:
  /* Inherit the constructors */
  using UcxContext::UcxContext;

  // Define a constructor that fully initializes the object.
  PyUcxContext(Fragment* fragment, std::shared_ptr<UcxEntitySerializer> serializer = nullptr,
               const std::string& name = "ucx_context")
      : UcxContext() {
    if (serializer) { this->add_arg(Arg{"serializer", serializer}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};
// End of trampoline classes for handling Python kwargs

PYBIND11_MODULE(_network_contexts, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _network_contexts
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<UcxContext,
             PyUcxContext,
             gxf::GXFNetworkContext,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<UcxContext>>(m, "UcxContext", doc::UcxContext::doc_UcxContext)
      .def(py::init<Fragment*, std::shared_ptr<UcxEntitySerializer>, const std::string&>(),
           "fragment"_a,
           "serializer"_a = nullptr,
           "name"_a = "ucx_context"s,
           doc::UcxContext::doc_UcxContext_python)
      .def_property_readonly(
          "gxf_typename", &UcxContext::gxf_typename, doc::UcxContext::doc_gxf_typename);
}  // PYBIND11_MODULE
}  // namespace holoscan
