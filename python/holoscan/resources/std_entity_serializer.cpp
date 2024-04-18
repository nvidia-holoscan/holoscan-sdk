/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./std_entity_serializer_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/std_entity_serializer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

class PyStdEntitySerializer : public StdEntitySerializer {
 public:
  /* Inherit the constructors */
  using StdEntitySerializer::StdEntitySerializer;

  // Define a constructor that fully initializes the object.
  explicit PyStdEntitySerializer(Fragment* fragment,
                                 const std::string& name = "std_entity_serializer")
      : StdEntitySerializer() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_std_entity_serializer(py::module_& m) {
  py::class_<StdEntitySerializer,
             PyStdEntitySerializer,
             gxf::GXFResource,
             std::shared_ptr<StdEntitySerializer>>(
      m, "StdEntitySerializer", doc::StdEntitySerializer::doc_StdEntitySerializer)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "std_entity_serializer"s,
           doc::StdEntitySerializer::doc_StdEntitySerializer_python)
      .def_property_readonly("gxf_typename",
                             &StdEntitySerializer::gxf_typename,
                             doc::StdEntitySerializer::doc_gxf_typename)
      .def("setup", &StdEntitySerializer::setup, "spec"_a, doc::StdEntitySerializer::doc_setup)
      .def(
          "initialize", &StdEntitySerializer::initialize, doc::StdEntitySerializer::doc_initialize);
}
}  // namespace holoscan
