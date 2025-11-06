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
#include <string>
#include <variant>

#include "../core/component_util.hpp"
#include "./entity_serializers_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/resources/gxf/ucx_entity_serializer.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyUcxEntitySerializer : public UcxEntitySerializer {
 public:
  /* Inherit the constructors */
  using UcxEntitySerializer::UcxEntitySerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxEntitySerializer(
      std::variant<Fragment*, Subgraph*> fragment_or_subgraph,
      // std::vector<std::shared_ptr<holoscan::Resource>> component_serializers = {},
      bool verbose_warning = false, const std::string& name = "ucx_entity_buffer")
      : UcxEntitySerializer(ArgList{
            Arg{"verbose_warning", verbose_warning},
        }) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_entity_serializers(py::module_& m) {
  py::class_<UcxEntitySerializer,
             PyUcxEntitySerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxEntitySerializer>>(
      m, "UcxEntitySerializer", doc::UcxEntitySerializer::doc_UcxEntitySerializer)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    // std::vector<std::shared_ptr<holoscan::Resource>>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           // "component_serializers"_a = std::vector<std::shared_ptr<holoscan::Resource>>{},
           "verbose_warning"_a = false,
           "name"_a = "ucx_entity_serializer"s,
           doc::UcxEntitySerializer::doc_UcxEntitySerializer);
}
}  // namespace holoscan
