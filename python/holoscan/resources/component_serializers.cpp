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
#include "./component_serializers_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/std_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_component_serializer.hpp"
#include "holoscan/core/resources/gxf/ucx_holoscan_component_serializer.hpp"
#include "holoscan/core/subgraph.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyStdComponentSerializer : public StdComponentSerializer {
 public:
  /* Inherit the constructors */
  using StdComponentSerializer::StdComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyStdComponentSerializer(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                    const std::string& name = "std_component_serializer") {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyUcxComponentSerializer : public UcxComponentSerializer {
 public:
  /* Inherit the constructors */
  using UcxComponentSerializer::UcxComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxComponentSerializer(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                    std::shared_ptr<holoscan::Allocator> allocator = nullptr,
                                    const std::string& name = "ucx_component_serializer") {
    if (allocator) {
      this->add_arg(Arg{"allocator", allocator});
    }
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyUcxHoloscanComponentSerializer : public UcxHoloscanComponentSerializer {
 public:
  /* Inherit the constructors */
  using UcxHoloscanComponentSerializer::UcxHoloscanComponentSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxHoloscanComponentSerializer(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<holoscan::Allocator> allocator = nullptr,
      const std::string& name = "ucx_holoscan_component_serializer") {
    if (allocator) {
      this->add_arg(Arg{"allocator", allocator});
    }
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_component_serializers(py::module_& m) {
  py::class_<StdComponentSerializer,
             PyStdComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<StdComponentSerializer>>(
      m, "StdComponentSerializer", doc::StdComponentSerializer::doc_StdComponentSerializer)
      .def(py::init<std::variant<Fragment*, Subgraph*>, const std::string&>(),
           "fragment"_a,
           "name"_a = "standard_component_serializer"s,
           doc::StdComponentSerializer::doc_StdComponentSerializer)
      .def("initialize",
           &StdComponentSerializer::initialize,
           doc::StdComponentSerializer::doc_initialize);

  py::class_<UcxComponentSerializer,
             PyUcxComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxComponentSerializer>>(
      m, "UcxComponentSerializer", doc::UcxComponentSerializer::doc_UcxComponentSerializer)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "name"_a = "ucx_component_serializer"s,
           doc::UcxComponentSerializer::doc_UcxComponentSerializer);

  py::class_<UcxHoloscanComponentSerializer,
             PyUcxHoloscanComponentSerializer,
             gxf::GXFResource,
             std::shared_ptr<UcxHoloscanComponentSerializer>>(
      m,
      "UcxHoloscanComponentSerializer",
      doc::UcxHoloscanComponentSerializer::doc_UcxHoloscanComponentSerializer)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "name"_a = "ucx_component_serializer"s,
           doc::UcxHoloscanComponentSerializer::doc_UcxHoloscanComponentSerializer_python);
}
}  // namespace holoscan
