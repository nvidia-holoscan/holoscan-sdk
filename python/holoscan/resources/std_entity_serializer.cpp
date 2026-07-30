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
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_resource.hpp>
#include <holoscan/core/resources/gxf/std_entity_serializer.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./std_entity_serializer_pydoc.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyStdEntitySerializer : public StdEntitySerializer {
 public:
  /* Inherit the constructors */
  using StdEntitySerializer::StdEntitySerializer;

  // Define a constructor that fully initializes the object.
  explicit PyStdEntitySerializer(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      const std::string& name = resource_default_name_v<StdEntitySerializer>) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_std_entity_serializer(py::module_& m) {
  py::class_<StdEntitySerializer,
             PyStdEntitySerializer,
             gxf::GXFResource,
             std::shared_ptr<StdEntitySerializer>>(
      m, "StdEntitySerializer", doc::StdEntitySerializer::doc_StdEntitySerializer)
      .def(py::init<std::variant<Fragment*, Subgraph*>, const std::string&>(),
           "fragment"_a,
           "name"_a = std::string(resource_default_name_v<StdEntitySerializer>),
           doc::StdEntitySerializer::doc_StdEntitySerializer);
}
}  // namespace holoscan
