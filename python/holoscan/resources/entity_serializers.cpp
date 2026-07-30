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
#include <holoscan/core/resources/gxf/ucx_entity_serializer.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./entity_serializers_pydoc.hpp"

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
      bool verbose_warning = false,
      const std::string& name = resource_default_name_v<UcxEntitySerializer>)
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
           "name"_a = std::string(resource_default_name_v<UcxEntitySerializer>),
           doc::UcxEntitySerializer::doc_UcxEntitySerializer);
}
}  // namespace holoscan
