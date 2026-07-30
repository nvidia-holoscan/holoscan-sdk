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
#include <holoscan/core/resources/gxf/serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/ucx_serialization_buffer.hpp>
#include <holoscan/core/subgraph.hpp>
#include "../core/component_util.hpp"
#include "./serialization_buffers_pydoc.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PySerializationBuffer : public SerializationBuffer {
 public:
  /* Inherit the constructors */
  using SerializationBuffer::SerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PySerializationBuffer(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<holoscan::Allocator> allocator = nullptr,
      size_t buffer_size = kDefaultSerializationBufferSize,
      const std::string& name = resource_default_name_v<SerializationBuffer>)
      : SerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) {
      this->add_arg(Arg{"allocator", allocator});
    }
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

class PyUcxSerializationBuffer : public UcxSerializationBuffer {
 public:
  /* Inherit the constructors */
  using UcxSerializationBuffer::UcxSerializationBuffer;

  // Define a constructor that fully initializes the object.
  explicit PyUcxSerializationBuffer(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<holoscan::Allocator> allocator = nullptr,
      size_t buffer_size = kDefaultSerializationBufferSize,
      const std::string& name = resource_default_name_v<UcxSerializationBuffer>)
      : UcxSerializationBuffer(ArgList{
            Arg{"buffer_size", buffer_size},
        }) {
    if (allocator) {
      this->add_arg(Arg{"allocator", allocator});
    }
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_serialization_buffers(py::module_& m) {
  py::class_<SerializationBuffer,
             PySerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<SerializationBuffer>>(
      m, "SerializationBuffer", doc::SerializationBuffer::doc_SerializationBuffer)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<holoscan::Allocator>,
                    size_t,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = std::string(resource_default_name_v<SerializationBuffer>),
           doc::SerializationBuffer::doc_SerializationBuffer);

  py::class_<UcxSerializationBuffer,
             PyUcxSerializationBuffer,
             gxf::GXFResource,
             std::shared_ptr<UcxSerializationBuffer>>(
      m, "UcxSerializationBuffer", doc::UcxSerializationBuffer::doc_UcxSerializationBuffer)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::shared_ptr<holoscan::Allocator>,
                    size_t,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "buffer_size"_a = kDefaultSerializationBufferSize,
           "name"_a = std::string(resource_default_name_v<UcxSerializationBuffer>),
           doc::UcxSerializationBuffer::doc_UcxSerializationBuffer);
}
}  // namespace holoscan
