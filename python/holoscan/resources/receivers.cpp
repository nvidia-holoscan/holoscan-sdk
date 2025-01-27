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

#include <cstdint>
#include <memory>
#include <string>

#include "./receivers_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/receiver.hpp"
#include "holoscan/core/resources/gxf/ucx_receiver.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyDoubleBufferReceiver : public DoubleBufferReceiver {
 public:
  /* Inherit the constructors */
  using DoubleBufferReceiver::DoubleBufferReceiver;

  // Define a constructor that fully initializes the object.
  explicit PyDoubleBufferReceiver(Fragment* fragment, uint64_t capacity = 1UL,
                                  uint64_t policy = 2UL,
                                  const std::string& name = "double_buffer_receiver")
      : DoubleBufferReceiver(ArgList{Arg{"capacity", capacity}, Arg{"policy", policy}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

class PyUcxReceiver : public UcxReceiver {
 public:
  /* Inherit the constructors */
  using UcxReceiver::UcxReceiver;

  // Define a constructor that fully initializes the object.
  explicit PyUcxReceiver(Fragment* fragment,
                         std::shared_ptr<UcxSerializationBuffer> buffer = nullptr,
                         uint64_t capacity = 1UL, uint64_t policy = 2UL,
                         const std::string& address = std::string("0.0.0.0"),
                         uint32_t port = kDefaultUcxPort, const std::string& name = "ucx_receiver")
      : UcxReceiver(ArgList{Arg{"capacity", capacity},
                            Arg{"policy", policy},
                            Arg{"address", address},
                            Arg{"port", port}}) {
    if (buffer) { this->add_arg(Arg{"buffer", buffer}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_receivers(py::module_& m) {
  py::class_<Receiver, gxf::GXFResource, std::shared_ptr<Receiver>>(
      m, "Receiver", doc::Receiver::doc_Receiver)
      .def(py::init<>(), doc::Receiver::doc_Receiver)
      .def_property_readonly("capacity", &Receiver::capacity, doc::Receiver::doc_capacity)
      .def_property_readonly("size", &Receiver::size, doc::Receiver::doc_size)
      .def_property_readonly("back_size", &Receiver::back_size, doc::Receiver::doc_back_size);

  py::class_<DoubleBufferReceiver,
             PyDoubleBufferReceiver,
             Receiver,
             std::shared_ptr<DoubleBufferReceiver>>(
      m, "DoubleBufferReceiver", doc::DoubleBufferReceiver::doc_DoubleBufferReceiver)
      .def(py::init<Fragment*, uint64_t, uint64_t, const std::string&>(),
           "fragment"_a,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "name"_a = "double_buffer_receiver"s,
           doc::DoubleBufferReceiver::doc_DoubleBufferReceiver);

  py::class_<UcxReceiver, PyUcxReceiver, Receiver, std::shared_ptr<UcxReceiver>>(
      m, "UcxReceiver", doc::UcxReceiver::doc_UcxReceiver)
      .def(py::init<Fragment*,
                    std::shared_ptr<UcxSerializationBuffer>,
                    uint64_t,
                    uint64_t,
                    const std::string&,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "buffer"_a = nullptr,
           "capacity"_a = 1UL,
           "policy"_a = 2UL,
           "address"_a = std::string("0.0.0.0"),
           "port"_a = kDefaultUcxPort,
           "name"_a = "ucx_receiver"s,
           doc::UcxReceiver::doc_UcxReceiver);
}
}  // namespace holoscan
