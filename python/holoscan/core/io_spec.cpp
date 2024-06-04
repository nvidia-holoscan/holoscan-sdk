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

#include "io_spec.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "holoscan/core/condition.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "io_spec_pydoc.hpp"
#include "kwarg_handling.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void init_io_spec(py::module_& m) {
  py::class_<IOSpec, std::shared_ptr<IOSpec>> iospec(
      m, "IOSpec", R"doc(I/O specification class.)doc");

  py::enum_<IOSpec::IOType>(iospec, "IOType", doc::IOType::doc_IOType)
      .value("INPUT", IOSpec::IOType::kInput)
      .value("OUTPUT", IOSpec::IOType::kOutput);

  py::enum_<IOSpec::ConnectorType>(iospec, "ConnectorType", doc::ConnectorType::doc_ConnectorType)
      .value("DEFAULT", IOSpec::ConnectorType::kDefault)
      .value("DOUBLE_BUFFER", IOSpec::ConnectorType::kDoubleBuffer)
      .value("UCX", IOSpec::ConnectorType::kUCX);

  iospec
      .def(py::init<OperatorSpec*, const std::string&, IOSpec::IOType>(),
           "op_spec"_a,
           "name"_a,
           "io_type"_a,
           doc::IOSpec::doc_IOSpec)
      .def_property_readonly("name", &IOSpec::name, doc::IOSpec::doc_name)
      .def_property_readonly("io_type", &IOSpec::io_type, doc::IOSpec::doc_io_type)
      .def_property_readonly(
          "connector_type", &IOSpec::connector_type, doc::IOSpec::doc_connector_type)
      .def_property_readonly("conditions", &IOSpec::conditions, doc::IOSpec::doc_conditions)
      .def(
          "condition",
          [](IOSpec& io_spec, const ConditionType& kind, const py::kwargs& kwargs) {
            return io_spec.condition(kind, kwargs_to_arglist(kwargs));
          },
          doc::IOSpec::doc_condition)
      // TODO: sphinx API doc build complains if more than one connector
      //       method has a docstring specified. For now just set the docstring for the
      //       first overload only and add information about the rest in the Notes section.
      .def(
          "connector",
          [](IOSpec& io_spec, const IOSpec::ConnectorType& kind, const py::kwargs& kwargs) {
            return io_spec.connector(kind, kwargs_to_arglist(kwargs));
          },
          doc::IOSpec::doc_connector)
      // using lambdas for overloaded connector methods because py::overload_cast didn't work
      .def("connector", [](IOSpec& io_spec) { return io_spec.connector(); })
      .def("connector",
           [](IOSpec& io_spec, std::shared_ptr<Resource> connector) {
             return io_spec.connector(connector);
           })
      .def("__repr__",
           // use py::object and obj.cast to avoid a segfault if object has not been initialized
           [](const IOSpec& iospec) { return iospec.description(); });
}
}  // namespace holoscan
