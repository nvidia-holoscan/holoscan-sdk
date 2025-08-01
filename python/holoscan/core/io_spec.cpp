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

#include "io_spec.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/condition.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "io_spec_pydoc.hpp"
#include "kwarg_handling.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

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
      .value("ASYNC_BUFFER", IOSpec::ConnectorType::kAsyncBuffer)
      .value("UCX", IOSpec::ConnectorType::kUCX);

  py::enum_<IOSpec::QueuePolicy>(iospec, "QueuePolicy", doc::QueuePolicy::doc_QueuePolicy)
      .value("POP", IOSpec::QueuePolicy::kPop)
      .value("REJECT", IOSpec::QueuePolicy::kReject)
      .value("FAULT", IOSpec::QueuePolicy::kFault);

  py::class_<IOSpec::IOSize, std::shared_ptr<IOSpec::IOSize>>(
      iospec, "IOSize", doc::IOSpec::IOSize::doc_IOSize)
      .def(py::init<int64_t>(), "size"_a)
      .def_property("size",
                    py::overload_cast<>(&IOSpec::IOSize::size, py::const_),
                    py::overload_cast<int64_t>(&IOSpec::IOSize::size),
                    doc::IOSpec::IOSize::doc_size)
      // Define int conversion from IOSize
      .def("__int__",
           [](const IOSpec::IOSize& io_size) { return static_cast<int>(io_size.size()); })
      .def("__repr__",
           [](const IOSpec::IOSize& io_size) { return fmt::format("IOSize({})", io_size.size()); });

  iospec
      .def(py::init<OperatorSpec*, const std::string&, IOSpec::IOType>(),
           "op_spec"_a,
           "name"_a,
           "io_type"_a,
           doc::IOSpec::doc_IOSpec)
      .def_property_readonly(
          "name", &IOSpec::name, doc::IOSpec::doc_name, py::return_value_policy::reference_internal)
      .def_property_readonly("io_type", &IOSpec::io_type, doc::IOSpec::doc_io_type)
      .def_property_readonly(
          "connector_type", &IOSpec::connector_type, doc::IOSpec::doc_connector_type)
      .def_property_readonly("conditions",
                             &IOSpec::conditions,
                             doc::IOSpec::doc_conditions,
                             py::return_value_policy::reference_internal)
      .def(
          "condition",
          // Note: The return type needs to be specified explicitly because pybind11 can't deduce it
          [](IOSpec& io_spec, const ConditionType& kind, const py::kwargs& kwargs) -> IOSpec& {
            return io_spec.condition(kind, kwargs_to_arglist(kwargs));
          },
          doc::IOSpec::doc_condition,
          py::return_value_policy::reference_internal)
      // TODO(unknown): sphinx API doc build complains if more than one connector
      // method has a docstring specified. For now just set the docstring for the
      // first overload only and add information about the rest in the Notes section.
      .def(
          "connector",
          // Note: The return type needs to be specified explicitly because pybind11 can't deduce it
          [](IOSpec& io_spec, const IOSpec::ConnectorType& kind, const py::kwargs& kwargs)
              -> IOSpec& { return io_spec.connector(kind, kwargs_to_arglist(kwargs)); },
          doc::IOSpec::doc_connector,
          py::return_value_policy::reference_internal)
      // using lambdas for overloaded connector methods because py::overload_cast didn't work
      .def("connector", [](IOSpec& io_spec) { return io_spec.connector(); })
      .def("connector",
           [](IOSpec& io_spec, std::shared_ptr<Resource> connector) {
             return io_spec.connector(std::move(connector));
           })
      .def_property("queue_size",
                    py::overload_cast<>(&IOSpec::queue_size, py::const_),
                    py::overload_cast<int64_t>(&IOSpec::queue_size),
                    doc::IOSpec::doc_queue_size)
      .def_property("queue_policy",
                    py::overload_cast<>(&IOSpec::queue_policy, py::const_),
                    py::overload_cast<IOSpec::QueuePolicy>(&IOSpec::queue_policy),
                    doc::IOSpec::doc_queue_policy)
      .def("__repr__",
           // use py::object and obj.cast to avoid a segfault if object has not been initialized
           [](const IOSpec& iospec) { return iospec.description(); });

  // Define IOSize constants in IOSpec module
  iospec
      .def_property_readonly_static(
          "ANY_SIZE", [](const py::object&) { return IOSpec::kAnySize; }, "Any size")
      .def_property_readonly_static(
          "PRECEDING_COUNT",
          [](const py::object&) { return IOSpec::kPrecedingCount; },
          "Number of preceding connections")
      .def_property_readonly_static(
          "SIZE_ONE", [](const py::object&) { return IOSpec::kSizeOne; }, "Size one");
}
}  // namespace holoscan
