/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "messagelabel.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/messagelabel.hpp>
#include "messagelabel_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_messagelabel(py::module_& m) {
  // Bind OperatorTimestampLabel struct
  py::class_<OperatorTimestampLabel>(
      m, "OperatorTimestampLabel", doc::OperatorTimestampLabel::doc_OperatorTimestampLabel)
      .def(py::init<>(), doc::OperatorTimestampLabel::doc_OperatorTimestampLabel_default)
      .def(py::init<const std::string&>(),
           "op_name"_a,
           doc::OperatorTimestampLabel::doc_OperatorTimestampLabel_name)
      .def(py::init<const std::string&, int64_t, int64_t>(),
           "op_name"_a,
           "rec_timestamp"_a,
           "pub_timestamp"_a,
           doc::OperatorTimestampLabel::doc_OperatorTimestampLabel_full)
      .def_readwrite("operator_name",
                     &OperatorTimestampLabel::operator_name,
                     doc::OperatorTimestampLabel::doc_operator_name)
      .def_readwrite("rec_timestamp",
                     &OperatorTimestampLabel::rec_timestamp,
                     doc::OperatorTimestampLabel::doc_rec_timestamp)
      .def_readwrite("pub_timestamp",
                     &OperatorTimestampLabel::pub_timestamp,
                     doc::OperatorTimestampLabel::doc_pub_timestamp)
      .def("set_pub_timestamp_to_current",
           &OperatorTimestampLabel::set_pub_timestamp_to_current,
           doc::OperatorTimestampLabel::doc_set_pub_timestamp_to_current)
      .def("__repr__", [](const OperatorTimestampLabel& label) {
        return "<OperatorTimestampLabel: op='" + label.operator_name +
               "', rec=" + std::to_string(label.rec_timestamp) +
               ", pub=" + std::to_string(label.pub_timestamp) + ">";
      });

  // Bind MessageLabel class
  py::class_<MessageLabel>(m, "MessageLabel", doc::MessageLabel::doc_MessageLabel)
      .def(py::init<>(), doc::MessageLabel::doc_MessageLabel_default)
      .def(py::init<const std::vector<MessageLabel::TimestampedPath>&>(),
           "paths"_a,
           doc::MessageLabel::doc_MessageLabel_paths)
      .def("num_paths", &MessageLabel::num_paths, doc::MessageLabel::doc_num_paths)
      .def("get_all_path_names",
           &MessageLabel::get_all_path_names,
           doc::MessageLabel::doc_get_all_path_names)
      .def_property_readonly(
          "paths",
          static_cast<const std::vector<MessageLabel::TimestampedPath>& (MessageLabel::*)() const>(
              &MessageLabel::paths),
          doc::MessageLabel::doc_paths)
      .def("get_e2e_latency",
           &MessageLabel::get_e2e_latency,
           "index"_a,
           doc::MessageLabel::doc_get_e2e_latency)
      .def("get_e2e_latency_ms",
           &MessageLabel::get_e2e_latency_ms,
           "index"_a,
           doc::MessageLabel::doc_get_e2e_latency_ms)
      .def("get_path", &MessageLabel::get_path, "index"_a, doc::MessageLabel::doc_get_path)
      .def("get_path_name",
           &MessageLabel::get_path_name,
           "index"_a,
           doc::MessageLabel::doc_get_path_name)
      .def("get_operator",
           &MessageLabel::get_operator,
           "path_index"_a,
           "op_index"_a,
           doc::MessageLabel::doc_get_operator,
           py::return_value_policy::reference_internal)
      .def("has_operator",
           &MessageLabel::has_operator,
           "op_name"_a,
           doc::MessageLabel::doc_has_operator)
      .def("to_string",
           static_cast<std::string (MessageLabel::*)() const>(&MessageLabel::to_string),
           doc::MessageLabel::doc_to_string)
      .def("print_all", &MessageLabel::print_all, doc::MessageLabel::doc_print_all)
      .def("__repr__",
           [](const MessageLabel& label) {
             return "<MessageLabel: " + std::to_string(label.num_paths()) + " paths>";
           })
      .def("__str__", static_cast<std::string (MessageLabel::*)() const>(&MessageLabel::to_string));
}

}  // namespace holoscan
