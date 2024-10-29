/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/numpy.h>  // for py::dtype
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::optional, std::variant

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyPingTensorTxOp : public holoscan::ops::PingTensorTxOp {
 public:
  /* Inherit the constructors */
  using PingTensorTxOp::PingTensorTxOp;

  // Define a constructor that fully initializes the object.
  PyPingTensorTxOp(Fragment* fragment, const py::args& args,
                   std::optional<std::shared_ptr<::holoscan::Allocator>> allocator = std::nullopt,
                   const std::string& storage_type = "system"s,
                   std::optional<int32_t> batch_size = std::nullopt, int32_t rows = 32,
                   std::optional<int32_t> columns = 64,
                   std::optional<int32_t> channels = std::nullopt,
                   const std::variant<std::string, py::dtype>& dtype = "uint8_t",
                   const std::string& tensor_name = "tensor",
                   const std::string& name = "ping_tensor_tx")
      : PingTensorTxOp(ArgList{Arg{"storage_type", storage_type},
                               Arg{"rows", rows},
                               Arg{"tensor_name", tensor_name}}) {
    add_positional_condition_and_resource_args(this, args);
    if (allocator.has_value()) { this->add_arg(Arg{"allocator", allocator.value()}); }
    if (batch_size.has_value()) { this->add_arg(Arg{"batch_size", batch_size.value()}); }
    if (batch_size.has_value()) { this->add_arg(Arg{"columns", columns.value()}); }
    if (batch_size.has_value()) { this->add_arg(Arg{"channels", channels.value()}); }
    if (std::holds_alternative<std::string>(dtype)) {
      this->add_arg(Arg("data_type", std::get<std::string>(dtype)));
    } else {
      auto dt = std::get<py::dtype>(dtype);
      std::string data_type;
      auto dtype_name = dt.attr("name").cast<std::string>();
      if (dtype_name == "float16" || dtype_name == "float32") {
        // currently promoting float16 scalars to float
        data_type = "float";
      } else if (dtype_name == "float64") {
        data_type = "double";
      } else if (dtype_name == "int8") {
        data_type = "int8_t";
      } else if (dtype_name == "int16") {
        data_type = "int16_t";
      } else if (dtype_name == "int32") {
        data_type = "int32_t";
      } else if (dtype_name == "int64") {
        data_type = "int64_t";
      } else if (dtype_name == "bool" || dtype_name == "uint8") {
        data_type = "uint8_t";
      } else if (dtype_name == "uint16") {
        data_type = "uint16_t";
      } else if (dtype_name == "uint32") {
        data_type = "uint32_t";
      } else if (dtype_name == "uint64") {
        data_type = "uint64_t";
      } else if (dtype_name == "complex64") {
        data_type = "complex<float>";
      } else if (dtype_name == "complex128") {
        data_type = "complex<double>";
      } else {
        throw std::runtime_error(fmt::format("unsupported numpy dtype with name: {}", dtype_name));
      }
      this->add_arg(Arg("data_type", data_type));
    }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_ping_tensor_tx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK PingTensorTxOp Python Bindings
        --------------------------------------------------
        .. currentmodule:: _ping_tensor_tx
    )pbdoc";

  py::class_<PingTensorTxOp, PyPingTensorTxOp, Operator, std::shared_ptr<PingTensorTxOp>>(
      m, "PingTensorTxOp", doc::PingTensorTxOp::doc_PingTensorTxOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::optional<std::shared_ptr<::holoscan::Allocator>>,
                    const std::string&,
                    std::optional<int32_t>,
                    int32_t,
                    std::optional<int32_t>,
                    std::optional<int32_t>,
                    const std::variant<std::string, py::dtype>,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = py::none(),
           "storage_type"_a = "system"s,
           "batch_size"_a = py::none(),
           "rows"_a = 32,
           "columns"_a = 64,
           "channels"_a = py::none(),
           "dtype"_a = "uint8_t"s,
           "tensor_name"_a = "tensor"s,
           "name"_a = "video_stream_replayer"s,
           doc::PingTensorTxOp::doc_PingTensorTxOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
