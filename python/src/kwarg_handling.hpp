/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_PYTHON_PYBIND11_CORE_KWARG_HANDLING_HPP
#define HOLOSCAN_PYTHON_PYBIND11_CORE_KWARG_HANDLING_HPP

#include <pybind11/numpy.h>  // py::dtype
#include <pybind11/pybind11.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/parameter.hpp"

namespace py = pybind11;

namespace holoscan {

void set_scalar_arg_via_dtype(const py::object&, const py::dtype&, Arg&);

template <typename T>
void set_vector_arg_via_numpy_array(const py::array&, Arg&);

template <typename T>
void set_vector_arg_via_py_sequence(const py::sequence&, Arg&);

void set_vector_arg_via_iterable(const py::object&, Arg&);
Arg py_object_to_arg(py::object, std::string);
ArgList kwargs_to_arglist(const py::kwargs&);
py::object yaml_node_to_py_object(YAML::Node node);
py::object arg_to_py_object(Arg&);
py::dict arglist_to_kwargs(ArgList&);

}  // namespace holoscan

#endif /* HOLOSCAN_PYTHON_PYBIND11_CORE_KWARG_HANDLING_HPP */
