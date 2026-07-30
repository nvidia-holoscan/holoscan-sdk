/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_KWARG_HANDLING_HPP
#define PYHOLOSCAN_CORE_KWARG_HANDLING_HPP

#include <pybind11/numpy.h>  // py::array, py::dtype
#include <pybind11/pybind11.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/parameter.hpp>

namespace py = pybind11;

namespace holoscan {

void init_kwarg_handling(py::module_&);

void set_scalar_arg_via_dtype(const py::object&, const py::dtype&, Arg&);

template <typename T>
void set_vector_arg_via_numpy_array(const py::array&, Arg&);

template <typename T>
void set_vector_arg_via_py_sequence(const py::sequence&, Arg&);

void set_vector_arg_via_iterable(const py::object&, Arg&);
Arg py_object_to_arg(py::object, const std::string&);
ArgList kwargs_to_arglist(const py::kwargs&);
py::object yaml_node_to_py_object(const YAML::Node& node);
py::object arg_to_py_object(Arg&);
py::dict arglist_to_kwargs(ArgList&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_KWARG_HANDLING_HPP */
