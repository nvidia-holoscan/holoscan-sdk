/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYBIND11_CORE_ARG_HPP
#define PYBIND11_CORE_ARG_HPP

#include <pybind11/pybind11.h>

#include <unordered_map>

#include "holoscan/core/arg.hpp"

namespace py = pybind11;

namespace holoscan {

void init_arg(py::module_&);

static const std::unordered_map<ArgElementType, const char*> element_type_namemap{
    {ArgElementType::kCustom, "CUSTOM"},
    {ArgElementType::kBoolean, "BOOLEAN"},
    {ArgElementType::kInt8, "INT8"},
    {ArgElementType::kUnsigned8, "UNSIGNED8"},
    {ArgElementType::kInt16, "INT16"},
    {ArgElementType::kUnsigned16, "UNSIGNED16"},
    {ArgElementType::kInt32, "INT32"},
    {ArgElementType::kUnsigned32, "UNSIGNED32"},
    {ArgElementType::kInt64, "INT64"},
    {ArgElementType::kUnsigned64, "UNSIGNED64"},
    {ArgElementType::kFloat32, "FLOAT32"},
    {ArgElementType::kFloat64, "FLOAT64"},
    {ArgElementType::kString, "STRING"},
    {ArgElementType::kHandle, "HANDLE"},
    {ArgElementType::kYAMLNode, "YAML_NODE"},
    {ArgElementType::kIOSpec, "IO_SPEC"},
    {ArgElementType::kCondition, "CONDITION"},
    {ArgElementType::kResource, "RESOURCE"},
};

static const std::unordered_map<ArgContainerType, const char*> container_type_namemap{
    {ArgContainerType::kNative, "NATIVE"},
    {ArgContainerType::kVector, "VECTOR"},
    {ArgContainerType::kArray, "ARRAY"},
};

}  // namespace holoscan

#endif /* PYBIND11_CORE_ARG_HPP */
