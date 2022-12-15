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

#ifndef PYBIND11_CORE_DL_CONVERTER_HPP
#define PYBIND11_CORE_DL_CONVERTER_HPP

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

// Forward declaration
class DLManagedTensorCtx;

/**
 * @brief Structure to hold the context of a DLManagedTensor.
 *
 * This structure is used to hold the context of a DLManagedTensor for array interface.
 */
struct ArrayInterfaceMemoryBuffer {
  py::object obj_ref;               ///< Reference to the Python object that owns the memory buffer.
  std::vector<int64_t> dl_shape;    ///< Shape of the DLManagedTensor.
  std::vector<int64_t> dl_strides;  ///< Strides of the DLManagedTensor.
};

/**
 * @brief Set the array interface object of a Python object.
 *
 * This method sets `__array_interface__` or `__cuda_array_interface__` attribute of a Python
 * object.
 *
 * @param obj The Python object to set the array interface object.
 * @param ctx The context of the DLManagedTensor.
 */
void set_array_interface(const py::object& obj, std::shared_ptr<DLManagedTensorCtx> ctx);

/**
 * @brief Set the dlpack interface object
 *
 * This method sets `__dlpack__` and `__dlpack_device__` attribute of a Python object.
 *
 * @param obj The Python object to set the dlpack interface object.
 * @param ctx The context of the DLManagedTensor.
 */
void set_dlpack_interface(const py::object& obj, std::shared_ptr<DLManagedTensorCtx> ctx);

/**
 * @brief Provide `__dlpack__` method
 *
 * @param tensor The tensor to provide the `__dlpack__` method.
 * @param stream The client stream to use for the `__dlpack__` method.
 * @return The PyCapsule object.
 */
py::capsule py_dlpack(Tensor* tensor, py::object stream);

/**
 * @brief Provide `__dlpack_device__` method
 *
 * @param tensor The tensor to provide the `__dlpack_device__` method.
 * @return The tuple of device type and device id.
 */
py::tuple py_dlpack_device(Tensor* tensor);

/**
 * @brief Convert std::vector to pybind11::tuple
 *
 * @tparam PT The type of the elements in the tuple.
 * @tparam T The type of the elements in the vector.
 * @param vec The vector to convert.
 * @return The Python tuple object.
 */
template <typename PT, typename T>
pybind11::tuple vector2pytuple(const std::vector<T>& vec) {
  py::tuple result(vec.size());
  int counter = 0;
  for (auto& item : vec) {
    PyTuple_SET_ITEM(result.ptr(),
                     counter++,
                     pybind11::reinterpret_steal<pybind11::object>(
                         pybind11::detail::make_caster<PT>::cast(
                             std::forward<PT>(item),
                             pybind11::return_value_policy::automatic_reference,
                             nullptr))
                         .release()
                         .ptr());
  }
  return result;
}

/**
 * @brief Convert an array to pybind11::tuple
 *
 * @tparam PT The type of the elements in the tuple.
 * @tparam T The type of the elements in the array.
 * @param vec The vector to convert.
 * @return The Python tuple object.
 */
template <typename PT, typename T>
pybind11::tuple array2pytuple(const T* arr, size_t length) {
  py::tuple result(length);
  for (int index = 0; index < length; ++index) {
    const auto& value = arr[index];
    PyTuple_SET_ITEM(result.ptr(),
                     index,
                     pybind11::reinterpret_steal<pybind11::object>(
                         pybind11::detail::make_caster<PT>::cast(
                             std::forward<PT>(value),
                             pybind11::return_value_policy::automatic_reference,
                             nullptr))
                         .release()
                         .ptr());
  }
  return result;
}

}  // namespace holoscan

#endif /* PYBIND11_CORE_DL_CONVERTER_HPP */
