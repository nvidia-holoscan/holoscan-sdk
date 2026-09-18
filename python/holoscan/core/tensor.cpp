/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include <dlpack/dlpack.h>

#include <holoscan/core/domain/tensor.hpp>

#include "dl_converter.hpp"
#include "enum_util.hpp"
#include "tensor_pydoc.hpp"

namespace holoscan {

namespace {

std::string dldevicetype_name(DLDeviceType device_type) {
  return python::cpp_enum_value_name(device_type);
}

std::string dldatatypecode_name(uint8_t code) {
  return python::enum_value_name(static_cast<DLDataTypeCode>(code));
}

// Parse an optional Python 2-tuple of ints (used for __dlpack__'s max_version / dl_device kwargs).
std::optional<std::tuple<int, int>> parse_int_pair(const nb::object& obj,
                                                   const char* error_message) {
  if (obj.is_none()) {
    return std::nullopt;
  }
  if (!nb::isinstance<nb::tuple>(obj)) {
    throw nb::value_error(error_message);
  }
  auto pair = nb::cast<nb::tuple>(obj);
  if (pair.size() != 2) {
    throw nb::value_error(error_message);
  }
  return std::make_tuple(nb::cast<int>(pair[0]), nb::cast<int>(pair[1]));
}

}  // namespace

void init_tensor(nb::module_& m) {
  // is_arithmetic so the members convert to / compare with the underlying int (these are DLPack
  // integer codes), matching how they're used at the DLPack boundary.
  python::export_enum<DLDeviceType>(m, "DLDeviceType", nb::is_arithmetic(), doc::DLDeviceType::doc);

  python::export_enum<DLDataTypeCode>(
      m, "DLDataTypeCode", nb::is_arithmetic(), doc::DLDataTypeCode::doc);

  nb::class_<DLDevice>(m, "DLDevice", doc::DLDevice::doc)
      .def(
          "__init__",
          [](DLDevice* self, DLDeviceType device_type, int32_t device_id) {
            new (self) DLDevice{device_type, device_id};
          },
          nb::arg("device_type"),
          nb::arg("device_id"))
      .def_ro("device_type", &DLDevice::device_type, doc::DLDevice::device_type)
      .def_ro("device_id", &DLDevice::device_id, doc::DLDevice::device_id)
      .def("__repr__", [](const DLDevice& device) {
        return std::string("DLDevice(device_type=") + dldevicetype_name(device.device_type) +
               ", device_id=" + std::to_string(device.device_id) + ")";
      });

  nb::class_<DLDataType>(m, "DLDataType", doc::DLDataType::doc)
      .def(
          "__init__",
          [](DLDataType* self, uint8_t code, uint8_t bits, uint16_t lanes) {
            new (self) DLDataType{code, bits, lanes};
          },
          nb::arg("code"),
          nb::arg("bits"),
          nb::arg("lanes"))
      .def_rw("code", &DLDataType::code, doc::DLDataType::code)
      .def_rw("bits", &DLDataType::bits, doc::DLDataType::bits)
      .def_rw("lanes", &DLDataType::lanes, doc::DLDataType::lanes)
      .def("__repr__", [](const DLDataType& dtype) {
        return std::string("DLDataType(code=") + dldatatypecode_name(dtype.code) +
               ", bits=" + std::to_string(dtype.bits) + ", lanes=" + std::to_string(dtype.lanes) +
               ")";
      });

  auto tensor_class = nb::class_<Tensor>(m, "Tensor", doc::Tensor::doc);
  tensor_class.def_prop_ro("ndim", &Tensor::ndim, doc::Tensor::ndim)
      .def_prop_ro(
          "shape",
          [](const Tensor& tensor) { return tensor_shape_tuple(tensor); },
          doc::Tensor::shape)
      .def_prop_ro(
          "strides",
          [](const Tensor& tensor) { return tensor_strides_tuple(tensor); },
          doc::Tensor::strides)
      .def_prop_ro("size", &Tensor::size, doc::Tensor::size)
      .def_prop_ro(
          "itemsize",
          [](const Tensor& tensor) { return static_cast<int>(tensor.itemsize()); },
          doc::Tensor::itemsize)
      .def_prop_ro("nbytes", &Tensor::nbytes, doc::Tensor::nbytes)
      .def_prop_ro(
          "byte_offset",
          [](const Tensor& tensor) { return tensor.byte_offset(); },
          doc::Tensor::byte_offset)
      .def_prop_ro("device", &Tensor::device, doc::Tensor::device)
      .def_prop_ro("dtype", &Tensor::dtype, doc::Tensor::dtype)
      .def_prop_ro(
          "data",
          [](const Tensor& tensor) { return reinterpret_cast<uintptr_t>(tensor.data()); },
          doc::Tensor::data)
      .def("is_contiguous", &Tensor::is_contiguous, doc::Tensor::is_contiguous)
      .def_prop_ro("is_read_only", &Tensor::is_read_only, doc::Tensor::is_read_only)
      // Array-interface export (host -> __array_interface__, device -> __cuda_array_interface__).
      // Each getter raises AttributeError when the device doesn't match, so numpy/cupy duck-typing
      // (hasattr) resolves to the right protocol.
      .def_prop_ro("__array_interface__", &array_interface, doc::Tensor::array_interface)
      .def_prop_ro(
          "__cuda_array_interface__", &cuda_array_interface, doc::Tensor::cuda_array_interface)
      // DLPack export protocol.
      .def(
          "__dlpack__",
          [](Tensor& self,
             const nb::object& stream,
             const nb::object& max_version,
             const nb::object& dl_device,
             const nb::object& copy) {
            auto cpp_max_version =
                parse_int_pair(max_version, "max_version must be a tuple of (major, minor)");
            auto cpp_dl_device =
                parse_int_pair(dl_device, "dl_device must be a tuple of (device_type, device_id)");
            std::optional<bool> cpp_copy;
            if (!copy.is_none()) {
              cpp_copy = nb::cast<bool>(copy);
            }
            return py_dlpack(&self, stream, cpp_max_version, cpp_dl_device, cpp_copy);
          },
          nb::kw_only(),
          nb::arg("stream") = nb::none(),
          nb::arg("max_version") = nb::none(),
          nb::arg("dl_device") = nb::none(),
          nb::arg("copy") = nb::none(),
          doc::Tensor::dlpack)
      .def(
          "__dlpack_device__",
          [](Tensor& self) { return py_dlpack_device(&self); },
          doc::Tensor::dlpack_device)
      // DLPack import (GIL-safe ownership; see dl_converter.cpp). as_tensor accepts DLPack or an
      // array interface (DLPack preferred); from_dlpack is DLPack-only.
      .def_static("as_tensor", &as_tensor, nb::arg("obj"), doc::Tensor::as_tensor)
      .def_static("from_dlpack", &tensor_from_dlpack, nb::arg("obj"), doc::Tensor::from_dlpack);

  // DLPack 1.3 fast C exchange API: publish the function-pointer table on the Tensor type.
  tensor_class.attr("__dlpack_c_exchange_api__") = dlpack_c_exchange_api_capsule();
  m.def("_test_tensor_metadata_projection",
        &test_tensor_metadata_projection,
        nb::arg("backing"),
        nb::arg("shape"),
        nb::arg("strides").none(),
        nb::arg("dtype"),
        nb::arg("byte_offset"),
        "Test-only helper that projects synthetic metadata over a live Tensor backing.");
  m.def("_test_dlpack_c_exchange_api",
        &test_dlpack_c_exchange_api,
        nb::arg("tensor"),
        "Test-only helper that directly exercises the DLPack C exchange API.");

  m.def("is_tensor_like", &is_tensor_like, nb::arg("obj"), doc::is_tensor_like);
}

}  // namespace holoscan
