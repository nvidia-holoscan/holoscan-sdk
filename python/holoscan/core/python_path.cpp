/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "python_path.hpp"

#include <nanobind/nanobind.h>

#include <cstddef>
#include <filesystem>
#include <string>

namespace holoscan {

std::filesystem::path python_filesystem_path(nb::handle value) {
  const nb::object path = nb::steal<nb::object>(PyOS_FSPath(value.ptr()));
  if (!path.is_valid()) {
    throw nb::python_error();
  }
  PyObject* encoded_path_object = nullptr;
  if (PyUnicode_FSConverter(path.ptr(), &encoded_path_object) == 0) {
    throw nb::python_error();
  }
  const nb::bytes encoded_path = nb::steal<nb::bytes>(encoded_path_object);
  char* path_bytes = nullptr;
  Py_ssize_t path_size = 0;
  if (PyBytes_AsStringAndSize(encoded_path.ptr(), &path_bytes, &path_size) != 0) {
    throw nb::python_error();
  }
  return std::filesystem::path{std::string{path_bytes, static_cast<std::size_t>(path_size)}};
}

}  // namespace holoscan
