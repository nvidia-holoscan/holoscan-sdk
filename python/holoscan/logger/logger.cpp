/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <memory>

#include <holoscan/logger/logger.hpp>
#include "logger_pydoc.hpp"

namespace py = pybind11;

namespace holoscan {

PYBIND11_MODULE(_logger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Logger Python Bindings
        -----------------------------------
        .. currentmodule:: _logger
    )pbdoc";

  py::enum_<LogLevel>(m, "LogLevel", doc::Logger::doc_LogLevel)
      .value("TRACE", LogLevel::TRACE)
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARN", LogLevel::WARN)
      .value("ERROR", LogLevel::ERROR)
      .value("CRITICAL", LogLevel::CRITICAL)
      .value("OFF", LogLevel::OFF);

  m.def("set_log_level", &set_log_level, doc::Logger::doc_set_log_level);
  m.def("log_level", &log_level, doc::Logger::doc_log_level);
  m.def("set_log_pattern", &set_log_pattern, doc::Logger::doc_set_log_pattern);
}  // PYBIND11_MODULE
}  // namespace holoscan
