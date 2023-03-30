/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include <memory>

#include "holoscan/logger/logger.hpp"
#include "logger_pydoc.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

PYBIND11_MODULE(_logger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _logger
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<LogLevel>(m, "LogLevel", doc::Logger::doc_LogLevel)
      .value("TRACE", LogLevel::TRACE)
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARN", LogLevel::WARN)
      .value("ERROR", LogLevel::ERROR)
      .value("CRITICAL", LogLevel::CRITICAL)
      .value("OFF", LogLevel::OFF);

  m.def("load_env_log_level", &load_env_log_level, doc::Logger::doc_load_env_log_level);
  m.def("set_log_level", &set_log_level, doc::Logger::doc_set_log_level);
  m.def("log_level", &log_level, doc::Logger::doc_log_level);
  m.def("set_log_pattern", &set_log_pattern, doc::Logger::doc_set_log_pattern);
  m.def("enable_backtrace", &Logger::enable_backtrace, doc::Logger::doc_enable_backtrace);
  m.def("disable_backtrace", &Logger::disable_backtrace, doc::Logger::doc_disable_backtrace);
  m.def("dump_backtrace", &Logger::dump_backtrace, doc::Logger::doc_enable_backtrace);
  m.def("should_backtrace", &Logger::should_backtrace, doc::Logger::doc_enable_backtrace);
  m.def("flush", &Logger::flush, doc::Logger::doc_flush);
  m.def("flush_level", &Logger::flush_level, doc::Logger::doc_flush_level);
  m.def("flush_on", &Logger::flush_on, doc::Logger::doc_flush_on);
}  // PYBIND11_MODULE
}  // namespace holoscan
