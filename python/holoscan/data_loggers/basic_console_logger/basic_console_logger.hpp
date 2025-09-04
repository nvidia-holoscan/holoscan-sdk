/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP
#define PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "../../core/gil_guarded_pyobject.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/gxf_console_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

namespace py = pybind11;

namespace holoscan::data_loggers {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the resource.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the resource's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_resource<ResourceT>
 */

class PySimpleTextSerializer : public SimpleTextSerializer {
 public:
  /* Inherit the constructors */
  using SimpleTextSerializer::SimpleTextSerializer;

  // Define a constructor that fully initializes the object.
  PySimpleTextSerializer(Fragment* fragment, int64_t max_elements = 10,
                         int64_t max_metadata_items = 10, bool log_video_buffer_content = false,
                         bool log_python_object_contents = true,
                         const std::string& name = "simple_text_serializer");

  void initialize() override;
  void setup(ComponentSpec& spec) override;

 private:
  Parameter<bool> log_python_object_contents_;
  void register_gil_guarded_pyobject_encoder();
};

class PyBasicConsoleLogger : public BasicConsoleLogger {
 public:
  /* Inherit the constructors */
  using BasicConsoleLogger::BasicConsoleLogger;

  // Define a constructor that fully initializes the object.
  PyBasicConsoleLogger(Fragment* fragment,
                       std::shared_ptr<SimpleTextSerializer> serializer = nullptr,
                       bool log_inputs = true, bool log_outputs = true, bool log_metadata = true,
                       bool log_tensor_data_content = true, bool use_scheduler_clock = true,
                       std::optional<std::shared_ptr<Resource>> clock = std::nullopt,
                       const std::vector<std::string>& allowlist_patterns = {},
                       const std::vector<std::string>& denylist_patterns = {},
                       const std::string& name = "basic_console_logger");

  void initialize() override;
};

class PyGXFConsoleLogger : public GXFConsoleLogger {
 public:
  /* Inherit the constructors */
  using GXFConsoleLogger::GXFConsoleLogger;

  // Define a constructor that fully initializes the object.
  PyGXFConsoleLogger(Fragment* fragment, std::shared_ptr<SimpleTextSerializer> serializer = nullptr,
                     bool log_inputs = true, bool log_outputs = true, bool log_metadata = true,
                     bool log_tensor_data_content = true, bool use_scheduler_clock = true,
                     std::optional<std::shared_ptr<Resource>> clock = std::nullopt,
                     const std::vector<std::string>& allowlist_patterns = {},
                     const std::vector<std::string>& denylist_patterns = {},
                     const std::string& name = "gxf_basic_console_logger");

  void initialize() override;
};

}  // namespace holoscan::data_loggers

#endif /* PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP */
