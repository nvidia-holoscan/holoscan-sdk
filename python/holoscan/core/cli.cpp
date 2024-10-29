/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "cli_pydoc.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/cli_options.hpp"
#include "holoscan/core/executors/gxf/gxf_parameter_adaptor.hpp"
#include "kwarg_handling.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_cli(py::module_& m) {
  // CLIOptions data structure
  py::class_<CLIOptions>(m, "CLIOptions")
      .def(py::init<bool,
                    bool,
                    const std::string&,
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "run_driver"_a = false,
           "run_worker"_a = false,
           "driver_address"_a = "",
           "worker_address"_a = "",
           "worker_targets"_a = std::vector<std::string>(),
           "config_path"_a = std::string(),
           doc::CLIOptions::doc_CLIOptions)
      .def_readwrite("run_driver", &CLIOptions::run_driver, doc::CLIOptions::doc_run_driver)
      .def_readwrite("run_worker", &CLIOptions::run_worker, doc::CLIOptions::doc_run_worker)
      .def_readwrite(
          "driver_address", &CLIOptions::driver_address, doc::CLIOptions::doc_driver_address)
      .def_readwrite(
          "worker_address", &CLIOptions::worker_address, doc::CLIOptions::doc_worker_address)
      .def_readwrite(
          "worker_targets", &CLIOptions::worker_targets, doc::CLIOptions::doc_worker_targets)
      .def_readwrite("config_path", &CLIOptions::config_path, doc::CLIOptions::doc_config_path)
      .def("print", &CLIOptions::print, doc::CLIOptions::doc_print)
      .def("__repr__", [](const CLIOptions& options) {
        return fmt::format(
            "<holoscan.core.CLIOptions: run_driver:{} run_worker:{} driver_address:'{}' "
            "worker_address:'{}' worker_targets:{} config_path:'{}'>",
            options.run_driver ? "True" : "False",
            options.run_worker ? "True" : "False",
            options.driver_address,
            options.worker_address,
            fmt::join(options.worker_targets, ","),
            options.config_path);
      });
}

}  // namespace holoscan
