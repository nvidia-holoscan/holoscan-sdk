/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/cli_parser.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "CLI/Config.hpp"
#include "CLI/Formatter.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

void CLIParser::initialize(std::string app_description, std::string app_version) {
  // Set the application description and version.
  app_.description(std::move(app_description));
  app_.set_version_flag("--version", app_version, "Show the version of the application.");

  if (!is_initialized_) {
    app_.add_flag("--driver",
                  options_.run_driver,
                  "Run the App Driver on the current machine. Can be used together with the "
                  "'--worker' option "
                  "to run both the App Driver and the App Worker on the same machine.");
    app_.add_flag("--worker", options_.run_worker, "Run the App Worker.");
    app_.add_option(
        "--address",
        options_.driver_address,
        "Address ('[<IP or hostname>][:<port>]') of the App Driver. If not specified, "
        "the App Driver uses the default host address ('0.0.0.0') with the default port "
        "number ('8765').");
    app_.add_option(
        "--worker-address",
        options_.worker_address,
        "The address (`[<IP or hostname>][:<port>]`) of the App Worker. If not specified, the App "
        "Worker uses the default host address ('0.0.0.0') with a randomly chosen port number "
        "between 10000 and 32767 that is not currently in use.");
    app_.add_option(
            "--fragments",
            options_.worker_targets,
            "Comma-separated names of the fragments to be executed by the App Worker. If not "
            "specified, only one fragment (selected by the App Driver) will be executed. 'all' "
            "can be used to run all the fragments.")
        ->delimiter(',')
        ->allow_extra_args(false);
    app_.add_option(
        "--config",
        options_.config_path,
        "Path to the configuration file. This will override the configuration file path "
        "configured in the application code (before run() is called).");

    // Allow extra arguments (ie, don't throw an error)
    app_.allow_extras(true);

    is_initialized_ = true;
  }
  has_error_ = false;
}

std::vector<std::string>& CLIParser::parse(std::vector<std::string>& argv) {
  try {
    if (!argv.empty()) {
      // Set the application name
      std::string app_name;
      if (argv.size() > 1 && argv[1].find(".py") != std::string::npos) {
        // For Python applications argv[0] is the Python binary and argv[1] is the .py file name
        app_name = argv[1];
      } else {
        app_name = argv[0];
      }
      // retain only the binary name without the full path
      try {
        app_name = std::filesystem::path(app_name).filename();
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN(
            "Unable to extract filename from app_name: {}. Exception: {}", app_name, e.what());
      }
      app_.name(app_name);
    }

    // CLI::App::parse() accepts the arguments in the reverse order
    std::reverse(argv.begin(), argv.end());
    app_.parse(argv);
  } catch (const CLI::CallForHelp& e) {
    // In case of -h or --help, print the help message and exit
    // The following code is explicit statements of `std::exit(app_.exit(e));`
    std::cout << app_.help("", CLI::AppFormatMode::All);
    std::exit(0);
  } catch (const CLI::CallForVersion& v) {
    // In case of --version, print the version and exit
    // The following code is explicit statements of `std::exit(app_.exit(v));`
    std::cout << v.what() << "\n";
    std::exit(0);
  } catch (const CLI::ExtrasError& e) {
    // It won't reach here since `allow_extras` is set to `true`
    // (`app_.allow_extras(true);`) in the `initialize()` function
  } catch (const CLI::ParseError& e) {
    // Print the error message and set the error flag
    HOLOSCAN_LOG_ERROR("{}", e.what());
    has_error_ = true;
  }
  if (app_.get_allow_extras()) {
    // Store the remaining arguments
    argv = app_.remaining();
  }
  return argv;
}

bool CLIParser::CLIParser::has_error() const {
  return has_error_;
}

CLIOptions& CLIParser::options() {
  return options_;
}

}  // namespace holoscan
