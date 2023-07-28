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

#ifndef HOLOSCAN_CORE_CLI_PARSER_HPP
#define HOLOSCAN_CORE_CLI_PARSER_HPP

#include <string>
#include <vector>

#include "CLI/App.hpp"

#include "cli_options.hpp"

namespace holoscan {

/**
 * @brief CLI Parser class.
 *
 * This class is used to parse the command line arguments. It uses CLI11 library internally.
 *
 * The application binary (in C++) or the Python script (in Python) can be executed with various
 * command-line options to run the App Driver and/or the App Worker:
 *
 * - `--driver`: Run the App Driver on the current machine. Can be used together with the `--worker`
 * option to run both the App Driver and the App Worker on the same machine.
 * - `--worker`: Run the App Worker.
 * - `--address`: The address (`[<IPv4 address or hostname>][:<port>]`) of the App Driver. If not
 * specified, the App Driver uses the default host address (0.0.0.0) with the default port number
 * (8765).
 * - `--worker-address`: The address (`[<IPv4 address or hostname>][:<port>]`) of the App Worker. If
 * not specified, the App Worker uses the default host address (0.0.0.0) with the default port
 * number randomly chosen from unused ports (between 10000 and 32767).
 * - `--fragments`: The comma-separated names of the fragments to be executed by the App Worker. If
 * not specified, only one fragment (selected by the App Driver) will be executed. `all` can be used
 * to run all the fragments.
 * - `--config`: The path to the configuration file. This will override the configuration file path
 * configured in the application code (before run() is called).
 *
 * If neither `--driver` nor `--worker` is specified, the application will run the application
 * without the App Driver and the App Worker, as if the application were running in the single-node
 * without network communication. Connections between fragments are replaced with the standard
 * intra-fragment connections (double-buffered transmitter/receiver) used for operators.
 */
class CLIParser {
 public:
  /**
   * @brief Construct a new CLIParser object.
   */
  CLIParser() = default;

  /**
   * @brief Initialize the CLI Parser.
   *
   * Set the application description and name and add options/flags for parsing.
   *
   * @param app_description The description of the application.
   */
  void initialize(std::string app_description = "", std::string app_version = "0.0.0");

  /**
   * @brief Parse the command line arguments.
   *
   * Parse the command line arguments and return the remaining arguments.   *
   * Note that the provided vector 'argv' will be modified.
   *
   * @param argv The reference to the vector of strings that contains the command line arguments.
   * @return The reference to the vector of strings that contains the remaining arguments (same as
   * 'argv').
   */
  std::vector<std::string>& parse(std::vector<std::string>& argv);

  /**
   * @brief Check if there is an error during parsing.
   *
   * @return true If there is an error during parsing.
   */
  bool has_error() const;

  /**
   * @brief Get the reference of the CLIOptions struct.
   *
   * @return The reference of the CLIOptions struct.
   */
  CLIOptions& options();

 protected:
  CLI::App app_;                 ///< The CLI11 application object.
  bool is_initialized_ = false;  ///< The flag to check if the parser is initialized.
  bool has_error_ = false;       ///< The flag to check if there is an error during parsing.
  CLIOptions options_;           ///< The CLI options.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CLI_PARSER_HPP */
