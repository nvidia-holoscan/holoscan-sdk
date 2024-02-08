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

#ifndef PYHOLOSCAN_CORE_CLI_PYDOC_HPP
#define PYHOLOSCAN_CORE_CLI_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CLIOptions {

// Constructor
PYDOC(CLIOptions, R"doc(
CLIOptions class.
)doc")

PYDOC(run_driver, R"doc(
The flag to run the App Driver.
)doc")

PYDOC(run_worker, R"doc(
The flag to run the App Worker.
)doc")

PYDOC(driver_address, R"doc(
The address of the App Driver.
)doc")

PYDOC(worker_address, R"doc(
The address of the App Worker.
)doc")

PYDOC(worker_targets, R"doc(
The list of fragments for the App Worker.

Returns
-------
worker_targets : list of str
)doc")

PYDOC(config_path, R"doc(
The path to the configuration file.
)doc")

PYDOC(print, R"doc(
Print the CLI Options.
)doc")

}  // namespace CLIOptions

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_CLI_PYDOC_HPP
