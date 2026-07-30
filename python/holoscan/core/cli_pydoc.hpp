/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
