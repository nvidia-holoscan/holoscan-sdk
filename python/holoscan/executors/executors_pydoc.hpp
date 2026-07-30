/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_EXECUTORS_PYDOC_HPP
#define PYHOLOSCAN_EXECUTORS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFExecutor {

// Constructor
PYDOC(GXFExecutor, R"doc(
GXF-based executor class.
)doc")

PYDOC(GXFExecutor_app, R"doc(
GXF-based executor class.

Parameters
----------
app : holoscan.core.Fragment
    The fragment associated with the executor.

)doc")

}  // namespace GXFExecutor

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_EXECUTORS_PYDOC_HPP
