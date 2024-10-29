/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CudaStreamCondition {

PYDOC(CudaStreamCondition, R"doc(
Condition class to indicate data availability on CUDA stream completion.

This condition will register a callback function which will be called once the work on the
specified CUDA stream completes indicating that the data is available for consumption.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with.
name : str, optional
    The name of the condition.
)doc")

PYDOC(receiver, R"doc(
The receiver associated with the condition.
)doc")

}  // namespace CudaStreamCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_CUDA_STREAM_PYDOC_HPP */
