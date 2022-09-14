/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <cuda.h>

#include <sstream>

namespace clara::holoviz
{

/**
 * Cuda driver API error check helper
 */
#define CudaCheck(FUNC)                                                        \
    {                                                                          \
        const CUresult result = FUNC;                                          \
        if (result != CUDA_SUCCESS)                                            \
        {                                                                      \
            const char *error_name = "";                                       \
            cuGetErrorName(result, &error_name);                               \
            const char *error_string = "";                                     \
            cuGetErrorString(result, &error_string);                           \
            std::stringstream buf;                                             \
            buf << "Cuda driver error " << error_name << ": " << error_string; \
            throw std::runtime_error(buf.str().c_str());                       \
        }                                                                      \
    }

} // namespace clara::holoviz
