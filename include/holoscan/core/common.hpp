/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_COMMON_HPP
#define HOLOSCAN_CORE_COMMON_HPP

#include "./errors.hpp"
#include "./expected.hpp"
#include "./forward_def.hpp"
// clang-format off

// Include parameter.hpp before logger.hpp for supporting holoscan::Parameter<T>
// with fmt::format.
#include "./parameter.hpp"
#include <holoscan/logger/logger.hpp>

// clang-format on

#endif /* HOLOSCAN_CORE_COMMON_HPP */
