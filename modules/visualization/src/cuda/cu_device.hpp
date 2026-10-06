/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <array>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

namespace holoviz::cuda {

[[nodiscard]] expected<int> get_cuda_ordinal(const std::array<uint8_t, 16>& uuid);

}  // namespace holoviz::cuda
