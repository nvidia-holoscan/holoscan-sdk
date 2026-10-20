/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <holoviz/common.hpp>

namespace holoviz::cuda {

struct CuArrayFormat {
  CUarray_format format;
  unsigned num_channels;
};

/// Map a holoviz Format to the CUDA array format and channel count needed by
/// CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC. Returns an error for formats
/// that have no CUDA array equivalent (BGRA, 10-bit packed, 3-channel float,
/// depth).
[[nodiscard]] expected<CuArrayFormat> from_holoviz_format(Format format);

}  // namespace holoviz::cuda
