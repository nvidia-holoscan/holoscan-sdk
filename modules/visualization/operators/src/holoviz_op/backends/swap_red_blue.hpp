// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace holoscan::visualization::backends {

/// Enqueue an in-place swap of bytes 0 and 2 of every 4-byte pixel in the pitch-linear image at
/// \p data (RGBA8 <-> BGRA8) on \p stream. Byte-wise, so \p data and \p pitch need no alignment
/// beyond what the copy that produced them required.
cudaError_t launch_swap_red_blue_inplace(CUdeviceptr data, size_t pitch, uint32_t width,
                                         uint32_t height, CUstream stream);

}  // namespace holoscan::visualization::backends
