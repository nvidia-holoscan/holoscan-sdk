// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace holoscan::visualization::backends {

// Enqueue expansion directly into the exported image, on the backend's upload stream.
cudaError_t launch_rgb_to_rgba(CUdeviceptr source, size_t pitch, uint32_t width, uint32_t height,
                               CUsurfObject destination, CUstream stream);

}  // namespace holoscan::visualization::backends
