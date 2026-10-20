/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>

/// Write one sub-frame band of an animated RGBA8 test pattern into `dst` (pitched device
/// memory of `band_height` rows, `dst_pitch` bytes apart) — the stand-in for a band-sized
/// tensor arriving from a sensor. The pattern is keyed on the band's position in the full
/// frame (`y0`, `full_height`) and on `frame`, so the assembled bands form one continuous
/// moving gradient; the band's first row is drawn white to make the band tiling visible.
/// Asynchronous; ordered by `stream`.
void launch_band_pattern(CUdeviceptr dst, size_t dst_pitch, uint32_t width, uint32_t band_height,
                         uint32_t y0, uint32_t full_height, uint32_t frame, CUstream stream);
