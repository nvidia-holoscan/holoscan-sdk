/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

namespace holoscan {

// Native Holoscan 5 storage-space tag. The integer values intentionally match
// the Holoscan 4.x/GXF values for the overlapping storage classes.
enum struct MemoryStorageType { kHost = 0, kDevice = 1, kSystem = 2, kCudaManaged = 3 };

}  // namespace holoscan
