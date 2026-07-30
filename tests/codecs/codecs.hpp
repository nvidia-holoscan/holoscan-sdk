/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdint>

namespace holoscan {

struct Coordinate {
  float x;
  float y;
  float z;
};

// note: don't have to explicitly define codec<Coordinate> for this POD type

// Intentionally place members of different size in non-optimal order to result in a struct
// that is not tightly packed. On my system this struct occupies 24 bytes.
// Automatically inserted to align to 8-byte boundaries.
struct MixedType {
  uint8_t a;
  float b;
  void* c;
  int16_t d;
};

// note: don't have to explicitly define codec<MixedType> for this POD type

}  // namespace holoscan
