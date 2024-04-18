/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>

#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/codecs.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

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
