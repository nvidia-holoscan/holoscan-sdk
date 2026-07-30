/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/codec_registry.hpp>

namespace holoscan {

CodecRegistry& CodecRegistry::get_instance() {
  static CodecRegistry instance;
  return instance;
}

}  // namespace holoscan
