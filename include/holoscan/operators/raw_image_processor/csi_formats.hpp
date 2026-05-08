/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_IMAGE_PROCESSOR_CSI_FORMATS_HPP
#define HOLOSCAN_OPERATORS_IMAGE_PROCESSOR_CSI_FORMATS_HPP

namespace holoscan::csi {

/**
 * CSI ingress data comes in one of these formats.  See the
 * CSI-2 specification for details.
 */
enum class PixelFormat {
  /** One byte per pixel */
  RAW_8 = 0,
  /** 10 bits per pixel; arranged as 5 bytes representing 4 pixels. */
  RAW_10 = 1,
  /** 12 bits per pixel; arranged as 3 bytes representing 2 pixels. */
  RAW_12 = 2,
};

// Bayer Format Enum
enum class BayerFormat {
  // NOTE: These values must match NppiBayerGridPosition from NPP; see
  // https://docs.nvidia.com/cuda/npp/nppdefs.html#c.NppiBayerGridPosition
  BGGR = 0,
  RGGB = 1,
  GBRG = 2,
  GRBG = 3
};

}  // namespace holoscan::csi

#endif /* HOLOSCAN_OPERATORS_IMAGE_PROCESSOR_CSI_FORMATS_HPP */
