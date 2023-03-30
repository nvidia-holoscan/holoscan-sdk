/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <driver_types.h>

#include <cstdint>

namespace holoscan::ops {
namespace segmentation_postprocessor {

struct Shape {
  int32_t height;
  int32_t width;
  int32_t channels;
};

enum NetworkOutputType {
  kSigmoid,
  kSoftmax,
};

enum DataFormat {
  kNCHW,
  kHWC,
  kNHWC,
};

typedef uint8_t output_type_t;

void cuda_postprocess(enum NetworkOutputType network_output_type, enum DataFormat data_format,
                      Shape shape, const float* input, output_type_t* output,
                      cudaStream_t cuda_stream = cudaStreamDefault);

}  // namespace segmentation_postprocessor
}  // namespace holoscan::ops
