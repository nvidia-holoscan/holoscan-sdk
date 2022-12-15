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
#include <cstdint>
#include <limits>

namespace nvidia {
namespace holoscan {
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

static constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

void cuda_postprocess(enum NetworkOutputType network_output_type, enum DataFormat data_format,
                      Shape shape, const float* input, output_type_t* output);

}  // namespace segmentation_postprocessor
}  // namespace holoscan
}  // namespace nvidia
