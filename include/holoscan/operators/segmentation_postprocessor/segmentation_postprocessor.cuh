/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
