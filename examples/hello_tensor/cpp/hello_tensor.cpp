/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for installed-SDK consumers of holoscan::Tensor: builds a host Tensor from a DLPack
// context and queries its metadata. This pulls in the PUBLIC header-only dependency dlpack
// (tensor.hpp / dlpack.hpp include <dlpack/dlpack.h>). If find_package(holoscan) cannot resolve
// dlpack::dlpack, or the dlpack / core-domain headers are not installed, this fails to
// configure/compile against the install tree -- which is what we want CI to catch. Uses host
// (kDLCPU) data only, so it needs no CUDA device.

#include <holoscan/core/domain/tensor.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

int main() {
  std::vector<float> data{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {2, 3};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;

  const holoscan::Tensor tensor(std::move(context));

  const std::vector<int64_t> shape = tensor.shape();
  if (tensor.ndim() != 2 || shape.size() != 2 || shape[0] != 2 || shape[1] != 3) {
    std::cerr << "holoscan tensor sample FAILED: unexpected ndim/shape\n";
    return 1;
  }

  std::cout << "tensor ndim=" << tensor.ndim() << " shape=[" << shape[0] << "," << shape[1]
            << "]\n";
  std::cout << "holoscan tensor sample complete\n";
  return 0;
}
