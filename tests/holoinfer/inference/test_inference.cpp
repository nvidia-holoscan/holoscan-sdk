/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_core.hpp"

#include <memory>
#include <string>
#include <utility>

void HoloInferTests::inference_tests() {
  std::string test_module = "Inference tests";
  backend = "trt";

  // Test: TRT backend, Empty input data
  auto status = prepare_for_inference();
  auto dmap = std::move(inference_specs_->data_per_tensor_);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 1, test_identifier_infer.at(1), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_ = std::move(dmap);

  // Test: TRT backend, Empty inference parameters
  clear_specs();
  setup_specifications();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 2, test_identifier_infer.at(2), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Missing input tensor
  status = prepare_for_inference();
  auto dm = std::move(inference_specs_->data_per_tensor_.at("plax_cham_pre_proc"));
  inference_specs_->data_per_tensor_.erase("plax_cham_pre_proc");
  status = do_inference();
  holoinfer_assert(
      status, test_module, 3, test_identifier_infer.at(3), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.insert({"plax_cham_pre_proc", dm});

  // Test: TRT backend, Missing output tensor
  status = prepare_for_inference();
  dm = std::move(inference_specs_->output_per_model_.at("aortic_infer"));
  inference_specs_->output_per_model_.erase("aortic_infer");
  status = do_inference();
  holoinfer_assert(
      status, test_module, 4, test_identifier_infer.at(4), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.insert({"aortic_infer", dm});

  // Test: TRT backend, Empty input cuda buffer 1
  auto dbs = inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->device_buffer->size();
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->device_buffer->resize(0);
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->device_buffer = nullptr;
  status = do_inference();
  holoinfer_assert(
      status, test_module, 5, test_identifier_infer.at(5), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->device_buffer =
      std::make_shared<HoloInfer::DeviceBuffer>();

  // Test: TRT backend, Empty input cuda buffer 2
  status = do_inference();
  holoinfer_assert(
      status, test_module, 6, test_identifier_infer.at(6), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->device_buffer->resize(dbs);

  // Test: TRT backend, Empty output cuda buffer 1
  dbs = inference_specs_->output_per_model_.at("aortic_infer")->device_buffer->size();
  inference_specs_->output_per_model_.at("aortic_infer")->device_buffer->resize(0);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 7, test_identifier_infer.at(7), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Empty output cuda buffer 2
  inference_specs_->output_per_model_.at("aortic_infer")->device_buffer = nullptr;
  status = do_inference();
  holoinfer_assert(
      status, test_module, 8, test_identifier_infer.at(8), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("aortic_infer")->device_buffer =
      std::make_shared<HoloInfer::DeviceBuffer>();

  // Test: TRT backend, Empty output cuda buffer 3
  status = do_inference();
  holoinfer_assert(
      status, test_module, 9, test_identifier_infer.at(9), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("aortic_infer")->device_buffer->resize(dbs);

  // Test: TRT backend, Basic end-to-end cuda inference
  status = do_inference();
  holoinfer_assert(
      status, test_module, 10, test_identifier_infer.at(10), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Basic sequential end-to-end cuda inference
  parallel_inference = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 11, test_identifier_infer.at(11), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Input on host inference
  input_on_cuda = false;
  parallel_inference = true;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 12, test_identifier_infer.at(12), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Output on host inference
  input_on_cuda = true;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 13, test_identifier_infer.at(13), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Input/Output on host inference
  input_on_cuda = false;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 14, test_identifier_infer.at(14), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Empty host input
  size_t re_dbs = 0;
  dbs = inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.size();
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.resize(re_dbs);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 15, test_identifier_infer.at(15), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.resize(dbs);

  // Test: TRT backend, Empty host output
  dbs = inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.size();
  inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(re_dbs);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 16, test_identifier_infer.at(16), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(dbs);

  if (use_onnxruntime) {
    // Test: ONNX backend, Basic parallel inference on CPU
    input_on_cuda = false;
    output_on_cuda = false;
    infer_on_cpu = true;
    backend = "onnxrt";
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     17,
                     test_identifier_infer.at(17),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic sequential inference on CPU
    parallel_inference = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     18,
                     test_identifier_infer.at(18),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    if (is_x86_64) {
      // Test: ONNX backend, Basic sequential inference on GPU
      infer_on_cpu = false;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       19,
                       test_identifier_infer.at(19),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: ONNX backend, Basic parallel inference on GPU
      parallel_inference = true;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       20,
                       test_identifier_infer.at(20),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: ONNX backend, Empty host input
      dbs = inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.size();
      inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.resize(0);
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       21,
                       test_identifier_infer.at(21),
                       HoloInfer::holoinfer_code::H_ERROR);
      inference_specs_->data_per_tensor_.at("plax_cham_pre_proc")->host_buffer.resize(dbs);

      // Test: ONNX backend, Empty host output
      dbs = inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.size();
      inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(0);
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       22,
                       test_identifier_infer.at(22),
                       HoloInfer::holoinfer_code::H_ERROR);
      inference_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(dbs);
    } else {
      // Test: ONNX backend on ARM, Basic sequential inference on GPU
      infer_on_cpu = false;
      status = prepare_for_inference();
      holoinfer_assert(status,
                       test_module,
                       23,
                       test_identifier_infer.at(23),
                       HoloInfer::holoinfer_code::H_ERROR);
    }

    // Multi-GPU tests
    cudaDeviceProp device_prop;
    auto dev_id = 1;
    backend = "trt";
    auto cstatus = cudaGetDeviceProperties(&device_prop, dev_id);
    device_map.at("plax_chamber") = "1";

    if (cstatus == cudaSuccess) {
      // Test: TRT backend, Basic sequential inference on multi-GPU
      input_on_cuda = true;
      output_on_cuda = true;
      parallel_inference = false;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       27,
                       test_identifier_infer.at(27),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: TRT backend, Basic parallel inference on multi-GPU
      parallel_inference = true;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       28,
                       test_identifier_infer.at(28),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: TRT backend, Parallel inference on multi-GPU with I/O on host
      input_on_cuda = false;
      output_on_cuda = false;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       29,
                       test_identifier_infer.at(29),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: TRT backend, Parallel inference on multi-GPU with Input on host
      input_on_cuda = false;
      output_on_cuda = true;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       30,
                       test_identifier_infer.at(30),
                       HoloInfer::holoinfer_code::H_SUCCESS);

      // Test: TRT backend, Parallel inference on multi-GPU with Output on host
      input_on_cuda = true;
      output_on_cuda = false;
      status = prepare_for_inference();
      status = do_inference();
      holoinfer_assert(status,
                       test_module,
                       31,
                       test_identifier_infer.at(31),
                       HoloInfer::holoinfer_code::H_SUCCESS);
    }
    device_map.at("plax_chamber") = "0";

    if (is_x86_64) {
      device_map.at("aortic_stenosis") = "1";
      if (cstatus == cudaSuccess) {
        // Test: ONNX backend, Basic sequential inference on multi-GPU
        status = prepare_for_inference();
        status = do_inference();
        holoinfer_assert(status,
                         test_module,
                         24,
                         test_identifier_infer.at(24),
                         HoloInfer::holoinfer_code::H_SUCCESS);

        // Test: ONNX backend, Basic parallel inference on multi-GPU
        parallel_inference = true;
        status = prepare_for_inference();
        status = do_inference();
        holoinfer_assert(status,
                         test_module,
                         26,
                         test_identifier_infer.at(26),
                         HoloInfer::holoinfer_code::H_SUCCESS);
      } else {
        // Test: ONNX backend, Inference single GPU with multi-GPU settings
        status = prepare_for_inference();
        holoinfer_assert(status,
                         test_module,
                         25,
                         test_identifier_infer.at(25),
                         HoloInfer::holoinfer_code::H_ERROR);
      }
      device_map.at("aortic_stenosis") = "0";
    }
  }
}
