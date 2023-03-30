/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multiai_tests.hpp"

void holoinfer_assert(const HoloInfer::InferStatus& status, const std::string& module,
                      const std::string& test_name, HoloInfer::holoinfer_code assert_type) {
  status.display_message();
  unsigned int current_test = test_count++ + 1;

  if (status.get_code() != assert_type) {
    std::cout << "Test " << current_test << ": " << test_name << " in " << module << " -> FAIL."
              << std::endl;
    std::exit(1);
  }
  std::cout << "Test " << current_test << ": " << test_name << " in " << module << " -> PASS."
            << std::endl;
}

void clear_specs() {
  multiai_specs_.reset();
  holoscan_infer_context_.reset();
}

HoloInfer::InferStatus create_specifications() {
  // Create multiai specification structure
  multiai_specs_ = std::make_shared<HoloInfer::MultiAISpecs>(backend,
                                                             model_path_map,
                                                             inference_map,
                                                             is_engine_path,
                                                             infer_on_cpu,
                                                             parallel_inference,
                                                             enable_fp16,
                                                             input_on_cuda,
                                                             output_on_cuda);

  holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();
  auto status = holoscan_infer_context_->set_inference_params(multiai_specs_);
  return status;
}

HoloInfer::InferStatus call_parameter_check() {
  return HoloInfer::multiai_inference_validity_check(
      model_path_map, pre_processor_map, inference_map, in_tensor_names, out_tensor_names);
}

void parameter_test() {
  std::string test_module = "Parameter test";

  // Check for the validity of parameters from configuration

  auto input_tensor_names = std::move(in_tensor_names);
  std::string test_name = "in_tensor_names check 1";
  auto status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  in_tensor_names = std::move(input_tensor_names);

  auto output_tensor_names = std::move(out_tensor_names);
  test_name = "out_tensor_names check 1";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  out_tensor_names = std::move(output_tensor_names);

  model_path_map.insert({"test-dummy", "path-dummy"});
  test_name = "model_path_map check 1";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  model_path_map.at("test-dummy") = model_path_map.at("plax_chamber");
  test_name = "model_path_map check 2";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  model_path_map.erase("test-dummy");
  pre_processor_map.insert({"test-dummy", {"data-dummy"}});
  test_name = "pre_processor_map check 1";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  pre_processor_map.erase("test-dummy");
  pre_processor_map.at("plax_chamber").push_back("different-tensor-name");
  test_name = "pre_processor_map check 2";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  pre_processor_map.at("plax_chamber").pop_back();

  inference_map.insert({"test-dummy", "data-dummy"});
  test_name = "inference_map check 1";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  inference_map.erase("test-dummy");
  std::string original_name = inference_map.at("plax_chamber");
  inference_map.at("plax_chamber") = "different-tensor-name";
  test_name = "inference_map check 2";
  status = call_parameter_check();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  inference_map.at("plax_chamber") = original_name;

  status = call_parameter_check();
  test_name = "Input parameter set check";
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);
}

void parameter_setup_test() {
  std::string test_module = "Inference setup";

  std::string test_name = "TRT backend, Empty model path check 1";
  auto mmm = std::move(model_path_map);
  auto status = create_specifications();
  clear_specs();
  model_path_map = std::move(mmm);
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "TRT backend, Backend type";
  backend = "tensorflow";
  status = create_specifications();
  clear_specs();
  backend = "trt";
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "TRT backend, Backend unsupported";
  backend = "pytorch";
  status = create_specifications();
  clear_specs();
  backend = "trt";
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "Inference map, missing entry";
  auto imap = std::move(inference_map);
  status = create_specifications();
  clear_specs();
  inference_map = std::move(imap);
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "ONNX backend, Input cuda buffer test";
  backend = "onnxrt";
  status = create_specifications();
  clear_specs();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "ONNX backend, Input cuda buffer test";
  is_engine_path = true;
  status = create_specifications();
  clear_specs();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "ONNX backend, Default";
  if (!is_x86_64) { infer_on_cpu = true; }
  is_engine_path = false;
  input_on_cuda = false;
  output_on_cuda = false;
  status = create_specifications();
  clear_specs();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  backend = "trt";
  if (!is_x86_64) { infer_on_cpu = false; }
  input_on_cuda = true;
  output_on_cuda = true;
  test_name = "TRT backend, Default check 1";
  status = create_specifications();
  clear_specs();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);
}

HoloInfer::InferStatus prepare_for_inference() {
  clear_specs();

  auto status = create_specifications();

  for (const auto td : in_tensor_dimensions) {
    auto db = std::make_shared<HoloInfer::DataBuffer>();
    size_t buffer_size =
        std::accumulate(td.second.begin(), td.second.end(), 1, std::multiplies<size_t>());

    db->host_buffer.resize(buffer_size);
    db->device_buffer->resize(buffer_size);

    std::vector<float> data_in(buffer_size, 0);
    db->host_buffer = std::move(data_in);

    multiai_specs_->data_per_tensor_.insert({td.first, std::move(db)});
  }

  return status;
}

HoloInfer::InferStatus do_mapping() {
  return HoloInfer::map_data_to_model_from_tensor(
      pre_processor_map, multiai_specs_->data_per_model_, multiai_specs_->data_per_tensor_);
}

HoloInfer::InferStatus do_inference() {
  return holoscan_infer_context_->execute_inference(multiai_specs_->data_per_model_,
                                                    multiai_specs_->output_per_model_);
}

void inference_tests() {
  std::string test_module = "Inference tests";

  std::string test_name = "TRT backend, Empty preprocessor map";
  auto ppm = std::move(pre_processor_map);
  auto status = do_mapping();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  pre_processor_map = std::move(ppm);

  test_name = "TRT backend, Empty input data";
  status = prepare_for_inference();
  auto dmap = std::move(multiai_specs_->data_per_tensor_);
  status = do_mapping();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->data_per_tensor_ = std::move(dmap);

  test_name = "TRT backend, Missing input tensor";
  auto dm = std::move(multiai_specs_->data_per_tensor_.at("plax_cham_pre_proc"));
  multiai_specs_->data_per_tensor_.erase("plax_cham_pre_proc");
  status = do_mapping();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->data_per_tensor_.insert({"plax_cham_pre_proc", dm});

  test_name = "TRT backend, Empty input cuda buffer 1";
  multiai_specs_->data_per_model_.clear();
  status = do_mapping();
  auto dbs = multiai_specs_->data_per_model_.at("plax_chamber")->device_buffer->size();
  multiai_specs_->data_per_model_.at("plax_chamber")->device_buffer->resize(0);
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "TRT backend, Empty input cuda buffer 2";
  multiai_specs_->data_per_model_.at("plax_chamber")->device_buffer = nullptr;
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->data_per_model_.at("plax_chamber")->device_buffer =
      std::make_shared<HoloInfer::DeviceBuffer>();

  test_name = "TRT backend, Empty input cuda buffer 3";
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->data_per_model_.at("plax_chamber")->device_buffer->resize(dbs);

  test_name = "TRT backend, Empty output cuda buffer 1";
  dbs = multiai_specs_->output_per_model_.at("aortic_infer")->device_buffer->size();
  multiai_specs_->output_per_model_.at("aortic_infer")->device_buffer->resize(0);
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);

  test_name = "TRT backend, Empty output cuda buffer 2";
  multiai_specs_->output_per_model_.at("aortic_infer")->device_buffer = nullptr;
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->output_per_model_.at("aortic_infer")->device_buffer =
      std::make_shared<HoloInfer::DeviceBuffer>();

  test_name = "TRT backend, Empty output cuda buffer 3";
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->output_per_model_.at("aortic_infer")->device_buffer->resize(dbs);

  test_name = "TRT backend, Basic end-to-end cuda inference";
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "TRT backend, Basic sequential end-to-end cuda inference";
  parallel_inference = false;
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "TRT backend, Input on host inference";
  input_on_cuda = false;
  parallel_inference = true;
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "TRT backend, Output on host inference";
  input_on_cuda = true;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "TRT backend, Input/Output on host inference";
  input_on_cuda = false;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "TRT backend, Empty host input";
  dbs = multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.size();
  multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.resize(0);
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.resize(dbs);

  test_name = "TRT backend, Empty host output";
  dbs = multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.size();
  multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(0);
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(dbs);

  test_name = "ONNX backend, Basic parallel inference on CPU";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = true;
  backend = "onnxrt";
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  test_name = "ONNX backend, Basic sequential inference on CPU";
  parallel_inference = false;
  status = prepare_for_inference();
  status = do_mapping();
  status = do_inference();
  holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

  if (is_x86_64) {
    test_name = "ONNX backend, Basic sequential inference on GPU";
    infer_on_cpu = false;
    status = prepare_for_inference();
    status = do_mapping();
    status = do_inference();
    holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

    test_name = "ONNX backend, Basic parallel inference on GPU";
    parallel_inference = true;
    status = prepare_for_inference();
    status = do_mapping();
    status = do_inference();
    holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_SUCCESS);

    test_name = "ONNX backend, Empty host input";
    dbs = multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.size();
    multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.resize(0);
    status = do_inference();
    holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
    multiai_specs_->data_per_model_.at("plax_chamber")->host_buffer.resize(dbs);

    test_name = "ONNX backend, Empty host output";
    dbs = multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.size();
    multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(0);
    status = do_inference();
    holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
    multiai_specs_->output_per_model_.at("aortic_infer")->host_buffer.resize(dbs);
  } else {
    test_name = "ONNX backend on ARM, Basic sequential inference on GPU";
    infer_on_cpu = false;
    status = prepare_for_inference();
    holoinfer_assert(status, test_module, test_name, HoloInfer::holoinfer_code::H_ERROR);
  }
}

int main() {
  parameter_test();
  parameter_setup_test();
  inference_tests();
  clear_specs();

  return 0;
}
