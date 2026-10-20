/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests for InferenceOp's Phase-2 fast-path dispatch machinery:
//  1. The pure DispatchKind selector (fast_path_eligibility.hpp) — no GPU required.
//  2. The pure CudaStreamPool-insufficiency demotion policy — no GPU required.
//  3. End-to-end exercises of the module-level fast dispatch entry points
//     (InferContext::execute_inference_fast / _seq_fast / _par_fast) via the existing
//     HoloInferTests fixture, including the "pool too small" graceful-demotion path.

#include "test_core.hpp"

#include <holoscan/operators/inference/fast_path_eligibility.hpp>

#include <map>
#include <string>
#include <vector>

namespace eligibility = holoscan::ops::inference_eligibility;
using DispatchKind = eligibility::DispatchKind;

// =============================================================================
// select_dispatch_kind() — pure function, no GPU required
// =============================================================================

TEST(FastPathSelection, SingleTrtModel_SelectsKFast) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "model_1.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}}};

  auto sel = eligibility::select_dispatch_kind(model_path_map,
                                               input_map,
                                               output_map,
                                               "trt",
                                               {},
                                               /*input_on_cuda=*/true,
                                               /*dynamic_input_dims=*/false,
                                               /*parallel_inference=*/true);
  EXPECT_EQ(sel.kind, DispatchKind::kFast);
  EXPECT_EQ(sel.fast_single_model_name, "model_1");
}

TEST(FastPathSelection, SingleTrtModel_DynamicDims_SelectsKDynFast) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "model_1.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, /*dynamic_input_dims=*/true, true);
  EXPECT_EQ(sel.kind, DispatchKind::kDynFast);
}

TEST(FastPathSelection, MultiTrtModel_Sequential_SelectsKSeqFast) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m2_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, false, /*parallel_inference=*/false);
  EXPECT_EQ(sel.kind, DispatchKind::kSeqFast);
}

TEST(FastPathSelection, MultiTrtModel_Parallel_SelectsKParFast) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m2_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, false, /*parallel_inference=*/true);
  EXPECT_EQ(sel.kind, DispatchKind::kParFast);
}

TEST(FastPathSelection, MultiTrtModel_Parallel_DynamicDims_SelectsKDynParFast) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m2_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, true, true);
  EXPECT_EQ(sel.kind, DispatchKind::kDynParFast);
}

TEST(FastPathSelection, ChainedModels_FallsBackToKStandard) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  // model_2 consumes model_1's output — chained.
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m1_out"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, false, true);
  EXPECT_EQ(sel.kind, DispatchKind::kStandard);
}

TEST(FastPathSelection, ChainedModels_DynamicDims_FallsBackToKStandard) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m1_out"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, true, true, true);
  EXPECT_EQ(sel.kind, DispatchKind::kStandard);
}

TEST(FastPathSelection, HostInput_FallsBackToKStandard) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "trt", {}, /*input_on_cuda=*/false, false, true);
  EXPECT_EQ(sel.kind, DispatchKind::kStandard);
}

TEST(FastPathSelection, NonTrtBackend_FallsBackToKStandard) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "onnxrt", {}, true, false, true);
  EXPECT_EQ(sel.kind, DispatchKind::kStandard);
}

TEST(FastPathSelection, MixedBackendMap_FallsBackToKStandard) {
  const std::map<std::string, std::string> model_path_map{{"model_1", "m1.onnx"},
                                                          {"model_2", "m2.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{"model_1", {"m1_in"}},
                                                                  {"model_2", {"m2_in"}}};
  const std::map<std::string, std::vector<std::string>> output_map{{"model_1", {"m1_out"}},
                                                                   {"model_2", {"m2_out"}}};
  const std::map<std::string, std::string> backend_map{{"model_1", "trt"}, {"model_2", "onnxrt"}};

  auto sel = eligibility::select_dispatch_kind(
      model_path_map, input_map, output_map, "", backend_map, true, false, true);
  EXPECT_EQ(sel.kind, DispatchKind::kStandard);
}

// =============================================================================
// demote_if_stream_pool_insufficient() — pure function, no GPU required
// =============================================================================

TEST(FastPathStreamPoolDemotion, SufficientPool_KeepsKParFast) {
  auto kind =
      eligibility::demote_if_stream_pool_insufficient(DispatchKind::kParFast, true, /*max=*/4, 2);
  EXPECT_EQ(kind, DispatchKind::kParFast);
}

TEST(FastPathStreamPoolDemotion, UnboundedPool_KeepsKParFast) {
  auto kind =
      eligibility::demote_if_stream_pool_insufficient(DispatchKind::kParFast, true, /*max=*/0, 8);
  EXPECT_EQ(kind, DispatchKind::kParFast);
}

TEST(FastPathStreamPoolDemotion, UndersizedPool_DemotesToKSeqFast) {
  auto kind =
      eligibility::demote_if_stream_pool_insufficient(DispatchKind::kParFast, true, /*max=*/1, 2);
  EXPECT_EQ(kind, DispatchKind::kSeqFast);
}

TEST(FastPathStreamPoolDemotion, AbsentPool_DemotesToKSeqFast) {
  auto kind = eligibility::demote_if_stream_pool_insufficient(DispatchKind::kParFast, false, 0, 2);
  EXPECT_EQ(kind, DispatchKind::kSeqFast);
}

TEST(FastPathStreamPoolDemotion, UndersizedPool_DynParFast_DemotesToKDynSeqFast) {
  auto kind = eligibility::demote_if_stream_pool_insufficient(DispatchKind::kDynParFast,
                                                              true,
                                                              /*max=*/1,
                                                              3);
  EXPECT_EQ(kind, DispatchKind::kDynSeqFast);
}

TEST(FastPathStreamPoolDemotion, NonParallelKinds_PassThroughUnchanged) {
  EXPECT_EQ(eligibility::demote_if_stream_pool_insufficient(DispatchKind::kFast, false, 0, 1),
            DispatchKind::kFast);
  EXPECT_EQ(eligibility::demote_if_stream_pool_insufficient(DispatchKind::kSeqFast, false, 0, 2),
            DispatchKind::kSeqFast);
  EXPECT_EQ(eligibility::demote_if_stream_pool_insufficient(DispatchKind::kStandard, false, 0, 2),
            DispatchKind::kStandard);
}

// =============================================================================
// Module-level fast-dispatch entry points — GPU + TRT required (HoloInferTests fixture)
// =============================================================================

TEST_F(HoloInferTests, FastPath_KFast_SingleModel_BasicInference) {
  backend = "trt";
  model_path_map = {{"model_1", model_folder + "identity_model.onnx"}};
  input_map = {{"model_1", {"m1_pre_proc"}}};
  output_map = {{"model_1", {"m1_infer"}}};
  in_tensor_dimensions = {{"m1_pre_proc", {3, 256, 256}}};

  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);

  status = holoscan_infer_context_->execute_inference_fast(inference_specs_, 0, "model_1");
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, FastPath_KSeqFast_MultiModel_BasicInference) {
  backend = "trt";
  parallel_inference = false;
  fast_multi_model = true;

  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);

  status = holoscan_infer_context_->execute_inference_seq_fast(inference_specs_, 0);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, FastPath_KParFast_MultiModel_BasicInference) {
  backend = "trt";
  parallel_inference = true;
  fast_multi_model = true;
  stream_pool_limit = -1;  // unlimited — pool can supply 2 distinct streams

  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);

  // First call builds the I/O cache and runs sequentially (see infer_manager_fast_paths.cpp);
  // the second call exercises the actual inline-parallel fan-out/fan-in.
  status = holoscan_infer_context_->execute_inference_par_fast(inference_specs_, 0);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
  status = holoscan_infer_context_->execute_inference_par_fast(inference_specs_, 0);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);

  // The test stream-pool stand-in must have been asked for exactly one stream per model.
  EXPECT_EQ(stream_pool_calls_, static_cast<int32_t>(model_path_map.size()));
}

TEST_F(HoloInferTests, FastPath_KParFast_UndersizedStreamPool_DemotesGracefully) {
  backend = "trt";
  parallel_inference = true;
  fast_multi_model = true;
  // Only 1 distinct stream available for 2 models — HoloInfer's set_inference_params()
  // must detect this and demote to the work_queue/canonical dispatch internally, leaving
  // par_streams_ empty. execute_inference_par_fast() must then fall back to
  // execute_inference() and still succeed (see infer_context.cpp).
  stream_pool_limit = 1;

  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);

  status = holoscan_infer_context_->execute_inference_par_fast(inference_specs_, 0);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, FastPath_KDynFast_SingleModel_BasicInference) {
  backend = "trt";
  model_path_map = {{"model_1", model_folder + "identity_model_dynamic.onnx"}};
  input_map = {{"model_1", {"m1_pre_proc"}}};
  output_map = {{"model_1", {"m1_infer"}}};
  in_tensor_dimensions = {{"m1_pre_proc", {2, 256, 256}}};
  batch_sizes = {{"model_1", {"1, 2, 8"}}};
  dynamic_inputs = true;

  cleanup_engines();
  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }

  status = holoscan_infer_context_->execute_inference_dyn_fast(inference_specs_, 0, "model_1");
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
  cleanup_engines();
}
