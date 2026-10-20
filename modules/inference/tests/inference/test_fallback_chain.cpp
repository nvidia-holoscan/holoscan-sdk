/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fallback ladder tests.
//
// Each test asserts three things for its rung:
//   1. detection      — the blocking condition is spotted,
//   2. degradation    — the next-safest configuration is selected (never a throw, never silence),
//   3. observability  — the standard "fallback triggered — <REASON_KEY>" block is emitted.
//
// (3) is checked through FallbackLogRecorder, which stores the exact string handed to
// HOLOSCAN_LOG_WARN / HOLOSCAN_LOG_INFO (the Holoscan logger exposes no sink-hooking API, so the
// recorder is the seam).

#include "test_core.hpp"

#include <holoscan/operators/inference/fallback_ladder.hpp>

#include "manager/fallback_runtime.hpp"

#include <string>

namespace eligibility = holoscan::ops::inference_eligibility;
using DispatchKind = eligibility::DispatchKind;
using CaptureProtectionOutcome = eligibility::CaptureProtectionOutcome;
using FallbackReason = eligibility::FallbackReason;
using FallbackSeverity = eligibility::FallbackSeverity;
using eligibility::FallbackLogRecorder;

namespace {

/// Two independent (non-chained) TRT models — the baseline eligible multi-model config.
eligibility::FallbackContext two_model_context() {
  eligibility::FallbackContext ctx;
  ctx.model_path_map = {{"model_1", "m1.onnx"}, {"model_2", "m2.onnx"}};
  ctx.input_map = {{"model_1", {"m1_in"}}, {"model_2", {"m2_in"}}};
  ctx.output_map = {{"model_1", {"m1_out"}}, {"model_2", {"m2_out"}}};
  ctx.backend = "trt";
  ctx.input_on_cuda = true;
  ctx.dynamic_input_dims = false;
  ctx.parallel_inference = true;
  ctx.enable_cuda_graphs = true;
  ctx.stream_pool_present = true;
  ctx.stream_pool_max_size = 4;
  return ctx;
}

/// Emits a resolved decision and returns the concatenated log text for grepping.
std::string emit_and_capture(const eligibility::FallbackDecision& decision) {
  FallbackLogRecorder::clear();
  eligibility::emit_fallback_decision(decision);
  std::string all;
  for (const auto& record : FallbackLogRecorder::snapshot()) {
    all += record;
    all += "\n";
  }
  return all;
}

/// Finds the step for `reason`, or nullptr.
const eligibility::FallbackStep* find_step(const eligibility::FallbackDecision& decision,
                                           FallbackReason reason) {
  for (const auto& step : decision.chain) {
    if (step.reason == reason) {
      return &step;
    }
  }
  return nullptr;
}

/// True when `haystack` contains the standard header line for `reason`.
bool logged(const std::string& haystack, FallbackReason reason) {
  const std::string needle = std::string("fallback triggered — ") + eligibility::reason_key(reason);
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

// =============================================================================
// A — kDynParFast + CUDA graphs + no CudaGreenContextPool: cascade instead of throwing.
// =============================================================================

TEST(FallbackLadder, Fallback_A_DynParFastGraphsNoGreenContext_Cascades) {
  auto ctx = two_model_context();
  ctx.dynamic_input_dims = true;
  ctx.enable_cuda_graphs = true;
  ctx.green_context_pool_present = false;

  // Pre-condition: this is exactly the configuration the concurrent-capture guard refuses to start
  // on.
  EXPECT_EQ(
      eligibility::decide_capture_protection(DispatchKind::kDynParFast, true, false, 0).outcome,
      CaptureProtectionOutcome::kRefuse);

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);

  // Degradation: kDynParFast -> kDynSeqFast, graphs kept, and never a refusal.
  EXPECT_EQ(decision.primary_choice, DispatchKind::kDynParFast);
  EXPECT_EQ(decision.final_choice, DispatchKind::kDynSeqFast);
  EXPECT_NE(decision.final_protection, CaptureProtectionOutcome::kRefuse);
  EXPECT_TRUE(decision.enable_cuda_graphs);

  const auto* step = find_step(decision, FallbackReason::kDynParFastGraphsNoGreenContext);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->from, DispatchKind::kDynParFast);
  EXPECT_EQ(step->to, DispatchKind::kDynSeqFast);
  EXPECT_EQ(step->severity, FallbackSeverity::kWarn);
  // The remaining rung (graphs off -> kStandard) is documented in the message.
  EXPECT_NE(step->reason_detail.find("kStandard"), std::string::npos);
  EXPECT_FALSE(step->fix.empty());

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kDynParFastGraphsNoGreenContext)) << log;
  EXPECT_NE(log.find("requested: kDynParFast"), std::string::npos) << log;
  EXPECT_NE(log.find("using:     kDynSeqFast"), std::string::npos) << log;
}

// =============================================================================
// B — CudaStreamPool.max_size < N models: demotion path, now with an explicit WARN.
// (The plain demotion policy itself is still covered by
// FastPathStreamPoolDemotion.UndersizedPool_DemotesToKSeqFast in test_fast_path_dispatch.cpp.)
// =============================================================================

TEST(FallbackLadder, Fallback_B_UndersizedStreamPool_DemotesToSeqFast) {
  auto ctx = two_model_context();
  ctx.stream_pool_present = true;
  ctx.stream_pool_max_size = 1;  // 1 stream for 2 models

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);

  EXPECT_EQ(decision.primary_choice, DispatchKind::kParFast);
  EXPECT_EQ(decision.final_choice, DispatchKind::kSeqFast);

  const auto* step = find_step(decision, FallbackReason::kStreamPoolInsufficient);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->severity, FallbackSeverity::kWarn);
  EXPECT_NE(step->reason_detail.find("max_size"), std::string::npos);
  EXPECT_NE(step->fix.find("at least 2"), std::string::npos);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kStreamPoolInsufficient)) << log;
  EXPECT_NE(log.find("using:     kSeqFast"), std::string::npos) << log;
}

TEST(FallbackLadder, Fallback_B_AbsentStreamPool_DemotesToSeqFast) {
  auto ctx = two_model_context();
  ctx.stream_pool_present = false;
  ctx.stream_pool_max_size = 0;

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kSeqFast);
  const auto* step = find_step(decision, FallbackReason::kStreamPoolInsufficient);
  ASSERT_NE(step, nullptr);
  EXPECT_NE(step->reason_detail.find("no cuda_stream_pool"), std::string::npos);
}

// =============================================================================
// C — CudaGreenContextPool with fewer partitions than models.
// =============================================================================

TEST(FallbackLadder, Fallback_C_GreenContextTooFewPartitions_DemotesToMutex) {
  auto ctx = two_model_context();
  ctx.model_path_map["model_3"] = "m3.onnx";
  ctx.input_map["model_3"] = {"m3_in"};
  ctx.output_map["model_3"] = {"m3_out"};
  ctx.green_context_pool_present = true;
  ctx.green_context_num_partitions = 2;  // 2 partitions for 3 models

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);

  // Static shapes: stay parallel, but protect capture with prewarm + the serialization mutex
  // instead of (unavailable) per-model context isolation.
  EXPECT_EQ(decision.final_choice, DispatchKind::kParFast);
  EXPECT_EQ(decision.final_protection, CaptureProtectionOutcome::kMutexProtected);

  const auto* step = find_step(decision, FallbackReason::kGreenContextPartitionsInsufficient);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->from_protection, CaptureProtectionOutcome::kGreenContextIsolated);
  EXPECT_EQ(step->to_protection, CaptureProtectionOutcome::kMutexProtected);
  EXPECT_EQ(step->severity, FallbackSeverity::kWarn);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kGreenContextPartitionsInsufficient)) << log;
  EXPECT_NE(log.find("using:     kParFast + mutex-based"), std::string::npos) << log;
}

TEST(FallbackLadder, Fallback_C_GreenContextTooFewPartitions_DynamicGoesSequential) {
  auto ctx = two_model_context();
  ctx.dynamic_input_dims = true;
  ctx.green_context_pool_present = true;
  ctx.green_context_num_partitions = 1;  // 1 partition for 2 models

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);

  // Dynamic shapes cannot prewarm, so the mutex net is not durable -> go sequential.
  EXPECT_EQ(decision.final_choice, DispatchKind::kDynSeqFast);
  ASSERT_NE(find_step(decision, FallbackReason::kGreenContextPartitionsInsufficient), nullptr);
  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kGreenContextPartitionsInsufficient)) << log;
}

// =============================================================================
// D — cuCtxPushCurrent() fails at runtime dispatch.
// =============================================================================

// The manager-side reaction (drop isolation, re-engage the runtime capture-serialization mutex)
// is observable without a GPU: raising the process-wide flag must make a green-context-isolated
// dispatch report the degradation.
TEST(FallbackLadder, Fallback_D_RuntimeFlag_EmitsMutexProtectedDowngrade) {
  FallbackLogRecorder::clear();
  holoscan::inference::emit_runtime_fallback(FallbackReason::kGreenContextPushFailed,
                                             DispatchKind::kParFast,
                                             CaptureProtectionOutcome::kGreenContextIsolated,
                                             CaptureProtectionOutcome::kMutexProtected,
                                             "cuCtxPushCurrent() failed",
                                             "verify the pool outlives the operator");
  EXPECT_TRUE(FallbackLogRecorder::contains(FallbackReason::kGreenContextPushFailed));
  const auto records = FallbackLogRecorder::snapshot();
  ASSERT_FALSE(records.empty());
  EXPECT_NE(records.back().find("requested: kParFast + green-context-isolation"), std::string::npos)
      << records.back();
  EXPECT_NE(records.back().find("using:     kParFast + mutex-based"), std::string::npos)
      << records.back();
}

// =============================================================================
// G — CUDA graphs requested but the supplied stream is blocking: the auto-swap is announced.
// =============================================================================

TEST_F(HoloInferTests, Fallback_G_BlockingStream_LogsSwap) {
  backend = "trt";
  model_path_map = {{"model_1", model_folder + "identity_model.onnx"}};
  input_map = {{"model_1", {"m1_pre_proc"}}};
  output_map = {{"model_1", {"m1_infer"}}};
  in_tensor_dimensions = {{"m1_pre_proc", {3, 256, 256}}};
  use_cuda_graphs = true;
  // The fixture's stream-pool stand-in hands out plain cudaStreamCreate() streams, which are
  // BLOCKING streams — exactly the condition case G reports. `provide_backend_stream_pool` wires
  // it straight to the TrtInfer context (fast_multi_model would route it to par_streams_ instead).
  fast_multi_model = false;
  provide_backend_stream_pool = true;
  stream_pool_limit = -1;

  FallbackLogRecorder::clear();
  auto status = prepare_for_inference();
  ASSERT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);

  EXPECT_TRUE(FallbackLogRecorder::contains(FallbackReason::kBlockingStreamSwapped))
      << "expected the blocking-stream auto-swap to be reported at InferenceOp level";
  // Informational, not a warning.
  EXPECT_EQ(eligibility::default_severity(FallbackReason::kBlockingStreamSwapped),
            FallbackSeverity::kInfo);
}

// =============================================================================
// H — chained models: explicit reason + list + suggested fix.
// =============================================================================

TEST(FallbackLadder, Fallback_H_ChainedModels_ExplicitReason) {
  auto ctx = two_model_context();
  ctx.input_map["model_2"] = {"m1_out"};  // model_2 consumes model_1's output

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kStandard);

  const auto* step = find_step(decision, FallbackReason::kChainedModels);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->severity, FallbackSeverity::kWarn);
  EXPECT_NE(step->reason_detail.find("chained models detected"), std::string::npos);
  EXPECT_NE(step->reason_detail.find("model_1 -> model_2 (m1_out)"), std::string::npos);
  EXPECT_NE(step->fix.find("separate InferenceOps"), std::string::npos);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kChainedModels)) << log;
  EXPECT_NE(log.find("using:     kStandard"), std::string::npos) << log;
}

// =============================================================================
// I — host inputs.
// =============================================================================

TEST(FallbackLadder, Fallback_I_HostInputs_LogsStandard) {
  auto ctx = two_model_context();
  ctx.input_on_cuda = false;

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kStandard);

  const auto* step = find_step(decision, FallbackReason::kHostInputs);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->severity, FallbackSeverity::kInfo);
  EXPECT_NE(step->reason_detail.find("kStandard staging"), std::string::npos);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kHostInputs)) << log;
}

// =============================================================================
// J — non-TRT backend.
// =============================================================================

TEST(FallbackLadder, Fallback_J_NonTrtBackend_LogsStandard) {
  auto ctx = two_model_context();
  ctx.backend.clear();
  ctx.backend_map = {{"model_1", "trt"}, {"model_2", "onnxrt"}};

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kStandard);

  const auto* step = find_step(decision, FallbackReason::kNonTrtBackend);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->severity, FallbackSeverity::kInfo);
  EXPECT_NE(step->reason_detail.find("fast paths require all-TRT"), std::string::npos);
  EXPECT_NE(step->reason_detail.find("model_2=onnxrt"), std::string::npos);
  EXPECT_EQ(step->reason_detail.find("model_1=trt"), std::string::npos)
      << "only the offending models should be listed";

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kNonTrtBackend)) << log;
}

// =============================================================================
// K — parallel_inference honored but a sequential kind was selected.
// =============================================================================

TEST(FallbackLadder, Fallback_K_ParallelRequestedButSerialSelected_Logs) {
  auto ctx = two_model_context();
  ctx.parallel_inference = true;
  ctx.stream_pool_present = true;
  ctx.stream_pool_max_size = 1;  // forces the B demotion, which is what defeats parallelism

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kSeqFast);

  const auto* step = find_step(decision, FallbackReason::kParallelRequestedSerialSelected);
  ASSERT_NE(step, nullptr);
  EXPECT_EQ(step->severity, FallbackSeverity::kWarn);
  EXPECT_NE(step->reason_detail.find("parallel_inference=true honored -> kSeqFast selected"),
            std::string::npos);
  // K names the rung that actually caused it.
  EXPECT_NE(step->reason_detail.find("B_STREAM_POOL_INSUFFICIENT"), std::string::npos);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kParallelRequestedSerialSelected)) << log;
}

TEST(FallbackLadder, Fallback_K_NotEmittedWhenParallelIsObtained) {
  const auto decision = eligibility::resolve_dispatch_with_fallback(two_model_context());
  EXPECT_EQ(decision.final_choice, DispatchKind::kParFast);
  EXPECT_TRUE(decision.chain.empty());
  EXPECT_FALSE(decision.has(FallbackReason::kParallelRequestedSerialSelected));
}

// =============================================================================
// Composite — several blockers at once; the whole chain is emitted, in order.
// =============================================================================

TEST(FallbackLadder, Fallback_Composite_MultipleReasons_LogsAll) {
  auto ctx = two_model_context();
  ctx.backend.clear();
  ctx.backend_map = {{"model_1", "trt"}, {"model_2", "onnxrt"}};  // J
  ctx.input_on_cuda = false;                                      // I
  ctx.input_map["model_2"] = {"m1_out"};                          // H
  ctx.parallel_inference = true;                                  // K

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kStandard);
  ASSERT_EQ(decision.chain.size(), 4u);
  EXPECT_EQ(decision.chain[0].reason, FallbackReason::kNonTrtBackend);
  EXPECT_EQ(decision.chain[1].reason, FallbackReason::kHostInputs);
  EXPECT_EQ(decision.chain[2].reason, FallbackReason::kChainedModels);
  EXPECT_EQ(decision.chain[3].reason, FallbackReason::kParallelRequestedSerialSelected);

  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kNonTrtBackend)) << log;
  EXPECT_TRUE(logged(log, FallbackReason::kHostInputs)) << log;
  EXPECT_TRUE(logged(log, FallbackReason::kChainedModels)) << log;
  EXPECT_TRUE(logged(log, FallbackReason::kParallelRequestedSerialSelected)) << log;

  // Emission order must match the chain order.
  const auto j_pos = log.find(eligibility::reason_key(FallbackReason::kNonTrtBackend));
  const auto i_pos = log.find(eligibility::reason_key(FallbackReason::kHostInputs));
  const auto h_pos = log.find(eligibility::reason_key(FallbackReason::kChainedModels));
  EXPECT_LT(j_pos, i_pos);
  EXPECT_LT(i_pos, h_pos);

  // The one-line summary lists every key.
  EXPECT_NE(decision.user_message.find("J_NON_TRT_BACKEND"), std::string::npos);
  EXPECT_NE(decision.user_message.find("H_CHAINED_MODELS"), std::string::npos);
}

TEST(FallbackLadder, Fallback_Composite_StreamPoolThenGreenContext_ChainsInOrder) {
  auto ctx = two_model_context();
  ctx.dynamic_input_dims = true;
  ctx.enable_cuda_graphs = true;
  ctx.green_context_pool_present = false;  // A applies...
  ctx.stream_pool_present = true;
  ctx.stream_pool_max_size = 1;  // ...but B fires first and already goes sequential

  const auto decision = eligibility::resolve_dispatch_with_fallback(ctx);
  EXPECT_EQ(decision.final_choice, DispatchKind::kDynSeqFast);
  ASSERT_GE(decision.chain.size(), 1u);
  EXPECT_EQ(decision.chain[0].reason, FallbackReason::kStreamPoolInsufficient);
  // A must NOT also fire: the config is no longer kDynParFast once B demoted it.
  EXPECT_FALSE(decision.has(FallbackReason::kDynParFastGraphsNoGreenContext));
  EXPECT_TRUE(decision.has(FallbackReason::kParallelRequestedSerialSelected));
  const auto log = emit_and_capture(decision);
  EXPECT_TRUE(logged(log, FallbackReason::kStreamPoolInsufficient)) << log;
}

// =============================================================================
// Message-format contract: every rung renders the 5-line block with all four fields.
// =============================================================================

TEST(FallbackLadder, FallbackMessage_HasStandardLayout) {
  eligibility::FallbackStep step;
  step.reason = FallbackReason::kStreamPoolInsufficient;
  step.from = DispatchKind::kParFast;
  step.to = DispatchKind::kSeqFast;
  step.from_protection = CaptureProtectionOutcome::kMutexProtected;
  step.to_protection = CaptureProtectionOutcome::kNotApplicable;
  step.reason_detail = "pool too small";
  step.fix.clear();

  const auto message = eligibility::format_fallback_message(step);
  EXPECT_NE(message.find("InferenceOp/HoloInfer: fallback triggered — "
                         "B_STREAM_POOL_INSUFFICIENT"),
            std::string::npos);
  EXPECT_NE(message.find("\n    requested: kParFast + mutex-based"), std::string::npos);
  EXPECT_NE(message.find("\n    using:     kSeqFast + no-capture-protection"), std::string::npos);
  EXPECT_NE(message.find("\n    reason:    pool too small"), std::string::npos);
  EXPECT_NE(message.find("\n    fix:       none"), std::string::npos);
}
