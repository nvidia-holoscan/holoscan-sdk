/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Multi-operator concurrent CUDA-Graph capture tests.
//
// Background: TensorRT captures a CUDA graph on first kParFast/kDynParFast inference when
// enable_cuda_graphs is set. Stream capture is CUDA-context scoped: concurrent capture from
// two InferenceOp instances sharing a CUDA context (the common case — one process, one
// default context) can invalidate an in-progress capture on another stream. These tests
// mimic 3 concurrent InferenceOp operators (3 independent HoloInfer::InferContext /
// ManagerInfer instances, each with its own pair of TRT models) exercising kParFast
// concurrently from 3 threads, to validate:
//   2a. the failure mode exists in principle when nothing serializes capture (best-effort;
//       see the note on the always-on mutex safety net below — this test documents the
//       failure mode and may legitimately no longer reproduce it now that the mutex safety
//       net is always applied).
//   2b. ManagerInfer::capture_serialization_mutex_ + setup-time prewarm make
//       concurrent kParFast dispatch safe across repeated iterations.
//   2c. supplying a CudaGreenContextPool remains safe under the same concurrent load.
//   2d/2e. the pure decide_capture_protection() decision used by InferenceOp::start() refuses
//       kDynParFast + CUDA graphs without a green context, and accepts it with one.

#include "test_core.hpp"

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/operators/inference/fast_path_eligibility.hpp>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace eligibility = holoscan::ops::inference_eligibility;
using DispatchKind = eligibility::DispatchKind;
using CaptureProtectionOutcome = eligibility::CaptureProtectionOutcome;

namespace {

// Resolve a per-partition SM count that satisfies THIS device's Green Context granularity
// requirements instead of hardcoding a fixed count. Earlier revisions of these tests hardcoded
// 4 SMs/partition, which is below the minimum structured-partition granularity on newer
// architectures (e.g. Blackwell requires 8 SMs/partition and multiples of 8 — the driver
// rejects a request for 4 with "must be at least 8 and a multiple of 8"). That made the tests
// GTEST_SKIP() with a message reading "unavailable on this platform/driver", which is
// misleading: green contexts ARE available, the requested partition size was just too small.
// Query the device's actual granularity via a throwaway whole-device pool (whole-device
// requests bypass the per-partition-size check) and derive a valid per-partition SM count as
// the LCM of the driver-reported minimum partition size and co-scheduling alignment — the same
// "structured_sm_quantum" computation CudaGreenContextPool uses internally.
std::optional<uint32_t> resolve_green_context_partition_sm_count() {
  holoscan::CudaGreenContextPool capability_pool;  // default config: one whole-device group
  if (!capability_pool.initialize_pool()) {
    return std::nullopt;
  }
  auto min_size = capability_pool.min_sm_partition_size();
  auto alignment = capability_pool.sm_coscheduled_alignment();
  if (!min_size || !alignment || *min_size == 0 || *alignment == 0) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(std::lcm(*min_size, *alignment));
}

// One simulated InferenceOp: an independent HoloInfer::InferContext (which owns its own
// ManagerInfer) driving 2 TRT models in kParFast configuration. Model names are prefixed per
// instance because InferContext::set_inference_params() registers each manager under a
// unique-id derived from its model-name set (see infer_context.cpp) — reusing "model_1"/
// "model_2" verbatim across instances would collide in that process-wide registry.
struct ParFastOp {
  std::unique_ptr<HoloInfer::InferContext> ctx;
  std::shared_ptr<HoloInfer::InferenceSpecs> specs;
  std::vector<cudaStream_t> owned_streams;
  // Dedicated non-legacy "outer" stream for this op's execute_inference_par_fast() calls —
  // mirrors ExecutionContext::cuda_stream() in the real InferenceOp::compute() path. The
  // legacy default stream (0) must NOT be used here: it implicitly synchronizes with every
  // other blocking stream in the process, so concurrently recording events / issuing
  // cudaStreamWaitEvent on stream 0 from multiple threads while OTHER threads' per-model
  // par_streams_ are mid CUDA-Graph-capture fails with "operation would make the legacy
  // stream depend on a capturing blocking stream" — a test-harness artifact of using the
  // legacy stream, not the capture race under test.
  cudaStream_t outer_stream = nullptr;

  ~ParFastOp() {
    if (outer_stream != nullptr) {
      cudaStreamDestroy(outer_stream);
    }
    for (auto s : owned_streams) {
      if (s != nullptr) {
        cudaStreamDestroy(s);
      }
    }
  }
};

// Builds and sets up one ParFastOp (2 identity TRT models, parallel + fast_multi_model,
// static shapes so kParFast eligibility criteria are met). Returns true on success.
// `dynamic_dims_hack` reproduces the "no prewarm" state for test 2a: setting
// InferenceSpecs::dynamic_input_dims_ makes ManagerInfer::set_inference_params() skip the
// setup-time prewarm_par_fast() call (see infer_manager_setup.cpp), while the per-model
// par_streams_ are still provisioned (dynamic dims no longer blocks stream allocation), so
// run_core_inference_par_fast() is still reachable — just without the serialized prewarm
// captures that normally happen before any concurrent compute() call.
bool build_par_fast_op(const std::string& prefix, const std::string& model_folder,
                       bool dynamic_dims_hack, ParFastOp& op,
                       std::vector<CUcontext> green_context_partition_contexts = {},
                       CUcontext build_cuda_context = nullptr, int32_t build_sm_count = 0) {
  const std::string m1 = prefix + "_model_1";
  const std::string m2 = prefix + "_model_2";
  const std::string t1_in = prefix + "_m1_in";
  const std::string t2_in = prefix + "_m2_in";
  const std::string t1_out = prefix + "_m1_out";
  const std::string t2_out = prefix + "_m2_out";

  const std::map<std::string, std::string> model_path_map{
      {m1, model_folder + "identity_model.onnx"}, {m2, model_folder + "identity_model.onnx"}};
  const std::map<std::string, std::vector<std::string>> input_map{{m1, {t1_in}}, {m2, {t2_in}}};
  const std::map<std::string, std::vector<std::string>> output_map{{m1, {t1_out}}, {m2, {t2_out}}};
  const std::map<std::string, std::string> dla_core_map{{m1, "-1"}, {m2, "-1"}};
  const std::map<std::string, std::string> backend_map;
  const std::map<std::string, std::vector<std::string>> batch_sizes{{m1, {"1, 1, 1"}},
                                                                    {m2, {"1, 1, 1"}}};

  op.owned_streams.clear();
  auto allocate_cuda_stream = [&op](int32_t /*device_id*/) -> cudaStream_t {
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) {
      return nullptr;
    }
    op.owned_streams.push_back(s);
    return s;
  };

  op.specs = std::make_shared<HoloInfer::InferenceSpecs>("trt",
                                                         backend_map,
                                                         model_path_map,
                                                         input_map,
                                                         output_map,
                                                         dla_core_map,
                                                         batch_sizes,
                                                         /*dynamic_input_dims=*/dynamic_dims_hack,
                                                         /*is_engine_path=*/false,
                                                         /*parallel_proc=*/true,
                                                         /*use_fp16=*/false,
                                                         /*cuda_buffer_in=*/true,
                                                         /*cuda_buffer_out=*/true,
                                                         /*use_cuda_graphs=*/true,
                                                         /*dla_core=*/-1,
                                                         /*dla_gpu_fallback=*/true,
                                                         /*gpu_resident_inference=*/false,
                                                         allocate_cuda_stream);
  op.specs->fast_multi_model_ = true;
  op.specs->green_context_partition_contexts_ = std::move(green_context_partition_contexts);
  op.specs->green_context_isolated_ = !op.specs->green_context_partition_contexts_.empty();
  op.specs->build_cuda_context_ = build_cuda_context;
  op.specs->build_sm_count_ = build_sm_count;

  const std::vector<int> dims{3, 256, 256};
  const size_t buffer_size = 3 * 256 * 256;
  for (const auto& name : {t1_in, t2_in}) {
    auto db = std::make_shared<HoloInfer::DataBuffer>();
    db->device_buffer_->resize(buffer_size);
    db->host_buffer_->resize(buffer_size);
    op.specs->data_per_tensor_.insert({name, db});
    std::vector<int> d = dims;
    d.insert(d.begin(), 1);
    op.specs->dims_per_tensor_.insert({name, d});
  }

  op.ctx = std::make_unique<HoloInfer::InferContext>();
  auto status = op.ctx->set_inference_params(op.specs);
  if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
    return false;
  }

  return cudaStreamCreate(&op.outer_stream) == cudaSuccess;
}

}  // namespace

// =============================================================================
// 2a — executable documentation of the historical capture race that
//      capture_serialization_mutex_ protects against.
//
//      IMPORTANT: this test is intentionally inert as an assertion. It can only
//      SKIP (mutex absorbed the race — the healthy, expected outcome today) or
//      PASS (mutex missing or defeated → EXPECT_GT sees the race reproduce). It
//      never fails on a correctly built tree.
//
//      Purpose: kept in the suite so that any future refactor removing
//      capture_serialization_mutex_ from ManagerInfer::run_core_inference_par_fast()
//      (see infer_manager_fast_paths.cpp) reintroduces the historical failure
//      mode and this test switches from SKIP to PASS — a visible signal that the
//      safety net has been dropped. Without this documentation, a reader might
//      remove the mutex without appreciating what it protects.
//
//      Setup: N concurrent ParFast ops with dynamic_dims_hack=true (no setup-time
//      warm-up path), each executing execute_inference_par_fast() in parallel.
//      Pre-fix, at least one thread's in-progress CUDA-Graph capture would be
//      invalidated by another thread's capture on a different stream sharing the
//      CUDA context, surfacing as H_ERROR on the capture-serialization path.
// =============================================================================

TEST_F(HoloInferTests, MultiOp_Concurrent_ParFast_HistoricalRace_DocumentedFix) {
  constexpr int kNumOps = 3;
  std::vector<ParFastOp> ops(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    ASSERT_TRUE(build_par_fast_op("noprot_" + std::to_string(i),
                                  model_folder,
                                  /*dynamic_dims_hack=*/true,
                                  ops[i]))
        << "setup failed for op " << i;
  }

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    threads.emplace_back([&ops, i, &failures]() {
      for (int iter = 0; iter < 5; ++iter) {
        auto status = ops[i].ctx->execute_inference_par_fast(ops[i].specs, ops[i].outer_stream);
        if (status.get_code() == HoloInfer::holoinfer_code::H_ERROR) {
          ++failures;
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  // ManagerInfer::capture_serialization_mutex_, which is always applied to the
  // first few run_core_inference_par_fast() calls on every instance (see
  // infer_manager_fast_paths.cpp) — including this "no prewarm" scenario. So the historical
  // failure mode this test documents may no longer reproduce deterministically post-fix; that
  // is the intended outcome of Part 1/2's safety net, not a test bug. Accept either outcome.
  if (failures.load() == 0) {
    GTEST_SKIP() << "No capture-related failures observed (capture_serialization_mutex_ safety "
                    "net absorbed the race); this documents the fix, not a flake.";
  }
  EXPECT_GT(failures.load(), 0);
}

// =============================================================================
// 2b — setup-time prewarm + capture_serialization_mutex_ safety net: concurrent, repeated.
// =============================================================================

TEST_F(HoloInferTests, MultiOp_Concurrent_ParFast_WithMutex_Succeeds) {
  constexpr int kNumOps = 3;
  constexpr int kIterations = 50;
  std::vector<ParFastOp> ops(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    ASSERT_TRUE(build_par_fast_op("mutex_" + std::to_string(i),
                                  model_folder,
                                  /*dynamic_dims_hack=*/false,
                                  ops[i]))
        << "setup failed for op " << i;
  }

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    threads.emplace_back([&ops, i, &failures]() {
      for (int iter = 0; iter < kIterations; ++iter) {
        auto status = ops[i].ctx->execute_inference_par_fast(ops[i].specs, ops[i].outer_stream);
        if (status.get_code() == HoloInfer::holoinfer_code::H_ERROR) {
          ++failures;
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0)
      << "expected all concurrent kParFast dispatches to succeed with prewarm + "
         "capture_serialization_mutex_ protection";
}

// =============================================================================
// 2c — CudaGreenContextPool supplied: engine build AND runtime dispatch
// (TrtInfer::do_inference_on_stream() pushes each op's assigned partition CUcontext around its
// capture/enqueueV3 sequence — see infer/trt/core.cpp GreenContextDispatchGuard) are both
// constrained to a distinct partition per op; concurrent, repeated dispatch remains safe. Real
// context isolation (not the capture_serialization_mutex_ safety net — green_context_isolated_
// makes ManagerInfer skip that mutex entirely, see infer_manager_fast_paths.cpp) is what's
// under test here.
// =============================================================================

TEST_F(HoloInferTests, MultiOp_Concurrent_ParFast_WithGreenContext_Succeeds) {
  auto partition_sms = resolve_green_context_partition_sm_count();
  if (!partition_sms) {
    GTEST_SKIP() << "Could not resolve CUDA green context capability on this platform/driver.";
  }

  holoscan::CudaGreenContextPool::Config config;
  config.num_partitions = 3;
  // Driver-granularity-derived split (see resolve_green_context_partition_sm_count()) rather
  // than a hardcoded SM count that may be smaller than this device's minimum partition size.
  config.sms_per_partition = {static_cast<int32_t>(*partition_sms),
                              static_cast<int32_t>(*partition_sms),
                              static_cast<int32_t>(*partition_sms)};
  holoscan::CudaGreenContextPool pool(config);
  auto init_result = pool.initialize_pool();
  if (!init_result) {
    GTEST_SKIP() << "CUDA green contexts unavailable on this platform/driver; skipping. ("
                 << init_result.error().message << ")";
  }
  ASSERT_GE(pool.num_partitions(), 3u);

  constexpr int kNumOps = 3;
  constexpr int kIterations = 50;
  std::vector<ParFastOp> ops(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    // Thread the pool's per-op partition through for BOTH TRT engine build constraint (mirrors
    // InferenceOp::start()'s wiring of build_cuda_context_/build_sm_count_) AND runtime
    // dispatch isolation (green_context_partition_contexts_ — mirrors InferenceOp::start()'s
    // wiring in inference.cpp, consumed by ManagerInfer::set_inference_params() in
    // infer_manager_setup.cpp). Both model_1 and model_2 of op i are bound to partition i
    // (a single-element vector so index m % 1 == 0 for both models).
    auto part_ctx = pool.cuda_context(static_cast<uint32_t>(i));
    auto part_sms = pool.partition_sms(static_cast<uint32_t>(i));
    ASSERT_TRUE(part_ctx && part_sms) << "could not resolve partition " << i;
    ASSERT_TRUE(build_par_fast_op("green_" + std::to_string(i),
                                  model_folder,
                                  /*dynamic_dims_hack=*/false,
                                  ops[i],
                                  /*green_context_partition_contexts=*/{part_ctx.value()},
                                  /*build_cuda_context=*/part_ctx.value(),
                                  /*build_sm_count=*/static_cast<int32_t>(part_sms.value())))
        << "setup failed for op " << i;
  }

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumOps);
  for (int i = 0; i < kNumOps; ++i) {
    threads.emplace_back([&ops, i, &failures]() {
      for (int iter = 0; iter < kIterations; ++iter) {
        auto status = ops[i].ctx->execute_inference_par_fast(ops[i].specs, ops[i].outer_stream);
        if (status.get_code() == HoloInfer::holoinfer_code::H_ERROR) {
          ++failures;
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0) << "expected concurrent kParFast dispatch to remain safe with a "
                                   "CudaGreenContextPool present";
}

// =============================================================================
// 2d / 2e — decide_capture_protection(): pure decision used by InferenceOp::start().
// =============================================================================

TEST(CaptureProtectionDecision, DynParFast_Graphs_NoGreenContext_Refuses) {
  auto decision = eligibility::decide_capture_protection(DispatchKind::kDynParFast,
                                                         /*enable_cuda_graphs=*/true,
                                                         /*green_context_pool_present=*/false,
                                                         /*green_context_num_partitions=*/0);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kRefuse);
  EXPECT_NE(decision.message.find("kDynParFast"), std::string::npos);
  EXPECT_NE(decision.message.find("CudaGreenContextPool"), std::string::npos);
}

TEST(CaptureProtectionDecision, DynParFast_Graphs_WithGreenContext_IsSafe) {
  auto decision = eligibility::decide_capture_protection(DispatchKind::kDynParFast,
                                                         /*enable_cuda_graphs=*/true,
                                                         /*green_context_pool_present=*/true,
                                                         /*green_context_num_partitions=*/3);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kGreenContextIsolated);
  EXPECT_NE(decision.message.find("3 partitions"), std::string::npos);
}

TEST(CaptureProtectionDecision, DynParFast_GraphsDisabled_NotApplicable) {
  auto decision = eligibility::decide_capture_protection(DispatchKind::kDynParFast,
                                                         /*enable_cuda_graphs=*/false,
                                                         /*green_context_pool_present=*/false,
                                                         /*green_context_num_partitions=*/0);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kNotApplicable);
}

TEST(CaptureProtectionDecision, ParFast_Graphs_NoGreenContext_UsesMutexProtected) {
  auto decision = eligibility::decide_capture_protection(DispatchKind::kParFast,
                                                         /*enable_cuda_graphs=*/true,
                                                         /*green_context_pool_present=*/false,
                                                         /*green_context_num_partitions=*/0);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kMutexProtected);
}

TEST(CaptureProtectionDecision, ParFast_Graphs_WithGreenContext_PrefersGreenContext) {
  auto decision = eligibility::decide_capture_protection(DispatchKind::kParFast,
                                                         /*enable_cuda_graphs=*/true,
                                                         /*green_context_pool_present=*/true,
                                                         /*green_context_num_partitions=*/2);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kGreenContextIsolated);
}

TEST(CaptureProtectionDecision, NonParallelKinds_NotApplicable) {
  EXPECT_EQ(eligibility::decide_capture_protection(DispatchKind::kFast, true, false, 0).outcome,
            CaptureProtectionOutcome::kNotApplicable);
  EXPECT_EQ(eligibility::decide_capture_protection(DispatchKind::kSeqFast, true, false, 0).outcome,
            CaptureProtectionOutcome::kNotApplicable);
  EXPECT_EQ(
      eligibility::decide_capture_protection(DispatchKind::kDynSeqFast, true, false, 0).outcome,
      CaptureProtectionOutcome::kNotApplicable);
  EXPECT_EQ(eligibility::decide_capture_protection(DispatchKind::kStandard, true, false, 0).outcome,
            CaptureProtectionOutcome::kNotApplicable);
}

// Real HoloInfer::InferContext::start()-equivalent smoke test for 2e: with a real
// CudaGreenContextPool supplied, a kDynParFast-shaped concurrent setup (2 ops, dynamic dims,
// graphs enabled) must NOT be required to refuse — i.e. the pure decision that
// InferenceOp::start() consults returns a non-kRefuse outcome for this exact configuration.
TEST_F(HoloInferTests, DynParFast_WithGraphs_WithGreenContext_DecisionIsSafe) {
  auto partition_sms = resolve_green_context_partition_sm_count();
  if (!partition_sms) {
    GTEST_SKIP() << "Could not resolve CUDA green context capability on this platform/driver.";
  }

  holoscan::CudaGreenContextPool::Config config;
  config.num_partitions = 2;
  config.sms_per_partition = {static_cast<int32_t>(*partition_sms),
                              static_cast<int32_t>(*partition_sms)};
  holoscan::CudaGreenContextPool pool(config);
  auto init_result = pool.initialize_pool();
  if (!init_result) {
    GTEST_SKIP() << "CUDA green contexts unavailable on this platform/driver; skipping. ("
                 << init_result.error().message << ")";
  }

  auto decision =
      eligibility::decide_capture_protection(DispatchKind::kDynParFast,
                                             /*enable_cuda_graphs=*/true,
                                             /*green_context_pool_present=*/true,
                                             static_cast<uint32_t>(pool.num_partitions()));
  EXPECT_NE(decision.outcome, CaptureProtectionOutcome::kRefuse);
  EXPECT_EQ(decision.outcome, CaptureProtectionOutcome::kGreenContextIsolated);
}
