/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests for green-context-aware TRT engine building.
//
// The feature ensures that when a CudaGreenContext (SM partition) is attached to
// InferenceOp, TRT engine building runs within that partition so tactic selection
// and kernel timing reflect the actual SM budget rather than the full GPU.
//
// What these tests cover
// ─────────────────────
// 1. NetworkOptions: default field values (nullptr / 0).
// 2. InferenceSpecs: default build context fields (nullptr / 0).
// 3. generate_engine_path: embeds build_sm_count in filename when non-zero.
// 4. generate_engine_path: uses full GPU SM count when build_sm_count == 0.
// 5. generate_engine_path: partitioned and full-GPU paths differ.
// 6. Full engine build with a real green context (skipped without CUDA 12.4+).

#include "test_core.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

// Green context APIs (CUgreenCtx, CUdevResourceDesc, CUdevResource,
// CU_GREEN_CTX_DEFAULT_STREAM, cuGreenCtxCreate, cuDevSmResourceSplitByCount, ...)
// were introduced in CUDA 12.4.  On older toolkits these symbols do not exist in
// <cuda.h>, so the entire file is skipped at compile time -- the runtime driver
// check (load_green_ctx_symbols) still applies at execution time as well.
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040

// Nothing to compile on pre-12.4 toolkits.  A translation unit needs at least one
// external symbol so linkers on some platforms don't complain about an empty .o.
namespace {
[[maybe_unused]] void holoinfer_green_context_build_test_skipped_pre_cuda_12_4() {}
}  // namespace

#else

// NetworkOptions is in the TRT-internal header; exposed via the test include path
// (modules/inference/src/infer/trt is added to HOLOINFER_TEST's include dirs).
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <utils.hpp>

namespace HoloInfer = holoscan::inference;

// ─────────────────────────────────────────────────────────────────────────────
// 1 + 2  Field defaults
// ─────────────────────────────────────────────────────────────────────────────

TEST(GreenContextBuildTest, NetworkOptions_DefaultBuildContextIsNull) {
  const HoloInfer::NetworkOptions opts;
  EXPECT_EQ(opts.build_cuda_context, nullptr);
}

TEST(GreenContextBuildTest, NetworkOptions_DefaultBuildSmCountIsZero) {
  const HoloInfer::NetworkOptions opts;
  EXPECT_EQ(opts.build_sm_count, 0);
}

TEST(GreenContextBuildTest, InferenceSpecs_DefaultBuildContextIsNull) {
  const HoloInfer::InferenceSpecs specs;
  EXPECT_EQ(specs.build_cuda_context_, nullptr);
}

TEST(GreenContextBuildTest, InferenceSpecs_DefaultBuildSmCountIsZero) {
  const HoloInfer::InferenceSpecs specs;
  EXPECT_EQ(specs.build_sm_count_, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3 – 5  Engine path generation
// All generate_engine_path tests require a GPU (cudaGetDeviceProperties).
// ─────────────────────────────────────────────────────────────────────────────

TEST(GreenContextBuildTest, EnginePath_BuildSmCountEmbeddedWhenNonZero) {
  HoloInfer::NetworkOptions opts;
  opts.device_index = 0;
  opts.use_fp16 = false;
  opts.build_sm_count = 16;

  std::string engine_path;
  const bool ok = HoloInfer::generate_engine_path(opts, "model.onnx", engine_path);
  ASSERT_TRUE(ok);

  // The SM count segment must be "16" (not the GPU's total SM count).
  EXPECT_NE(engine_path.find(".16.trt."), std::string::npos) << "engine path: " << engine_path;
}

TEST(GreenContextBuildTest, EnginePath_FullGpuSmCountUsedWhenBuildSmCountIsZero) {
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  const int full_sms = prop.multiProcessorCount;

  HoloInfer::NetworkOptions opts;
  opts.device_index = 0;
  opts.use_fp16 = false;
  opts.build_sm_count = 0;

  std::string engine_path;
  const bool ok = HoloInfer::generate_engine_path(opts, "model.onnx", engine_path);
  ASSERT_TRUE(ok);

  const std::string sm_segment = "." + std::to_string(full_sms) + ".trt.";
  EXPECT_NE(engine_path.find(sm_segment), std::string::npos)
      << "expected '" << sm_segment << "' in engine path: " << engine_path;
}

TEST(GreenContextBuildTest, EnginePath_PartitionedAndFullGpuPathsDiffer) {
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  const int full_sms = prop.multiProcessorCount;

  // Need at least 4 SMs to form a meaningful non-trivial partition.
  if (full_sms < 4) {
    GTEST_SKIP() << "GPU has fewer than 4 SMs; skipping path-difference test.";
  }

  HoloInfer::NetworkOptions opts_full;
  opts_full.device_index = 0;
  opts_full.build_sm_count = 0;  // full GPU

  HoloInfer::NetworkOptions opts_part;
  opts_part.device_index = 0;
  opts_part.build_sm_count = full_sms / 2;  // simulated partition

  std::string path_full, path_part;
  ASSERT_TRUE(HoloInfer::generate_engine_path(opts_full, "model.onnx", path_full));
  ASSERT_TRUE(HoloInfer::generate_engine_path(opts_part, "model.onnx", path_part));

  EXPECT_NE(path_full, path_part)
      << "full-GPU and partitioned engine paths must differ so cached engines do not collide";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6  Full build within a real green context
//
// Requires CUDA 12.4+ driver with green context support.  The test creates a
// green context pool that splits the device SMs in half, builds a TRT engine
// using that context, and verifies:
//   (a) the engine file is created,
//   (b) the engine path contains the partition SM count (not the full count).
// ─────────────────────────────────────────────────────────────────────────────

// Helper: dynamically resolve cuCtxFromGreenCtx (CUDA 12.4+, not present on
// older drivers).
static CUresult (*fn_cuCtxFromGreenCtx)(CUcontext*, CUgreenCtx) = nullptr;
static CUresult (*fn_cuGreenCtxCreate)(CUgreenCtx*, CUdevResourceDesc, CUdevice,
                                       unsigned int) = nullptr;
static CUresult (*fn_cuGreenCtxDestroy)(CUgreenCtx) = nullptr;
static CUresult (*fn_cuDevSmResourceSplitByCount)(CUdevResource*, unsigned int*,
                                                  const CUdevResource*, CUdevResource*,
                                                  unsigned int, unsigned int) = nullptr;
static CUresult (*fn_cuDevResourceGenerateDesc)(CUdevResourceDesc*, CUdevResource*,
                                                unsigned int) = nullptr;
static CUresult (*fn_cuDeviceGetDevResource)(CUdevice, CUdevResource*, CUdevResourceType) = nullptr;

static bool load_green_ctx_symbols() {
  // cuInit is needed before cuGetProcAddress
  if (cuInit(0) != CUDA_SUCCESS) {
    return false;
  }

  int driver_ver = 0;
  if (cuDriverGetVersion(&driver_ver) != CUDA_SUCCESS) {
    return false;
  }
  // Green context APIs require CUDA 12.4 (driver version 12400)
  if (driver_ver < 12040) {
    return false;
  }

  auto load = [&](const char* name, void** fn_ptr) -> bool {
    return cuGetProcAddress(name, fn_ptr, driver_ver, 0, nullptr) == CUDA_SUCCESS &&
           *fn_ptr != nullptr;
  };
  return load("cuCtxFromGreenCtx", reinterpret_cast<void**>(&fn_cuCtxFromGreenCtx)) &&
         load("cuGreenCtxCreate", reinterpret_cast<void**>(&fn_cuGreenCtxCreate)) &&
         load("cuGreenCtxDestroy", reinterpret_cast<void**>(&fn_cuGreenCtxDestroy)) &&
         load("cuDevSmResourceSplitByCount",
              reinterpret_cast<void**>(&fn_cuDevSmResourceSplitByCount)) &&
         load("cuDevResourceGenerateDesc",
              reinterpret_cast<void**>(&fn_cuDevResourceGenerateDesc)) &&
         load("cuDeviceGetDevResource", reinterpret_cast<void**>(&fn_cuDeviceGetDevResource));
}

TEST(GreenContextBuildTest, EngineBuiltWithinGreenContextPartition) {
  if (!load_green_ctx_symbols()) {
    GTEST_SKIP() << "CUDA driver < 12.4 or green context symbols unavailable; skipping.";
  }

  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  const int full_sms = prop.multiProcessorCount;
  // Align down to nearest multiple of 4 (minimum SM partition granularity)
  const int aligned_total = full_sms & ~3;
  if (aligned_total < 8) {
    GTEST_SKIP() << "GPU has fewer than 8 aligned SMs; cannot create two non-trivial partitions.";
  }
  const int partition_sms = std::max(4, (aligned_total / 2) & ~3);

  // ── Create a green context ────────────────────────────────────────────────
  CUdevice cu_device = 0;
  ASSERT_EQ(cuDeviceGet(&cu_device, 0), CUDA_SUCCESS);

  // Get the SM resource for the whole device
  CUdevResource dev_sm_resource{};
  ASSERT_EQ(fn_cuDeviceGetDevResource(cu_device, &dev_sm_resource, CU_DEV_RESOURCE_TYPE_SM),
            CUDA_SUCCESS);

  // Split: ask for one partition of partition_sms SMs
  CUdevResource partition_resource{};
  CUdevResource remainder_resource{};
  unsigned int nbGroups = 1;
  ASSERT_EQ(fn_cuDevSmResourceSplitByCount(&partition_resource,
                                           &nbGroups,
                                           &dev_sm_resource,
                                           &remainder_resource,
                                           static_cast<unsigned int>(partition_sms),
                                           0),
            CUDA_SUCCESS)
      << "cuDevSmResourceSplitByCount failed for " << partition_sms << " SMs";

  CUdevResourceDesc desc{};
  ASSERT_EQ(fn_cuDevResourceGenerateDesc(&desc, &partition_resource, 1), CUDA_SUCCESS);

  CUgreenCtx green_ctx = nullptr;
  // CU_GREEN_CTX_DEFAULT_STREAM (0x1) is required by cuGreenCtxCreate; passing 0
  // returns CUDA_ERROR_INVALID_VALUE regardless of every other parameter.
  ASSERT_EQ(fn_cuGreenCtxCreate(&green_ctx, desc, cu_device, CU_GREEN_CTX_DEFAULT_STREAM),
            CUDA_SUCCESS);

  CUcontext build_ctx = nullptr;
  ASSERT_EQ(fn_cuCtxFromGreenCtx(&build_ctx, green_ctx), CUDA_SUCCESS);

  // ── Build an engine using the green context ───────────────────────────────
  // Copy the ONNX model into a unique per-test-invocation directory under the
  // test's own temp path so this test cannot collide with the shared
  // modules/inference/tests/test_models/ engines cleaned up by
  // HoloInferTests::cleanup_engines() in test_core.cpp when run in parallel.
  const std::string src_onnx = "modules/inference/tests/test_models/identity_model.onnx";
  const std::filesystem::path unique_dir =
      std::filesystem::temp_directory_path() /
      ("holoinfer_gc_build_" + std::to_string(::getpid()) + "_" +
       std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  std::filesystem::create_directories(unique_dir);
  const std::filesystem::path onnx_copy = unique_dir / "identity_model.onnx";
  std::error_code copy_ec;
  std::filesystem::copy_file(
      src_onnx, onnx_copy, std::filesystem::copy_options::overwrite_existing, copy_ec);
  ASSERT_FALSE(copy_ec) << "failed to stage ONNX to unique dir: " << copy_ec.message();
  const std::string onnx_path = onnx_copy.string();

  HoloInfer::NetworkOptions opts;
  opts.device_index = 0;
  opts.use_fp16 = false;
  opts.build_cuda_context = build_ctx;
  opts.build_sm_count = partition_sms;

  std::string engine_path;
  ASSERT_TRUE(HoloInfer::generate_engine_path(opts, onnx_path, engine_path));

  HoloInfer::Logger logger;
  const bool built = HoloInfer::build_engine(onnx_path, engine_path, opts, logger);
  EXPECT_TRUE(built) << "build_engine failed with green context (" << partition_sms << " SMs)";

  if (built) {
    // Engine file must exist
    EXPECT_TRUE(std::filesystem::exists(engine_path)) << "engine file missing at: " << engine_path;

    // Engine filename must embed the partition SM count, not the full GPU count
    const std::string part_seg = "." + std::to_string(partition_sms) + ".trt.";
    const std::string full_seg = "." + std::to_string(full_sms) + ".trt.";
    EXPECT_NE(engine_path.find(part_seg), std::string::npos)
        << "partition SM count (" << partition_sms << ") not found in: " << engine_path;
    if (partition_sms != full_sms) {
      EXPECT_EQ(engine_path.find(full_seg), std::string::npos)
          << "full GPU SM count (" << full_sms
          << ") should NOT appear in partitioned path: " << engine_path;
    }
  }

  // Remove only this test's own temp tree (never touches the shared test_models/).
  std::error_code cleanup_ec;
  std::filesystem::remove_all(unique_dir, cleanup_ec);

  // Cleanup green context (best effort — ignore errors in teardown)
  cuCtxDestroy(build_ctx);
  fn_cuGreenCtxDestroy(green_ctx);
}

#endif  // CUDA_VERSION >= 12040
