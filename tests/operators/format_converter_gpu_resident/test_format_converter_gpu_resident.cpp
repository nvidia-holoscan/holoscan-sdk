/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <npp.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter_gpu_resident/format_converter_gpu_resident.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace holoscan {

static constexpr int32_t kTestWidth = 64;
static constexpr int32_t kTestHeight = 64;

// ================================================================================================
// Generic source / sink operators for the GPU-resident test pipeline
// ================================================================================================

class FmtSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FmtSourceGpuOp, GPUResidentOperator)
  FmtSourceGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(mem_size_,
               "mem_size",
               "Memory size",
               "Output buffer size in bytes",
               static_cast<size_t>(0));
    spec.device_output("out", 0);
  }

  void initialize() override {
    GPUResidentOperator::initialize();
    spec()->device_output("out", mem_size_.get());
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  Parameter<size_t> mem_size_;
};

class FmtSinkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(FmtSinkGpuOp, GPUResidentOperator)
  FmtSinkGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        mem_size_, "mem_size", "Memory size", "Input buffer size in bytes", static_cast<size_t>(0));
    spec.device_input("in", 0);
  }

  void initialize() override {
    GPUResidentOperator::initialize();
    spec()->device_input("in", mem_size_.get());
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {}

 private:
  Parameter<size_t> mem_size_;
};

// ================================================================================================
// NPP-based reference conversions (standalone, no operator overhead)
// ================================================================================================

static void ref_rgb888_to_rgba8888(const uint8_t* src_dev, uint8_t* dst_dev, int32_t w, int32_t h,
                                   uint8_t alpha, cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  int src_step = w * 3;
  int dst_step = w * 4;
  NppiSize roi = {w, h};
  int order[4] = {0, 1, 2, 3};
  nppiSwapChannels_8u_C3C4R_Ctx(src_dev, src_step, dst_dev, dst_step, roi, order, alpha, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "ref_rgb888_to_rgba8888 sync");
}

static void ref_rgba8888_to_rgb888(const uint8_t* src_dev, uint8_t* dst_dev, int32_t w, int32_t h,
                                   cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  int src_step = w * 4;
  int dst_step = w * 3;
  NppiSize roi = {w, h};
  int order[3] = {0, 1, 2};
  nppiSwapChannels_8u_C4C3R_Ctx(src_dev, src_step, dst_dev, dst_step, roi, order, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "ref_rgba8888_to_rgb888 sync");
}

static void ref_uint8_to_float32(const uint8_t* src_dev, float* dst_dev, int32_t w, int32_t h,
                                 float scale_min, float scale_max, cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  int src_step = w * 3;
  int dst_step = w * 3 * static_cast<int>(sizeof(float));
  NppiSize roi = {w, h};
  nppiScale_8u32f_C3R_Ctx(src_dev, src_step, dst_dev, dst_step, roi, scale_min, scale_max, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "ref_uint8_to_float32 sync");
}

static void ref_float32_to_uint8(const float* src_dev, uint8_t* dst_dev, int32_t w, int32_t h,
                                 float scale_min, float scale_max, cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  int src_step = w * 3 * static_cast<int>(sizeof(float));
  int dst_step = w * 3;
  NppiSize roi = {w, h};
  nppiScale_32f8u_C3R_Ctx(src_dev, src_step, dst_dev, dst_step, roi, scale_min, scale_max, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "ref_float32_to_uint8 sync");
}

static void ref_rgba8888_to_float32(const uint8_t* src_dev, float* dst_dev, int32_t w, int32_t h,
                                    float scale_min, float scale_max, cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  int src_step = w * 4;
  NppiSize roi = {w, h};
  int order[3] = {0, 1, 2};
  int ch3_step = w * 3;
  std::vector<uint8_t> tmp(static_cast<size_t>(w) * h * 3);
  uint8_t* tmp_dev = nullptr;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMallocAsync(reinterpret_cast<void**>(&tmp_dev), tmp.size(), stream),
      "ref_rgba8888_to_float32 tmp buffer");
  nppiSwapChannels_8u_C4C3R_Ctx(src_dev, src_step, tmp_dev, ch3_step, roi, order, ctx);
  int dst_step = w * 3 * static_cast<int>(sizeof(float));
  nppiScale_8u32f_C3R_Ctx(tmp_dev, ch3_step, dst_dev, dst_step, roi, scale_min, scale_max, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(tmp_dev, stream),
                                 "ref_rgba8888_to_float32 tmp free");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "ref_rgba8888_to_float32 sync");
}

static void ref_resize_rgb888_to_float32(const uint8_t* src_dev, float* dst_dev, int32_t src_w,
                                         int32_t src_h, int32_t dst_w, int32_t dst_h,
                                         float scale_min, float scale_max, cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  const NppiSize src_size = {src_w, src_h};
  const NppiRect src_roi = {0, 0, src_w, src_h};
  const NppiSize dst_size = {dst_w, dst_h};
  const NppiRect dst_roi = {0, 0, dst_w, dst_h};
  const int32_t src_step = src_w * 3;
  const int32_t resized_step = dst_w * 3;
  const int32_t dst_step = dst_w * 3 * static_cast<int32_t>(sizeof(float));

  uint8_t* resized_dev = nullptr;
  const size_t resized_bytes = static_cast<size_t>(dst_w) * dst_h * 3u;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMallocAsync(reinterpret_cast<void**>(&resized_dev), resized_bytes, stream),
      "ref_resize_rgb888_to_float32 resize buffer");
  nppiResize_8u_C3R_Ctx(src_dev,
                        src_step,
                        src_size,
                        src_roi,
                        resized_dev,
                        resized_step,
                        dst_size,
                        dst_roi,
                        NPPI_INTER_CUBIC,
                        ctx);
  nppiScale_8u32f_C3R_Ctx(
      resized_dev, resized_step, dst_dev, dst_step, dst_size, scale_min, scale_max, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(resized_dev, stream),
                                 "ref_resize_rgb888_to_float32 resize free");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream),
                                 "ref_resize_rgb888_to_float32 sync");
}

static void ref_resize_rgba8888_to_rgba8888(const uint8_t* src_dev, uint8_t* dst_dev, int32_t src_w,
                                            int32_t src_h, int32_t dst_w, int32_t dst_h,
                                            const int order[4], cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  const NppiSize src_size = {src_w, src_h};
  const NppiRect src_roi = {0, 0, src_w, src_h};
  const NppiSize dst_size = {dst_w, dst_h};
  const NppiRect dst_roi = {0, 0, dst_w, dst_h};
  const int32_t src_step = src_w * 4;
  const int32_t dst_step = dst_w * 4;

  nppiResize_8u_C4R_Ctx(src_dev,
                        src_step,
                        src_size,
                        src_roi,
                        dst_dev,
                        dst_step,
                        dst_size,
                        dst_roi,
                        NPPI_INTER_CUBIC,
                        ctx);
  nppiSwapChannels_8u_C4IR_Ctx(dst_dev, dst_step, dst_size, const_cast<int*>(order), ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream),
                                 "ref_resize_rgba8888_to_rgba8888 sync");
}

static void ref_resize_rgba8888_to_float32(const uint8_t* src_dev, float* dst_dev, int32_t src_w,
                                           int32_t src_h, int32_t dst_w, int32_t dst_h,
                                           const int order[3], float scale_min, float scale_max,
                                           cudaStream_t stream) {
  NppStreamContext ctx{};
  ctx.hStream = stream;
  const NppiSize src_size = {src_w, src_h};
  const NppiRect src_roi = {0, 0, src_w, src_h};
  const NppiSize dst_size = {dst_w, dst_h};
  const NppiRect dst_roi = {0, 0, dst_w, dst_h};
  const int32_t src_step = src_w * 4;
  const int32_t resized_step = dst_w * 4;
  const int32_t scratch_step = dst_w * 3;
  const int32_t dst_step = dst_w * 3 * static_cast<int32_t>(sizeof(float));

  uint8_t* resized_dev = nullptr;
  uint8_t* scratch_dev = nullptr;
  const size_t resized_bytes = static_cast<size_t>(dst_w) * dst_h * 4u;
  const size_t scratch_bytes = static_cast<size_t>(dst_w) * dst_h * 3u;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMallocAsync(reinterpret_cast<void**>(&resized_dev), resized_bytes, stream),
      "ref_resize_rgba8888_to_float32 resize buffer");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMallocAsync(reinterpret_cast<void**>(&scratch_dev), scratch_bytes, stream),
      "ref_resize_rgba8888_to_float32 scratch buffer");
  nppiResize_8u_C4R_Ctx(src_dev,
                        src_step,
                        src_size,
                        src_roi,
                        resized_dev,
                        resized_step,
                        dst_size,
                        dst_roi,
                        NPPI_INTER_CUBIC,
                        ctx);
  nppiSwapChannels_8u_C4C3R_Ctx(
      resized_dev, resized_step, scratch_dev, scratch_step, dst_size, const_cast<int*>(order), ctx);
  nppiScale_8u32f_C3R_Ctx(
      scratch_dev, scratch_step, dst_dev, dst_step, dst_size, scale_min, scale_max, ctx);
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(scratch_dev, stream),
                                 "ref_resize_rgba8888_to_float32 scratch free");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(resized_dev, stream),
                                 "ref_resize_rgba8888_to_float32 resize free");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream),
                                 "ref_resize_rgba8888_to_float32 sync");
}

// ================================================================================================
// Test fixture
// ================================================================================================

class FormatConverterGpuResidentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = HOLOSCAN_CUDA_CALL(cudaGetDeviceCount(&device_count));
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU-resident tests";
    }

    int device = 0;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&device), "cudaGetDevice");

    int memory_pools_supported = 0;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device),
        "cudaDeviceGetAttribute(cudaDevAttrMemoryPoolsSupported)");
    if (!memory_pools_supported) {
      GTEST_SKIP() << "CUDA device " << device
                   << " does not support stream-ordered memory allocation; "
                      "skipping format_converter_gpu_resident tests";
    }
  }

  static bool wait_for_graph_launch(Fragment& fragment, int timeout_sec = 5) {
    auto start = std::chrono::steady_clock::now();
    while (!fragment.gpu_resident().is_launched()) {
      if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(timeout_sec)) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return true;
  }

  static bool wait_for_result(Fragment& fragment, int max_checks = 10) {
    for (int i = 0; i < max_checks; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (fragment.gpu_resident().result_ready()) {
        return true;
      }
    }
    return false;
  }

  // Run a GPU-resident pipeline with random input and compare against a reference NPP conversion.
  // T_in  = host element type for input  (uint8_t or float)
  // T_out = host element type for output (uint8_t or float)
  template <typename T_in, typename T_out>
  void RunAndCompare(const std::string& in_dtype, const std::string& out_dtype, int32_t width,
                     int32_t height, int in_channels, int out_channels, size_t in_bytes,
                     size_t out_bytes, const std::vector<Arg>& extra_args,
                     std::function<void(const void* src_dev, void* dst_dev, int32_t w, int32_t h,
                                        cudaStream_t stream)>
                         ref_fn) {
    Fragment fragment;

    ArgList converter_args{
        Arg("width", width),
        Arg("height", height),
        Arg("in_dtype", in_dtype),
        Arg("out_dtype", out_dtype),
    };
    for (const auto& a : extra_args) {
      converter_args.add(a);
    }

    auto source = fragment.make_operator<FmtSourceGpuOp>("source", Arg("mem_size", in_bytes));
    auto converter =
        fragment.make_operator<ops::FormatConverterGpuResidentOp>("converter", converter_args);
    auto sink = fragment.make_operator<FmtSinkGpuOp>("sink", Arg("mem_size", out_bytes));

    fragment.add_flow(source, converter, {{"out", "in"}});
    fragment.add_flow(converter, sink, {{"out", "in"}});

    fragment.gpu_resident().timeout_ms(5000);
    fragment.gpu_resident().sync_with_host();

    auto future = fragment.run_async();
    ASSERT_TRUE(wait_for_graph_launch(fragment)) << "GPU-resident graph failed to launch";

    auto graph = fragment.graph_shared();
    auto* source_op = dynamic_cast<GPUResidentOperator*>(graph->find_node("source").get());
    auto* sink_op = dynamic_cast<GPUResidentOperator*>(graph->find_node("sink").get());
    ASSERT_NE(source_op, nullptr);
    ASSERT_NE(sink_op, nullptr);

    // Generate random input
    const size_t in_elem_count = in_bytes / sizeof(T_in);
    const size_t out_elem_count = out_bytes / sizeof(T_out);
    std::vector<T_in> host_input(in_elem_count);
    std::vector<T_out> host_output_gpu(out_elem_count);
    std::vector<T_out> host_output_ref(out_elem_count);

    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    for (size_t i = 0; i < in_elem_count; ++i) {
      if constexpr (std::is_same_v<T_in, uint8_t>) {
        host_input[i] = static_cast<uint8_t>(rand_r(&seed) % 256);
      } else if constexpr (std::is_same_v<T_in, float>) {
        host_input[i] = static_cast<float>(rand_r(&seed) % 256) / 255.0f;
      } else if constexpr (std::is_same_v<T_in, uint16_t>) {
        host_input[i] = static_cast<uint16_t>(rand_r(&seed) % 65536);
      }
    }

    // Reference buffers
    void* ref_in_dev = nullptr;
    void* ref_out_dev = nullptr;
    cudaStream_t ref_stream;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreate(&ref_stream), "ref cudaStreamCreate");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMallocAsync(&ref_in_dev, in_bytes, ref_stream),
                                   "cudaMallocAsync ref_in");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMallocAsync(&ref_out_dev, out_bytes, ref_stream),
                                   "cudaMallocAsync ref_out");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(ref_stream), "ref alloc stream sync");

    for (int iter = 0; iter < 2; ++iter) {
      for (size_t i = 0; i < in_elem_count; ++i) {
        if constexpr (std::is_same_v<T_in, uint8_t>) {
          host_input[i] = static_cast<uint8_t>(rand_r(&seed) % 256);
        } else if constexpr (std::is_same_v<T_in, float>) {
          host_input[i] = static_cast<float>(rand_r(&seed) % 256) / 255.0f;
        } else if constexpr (std::is_same_v<T_in, uint16_t>) {
          host_input[i] = static_cast<uint16_t>(rand_r(&seed) % 65536);
        }
      }

      void* src_addr = source_op->device_memory("out");
      ASSERT_NE(src_addr, nullptr);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpy(src_addr, host_input.data(), in_bytes, cudaMemcpyHostToDevice),
          "copy input to source");

      // Reference
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpyAsync(
              ref_in_dev, host_input.data(), in_bytes, cudaMemcpyHostToDevice, ref_stream),
          "copy input to ref");
      ref_fn(ref_in_dev, ref_out_dev, width, height, ref_stream);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpyAsync(
              host_output_ref.data(), ref_out_dev, out_bytes, cudaMemcpyDeviceToHost, ref_stream),
          "copy ref output");
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(ref_stream), "ref output sync");

      fragment.gpu_resident().data_ready();
      ASSERT_TRUE(wait_for_result(fragment)) << "result not ready, iteration " << iter;

      void* sink_addr = sink_op->device_memory("in");
      ASSERT_NE(sink_addr, nullptr);
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaMemcpy(host_output_gpu.data(), sink_addr, out_bytes, cudaMemcpyDeviceToHost),
          "copy GPU output");

      for (size_t i = 0; i < out_elem_count; ++i) {
        if constexpr (std::is_floating_point_v<T_out>) {
          ASSERT_NEAR(host_output_gpu[i], host_output_ref[i], 1e-5f)
              << "Mismatch at index " << i << " iter " << iter;
        } else {
          ASSERT_EQ(host_output_gpu[i], host_output_ref[i])
              << "Mismatch at index " << i << " iter " << iter;
        }
      }
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(ref_in_dev, ref_stream), "cudaFreeAsync ref_in");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeAsync(ref_out_dev, ref_stream), "cudaFreeAsync ref_out");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(ref_stream), "ref free stream sync");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamDestroy(ref_stream), "ref cudaStreamDestroy");
    fragment.gpu_resident().tear_down();
    future.get();
  }
};

// ================================================================================================
// Test: kNone  (uint8 rgb888 -> rgb888, identity copy)
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestNoneIdentityCopy) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t bytes = static_cast<size_t>(w) * h * 3;

  RunAndCompare<uint8_t, uint8_t>(
      "rgb888",
      "rgb888",
      w,
      h,
      3,
      3,
      bytes,
      bytes,
      {},
      [](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        HOLOSCAN_CUDA_CALL_THROW_ERROR(
            cudaMemcpyAsync(
                dst, src, static_cast<size_t>(w) * h * 3, cudaMemcpyDeviceToDevice, stream),
            "TestNoneIdentityCopy D2D copy");
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "TestNoneIdentityCopy sync");
      });
}

// ================================================================================================
// Test: kRGB888ToRGBA8888
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestRGB888ToRGBA8888) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 3;
  const size_t out_bytes = static_cast<size_t>(w) * h * 4;
  const uint8_t alpha = 200;

  RunAndCompare<uint8_t, uint8_t>(
      "rgb888",
      "rgba8888",
      w,
      h,
      3,
      4,
      in_bytes,
      out_bytes,
      {Arg("alpha_value", alpha)},
      [alpha](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        ref_rgb888_to_rgba8888(
            static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), w, h, alpha, stream);
      });
}

// ================================================================================================
// Test: kRGBA8888ToRGB888
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestRGBA8888ToRGB888) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 4;
  const size_t out_bytes = static_cast<size_t>(w) * h * 3;

  RunAndCompare<uint8_t, uint8_t>(
      "rgba8888",
      "rgb888",
      w,
      h,
      4,
      3,
      in_bytes,
      out_bytes,
      {},
      [](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        ref_rgba8888_to_rgb888(
            static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), w, h, stream);
      });
}

// ================================================================================================
// Test: kUnsigned8ToFloat32
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestUnsigned8ToFloat32) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 3;
  const size_t out_bytes = static_cast<size_t>(w) * h * 3 * sizeof(float);

  RunAndCompare<uint8_t, float>(
      "rgb888",
      "float32",
      w,
      h,
      3,
      3,
      in_bytes,
      out_bytes,
      {},
      [](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        ref_uint8_to_float32(
            static_cast<const uint8_t*>(src), static_cast<float*>(dst), w, h, 0.f, 1.f, stream);
      });
}

// ================================================================================================
// Test: kFloat32ToUnsigned8
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestFloat32ToUnsigned8) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 3 * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(w) * h * 3;

  RunAndCompare<float, uint8_t>(
      "float32",
      "rgb888",
      w,
      h,
      3,
      3,
      in_bytes,
      out_bytes,
      {},
      [](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        ref_float32_to_uint8(
            static_cast<const float*>(src), static_cast<uint8_t*>(dst), w, h, 0.f, 1.f, stream);
      });
}

// ================================================================================================
// Test: kRGBA8888ToFloat32 (4-channel uint8 -> 3-channel float32 via channel scratch)
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestRGBA8888ToFloat32) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 4;
  const size_t out_bytes = static_cast<size_t>(w) * h * 3 * sizeof(float);

  RunAndCompare<uint8_t, float>(
      "rgba8888",
      "float32",
      w,
      h,
      4,
      3,
      in_bytes,
      out_bytes,
      {},
      [](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        ref_rgba8888_to_float32(
            static_cast<const uint8_t*>(src), static_cast<float*>(dst), w, h, 0.f, 1.f, stream);
      });
}

// ================================================================================================
// Test: kRGB888ToRGBA8888 with out_channel_order (BGR->BGRA permutation)
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestRGB888ToRGBA8888WithChannelOrder) {
  const int32_t w = kTestWidth, h = kTestHeight;
  const size_t in_bytes = static_cast<size_t>(w) * h * 3;
  const size_t out_bytes = static_cast<size_t>(w) * h * 4;
  const uint8_t alpha = 255;
  std::vector<int> channel_order = {2, 1, 0, 3};

  RunAndCompare<uint8_t, uint8_t>(
      "rgb888",
      "rgba8888",
      w,
      h,
      3,
      4,
      in_bytes,
      out_bytes,
      {Arg("alpha_value", alpha), Arg("out_channel_order", channel_order)},
      [alpha](const void* src, void* dst, int32_t w, int32_t h, cudaStream_t stream) {
        NppStreamContext ctx{};
        ctx.hStream = stream;
        int src_step = w * 3;
        int dst_step = w * 4;
        NppiSize roi = {w, h};
        int order[4] = {2, 1, 0, 3};
        nppiSwapChannels_8u_C3C4R_Ctx(static_cast<const uint8_t*>(src),
                                      src_step,
                                      static_cast<uint8_t*>(dst),
                                      dst_step,
                                      roi,
                                      order,
                                      alpha,
                                      ctx);
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream),
                                       "TestRGB888ToRGBA8888WithChannelOrder sync");
      });
}

// ================================================================================================
// Test: resize + identity (uint8 rgb888 -> rgb888 with resize)
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestResizeIdentity) {
  const int32_t src_w = 128, src_h = 128;
  const int32_t dst_w = 64, dst_h = 64;
  const size_t in_bytes = static_cast<size_t>(src_w) * src_h * 3;
  const size_t out_bytes = static_cast<size_t>(dst_w) * dst_h * 3;

  RunAndCompare<uint8_t, uint8_t>(
      "rgb888",
      "rgb888",
      src_w,
      src_h,
      3,
      3,
      in_bytes,
      out_bytes,
      {Arg("resize_width", dst_w), Arg("resize_height", dst_h)},
      [src_w, src_h, dst_w, dst_h](
          const void* src, void* dst, int32_t /*w*/, int32_t /*h*/, cudaStream_t stream) {
        NppStreamContext ctx{};
        ctx.hStream = stream;
        NppiSize src_size = {src_w, src_h};
        NppiRect src_roi = {0, 0, src_w, src_h};
        NppiSize dst_size = {dst_w, dst_h};
        NppiRect dst_roi = {0, 0, dst_w, dst_h};
        int src_step = src_w * 3;
        int dst_step = dst_w * 3;
        nppiResize_8u_C3R_Ctx(static_cast<const uint8_t*>(src),
                              src_step,
                              src_size,
                              src_roi,
                              static_cast<uint8_t*>(dst),
                              dst_step,
                              dst_size,
                              dst_roi,
                              NPPI_INTER_CUBIC,
                              ctx);
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(stream), "TestResizeIdentity sync");
      });
}

// ================================================================================================
// Test: resize + rgb888 -> float32
//       Used by the HoloHub ultrasound segmentation preprocessor.
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestResizeRGB888ToFloat32) {
  const int32_t src_w = 128, src_h = 96;
  const int32_t dst_w = 64, dst_h = 48;
  const size_t in_bytes = static_cast<size_t>(src_w) * src_h * 3;
  const size_t out_bytes = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);

  RunAndCompare<uint8_t, float>(
      "rgb888",
      "float32",
      src_w,
      src_h,
      3,
      3,
      in_bytes,
      out_bytes,
      {Arg("resize_width", dst_w), Arg("resize_height", dst_h)},
      [src_w, src_h, dst_w, dst_h](
          const void* src, void* dst, int32_t /*w*/, int32_t /*h*/, cudaStream_t stream) {
        ref_resize_rgb888_to_float32(static_cast<const uint8_t*>(src),
                                     static_cast<float*>(dst),
                                     src_w,
                                     src_h,
                                     dst_w,
                                     dst_h,
                                     0.f,
                                     1.f,
                                     stream);
      });
}

// ================================================================================================
// Test: resize + rgba8888 -> rgba8888 with BGRA channel order
//       Used by the HoloHub deltacast overlay format converter.
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestResizeRGBA8888ToRGBA8888WithChannelOrder) {
  const int32_t src_w = 128, src_h = 96;
  const int32_t dst_w = 64, dst_h = 48;
  const size_t in_bytes = static_cast<size_t>(src_w) * src_h * 4;
  const size_t out_bytes = static_cast<size_t>(dst_w) * dst_h * 4;
  std::vector<int> channel_order = {2, 1, 0, 3};

  RunAndCompare<uint8_t, uint8_t>(
      "rgba8888",
      "rgba8888",
      src_w,
      src_h,
      4,
      4,
      in_bytes,
      out_bytes,
      {Arg("resize_width", dst_w),
       Arg("resize_height", dst_h),
       Arg("out_channel_order", channel_order)},
      [src_w, src_h, dst_w, dst_h](
          const void* src, void* dst, int32_t /*w*/, int32_t /*h*/, cudaStream_t stream) {
        const int order[4] = {2, 1, 0, 3};
        ref_resize_rgba8888_to_rgba8888(static_cast<const uint8_t*>(src),
                                        static_cast<uint8_t*>(dst),
                                        src_w,
                                        src_h,
                                        dst_w,
                                        dst_h,
                                        order,
                                        stream);
      });
}

// ================================================================================================
// Test: resize + rgba8888 -> float32 with BGR order and scale_max 255
//       Used by the HoloHub deltacast preprocessor.
// ================================================================================================

TEST_F(FormatConverterGpuResidentTest, TestResizeRGBA8888ToFloat32WithChannelOrderAndScale255) {
  const int32_t src_w = 128, src_h = 96;
  const int32_t dst_w = 64, dst_h = 48;
  const size_t in_bytes = static_cast<size_t>(src_w) * src_h * 4;
  const size_t out_bytes = static_cast<size_t>(dst_w) * dst_h * 3 * sizeof(float);
  std::vector<int> channel_order = {2, 1, 0};

  RunAndCompare<uint8_t, float>(
      "rgba8888",
      "float32",
      src_w,
      src_h,
      4,
      3,
      in_bytes,
      out_bytes,
      {Arg("resize_width", dst_w),
       Arg("resize_height", dst_h),
       Arg("out_channel_order", channel_order),
       Arg("scale_min", 0.f),
       Arg("scale_max", 255.f)},
      [src_w, src_h, dst_w, dst_h](
          const void* src, void* dst, int32_t /*w*/, int32_t /*h*/, cudaStream_t stream) {
        const int order[3] = {2, 1, 0};
        ref_resize_rgba8888_to_float32(static_cast<const uint8_t*>(src),
                                       static_cast<float*>(dst),
                                       src_w,
                                       src_h,
                                       dst_w,
                                       dst_h,
                                       order,
                                       0.f,
                                       255.f,
                                       stream);
      });
}

}  // namespace holoscan
