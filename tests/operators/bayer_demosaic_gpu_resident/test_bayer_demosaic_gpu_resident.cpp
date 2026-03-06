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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic_gpu_resident/bayer_demosaic_gpu_resident.hpp>
#include <holoscan/utils/cuda_macros.hpp>

namespace holoscan {

// Test constants
static constexpr int32_t kTestWidth = 64;
static constexpr int32_t kTestHeight = 64;

// ================================================================================================
// Helper operators for GPU-resident test pipeline
// ================================================================================================

// Source operator: provides input device memory for the Bayer demosaic
class BayerSourceGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(BayerSourceGpuOp, GPUResidentOperator)
  BayerSourceGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(width_, "width", "Width", "Image width", kTestWidth);
    spec.param(height_, "height", "Height", "Image height", kTestHeight);
    spec.param(element_size_, "element_size", "Element Size", "Bytes per pixel", 1);
    spec.device_output("out", 0);  // default is set to 0 which will be overwritten in initialize()
  }

  void initialize() override {
    GPUResidentOperator::initialize();
    size_t mem_size = static_cast<size_t>(width_.get()) * static_cast<size_t>(height_.get()) *
                      element_size_.get();
    spec()->device_output("out", mem_size);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Source operator does nothing in compute - data is copied externally
  }

 private:
  Parameter<int32_t> width_;
  Parameter<int32_t> height_;
  Parameter<int32_t> element_size_;
};

// Sink operator: consumes the demosaiced output
class BayerSinkGpuOp : public GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(BayerSinkGpuOp, GPUResidentOperator)
  BayerSinkGpuOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(width_, "width", "Width", "Image width", kTestWidth);
    spec.param(height_, "height", "Height", "Image height", kTestHeight);
    spec.param(out_channels_, "out_channels", "Output Channels", "Output channels (3 or 4)", 3);
    spec.param(element_size_, "element_size", "Element Size", "Bytes per pixel", 1);
    spec.device_input("in", 0);  // default is set to 0 which will be overwritten in initialize()
  }

  void initialize() override {
    GPUResidentOperator::initialize();
    size_t mem_size = static_cast<size_t>(width_.get()) * static_cast<size_t>(height_.get()) *
                      static_cast<size_t>(out_channels_.get()) * element_size_.get();
    spec()->device_input("in", mem_size);
  }

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // Sink operator does nothing in compute - data is read externally
  }

 private:
  Parameter<int32_t> width_;
  Parameter<int32_t> height_;
  Parameter<int32_t> out_channels_;
  Parameter<int32_t> element_size_;
};

// ================================================================================================
// Reference implementation using original NPP calls (mimics BayerDemosaicOp)
// ================================================================================================

template <typename T>
void run_reference_demosaic(const T* input_device, T* output_device, int32_t width, int32_t height,
                            bool generate_alpha, int alpha_value, NppiBayerGridPosition grid_pos,
                            cudaStream_t stream) {
  static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>,
                "run_reference_demosaic only supports uint8_t and uint16_t types");

  NppStreamContext npp_ctx{};
  npp_ctx.hStream = stream;

  int in_step = width * sizeof(T);
  int out_channels = generate_alpha ? 4 : 3;
  int out_step = width * out_channels * sizeof(T);
  NppiSize roi_size = {width, height};
  NppiRect roi_rect = {0, 0, width, height};

  if constexpr (std::is_same_v<T, uint8_t>) {
    if (generate_alpha) {
      nppiCFAToRGBA_8u_C1AC4R_Ctx(input_device,
                                  in_step,
                                  roi_size,
                                  roi_rect,
                                  output_device,
                                  out_step,
                                  grid_pos,
                                  NPPI_INTER_UNDEFINED,
                                  static_cast<Npp8u>(alpha_value),
                                  npp_ctx);
    } else {
      nppiCFAToRGB_8u_C1C3R_Ctx(input_device,
                                in_step,
                                roi_size,
                                roi_rect,
                                output_device,
                                out_step,
                                grid_pos,
                                NPPI_INTER_UNDEFINED,
                                npp_ctx);
    }
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    if (generate_alpha) {
      nppiCFAToRGBA_16u_C1AC4R_Ctx(input_device,
                                   in_step,
                                   roi_size,
                                   roi_rect,
                                   output_device,
                                   out_step,
                                   grid_pos,
                                   NPPI_INTER_UNDEFINED,
                                   static_cast<Npp16u>(alpha_value),
                                   npp_ctx);
    } else {
      nppiCFAToRGB_16u_C1C3R_Ctx(input_device,
                                 in_step,
                                 roi_size,
                                 roi_rect,
                                 output_device,
                                 out_step,
                                 grid_pos,
                                 NPPI_INTER_UNDEFINED,
                                 npp_ctx);
    }
  }
  cudaStreamSynchronize(stream);
}

// ================================================================================================
// Test fixture
// ================================================================================================

class BayerDemosaicGpuResidentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available, skipping GPU-resident tests";
    }
  }

  // Helper function to wait for GPU-resident graph to be launched
  static bool wait_for_graph_launch(Fragment& fragment, int timeout_sec = 5) {
    auto start_time = std::chrono::steady_clock::now();
    while (!fragment.gpu_resident().is_launched()) {
      if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(timeout_sec)) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return true;
  }

  // Helper function to wait for result ready
  static bool wait_for_result(Fragment& fragment, int max_checks = 10) {
    for (int i = 0; i < max_checks; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (fragment.gpu_resident().result_ready()) {
        return true;
      }
    }
    return false;
  }
};

// ================================================================================================
// Test Case 1: 8-bit with alpha generation
// ================================================================================================

TEST_F(BayerDemosaicGpuResidentTest, Test8BitWithAlpha) {
  const int32_t width = kTestWidth;
  const int32_t height = kTestHeight;
  const bool generate_alpha = true;
  const int alpha_value = 255;
  const int out_channels = 4;
  const NppiBayerGridPosition grid_pos = NPPI_BAYER_GBRG;

  // Create GPU-resident fragment
  Fragment fragment;

  auto source = fragment.make_operator<BayerSourceGpuOp>(
      "source", Arg("width", width), Arg("height", height), Arg("element_size", 1));

  auto demosaic = fragment.make_operator<ops::BayerDemosaicGpuResidentOp>(
      "demosaic",
      Arg("width", width),
      Arg("height", height),
      Arg("pixel_type", 0),  // uint8
      Arg("generate_alpha", generate_alpha),
      Arg("alpha_value", alpha_value),
      Arg("bayer_grid_pos", static_cast<int>(grid_pos)));

  auto sink = fragment.make_operator<BayerSinkGpuOp>("sink",
                                                     Arg("width", width),
                                                     Arg("height", height),
                                                     Arg("out_channels", out_channels),
                                                     Arg("element_size", 1));

  fragment.add_flow(source, demosaic, {{"out", "in"}});
  fragment.add_flow(demosaic, sink, {{"out", "in"}});

  // Set a timeout to prevent hanging
  fragment.gpu_resident().timeout_ms(5000);

  // This test reads back results to the host via cudaMemcpy between iterations.
  // Enable sync_with_host to ensure device writes are visible to the host.
  fragment.gpu_resident().sync_with_host();

  // Run fragment asynchronously
  auto future = fragment.run_async();

  // Wait for graph to launch
  ASSERT_TRUE(wait_for_graph_launch(fragment)) << "GPU-resident graph failed to launch";

  // Get operators to access device memory
  auto graph = fragment.graph_shared();
  auto source_node = graph->find_node("source");
  auto sink_node = graph->find_node("sink");
  ASSERT_NE(source_node, nullptr);
  ASSERT_NE(sink_node, nullptr);

  auto* source_op = dynamic_cast<GPUResidentOperator*>(source_node.get());
  auto* sink_op = dynamic_cast<GPUResidentOperator*>(sink_node.get());
  ASSERT_NE(source_op, nullptr);
  ASSERT_NE(sink_op, nullptr);

  // Generate random input data
  size_t input_size = width * height;
  size_t output_size = width * height * out_channels;
  std::vector<uint8_t> host_input(input_size);
  std::vector<uint8_t> host_output_gpu_resident(output_size);
  std::vector<uint8_t> host_output_reference(output_size);

  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  for (size_t i = 0; i < input_size; ++i) {
    host_input[i] = static_cast<uint8_t>(rand_r(&seed) % 256);
  }

  // Allocate reference buffers on device
  uint8_t* ref_input_device = nullptr;
  uint8_t* ref_output_device = nullptr;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_input_device, input_size * sizeof(uint8_t)),
                                 "Failed to allocate reference input buffer");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_output_device, output_size * sizeof(uint8_t)),
                                 "Failed to allocate reference output buffer");

  // Run two iterations to verify GPU-resident execution
  for (int iter = 0; iter < 2; ++iter) {
    // Generate new random input for each iteration
    for (size_t i = 0; i < input_size; ++i) {
      host_input[i] = static_cast<uint8_t>(rand_r(&seed) % 256);
    }

    // Copy input to GPU-resident source
    void* source_out_addr = source_op->device_memory("out");
    ASSERT_NE(source_out_addr, nullptr) << "Source output device memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(source_out_addr,
                                              host_input.data(),
                                              input_size * sizeof(uint8_t),
                                              cudaMemcpyHostToDevice),
                                   "Failed to copy input to GPU-resident source");

    // Compute reference output
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(ref_input_device,
                                              host_input.data(),
                                              input_size * sizeof(uint8_t),
                                              cudaMemcpyHostToDevice),
                                   "Failed to copy input to reference buffer");

    cudaStream_t ref_stream;
    cudaStreamCreate(&ref_stream);
    run_reference_demosaic<uint8_t>(ref_input_device,
                                    ref_output_device,
                                    width,
                                    height,
                                    generate_alpha,
                                    alpha_value,
                                    grid_pos,
                                    ref_stream);
    cudaStreamDestroy(ref_stream);

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_reference.data(),
                                              ref_output_device,
                                              output_size * sizeof(uint8_t),
                                              cudaMemcpyDeviceToHost),
                                   "Failed to copy reference output to host");

    // Trigger GPU-resident execution
    fragment.gpu_resident().data_ready();

    // Wait for result
    ASSERT_TRUE(wait_for_result(fragment))
        << "GPU-resident result not ready for iteration " << iter;

    // Read GPU-resident output
    void* sink_in_addr = sink_op->device_memory("in");
    ASSERT_NE(sink_in_addr, nullptr) << "Sink input device memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_gpu_resident.data(),
                                              sink_in_addr,
                                              output_size * sizeof(uint8_t),
                                              cudaMemcpyDeviceToHost),
                                   "Failed to copy GPU-resident output to host");

    // Compare outputs
    for (size_t i = 0; i < output_size; ++i) {
      ASSERT_EQ(host_output_gpu_resident[i], host_output_reference[i])
          << "Mismatch at index " << i << " in iteration " << iter;
    }

    HOLOSCAN_LOG_INFO("Iteration {} - 8-bit with alpha test passed", iter);
  }

  // Cleanup
  cudaFree(ref_input_device);
  cudaFree(ref_output_device);

  // Tear down GPU-resident fragment
  fragment.gpu_resident().tear_down();
  future.get();
}

// ================================================================================================
// Test Case 2: 16-bit without alpha generation
// ================================================================================================

TEST_F(BayerDemosaicGpuResidentTest, Test16BitNoAlpha) {
  const int32_t width = kTestWidth;
  const int32_t height = kTestHeight;
  const bool generate_alpha = true;
  const int alpha_value = 65535;
  const int out_channels = 4;
  const NppiBayerGridPosition grid_pos = NPPI_BAYER_GBRG;

  // Create GPU-resident fragment
  Fragment fragment;

  auto source =
      fragment.make_operator<BayerSourceGpuOp>("source",
                                               Arg("width", width),
                                               Arg("height", height),
                                               Arg("element_size", 2));  // uint16 = 2 bytes

  auto demosaic = fragment.make_operator<ops::BayerDemosaicGpuResidentOp>(
      "demosaic",
      Arg("width", width),
      Arg("height", height),
      Arg("pixel_type", 1),  // uint16
      Arg("generate_alpha", generate_alpha),
      Arg("alpha_value", alpha_value),
      Arg("bayer_grid_pos", static_cast<int>(grid_pos)));

  auto sink = fragment.make_operator<BayerSinkGpuOp>("sink",
                                                     Arg("width", width),
                                                     Arg("height", height),
                                                     Arg("out_channels", out_channels),
                                                     Arg("element_size", 2));

  fragment.add_flow(source, demosaic, {{"out", "in"}});
  fragment.add_flow(demosaic, sink, {{"out", "in"}});

  // Set a timeout to prevent hanging
  fragment.gpu_resident().timeout_ms(5000);

  // This test reads back results to the host via cudaMemcpy between iterations.
  // Enable sync_with_host to ensure device writes are visible to the host.
  fragment.gpu_resident().sync_with_host();

  // Run fragment asynchronously
  auto future = fragment.run_async();

  // Wait for graph to launch
  ASSERT_TRUE(wait_for_graph_launch(fragment)) << "GPU-resident graph failed to launch";

  // Get operators to access device memory
  auto graph = fragment.graph_shared();
  auto source_node = graph->find_node("source");
  auto sink_node = graph->find_node("sink");
  ASSERT_NE(source_node, nullptr);
  ASSERT_NE(sink_node, nullptr);

  auto* source_op = dynamic_cast<GPUResidentOperator*>(source_node.get());
  auto* sink_op = dynamic_cast<GPUResidentOperator*>(sink_node.get());
  ASSERT_NE(source_op, nullptr);
  ASSERT_NE(sink_op, nullptr);

  // Generate random input data
  size_t input_size = width * height;
  size_t output_size = width * height * out_channels;
  std::vector<uint16_t> host_input(input_size);
  std::vector<uint16_t> host_output_gpu_resident(output_size);
  std::vector<uint16_t> host_output_reference(output_size);

  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  for (size_t i = 0; i < input_size; ++i) {
    host_input[i] = static_cast<uint16_t>(rand_r(&seed) % 65536);
  }

  // Allocate reference buffers on device
  uint16_t* ref_input_device = nullptr;
  uint16_t* ref_output_device = nullptr;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_input_device, input_size * sizeof(uint16_t)),
                                 "Failed to allocate reference input buffer");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_output_device, output_size * sizeof(uint16_t)),
                                 "Failed to allocate reference output buffer");

  // Run two iterations to verify GPU-resident execution
  for (int iter = 0; iter < 2; ++iter) {
    // Generate new random input for each iteration
    for (size_t i = 0; i < input_size; ++i) {
      host_input[i] = static_cast<uint16_t>(rand_r(&seed) % 65536);
    }

    // Copy input to GPU-resident source
    void* source_out_addr = source_op->device_memory("out");
    ASSERT_NE(source_out_addr, nullptr) << "Source output device memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(source_out_addr,
                                              host_input.data(),
                                              input_size * sizeof(uint16_t),
                                              cudaMemcpyHostToDevice),
                                   "Failed to copy input to GPU-resident source");

    // Compute reference output
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(ref_input_device,
                                              host_input.data(),
                                              input_size * sizeof(uint16_t),
                                              cudaMemcpyHostToDevice),
                                   "Failed to copy input to reference buffer");

    cudaStream_t ref_stream;
    cudaStreamCreate(&ref_stream);
    run_reference_demosaic<uint16_t>(ref_input_device,
                                     ref_output_device,
                                     width,
                                     height,
                                     generate_alpha,
                                     alpha_value,
                                     grid_pos,
                                     ref_stream);
    cudaStreamDestroy(ref_stream);

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_reference.data(),
                                              ref_output_device,
                                              output_size * sizeof(uint16_t),
                                              cudaMemcpyDeviceToHost),
                                   "Failed to copy reference output to host");

    // Trigger GPU-resident execution
    fragment.gpu_resident().data_ready();

    // Wait for result
    ASSERT_TRUE(wait_for_result(fragment))
        << "GPU-resident result not ready for iteration " << iter;

    // Read GPU-resident output
    void* sink_in_addr = sink_op->device_memory("in");
    ASSERT_NE(sink_in_addr, nullptr) << "Sink input device memory is null";

    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_gpu_resident.data(),
                                              sink_in_addr,
                                              output_size * sizeof(uint16_t),
                                              cudaMemcpyDeviceToHost),
                                   "Failed to copy GPU-resident output to host");

    // Compare outputs
    for (size_t i = 0; i < output_size; ++i) {
      ASSERT_EQ(host_output_gpu_resident[i], host_output_reference[i])
          << "Mismatch at index " << i << " in iteration " << iter;
    }

    HOLOSCAN_LOG_INFO("Iteration {} - 16-bit with alpha test passed", iter);
  }

  // Cleanup
  cudaFree(ref_input_device);
  cudaFree(ref_output_device);

  // Tear down GPU-resident fragment
  fragment.gpu_resident().tear_down();
  future.get();
}

// ================================================================================================
// Test Case 3: Real image pattern with file output
// ================================================================================================

// Generate a Bayer pattern image with color gradients
// This creates a pattern where R/G/B vary across the image, demonstrating demosaic effect
// GBRG pattern: G B   (even rows)
//               R G   (odd rows)
static void generate_colorful_bayer_image(std::vector<uint8_t>& bayer_data, int32_t width,
                                          int32_t height) {
  bayer_data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

  for (int32_t y = 0; y < height; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      // Create color gradients: R increases left-to-right, B increases top-to-bottom
      // G is moderate throughout
      uint8_t r_val = static_cast<uint8_t>((x * 255) / (width - 1));
      uint8_t g_val = 128;
      uint8_t b_val = static_cast<uint8_t>((y * 255) / (height - 1));

      // Determine which color this pixel samples based on GBRG Bayer pattern
      bool even_row = (y % 2 == 0);
      bool even_col = (x % 2 == 0);

      uint8_t pixel_val;
      if (even_row) {
        // Even row: G B G B ...
        pixel_val = even_col ? g_val : b_val;
      } else {
        // Odd row: R G R G ...
        pixel_val = even_col ? r_val : g_val;
      }

      bayer_data[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
          pixel_val;
    }
  }
}

// Helper function to save RGB/RGBA data as PNG
static bool save_image_png(const std::string& filename, const std::vector<uint8_t>& data,
                           int32_t width, int32_t height, int channels) {
  // Create output directory if it doesn't exist
  std::filesystem::path filepath(filename);
  if (filepath.has_parent_path()) {
    std::filesystem::create_directories(filepath.parent_path());
  }

  int result =
      stbi_write_png(filename.c_str(), width, height, channels, data.data(), width * channels);
  return result != 0;
}

// Helper function to save Bayer raw data as PGM (Portable GrayMap) file
// PGM is a standard format with header, widely supported by image tools
// Format: P5 (binary PGM), includes width, height, and max value in header
static bool save_bayer_pgm(const std::string& filename, const std::vector<uint8_t>& data,
                           int32_t width, int32_t height) {
  std::filesystem::path filepath(filename);
  if (filepath.has_parent_path()) {
    std::filesystem::create_directories(filepath.parent_path());
  }

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Write PGM header: P5 = binary grayscale, dimensions, max value (255 for 8-bit)
  file << "P5\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return file.good();
}

TEST_F(BayerDemosaicGpuResidentTest, TestRealImageWithFileOutput) {
  // Use a larger, more realistic image size
  const int32_t width = 256;
  const int32_t height = 256;
  const bool generate_alpha = true;
  const int alpha_value = 255;
  const int out_channels = 4;
  const NppiBayerGridPosition grid_pos = NPPI_BAYER_GBRG;

  // Output directory for test images
  const std::string output_dir = "test_output/bayer_demosaic_gpu_resident";

  // Create GPU-resident fragment
  Fragment fragment;

  auto source = fragment.make_operator<BayerSourceGpuOp>(
      "source", Arg("width", width), Arg("height", height), Arg("element_size", 1));

  auto demosaic = fragment.make_operator<ops::BayerDemosaicGpuResidentOp>(
      "demosaic",
      Arg("width", width),
      Arg("height", height),
      Arg("pixel_type", 0),  // uint8
      Arg("generate_alpha", generate_alpha),
      Arg("alpha_value", alpha_value),
      Arg("bayer_grid_pos", static_cast<int>(grid_pos)));

  auto sink = fragment.make_operator<BayerSinkGpuOp>("sink",
                                                     Arg("width", width),
                                                     Arg("height", height),
                                                     Arg("out_channels", out_channels),
                                                     Arg("element_size", 1));

  fragment.add_flow(source, demosaic, {{"out", "in"}});
  fragment.add_flow(demosaic, sink, {{"out", "in"}});

  // Set a timeout to prevent hanging
  fragment.gpu_resident().timeout_ms(10000);

  // This test reads back results to the host via cudaMemcpy between iterations.
  // Enable sync_with_host to ensure device writes are visible to the host.
  fragment.gpu_resident().sync_with_host();

  // Run fragment asynchronously
  auto future = fragment.run_async();

  // Wait for graph to launch
  ASSERT_TRUE(wait_for_graph_launch(fragment, 10)) << "GPU-resident graph failed to launch";

  // Get operators to access device memory
  auto graph = fragment.graph_shared();
  auto source_node = graph->find_node("source");
  auto sink_node = graph->find_node("sink");
  ASSERT_NE(source_node, nullptr);
  ASSERT_NE(sink_node, nullptr);

  auto* source_op = dynamic_cast<GPUResidentOperator*>(source_node.get());
  auto* sink_op = dynamic_cast<GPUResidentOperator*>(sink_node.get());
  ASSERT_NE(source_op, nullptr);
  ASSERT_NE(sink_op, nullptr);

  // Generate synthetic Bayer pattern image
  size_t input_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  size_t output_size = input_size * static_cast<size_t>(out_channels);
  std::vector<uint8_t> host_input;
  std::vector<uint8_t> host_output_gpu_resident(output_size);
  std::vector<uint8_t> host_output_reference(output_size);

  generate_colorful_bayer_image(host_input, width, height);
  ASSERT_EQ(host_input.size(), input_size) << "Generated Bayer image has incorrect size";

  // Save the input Bayer image as PGM (standard grayscale format)
  std::string bayer_filename = output_dir + "/input_bayer.pgm";
  ASSERT_TRUE(save_bayer_pgm(bayer_filename, host_input, width, height))
      << "Failed to save input Bayer PGM file";
  HOLOSCAN_LOG_INFO("Saved input Bayer data to {}", bayer_filename);

  // Allocate reference buffers on device
  uint8_t* ref_input_device = nullptr;
  uint8_t* ref_output_device = nullptr;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_input_device, input_size * sizeof(uint8_t)),
                                 "Failed to allocate reference input buffer");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&ref_output_device, output_size * sizeof(uint8_t)),
                                 "Failed to allocate reference output buffer");

  // Copy input to GPU-resident source
  void* source_out_addr = source_op->device_memory("out");
  ASSERT_NE(source_out_addr, nullptr) << "Source output device memory is null";

  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpy(
          source_out_addr, host_input.data(), input_size * sizeof(uint8_t), cudaMemcpyHostToDevice),
      "Failed to copy input to GPU-resident source");

  // Compute reference output using original NPP calls
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(ref_input_device,
                                            host_input.data(),
                                            input_size * sizeof(uint8_t),
                                            cudaMemcpyHostToDevice),
                                 "Failed to copy input to reference buffer");

  cudaStream_t ref_stream;
  cudaStreamCreate(&ref_stream);
  run_reference_demosaic<uint8_t>(ref_input_device,
                                  ref_output_device,
                                  width,
                                  height,
                                  generate_alpha,
                                  alpha_value,
                                  grid_pos,
                                  ref_stream);
  cudaStreamDestroy(ref_stream);

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_reference.data(),
                                            ref_output_device,
                                            output_size * sizeof(uint8_t),
                                            cudaMemcpyDeviceToHost),
                                 "Failed to copy reference output to host");

  // Save reference demosaic output
  ASSERT_TRUE(save_image_png(
      output_dir + "/output_reference.png", host_output_reference, width, height, out_channels))
      << "Failed to save reference output image";
  HOLOSCAN_LOG_INFO("Saved reference output image to {}/output_reference.png", output_dir);

  // Trigger GPU-resident execution
  fragment.gpu_resident().data_ready();

  // Wait for result
  ASSERT_TRUE(wait_for_result(fragment, 20)) << "GPU-resident result not ready";

  // Read GPU-resident output
  void* sink_in_addr = sink_op->device_memory("in");
  ASSERT_NE(sink_in_addr, nullptr) << "Sink input device memory is null";

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy(host_output_gpu_resident.data(),
                                            sink_in_addr,
                                            output_size * sizeof(uint8_t),
                                            cudaMemcpyDeviceToHost),
                                 "Failed to copy GPU-resident output to host");

  // Save GPU-resident demosaic output
  ASSERT_TRUE(save_image_png(output_dir + "/output_gpu_resident.png",
                             host_output_gpu_resident,
                             width,
                             height,
                             out_channels))
      << "Failed to save GPU-resident output image";
  HOLOSCAN_LOG_INFO("Saved GPU-resident output image to {}/output_gpu_resident.png", output_dir);

  // Compare outputs - compute statistics for verification
  int64_t total_diff = 0;
  int32_t max_diff = 0;
  int32_t mismatch_count = 0;

  for (size_t i = 0; i < output_size; ++i) {
    int32_t diff = std::abs(static_cast<int32_t>(host_output_gpu_resident[i]) -
                            static_cast<int32_t>(host_output_reference[i]));
    if (diff > 0) {
      ++mismatch_count;
      total_diff += diff;
      max_diff = std::max(max_diff, diff);
    }
  }

  // For Bayer demosaic, outputs should be identical (same NPP implementation)
  EXPECT_EQ(mismatch_count, 0) << "Found " << mismatch_count << " mismatched pixels, "
                               << "max diff: " << max_diff << ", "
                               << "avg diff: "
                               << (mismatch_count > 0 ? static_cast<double>(total_diff) /
                                                            static_cast<double>(mismatch_count)
                                                      : 0.0);

  // Generate a difference image for visual inspection if there are differences
  if (mismatch_count > 0) {
    std::vector<uint8_t> diff_image(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      int32_t diff = std::abs(static_cast<int32_t>(host_output_gpu_resident[i]) -
                              static_cast<int32_t>(host_output_reference[i]));
      // Scale up difference for visibility (multiply by 10, cap at 255)
      diff_image[i] = static_cast<uint8_t>(std::min(255, diff * 10));
    }
    save_image_png(output_dir + "/output_difference.png", diff_image, width, height, out_channels);
    HOLOSCAN_LOG_INFO("Saved difference image to {}/output_difference.png", output_dir);
  }

  // Verify exact match
  for (size_t i = 0; i < output_size; ++i) {
    ASSERT_EQ(host_output_gpu_resident[i], host_output_reference[i])
        << "Mismatch at index " << i << " (pixel " << (i / out_channels) << ", channel "
        << (i % out_channels) << ")";
  }

  HOLOSCAN_LOG_INFO("TestRealImageWithFileOutput passed - GPU-resident output matches reference");
  HOLOSCAN_LOG_INFO("Output images saved to {} directory", output_dir);

  // Cleanup
  cudaFree(ref_input_device);
  cudaFree(ref_output_device);

  // Tear down GPU-resident fragment
  fragment.gpu_resident().tear_down();
  future.get();
}

}  // namespace holoscan
