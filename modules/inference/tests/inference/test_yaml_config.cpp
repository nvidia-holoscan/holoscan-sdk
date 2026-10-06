/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure CPU tests for InferenceOpParams::from_yaml() (see
// include/holoscan/operators/inference/inference.hpp /
// modules/inference/operators/inference/inference_yaml.cpp).
// None of these tests construct an InferenceOp or run a graph -- they only exercise YAML parsing,
// so they do not require GPU access and are registered as ordinary gtests.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <holoscan/operators/inference/inference.hpp>

namespace holoscan::ops {

namespace {

// Writes `content` to `path`, creating parent directories as needed.
void write_yaml(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open()) << "Failed to open " << path << " for writing";
  out << content;
  out.close();
}

}  // namespace

class YamlConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    base_dir_ = std::filesystem::temp_directory_path() / "holoinfer_yaml_config_test";
    std::filesystem::create_directories(base_dir_);
  }

  void TearDown() override {
    std::error_code error_code;
    std::filesystem::remove_all(base_dir_, error_code);
  }

  // Returns a path under a fresh, uniquely-named subdirectory of base_dir_ so tests never
  // collide with each other's temporary files.
  std::filesystem::path unique_path(const std::string& name) {
    static int counter = 0;
    return base_dir_ / (std::to_string(counter++) + "_" + name);
  }

  std::filesystem::path base_dir_;
};

// ---------------------------------------------------------------------------------------------
// Field-parsing round trips
// ---------------------------------------------------------------------------------------------

TEST_F(YamlConfigTest, YamlConfig_ModelPathMap_ParsesCorrectly) {
  const auto path = unique_path("model_path_map.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: /some/absolute/path/model.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_EQ(params.model_path_map.size(), 1U);
  EXPECT_EQ(params.model_path_map.at("model_a"), "/some/absolute/path/model.onnx");
}

TEST_F(YamlConfigTest, YamlConfig_ModelPathMap_ResolvesRelativeToConfigFile) {
  // Use a unique per-test subdirectory under base_dir_ so this test cannot race with
  // parallel tests or clobber unrelated files at a fixed shared location like /tmp/subdir.
  // The TearDown() call cleans up base_dir_ (and everything under it) automatically.
  const std::filesystem::path sub_dir = unique_path("resolve_relative_config_dir");
  std::filesystem::create_directories(sub_dir);
  const auto path = sub_dir / "config.yaml";
  write_yaml(path, R"(
model_path_map:
  model_a: models/x.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_EQ(params.model_path_map.size(), 1U);
  const std::filesystem::path resolved(params.model_path_map.at("model_a"));
  EXPECT_TRUE(resolved.is_absolute()) << resolved;
  EXPECT_EQ(resolved.lexically_normal(), (sub_dir / "models" / "x.onnx").lexically_normal());
}

TEST_F(YamlConfigTest, YamlConfig_ModelPathMap_AbsolutePathsUnchanged) {
  const auto path = unique_path("abs_path.yaml");
  const std::string absolute_model_path = (base_dir_ / "models" / "identity.onnx").string();
  write_yaml(path,
             "model_path_map:\n"
             "  model_a: " +
                 absolute_model_path +
                 "\n"
                 "input_map:\n"
                 "  model_a: [tensor_in]\n"
                 "output_map:\n"
                 "  model_a: [tensor_out]\n");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.model_path_map.at("model_a"), absolute_model_path);
}

TEST_F(YamlConfigTest, YamlConfig_BackendAndBackendMap_ParseCorrectly) {
  const auto path = unique_path("backend.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
  model_b: model_b.onnx
input_map:
  model_a: [tensor_a_in]
  model_b: [tensor_b_in]
output_map:
  model_a: [tensor_a_out]
  model_b: [tensor_b_out]
backend: trt
backend_map:
  model_a: trt
  model_b: trt
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.backend, "trt");
  ASSERT_EQ(params.backend_map.size(), 2U);
  EXPECT_EQ(params.backend_map.at("model_a"), "trt");
  EXPECT_EQ(params.backend_map.at("model_b"), "trt");
}

TEST_F(YamlConfigTest, YamlConfig_InputMapAndOutputMap_ParseCorrectly) {
  const auto path = unique_path("input_output_map.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in_1, tensor_in_2]
output_map:
  model_a: [tensor_out_1, tensor_out_2, tensor_out_3]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_EQ(params.input_map.size(), 1U);
  const std::vector<std::string> expected_inputs = {"tensor_in_1", "tensor_in_2"};
  EXPECT_EQ(params.input_map.at("model_a"), expected_inputs);

  ASSERT_EQ(params.output_map.size(), 1U);
  const std::vector<std::string> expected_outputs = {
      "tensor_out_1", "tensor_out_2", "tensor_out_3"};
  EXPECT_EQ(params.output_map.at("model_a"), expected_outputs);
}

TEST_F(YamlConfigTest, YamlConfig_TrtOptProfile_RoundTrips) {
  const auto path = unique_path("trt_opt_profile.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
trt_opt_profile:
  model_a: ["1,4,8"]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_EQ(params.trt_opt_profile.size(), 1U);
  const std::vector<std::string> expected = {"1,4,8"};
  EXPECT_EQ(params.trt_opt_profile.at("model_a"), expected);
}

TEST_F(YamlConfigTest, YamlConfig_TrtOptProfile_AbsentKeyLeavesEmpty) {
  const auto path = unique_path("no_trt_opt_profile.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_TRUE(params.trt_opt_profile.empty());
}

TEST_F(YamlConfigTest, YamlConfig_DlaCoreMap_RoundTrips) {
  const auto path = unique_path("dla_core_map.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
  model_b: model_b.onnx
input_map:
  model_a: [tensor_in_a]
  model_b: [tensor_in_b]
output_map:
  model_a: [tensor_out_a]
  model_b: [tensor_out_b]
dla_core_map:
  model_a: "0"
  model_b: "-1"
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_EQ(params.dla_core_map.size(), 2U);
  EXPECT_EQ(params.dla_core_map.at("model_a"), "0");
  EXPECT_EQ(params.dla_core_map.at("model_b"), "-1");
}

TEST_F(YamlConfigTest, YamlConfig_DlaCoreMap_AbsentKeyLeavesEmpty) {
  const auto path = unique_path("no_dla_core_map.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_TRUE(params.dla_core_map.empty());
}

TEST_F(YamlConfigTest, YamlConfig_MaxBatchSize_RoundTrips) {
  const auto path = unique_path("max_batch_size.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
max_batch_size: 512
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.max_batch_size, 512);
}

TEST_F(YamlConfigTest, YamlConfig_MaxBatchSize_AbsentKeyLeavesDefault) {
  const auto path = unique_path("no_max_batch_size.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.max_batch_size, 256);
}

TEST_F(YamlConfigTest, YamlConfig_MaxMemory_RoundTrips) {
  const auto path = unique_path("max_memory.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
max_memory: 2147483648
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.max_memory, 2147483648ULL);
}

TEST_F(YamlConfigTest, YamlConfig_MaxMemory_AbsentKeyLeavesDefault) {
  const auto path = unique_path("no_max_memory.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.max_memory, 10'000'000'000ULL);
}

TEST_F(YamlConfigTest, YamlConfig_AllBoolFlags_ParseCorrectly) {
  const auto path = unique_path("bool_flags.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
is_engine_path: true
input_on_cuda: false
output_on_cuda: false
transmit_on_cuda: false
enable_cuda_graphs: false
enable_fp16: true
parallel_inference: true
dynamic_input_dims: true
output_dynamic_dims: true
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_TRUE(params.is_engine_path);
  EXPECT_FALSE(params.input_on_cuda);
  EXPECT_FALSE(params.output_on_cuda);
  EXPECT_FALSE(params.transmit_on_cuda);
  EXPECT_FALSE(params.enable_cuda_graphs);
  EXPECT_TRUE(params.enable_fp16);
  EXPECT_TRUE(params.parallel_inference);
  EXPECT_TRUE(params.dynamic_input_dims);
  EXPECT_TRUE(params.output_dynamic_dims);
}

TEST_F(YamlConfigTest, YamlConfig_DlaCore_ParsesCorrectly) {
  const auto path = unique_path("dla_core.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
dla_core: 1
dla_gpu_fallback: false
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  EXPECT_EQ(params.dla_core, 1);
  EXPECT_FALSE(params.dla_gpu_fallback);
}

TEST_F(YamlConfigTest, YamlConfig_CudaStreamPool_SubMap_ConstructsResource) {
  const auto path = unique_path("cuda_stream_pool.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
cuda_stream_pool:
  max_size: 4
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_NE(params.cuda_stream_pool, nullptr);
}

TEST_F(YamlConfigTest, YamlConfig_CudaGreenContextPool_SubMap_ConstructsResource) {
  const auto path = unique_path("cuda_green_context_pool.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
cuda_green_context_pool:
  num_partitions: 2
)");

  const auto params = InferenceOpParams::from_yaml(path.string());
  ASSERT_NE(params.cuda_green_context_pool, nullptr);
}

// ---------------------------------------------------------------------------------------------
// Absent fields -> defaults
// ---------------------------------------------------------------------------------------------

TEST_F(YamlConfigTest, YamlConfig_MissingOptionalFields_UseDefaults) {
  const auto path = unique_path("minimal.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  const InferenceOpParams defaults{};
  const auto params = InferenceOpParams::from_yaml(path.string());

  EXPECT_EQ(params.backend, defaults.backend);
  EXPECT_TRUE(params.backend_map.empty());
  EXPECT_EQ(params.is_engine_path, defaults.is_engine_path);
  EXPECT_EQ(params.input_on_cuda, defaults.input_on_cuda);
  EXPECT_EQ(params.output_on_cuda, defaults.output_on_cuda);
  EXPECT_EQ(params.transmit_on_cuda, defaults.transmit_on_cuda);
  EXPECT_EQ(params.dynamic_input_dims, defaults.dynamic_input_dims);
  EXPECT_EQ(params.output_dynamic_dims, defaults.output_dynamic_dims);
  EXPECT_EQ(params.enable_fp16, defaults.enable_fp16);
  EXPECT_EQ(params.enable_cuda_graphs, defaults.enable_cuda_graphs);
  EXPECT_EQ(params.dla_core, defaults.dla_core);
  EXPECT_EQ(params.dla_gpu_fallback, defaults.dla_gpu_fallback);
  EXPECT_EQ(params.parallel_inference, defaults.parallel_inference);
  EXPECT_EQ(params.cuda_stream_pool, nullptr);
  EXPECT_EQ(params.cuda_green_context_pool, nullptr);
}

// ---------------------------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------------------------

TEST_F(YamlConfigTest, YamlConfig_MissingFile_Throws) {
  EXPECT_THROW(
      {
        [[maybe_unused]] auto params =
            InferenceOpParams::from_yaml("/nonexistent/path/to/config.yaml");
      },
      std::runtime_error);
}

TEST_F(YamlConfigTest, YamlConfig_MalformedYaml_Throws) {
  const auto path = unique_path("malformed.yaml");
  // Unterminated flow-mapping / mismatched brackets -> a YAML parse error.
  write_yaml(path, "model_path_map: [this is not valid: yaml{{{\n");

  EXPECT_THROW(
      { [[maybe_unused]] auto params = InferenceOpParams::from_yaml(path.string()); },
      std::runtime_error);
}

TEST_F(YamlConfigTest, YamlConfig_MissingRequiredField_Throws) {
  const auto path = unique_path("missing_required.yaml");
  write_yaml(path, R"(
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
)");

  EXPECT_THROW(
      { [[maybe_unused]] auto params = InferenceOpParams::from_yaml(path.string()); },
      std::runtime_error);
}

TEST_F(YamlConfigTest, YamlConfig_UnknownField_Ignored_OrLoggedAsWarning) {
  const auto path = unique_path("unknown_field.yaml");
  write_yaml(path, R"(
model_path_map:
  model_a: model_a.onnx
input_map:
  model_a: [tensor_in]
output_map:
  model_a: [tensor_out]
this_key_does_not_exist: 42
)");

  // InferenceOpParams::from_yaml() (modules/inference/operators/inference/inference_yaml.cpp) only
  // ever reads known keys via assign_if_present()/to_data_map()/to_data_vec_map() against explicit
  // key names; it never rejects a YAML document for containing an extra top-level key. Match that
  // actual (graceful-ignore) behavior: parsing succeeds and known fields are populated normally.
  InferenceOpParams params;
  EXPECT_NO_THROW({ params = InferenceOpParams::from_yaml(path.string()); });
  // model_a.onnx is relative, so from_yaml() resolves it against the config file's own
  // directory (see YamlConfig_ModelPathMap_ResolvesRelativeToConfigFile) regardless of the
  // unrelated unknown key.
  EXPECT_EQ(params.model_path_map.at("model_a"), (path.parent_path() / "model_a.onnx").string());
}

}  // namespace holoscan::ops
