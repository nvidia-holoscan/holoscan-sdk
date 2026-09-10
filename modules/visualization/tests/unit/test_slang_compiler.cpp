// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <gtest/gtest.h>

#include <holoviz/rhi/slang_compiler.hpp>

namespace holoviz::rhi {

namespace {

// Minimal vertex stage shared by every test source; the binding declarations under
// test are prepended per case.
constexpr const char* kStages = R"(
struct VSInput {
  float3 position;
};

struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)";

expected<ShaderBlob> compile(const std::string& declarations) {
  const std::string source = declarations + kStages;
  return compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
}

}  // namespace

TEST(SlangCompilerTest, ReflectsSingleBindingsOfMixedKinds) {
  const std::string source = std::string(R"(
StructuredBuffer<float> values;
Texture2D<float4> image;
SamplerState image_sampler;

struct VSInput {
  float3 position;
};

struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return image.Sample(image_sampler, float2(0.5, 0.5)) * values[0];
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_TRUE(result) << result.error().message;

  ASSERT_EQ(result->bindings.size(), 3u);
  for (const auto& binding : result->bindings) {
    EXPECT_EQ(binding.count, 1u) << binding.name;
    if (binding.name == "values") {
      EXPECT_EQ(binding.kind, BindingKind::kStorageBuffer);
    } else if (binding.name == "image") {
      EXPECT_EQ(binding.kind, BindingKind::kSampledImage);
    } else if (binding.name == "image_sampler") {
      EXPECT_EQ(binding.kind, BindingKind::kSampler);
    } else {
      ADD_FAILURE() << "unexpected binding: " << binding.name;
    }
  }
}

TEST(SlangCompilerTest, ReflectsStructVertexAttributes) {
  const std::string source = std::string(R"(
struct VSInput {
  float3 position : POSITION;
  float2 uv_coord : TEXCOORD0;
};

struct VSOutput {
  float4 position : SV_Position;
  float2 uv_coord : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 1.0);
  output.uv_coord = input.uv_coord;
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(input.uv_coord, 0.0, 1.0);
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_TRUE(result) << result.error().message;

  ASSERT_EQ(result->attributes.size(), 2u);
  EXPECT_EQ(result->attributes[0].name, "position");
  EXPECT_EQ(result->attributes[0].location, 0u);
  EXPECT_EQ(result->attributes[0].format, Format::kR32G32B32Sfloat);
  EXPECT_EQ(result->attributes[0].byte_size, 12u);
  EXPECT_EQ(result->attributes[1].name, "uv_coord");
  EXPECT_EQ(result->attributes[1].location, 1u);
  EXPECT_EQ(result->attributes[1].format, Format::kR32G32Sfloat);
  EXPECT_EQ(result->attributes[1].byte_size, 8u);
}

TEST(SlangCompilerTest, RejectsDirectVertexInput) {
  const std::string source = std::string(R"(
struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(float2 position : POSITION) {
  VSOutput output;
  output.position = float4(position, 0.0, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("direct vertex input"), std::string::npos)
      << result.error().message;
  EXPECT_NE(result.error().message.find("position"), std::string::npos) << result.error().message;
  EXPECT_NE(result.error().message.find("struct"), std::string::npos) << result.error().message;
}

TEST(SlangCompilerTest, RejectsMultipleDirectVertexInputs) {
  const std::string source = std::string(R"(
struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(float2 position : POSITION, float2 uv_coord : TEXCOORD0) {
  VSOutput output;
  output.position = float4(position + uv_coord, 0.0, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("direct vertex input"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, AcceptsDirectSystemValueInput) {
  const std::string source = std::string(R"(
struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(uint vertex_id : SV_VertexID) {
  VSOutput output;
  output.position = float4(float(vertex_id), 0.0, 0.0, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_TRUE(result->attributes.empty());
}

TEST(SlangCompilerTest, MixedStructAndDirectSystemValueInput) {
  const std::string source = std::string(R"(
struct VSInput {
  float3 position : POSITION;
};

struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input, uint vertex_id : SV_VertexID) {
  VSOutput output;
  output.position = float4(input.position * float(vertex_id), 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)");
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_TRUE(result) << result.error().message;
  ASSERT_EQ(result->attributes.size(), 1u);
  EXPECT_EQ(result->attributes[0].name, "position");
  EXPECT_EQ(result->attributes[0].format, Format::kR32G32B32Sfloat);
}

TEST(SlangCompilerTest, RejectsTextureArray) {
  auto result = compile("Texture2D<float4> images[2];\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("array resource bindings"), std::string::npos)
      << result.error().message;
  EXPECT_NE(result.error().message.find("images"), std::string::npos) << result.error().message;
}

TEST(SlangCompilerTest, RejectsSamplerArray) {
  auto result = compile("SamplerState samplers[2];\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("array resource bindings"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, RejectsStorageBufferArray) {
  auto result = compile("StructuredBuffer<float> buffers[3];\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("array resource bindings"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, RejectsUnsizedTextureArray) {
  auto result = compile("Texture2D<float4> images[];\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("array resource bindings"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, RejectsTexture1D) {
  auto result = compile("Texture1D<float4> image;\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("only 2D sampled images"), std::string::npos)
      << result.error().message;
  EXPECT_NE(result.error().message.find("image"), std::string::npos) << result.error().message;
}

TEST(SlangCompilerTest, RejectsTexture3D) {
  auto result = compile("Texture3D<float4> volume;\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("only 2D sampled images"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, RejectsTextureCube) {
  auto result = compile("TextureCube<float4> cubemap;\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("only 2D sampled images"), std::string::npos)
      << result.error().message;
}

TEST(SlangCompilerTest, RejectsTexture2DArrayBinding) {
  auto result = compile("Texture2DArray<float4> layers;\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("only plain Texture2D"), std::string::npos)
      << result.error().message;
  EXPECT_NE(result.error().message.find("layers"), std::string::npos) << result.error().message;
}

// Emits a Slang warning (implicit float-to-int truncation) without failing the
// compile; used to verify diagnostics from one operation never leak into the
// error message of a later, unrelated failure.
constexpr const char* kWarningFunction = R"(
int truncate_for_warning(float value) {
  return value;
}
)";

TEST(SlangCompilerTest, WarningSourceStillCompiles) {
  auto result = compile(kWarningFunction);
  ASSERT_TRUE(result) << result.error().message;
}

TEST(SlangCompilerTest, WarningThenLocalFailureReportsLocalError) {
  const std::string source = std::string(kWarningFunction) + R"(
struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(float2 position : POSITION) {
  VSOutput output;
  output.position = float4(position, 0.0, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return float4(1.0, 1.0, 1.0, 1.0);
}
)";
  auto result =
      compile_slang_shader(Backend::kVulkan, "vertex_main", "fragment_main", source.c_str());
  ASSERT_FALSE(result);
  // The reflection failure must report its own message, not a stale warning
  // diagnostic left over from an earlier successful operation.
  EXPECT_NE(result.error().message.find("direct vertex input"), std::string::npos)
      << result.error().message;
  EXPECT_EQ(result.error().message.find("warning"), std::string::npos) << result.error().message;
}

TEST(SlangCompilerTest, RejectsTexture2DMultisample) {
  auto result = compile("Texture2DMS<float4> msaa_image;\n");
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("only plain Texture2D"), std::string::npos)
      << result.error().message;
}

}  // namespace holoviz::rhi
