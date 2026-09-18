/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pipeline and the shader interface it is built from. Shaders are Slang-only
// (vertex + fragment; no compute, no uniform buffers — runtime data is push
// constants or storage buffers). A ShaderBlob carries the compiled IR
// together with its reflected interface, so pipeline creation and descriptor
// binding are validated against what the shader actually declares instead of
// a hand-maintained layout description.

#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz {

class BindGroupLayout;

enum class ShaderStage : uint8_t { kVertex, kFragment };

enum class ShaderStageMask : uint32_t {
  kNone = 0,
  kVertex = 1u << 0,
  kFragment = 1u << 1,
  kAllGraphics = kVertex | kFragment,
};
constexpr ShaderStageMask operator|(ShaderStageMask a, ShaderStageMask b) {
  return static_cast<ShaderStageMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr bool has_flag(ShaderStageMask v, ShaderStageMask f) {
  return (static_cast<uint32_t>(v) & static_cast<uint32_t>(f)) != 0;
}

enum class ShaderIR : uint8_t {
  kSpirv,
  // kDxil,
};

/// One reflected vertex-shader input. Attributes are packed into a single
/// interleaved vertex buffer (binding slot 0) in ascending location order;
/// the pipeline derives each attribute's offset and the total stride from
/// byte_size, so vertex data must be laid out to match.
struct AttributeDesc {
  std::string name;
  uint32_t location;
  Format format;
  uint32_t byte_size;
};

/// The bindable resource kinds. Deliberately minimal: no uniform buffers and
/// no compute leaves storage buffers, sampled textures, and samplers.
enum class BindingKind : uint8_t {
  kStorageBuffer,
  kSampler,
  kSampledImage,
};

/// One reflected shader resource binding (descriptor space 0 only). `name` is
/// the shader-source variable name, usable with Descriptor's by-name binds.
struct BindingDesc {
  std::string name;
  uint32_t index;
  BindingKind kind;
  /// Array size. Always 1 for now: compile_slang_shader() rejects array resource
  /// declarations because Descriptor has no array bind operations yet. The field
  /// is kept for forward compatibility with future descriptor-array support.
  uint32_t count;
};

/// Compiled shader stages plus their reflected interface, as produced by
/// compile_slang_shader() (or, later, an offline tool emitting the same
/// reflection).
struct ShaderBlob {
  ShaderIR ir = ShaderIR::kSpirv;
  std::vector<std::byte> vertex_bytes;
  std::vector<std::byte> fragment_bytes;
  std::string vertex_entry_point;
  std::string fragment_entry_point;
  std::vector<AttributeDesc> attributes;
  std::vector<BindingDesc> bindings;
  uint32_t push_constant_size = 0;
};

enum class PrimitiveTopology : uint8_t {
  kPointList,
  kLineList,
  kLineStrip,
  kTriangleList,
  kTriangleStrip,
};

enum class BlendFactor : uint8_t {
  kZero,
  kOne,
  kSrcAlpha,
  kOneMinusSrcAlpha,
  kDstAlpha,
  kOneMinusDstAlpha,
};
enum class BlendOp : uint8_t { kAdd, kMin, kMax };

struct BlendState {
  bool enable = false;
  // Defaults compose premultiplied alpha when enabled.
  BlendFactor src_color = BlendFactor::kOne;
  BlendFactor dst_color = BlendFactor::kOneMinusSrcAlpha;
  BlendOp color_op = BlendOp::kAdd;
  BlendFactor src_alpha = BlendFactor::kOne;
  BlendFactor dst_alpha = BlendFactor::kOneMinusSrcAlpha;
  BlendOp alpha_op = BlendOp::kAdd;
};

enum class CompareOp : uint8_t {
  kAlways,
  kLess,
  kLessEqual,
  kGreater,
  kGreaterEqual,
  kEqual,
  kNever,
};

struct DepthState {
  bool test = false;
  bool write = false;
  CompareOp compare = CompareOp::kAlways;
};

enum class CullMode : uint8_t { kNone, kFront, kBack, kFrontAndBack };

/// How polygons are rasterized. kFillRectangle rasterizes each triangle's
/// axis-aligned bounding rectangle (VK_NV_fill_rectangle) — e.g. one triangle
/// per screen-aligned quad, with attributes extrapolated by their plane
/// equations. Requires CapabilityReport::fill_rectangle; pipeline creation
/// fails on devices without it.
enum class PolygonMode : uint8_t { kFill, kFillRectangle };

struct RasterState {
  CullMode cull = CullMode::kNone;
  PolygonMode polygon_mode = PolygonMode::kFill;
};

enum class VertexStepMode : uint8_t { kVertex, kInstance };

/// Everything needed to build a graphics pipeline. Viewport and scissor are
/// always dynamic (set on the command list); attachment formats come from the
/// render pass the pipeline is first used with, not from this desc.
struct PipelineDesc {
  ShaderBlob shader_blob;

  VertexStepMode step = VertexStepMode::kVertex;
  PrimitiveTopology topology = PrimitiveTopology::kTriangleList;
  RasterState raster;
  DepthState depth;
  BlendState blend;

  const char* debug_name = nullptr;
};

namespace detail {
class PipelineImpl;
}

class Descriptor;

/// A graphics pipeline. Holds the shader's reflected interface; the actual
/// backend pipeline objects are created lazily per render pass on first bind
/// and cached (see CommandList::bind_pipeline).
class Pipeline {
 public:
  HOLOVIZ_EXPORT explicit Pipeline(std::unique_ptr<detail::PipelineImpl> impl);
  HOLOVIZ_EXPORT Pipeline();
  HOLOVIZ_EXPORT Pipeline(Pipeline&&) noexcept;
  HOLOVIZ_EXPORT Pipeline& operator=(Pipeline&&) noexcept;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  HOLOVIZ_EXPORT ~Pipeline();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Create a Descriptor shaped by this pipeline's reflected bindings. The
  /// pipeline must outlive descriptors created from it.
  [[nodiscard]] HOLOVIZ_EXPORT expected<Descriptor> create_descriptor();

  /// Resolve a shader-source variable name to its binding index.
  [[nodiscard]] HOLOVIZ_EXPORT expected<uint32_t> binding_name_to_index(
      const std::string& name) const;

  /// Vulkan: answers kVkDevice, kVkPipelineLayout, kVkDescriptorSetLayout.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::PipelineImpl& impl() { return *impl_; }
  const detail::PipelineImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::PipelineImpl> impl_;
};

}  // namespace holoviz
