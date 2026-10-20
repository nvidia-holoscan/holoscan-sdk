/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Definitions shared between Slang shaders and C++.
 *
 * A shader interface header is a file that both a `.slang` shader and the C++ code driving it
 * `#include`, so the push-constant block and any constants both sides rely on have exactly one
 * definition. This header provides what such a file needs to be valid in both languages:
 *
 *   - In Slang the vector types (`float2`, `uint4`, ...) and `uint` are built in; this header
 *     only defines the macros below, as no-ops where Slang needs nothing.
 *   - In C++ it defines the same type names in `holoviz::shader_interop`, laid out exactly as
 *     Slang lays a push-constant block out (std140-style: `float2` aligns to 8 bytes, `float4`
 *     and `uint4` to 16, and a struct's size rounds up to its largest member alignment).
 *
 * A shader interface header looks like this:
 *
 *   #pragma once
 *   #include <holoviz/shader_interop.hpp>
 *
 *   HOLOVIZ_SHADER_INTERFACE_BEGIN(my_app::detail)
 *
 *   HOLOVIZ_SHADER_CONSTANT uint kFlagInvert = 1u;
 *
 *   struct PushConstants {
 *       float2 scale;
 *       float2 translate;
 *       uint   flags;
 *       uint   pad;
 *   };
 *
 *   HOLOVIZ_SHADER_INTERFACE_END
 *
 * The shader includes it and declares `[[vk::push_constant]] ConstantBuffer<PushConstants> pc;`;
 * the C++ side includes it and pushes a `my_app::detail::PushConstants` value. Passing
 * `PUSH_CONSTANTS_TYPE`/`PUSH_CONSTANTS_INCLUDE` to `holoviz_generate_shader_header()` makes the
 * generated header `static_assert` the struct's size and every field offset against the
 * shader's reflected layout, so any drift between the two languages is a compile error.
 *
 * Layout rules for a shared struct (the generated asserts enforce them):
 *   - Use only the types provided here and `float`/`int`; `int` is 32-bit on both sides.
 *   - No `float3`/`int3`/`uint3`: Slang gives them 16-byte alignment but a 12-byte size, which
 *     C++ cannot express. Use a 4-component vector or separate scalars.
 *   - No `bool`: Slang stores it as 4 bytes, C++ as 1. Use `uint` flag bits.
 *   - No scalar or 2-component arrays (`float[N]`, `float2[N]`): std140 pads each array element
 *     to 16 bytes. Use `float4[N]`/`uint4[N]` and index `v[i >> 2][i & 3]`.
 *   - Where Slang would pad implicitly, spell the padding out (`uint pad;`) so both structs read
 *     the same and `sizeof` matches without surprises.
 *   - Nested structs are allowed as long as they follow the same rules.
 *   - One namespace per shader: `HOLOVIZ_SHADER_INTERFACE_BEGIN` opens an ordinary named
 *     namespace, so two interface headers in the same namespace define the same
 *     `PushConstants` twice (an ODR violation, and a redefinition error once both are seen by
 *     one translation unit). Open the namespace the generated header is emitted into
 *     (`holoviz_generate_shader_header(NAMESPACE ...)`) and nothing else shares it.
 *
 * These types are deliberately distinct from `holoviz::float2`/`holoviz::float4` in
 * <holoviz/common.hpp>: those are public API value types (`Color` is one) with natural
 * alignment, and changing their alignment would change the ABI of every struct holding them.
 */

#if defined(__cplusplus)

#include <cstddef>
#include <cstdint>

namespace holoviz::shader_interop {

/// Slang's `uint`.
using uint = std::uint32_t;

// Each vector type has the constructors Slang has (component-wise and, for the two-component
// types, from two scalars) and `operator[]`, so expressions such as `float2(x, y)` and
// `words[i >> 2][i & 3]` compile identically in both languages.

struct alignas(8) float2 {
  float x = 0.0F;
  float y = 0.0F;

  constexpr float2() = default;
  constexpr float2(float x_, float y_) : x(x_), y(y_) {}

  constexpr float& operator[](std::size_t i) { return i == 0 ? x : y; }
  constexpr const float& operator[](std::size_t i) const { return i == 0 ? x : y; }
};

struct alignas(16) float4 {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  float w = 0.0F;

  constexpr float4() = default;
  constexpr float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

  constexpr float& operator[](std::size_t i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
  constexpr const float& operator[](std::size_t i) const {
    return i == 0 ? x : i == 1 ? y : i == 2 ? z : w;
  }
};

struct alignas(8) int2 {
  std::int32_t x = 0;
  std::int32_t y = 0;

  constexpr int2() = default;
  constexpr int2(std::int32_t x_, std::int32_t y_) : x(x_), y(y_) {}

  constexpr std::int32_t& operator[](std::size_t i) { return i == 0 ? x : y; }
  constexpr const std::int32_t& operator[](std::size_t i) const { return i == 0 ? x : y; }
};

struct alignas(16) int4 {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
  std::int32_t w = 0;

  constexpr int4() = default;
  constexpr int4(std::int32_t x_, std::int32_t y_, std::int32_t z_, std::int32_t w_)
      : x(x_), y(y_), z(z_), w(w_) {}

  constexpr std::int32_t& operator[](std::size_t i) {
    return i == 0 ? x : i == 1 ? y : i == 2 ? z : w;
  }
  constexpr const std::int32_t& operator[](std::size_t i) const {
    return i == 0 ? x : i == 1 ? y : i == 2 ? z : w;
  }
};

struct alignas(8) uint2 {
  uint x = 0;
  uint y = 0;

  constexpr uint2() = default;
  constexpr uint2(uint x_, uint y_) : x(x_), y(y_) {}

  constexpr uint& operator[](std::size_t i) { return i == 0 ? x : y; }
  constexpr const uint& operator[](std::size_t i) const { return i == 0 ? x : y; }
};

struct alignas(16) uint4 {
  uint x = 0;
  uint y = 0;
  uint z = 0;
  uint w = 0;

  constexpr uint4() = default;
  constexpr uint4(uint x_, uint y_, uint z_, uint w_) : x(x_), y(y_), z(z_), w(w_) {}

  constexpr uint& operator[](std::size_t i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
  constexpr const uint& operator[](std::size_t i) const {
    return i == 0 ? x : i == 1 ? y : i == 2 ? z : w;
  }
};

// The std140 layout Slang applies to push-constant blocks, pinned so a toolchain that padded
// these types differently would fail here rather than in a shader's generated asserts.
static_assert(sizeof(float2) == 8 && alignof(float2) == 8);
static_assert(sizeof(float4) == 16 && alignof(float4) == 16);
static_assert(sizeof(int2) == 8 && alignof(int2) == 8);
static_assert(sizeof(int4) == 16 && alignof(int4) == 16);
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 8);
static_assert(sizeof(uint4) == 16 && alignof(uint4) == 16);
static_assert(sizeof(int) == 4, "Slang's int is 32-bit; the host int must match");

}  // namespace holoviz::shader_interop

/// Opens namespace `ns` and declares the shader vector types in it, so the struct and constant
/// definitions between BEGIN and END read as plain Slang. Using-declarations rather than a
/// using-directive: they make the names members of `ns`, so unqualified lookup stops there and
/// never reaches `holoviz::float2`/`holoviz::float4` from <holoviz/common.hpp> (which would be
/// ambiguous for any `ns` nested in `holoviz`).
#define HOLOVIZ_SHADER_INTERFACE_BEGIN(ns) \
  namespace ns {                           \
  using ::holoviz::shader_interop::float2; \
  using ::holoviz::shader_interop::float4; \
  using ::holoviz::shader_interop::int2;   \
  using ::holoviz::shader_interop::int4;   \
  using ::holoviz::shader_interop::uint;   \
  using ::holoviz::shader_interop::uint2;  \
  using ::holoviz::shader_interop::uint4;
#define HOLOVIZ_SHADER_INTERFACE_END }

/// A compile-time constant usable in both languages: `inline constexpr` in C++ (one definition
/// per program, usable in constant expressions), `static const` in Slang.
#define HOLOVIZ_SHADER_CONSTANT inline constexpr

#else  // Slang

// Slang has no namespaces to open here: shared definitions live at file scope, where the shader
// expects them. slangc predefines __SLANG__ in case an interface header needs to branch.
#define HOLOVIZ_SHADER_INTERFACE_BEGIN(ns)
#define HOLOVIZ_SHADER_INTERFACE_END
#define HOLOVIZ_SHADER_CONSTANT static const

#endif  // __cplusplus
