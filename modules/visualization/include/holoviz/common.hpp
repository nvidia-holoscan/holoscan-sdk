/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared vocabulary for HoloViz: formats, backend/policy enums, small
// geometry structs, and the NativeObject escape hatch. Every other HoloViz header
// builds on this one.

#pragma once

#include <cstdint>

#include <cuda.h>

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/logger.hpp>
#include <holoviz/present_mode.hpp>

namespace holoviz {

/// Version stamp for the public HoloViz headers; bump on incompatible API changes.
inline constexpr uint32_t kGfxHeaderVersion = 1;

/// Pixel/attribute formats. Deliberately a small, portable set: every entry
/// must lower cleanly to each planned backend, so new formats are added only
/// with a concrete use case. Whether a given adapter supports a format for a
/// given use is a runtime question — see CapabilityReport in adapter.hpp.
/// Numeric values are append-only once landed: CapabilityReport::format_caps
/// is indexed by them. This is distinct from ImageFormat
/// (<holoscan/visualization/holoviz_op.hpp>), the curated source-image set of the layer
/// API; multi-planar/packed YUV inputs there lower to one or more images in
/// these formats inside a RenderBackend implementation. Lowering NV12 will
/// need a kR8G8Unorm entry (the interleaved UV plane) appended here, with the
/// matching Vulkan conversion cases, once that backend lands.
enum class Format : uint32_t {
  kUnknown = 0,
  kR8Unorm,
  kR8G8B8A8Unorm,
  kB8G8R8A8Unorm,     ///< Present-friendly; CUDA import graphics-only.
  kR8G8B8A8Srgb,      ///< Graphics-only sampling decode.
  kA2B10G10R10Unorm,  ///< 10-bit; swapchain/RT (HDR10). Graphics-only.
  kR16G16B16A16Sfloat,
  kR32Uint,
  kR32Sfloat,
  kR32G32Sfloat,
  kR32G32B32Sfloat,  // Needed for vertex buffer formats
  kR32G32B32A32Sfloat,
  kD32Sfloat,  ///< Never exportable. Reserved for the 3D phase.
  kR16Unorm,   ///< 16-bit single-channel unorm (grayscale / depth-like sources); CUDA-importable.
  kR16G16B16A16Unorm,  ///< 64-bit RGBA, 16 bits per channel unorm; CUDA-importable.
};

/// Highest Format enumerator. Must always name the final enumerator; update it in the same
/// change that appends a format, otherwise capability probes silently skip the new format.
inline constexpr Format kLastFormat = Format::kR16G16B16A16Unorm;

/// Swapchain surface color space (what the display is told the pixels mean).
/// Distinct from the layer API's ColorSpace (<holoscan/visualization/holoviz_op.hpp>),
/// which is the caller-facing declaration on an image layer.
enum class ColorSpace : uint8_t {
  kSrgbNonlinear,
  kHdr10St2084,  ///< Requires hdr10_st2084 capability + kA2B10G10R10Unorm.
};

/// Rendering backend selector. Non-Vulkan values are reserved seams: the enum
/// slots exist so call sites are written backend-agnostically today, and a new
/// backend is only a factory case away.
enum class Backend : uint8_t {
  kAuto,    ///< 5.0: resolves to kVulkan.
  kVulkan,  ///< The 5.0 backend.
  // kD3D12, kVulkanSC, kcorVK    ///< Reserved;
};

/// Object-creation policy for a Device. The intended lifecycle is
/// create -> [setup: create/record everything] -> seal() -> [execute], with
/// kSealed rejecting post-seal creation to keep the steady state
/// allocation-free. 5.0 implements only kDynamic; requesting kSealed fails at
/// device creation.
enum class DevicePolicy : uint8_t {
  kDynamic,  ///< Post-seal creation succeeds with a diagnostic.
  kSealed,   ///< Post-seal creation fails.
};

/// Whether a resource is created with external-memory handles for sharing
/// with CUDA. Chosen at creation because it changes how the backend
/// allocates; it cannot be retrofitted onto a private resource.
enum class ShareMode : uint8_t {
  kPrivate,
  kExported,  ///< Born external; imported into CUDA at creation (Buffer::cuda_device_ptr(),
              ///< Image::cuda_mipmapped_array()).
  // kExportedCrossAdapter reserved for the D3D12 backend.
};

struct Extent2D {
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Rect2D {
  int32_t x = 0;
  int32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

// Lowercase on purpose: mirrors Slang/HLSL vector spelling.
struct float2 {
  float x = 0.0F, y = 0.0F;
};
struct float4 {
  float x = 0.0F, y = 0.0F, z = 0.0F, w = 0.0F;
};
using Color = float4;

/// Tags for the native (backend/window-system) handles reachable through
/// native_object(). Each wrapper answers only the types it actually holds and
/// returns an error for the rest.
enum class NativeObjectType : uint32_t {
  // Vulkan objects
  kVkInstance,
  kVkPhysicalDevice,
  kVkDevice,
  kVkQueue,
  kVkDeviceMemory,
  kVmaAllocation,
  kVkBuffer,
  kVkImage,
  kVkImageView,
  kVkSampler,
  kVkDescriptorSetLayout,
  kVkDescriptorSet,
  kVkDescriptorPool,
  kVkPipelineLayout,
  kVkRenderPass,
  kVkSemaphore,
  kVkFence,
  kVkCommandBuffer,
  kVkCommandPool,
  kVkQueryPool,
  kVkSurfaceKHR,
  kVkSwapchainKHR,
  kVkDisplayKHR,

  // GLFW objects. Handles come from HoloViz's private, statically linked GLFW instance and are
  // not interchangeable with handles from a GLFW the application links itself.
  kGLFWwindow,

  // Reserved: kD3D12Device, kD3D12CommandQueue, kD3D12Heap, kD3D12Resource,
  // kD3D12Fence, kD3D12GraphicsCommandList, kD3D12PipelineState, kDxgiAdapter,
  // kDxgiSwapChain, kWin32Hwnd, kWaylandDisplay, kWaylandSurface, kX11Display,
  // kX11Window, kXcbConnection, kQnxScreenWindow.
};

/// A borrowed backend handle, exposed for interop with code that speaks the
/// native API directly. The wrapper retains ownership: never destroy the
/// handle, and do not use it past the owning wrapper's lifetime.
struct NativeObject {
  NativeObjectType type{};
  uint64_t value = 0;

  constexpr bool valid() const { return value != 0; }
};

struct SubmitSyncToken {
  uint64_t timeline_value = 0;
};

/// Token for a signal enqueued outside the device timeline (ExternalSemaphore::signal()).
/// A default-constructed token is a no-op: Device::submit() skips it, so producers with
/// nothing to synchronize (e.g. a renderer that uploads nothing) can return one and callers
/// pass it through unchanged.
///
/// Lifetime: `impl` is a borrowed pointer into the ExternalSemaphore that produced the
/// token (like NativeObject, the wrapper retains ownership). The token must not be passed
/// to Device::submit() after that semaphore is destroyed.
struct ExternalSyncToken {
  uint64_t timeline_value = 0;
  Backend backend = Backend::kAuto;  // The backend that produced this token.
  void* impl = nullptr;              // Borrowed pointer to the producing semaphore's impl.
};

}  // namespace holoviz

// Fetch PROVIDER's native handle whose NativeObjectType member name matches the handle type
// (TYPE -> kTYPE), forward the error on failure, and declare `TYPE VAR_NAME` from the handle
// value. Only VAR_NAME escapes the macro's scope block; native handles are assumed
// default-constructible. PROVIDER must be an object (dereference pointers: *slice.buffer).
#define VIZ_UNWRAP_NATIVE(TYPE, VAR_NAME, PROVIDER)                           \
  TYPE VAR_NAME{};                                                            \
  {                                                                           \
    VIZ_UNWRAP(holoviz::NativeObject,                                         \
               VAR_NAME##_native_obj,                                         \
               (PROVIDER).native_object(holoviz::NativeObjectType::k##TYPE)); \
    VAR_NAME = reinterpret_cast<TYPE>(VAR_NAME##_native_obj.value);           \
  }

// Same, for native handles that are pointers to TYPE (e.g. GLFWwindow, future D3D interfaces):
// declares `TYPE* VAR_NAME`. A GLFWwindow obtained this way is only valid with HoloViz's own
// GLFW instance (see NativeObjectType::kGLFWwindow).
#define VIZ_UNWRAP_NATIVE_PTR(TYPE, VAR_NAME, PROVIDER)                       \
  TYPE* VAR_NAME = nullptr;                                                   \
  {                                                                           \
    VIZ_UNWRAP(holoviz::NativeObject,                                         \
               VAR_NAME##_native_obj,                                         \
               (PROVIDER).native_object(holoviz::NativeObjectType::k##TYPE)); \
    VAR_NAME = reinterpret_cast<TYPE*>(VAR_NAME##_native_obj.value);          \
  }
