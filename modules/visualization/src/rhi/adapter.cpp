/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoviz/rhi/adapter.hpp>

#include <utility>

#include "holoviz/expected.hpp"
#include "holoviz/rhi/common.hpp"
#include "holoviz/rhi/surface.hpp"

#include "detail/device_impl.hpp"  // IWYU pragma: keep
#include "vulkan/vk_adapter.hpp"

namespace holoviz::rhi {

inline Backend resolve(Backend requested, [[maybe_unused]] DevicePolicy policy) {
  if (requested != Backend::kAuto)
    return requested;
  return Backend::kVulkan;
}

expected<std::vector<Adapter>> enumerate_adapters(Backend backend) {
  switch (resolve(backend, DevicePolicy::kDynamic)) {
    case Backend::kVulkan:
      return vulkan::enumerate_adapters();
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<bool> Adapter::supports_surface(const Surface& surface) const {
  switch (backend) {
    case Backend::kVulkan:
      return vulkan::adapter_supports_surface(*this, surface);
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<NativeObject> Adapter::native_object(NativeObjectType type) const {
  switch (backend) {
    case Backend::kVulkan:
      return vulkan::adapter_native_object(*this, type);
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<Device> Adapter::create_device(const DeviceDesc& desc) const {
  if (desc.backend != Backend::kAuto && desc.backend != backend) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Adapter backend does not match requested device backend",
    });
  }

  switch (backend) {
    case Backend::kVulkan:
      return vulkan::adapter_create_device(*this, desc).map([](auto impl) {
        return Device{std::move(impl)};
      });
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<Device> create_device_for_surface(const DeviceDesc& device_desc, const Surface& surface) {
  VIZ_UNWRAP(std::vector<Adapter>, adapters, enumerate_adapters(device_desc.backend));

  for (const auto& adapter : adapters) {
    VIZ_UNWRAP(bool, supported, adapter.supports_surface(surface));

    if (!supported) {
      continue;
    }

    if (device_desc.cuda_device >= 0 && adapter.cuda_device_index != device_desc.cuda_device) {
      continue;
    }

    // Surface-aware creation: the device's queue family is selected for both graphics and
    // presentation to this surface, matching what supports_surface() just verified.
    switch (adapter.backend) {
      case Backend::kVulkan:
        return vulkan::adapter_create_device(adapter, device_desc, &surface).map([](auto impl) {
          return Device{std::move(impl)};
        });
      default:
        return make_unexpected(Error{
            .code = ErrorCode::kNotImplemented,
            .message = "[holoviz] Only the Vulkan backend is implemented",
        });
    }
  }
  return make_unexpected(Error{.code = ErrorCode::kNotFound,
                               .message = "[holoviz] No suitable adapter found for surface."});
}

}  // namespace holoviz::rhi
