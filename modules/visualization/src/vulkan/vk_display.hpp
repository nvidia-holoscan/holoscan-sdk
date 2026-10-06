/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

#include <holoviz/display.hpp>

#include "detail/surface_impl.hpp"
#include "display_systems/display_ownership.hpp"
#include "vk_display_surface.hpp"  // IWYU pragma: export
#include "vk_impls.hpp"            // IWYU pragma: keep

namespace holoviz::vulkan {

expected<std::vector<Display>> enumerate_displays(const Adapter& adapter);

expected<std::vector<DisplayMode>> display_modes(const Display& display);
expected<NativeObject> display_native_object(const Display& display, NativeObjectType type);

expected<std::unique_ptr<detail::SurfaceImpl>> display_create_surface(
    const Display& display, const DisplaySurfaceDesc& desc);

}  // namespace holoviz::vulkan
