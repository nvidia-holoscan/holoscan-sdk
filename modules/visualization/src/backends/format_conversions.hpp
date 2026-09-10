// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoviz/expected.hpp>
#include <holoviz/image_format.hpp>
#include <holoviz/rhi/common.hpp>

namespace holoviz::backends {

/// Conversions between the public layer-API format vocabulary (ImageFormat) and the RHI's
/// (rhi::Format). The mapping is intentionally partial in both directions: multi-planar and
/// packed YUV ImageFormats lower to one or more RHI images inside a RenderBackend
/// implementation (see the rhi::Format doc in <holoviz/rhi/common.hpp>), and most RHI
/// formats (float render targets, depth, vertex-attribute formats) have no public spelling.

/// Map a public ImageFormat to the single RHI format it corresponds to. Fails with
/// kInvalidArgument for kUnknown and for formats with no single-image RHI lowering
/// (kR8G8B8Unorm, kNV12, kYUYV422).
expected<rhi::Format> to_rhi_format(ImageFormat format);

/// Map an RHI format (e.g. a swapchain's actual format) to its public ImageFormat. Fails
/// with kFailure for RHI formats that have no public spelling.
expected<ImageFormat> to_image_format(rhi::Format format);

}  // namespace holoviz::backends
