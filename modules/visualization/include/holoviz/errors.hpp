/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace holoviz {

/// Error codes for HoloViz expected-return paths.
///
/// These values belong exclusively to HoloViz. Integrations translate every named meaning into
/// their own error domain and must not cast, assert numeric parity, or interpret diagnostic text.
/// Numeric values are append-only once landed.
///
/// The set is deliberately limited to what the module returns today; add a code when a call site
/// needs it, not speculatively.
enum class ErrorCode : std::uint32_t {
  kSuccess = 0,          ///< No error.
  kFailure = 1,          ///< Generic HoloViz error.
  kInvalidArgument = 4,  ///< Invalid argument.
  kNotFound = 5,         ///< Requested object was not found.
  kNotImplemented = 7,   ///< The requested behavior has not been implemented.
  kBackpressured = 8,    ///< A wait timed out; the caller may retry.
  kInvalidState = 9,     ///< Operation on an object in an invalid state (e.g. an empty handle).
  kVulkanError = 10,     ///< A Vulkan call failed; see the message for details.
  kCudaError = 11,       ///< A CUDA call failed; see the message for details.
  kOutOfDate = 12,       ///< A swapchain no longer matches its surface; recreate/resize and retry.
  kImageCountChanged = 13,     ///< A swapchain recreation changed image_count(); rebuild per-image
                               ///< resources and re-query before continuing.
  kFrameDropped = 14,          ///< The frame was submitted but a swapchain recreation consumed it
                               ///< before display; continue with the next frame.
  kNullArg = 15,               ///< A required argument was a null/empty handle.
  kPushConstantOverflow = 16,  ///< Push data exceeds the pipeline's (or device's) push-constant
                               ///< budget; see Device::capabilities().max_push_data_bytes.
};

struct Error {
  ErrorCode code = ErrorCode::kFailure;
  std::string message;
};

}  // namespace holoviz
