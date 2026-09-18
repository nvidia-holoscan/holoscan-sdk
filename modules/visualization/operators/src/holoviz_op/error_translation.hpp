// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared bridge from the HoloViz error/CUDA vocabulary into the Holoscan Core error domain, used by
// the operators in this module (HolovizOp). Module-private: not exported, not installed.

#pragma once

#include <cuda.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoviz/errors.hpp>

namespace holoscan::visualization {

/// Return a short diagnostic name for a holoviz::ErrorCode. For log/message text only — not a
/// stable identifier, and not a substitute for the holoscan::ErrorCode mapping in
/// translate_error().
[[nodiscard]] const char* holoviz_error_code_name(holoviz::ErrorCode code);

/// Translate a holoviz backend error into the holoscan error domain.
/// holoviz::Error and holoscan::Error are distinct types that must not be cast between each
/// other; this function performs an explicit best-effort mapping. \p what names the failing
/// operation (e.g. "RenderBackend::submit") and, together with the original HoloViz error code,
/// is folded into the returned message — holoscan::ErrorCode alone cannot represent every HoloViz
/// category (e.g. kVulkanError/kCudaError both collapse to kFailure), so the message is what
/// preserves that detail for diagnostics.
[[nodiscard]] holoscan::Error translate_error(const holoviz::Error& e, const char* what);

/// Translate a CUDA driver result into a kFailure holoscan::Error carrying the CUDA
/// name/description.
[[nodiscard]] holoscan::Error translate_cuda_error(CUresult result, const char* what);

/// Create a CUDA stream on \p context (pushed for the duration of the call if not current).
[[nodiscard]] holoscan::expected<CUstream, holoscan::Error> create_cuda_stream(CUcontext context);

/// Destroy a stream created by create_cuda_stream(); no-op success for a null stream.
[[nodiscard]] holoscan::expected<void, holoscan::Error> destroy_cuda_stream(CUcontext context,
                                                                            CUstream stream);

/// Throw a holoscan::RuntimeError from a holoviz backend failure, using the same mapping and
/// message shape as translate_error(). For start() paths, which have no error return channel.
[[noreturn]] void throw_holoviz_error(const holoviz::Error& e, const char* what);

}  // namespace holoscan::visualization
