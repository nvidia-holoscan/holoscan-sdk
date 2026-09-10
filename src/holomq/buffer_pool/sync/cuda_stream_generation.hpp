/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstdint>

namespace holomq {

/// Enqueues \c cuStreamWriteValue64 on \p stream to publish \p value at \p address (export fence).
/// Returns \c false if \p stream or \p address is invalid, or if the driver rejects the enqueue.
/// Callers must treat \c false as fail-closed (do not publish or import with a fence).
[[nodiscard]] bool stream_write_generation(CUstream_st* stream, CUdeviceptr address,
                                           std::uint64_t value) noexcept;

/// Enqueues \c cuStreamWaitValue64 on \p stream until \p address is \c >= \p value (monotonic
/// generation). Returns \c false if \p stream or \p address is invalid, or if the driver rejects
/// the enqueue. Callers must treat \c false as fail-closed (do not continue as if the wait ran).
[[nodiscard]] bool stream_wait_generation(CUstream_st* stream, CUdeviceptr address,
                                          std::uint64_t value) noexcept;

}  // namespace holomq
