/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

struct CUstream_st;

#include "buffer_pool/ref_count/shm_ref_count.hpp"

namespace holomq {

/// Non-null stream handle suitable for enqueueing CUDA stream operations, including a stream that
/// is being captured.
[[nodiscard]] bool is_usable_stream(CUstream_st* stream) noexcept;

/// Enqueue deferred \c release_ref on \p stream via \c cuLaunchHostFunc. On enqueue failure for a
/// valid stream, synchronize \p stream before calling \c release_ref. If stream validity or
/// completion cannot be established, keep the ref-count elevated to prevent unsafe slot reuse
/// (logged at critical severity).
///
/// \p mapping_owner must be non-null (programming error otherwise; logs and aborts). It pins the
/// ref-count SHM mapping until the host function runs (and during any synchronous fallback).
///
/// ABI note: exported signature uses \c ShmRefCountMappingPtr
/// (\c std::shared_ptr<ShmRefCountMapping>); callers outside holomq must link against the updated
/// \c defer_ref_release_on_stream export.
///
/// All buffers using \p stream must be destroyed before \c cuStreamDestroy(\p stream).
void defer_ref_release_on_stream(const ShmRefCountMappingPtr& mapping_owner,
                                 std::uint32_t slot_index, CUstream_st* stream) noexcept;

}  // namespace holomq
