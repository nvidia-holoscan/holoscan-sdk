/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/deferred_ref_release.hpp"

#include <holomq/logger.hpp>
#include "buffer_pool/ref_count/shm_ref_count.hpp"

#include <cuda.h>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace holomq {

bool is_usable_stream(CUstream_st* const stream) noexcept {
  if (stream == nullptr) {
    return false;
  }
  const CUresult rc = cuStreamQuery(reinterpret_cast<CUstream>(stream));
  return rc == CUDA_SUCCESS || rc == CUDA_ERROR_NOT_READY;
}

namespace {

struct DeferredRefReleasePayload {
  ShmRefCountMappingPtr mapping_owner;
  std::uint32_t slot_index;
};

void CUDAAPI deferred_ref_release_host_fn(void* user_data) {
  const auto payload = static_cast<DeferredRefReleasePayload*>(user_data);
  if (!payload->mapping_owner) {
    holomq::log_literal(holomq::LogLevel::Error,
                        "null ref-count mapping in deferred ref release host function; this is a "
                        "programming error");
    assert(payload->mapping_owner != nullptr);
    std::abort();
  }
  payload->mapping_owner->table().release_ref(payload->slot_index);
  delete payload;
}

/// Enqueue deferred release. Caller must validate \p mapping_owner, \p slot_index, and \p stream.
bool try_enqueue_deferred_ref_release(const ShmRefCountMappingPtr& mapping_owner,
                                      const std::uint32_t slot_index,
                                      CUstream_st* const stream) noexcept {
  auto payload = std::unique_ptr<DeferredRefReleasePayload>(
      new (std::nothrow) DeferredRefReleasePayload{mapping_owner, slot_index});
  if (!payload) {
    return false;
  }

  const CUresult rc = cuLaunchHostFunc(
      reinterpret_cast<CUstream>(stream), deferred_ref_release_host_fn, payload.get());
  if (rc != CUDA_SUCCESS) {
    return false;
  }
  [[maybe_unused]] DeferredRefReleasePayload* released_payload = payload.release();
  return true;
}

}  // namespace

void defer_ref_release_on_stream(const ShmRefCountMappingPtr& mapping_owner,
                                 const std::uint32_t slot_index,
                                 CUstream_st* const stream) noexcept {
  if (!mapping_owner) {
    log_literal(LogLevel::Error,
                "null ref-count mapping for deferred ref release; this is a programming error");
    assert(mapping_owner != nullptr);
    std::abort();
  }

  ShmRefCountTable& table = mapping_owner->table();

  if (!table.is_mapped() || slot_index >= table.buffer_count()) {
    log_literal(LogLevel::Critical,
                "invalid ref-count table state for deferred ref release; slot state cannot be "
                "released safely");
    return;
  }

  if (!is_usable_stream(stream)) {
    log_literal(LogLevel::Critical,
                "invalid stream for deferred ref release; pinning slot ref-count to prevent unsafe "
                "reuse");
    return;
  }

  if (try_enqueue_deferred_ref_release(mapping_owner, slot_index, stream)) {
    return;
  }

  log_literal(LogLevel::Warn,
              "deferred ref release enqueue failed; synchronizing stream before slot ref release");
  const CUresult sync_rc = cuStreamSynchronize(reinterpret_cast<CUstream>(stream));
  if (sync_rc != CUDA_SUCCESS) {
    log_literal(LogLevel::Critical,
                "cuStreamSynchronize failed before deferred-ref-release fallback; pinning slot "
                "ref-count to prevent unsafe reuse");
    return;
  }
  table.release_ref(slot_index);
}

}  // namespace holomq
