/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <cstdint>
#include <span>

#include <holoviz/query.hpp>

namespace holoviz::detail {

/// Backend interface for QueryPool; see the QueryPool handle for API docs.
class QueryPoolImpl {
 public:
  QueryPoolImpl(const QueryPoolImpl&) = delete;
  QueryPoolImpl& operator=(const QueryPoolImpl&) = delete;
  QueryPoolImpl(QueryPoolImpl&&) = delete;
  QueryPoolImpl& operator=(QueryPoolImpl&&) = delete;
  virtual ~QueryPoolImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  virtual uint32_t timestamp_count() const = 0;
  [[nodiscard]] virtual expected<void> results_ns(uint32_t first, uint32_t count,
                                                  std::span<uint64_t> out_ns) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  QueryPoolImpl() = default;
};

}  // namespace holoviz::detail
