/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/query.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/query_impl.hpp"

namespace holoviz::rhi {

QueryPool::QueryPool(std::unique_ptr<detail::QueryPoolImpl> impl) : impl_(std::move(impl)) {}
QueryPool::QueryPool() = default;
QueryPool::QueryPool(QueryPool&&) noexcept = default;
QueryPool& QueryPool::operator=(QueryPool&&) noexcept = default;
QueryPool::~QueryPool() = default;

expected<void> QueryPool::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<uint32_t> QueryPool::timestamp_count() const {
  if (!impl_) {
    return detail::null_impl_error("QueryPool::timestamp_count");
  }
  return impl_->timestamp_count();
}

expected<void> QueryPool::results_ns(uint32_t first, uint32_t count, std::span<uint64_t> out_ns) {
  if (!impl_) {
    return detail::null_impl_error("QueryPool::results_ns");
  }
  if (out_ns.size() < count) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "QueryPool::results_ns requires out_ns.size() >= count",
    });
  }
  return impl_->results_ns(first, count, out_ns);
}

expected<NativeObject> QueryPool::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("QueryPool::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
