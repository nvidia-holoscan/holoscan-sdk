/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <holomq/context.hpp>

#include <holoscan/channel/runtime.hpp>
#include "channel_tensor_sidecar_store.hpp"

namespace holoscan {

class ChannelRuntime::Impl {
 public:
  explicit Impl(holomq::Context&& context) : context_(std::move(context)) {}

  [[nodiscard]] expected<std::shared_ptr<TensorSidecarStore>, Error> tensor_sidecar_store(
      const std::string_view channel_name, const ChannelOptions& options) {
    try {
      const std::lock_guard<std::recursive_mutex> lock(tensor_sidecar_store_mutex_);
      const std::string name(channel_name);
      const auto existing = tensor_sidecar_stores_.find(name);
      if (existing != tensor_sidecar_stores_.end()) {
        if (existing->second->expected_recipient_count() != options.expected_recipient_count) {
          return make_unexpected(channel_internal::make_channel_error(
              ErrorCode::kInvalidArgument,
              "Tensor channel was reopened with a different expected recipient count"));
        }
        return existing->second;
      }
      if (manual_tensor_service_names_.contains(name)) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kDuplicateName,
            "manual Tensor publisher service identity is retired in this runtime"));
      }

      auto capacity = channel_internal::tensor_sidecar_record_capacity(options);
      if (!capacity) {
        return make_unexpected(std::move(capacity).error());
      }
      // Construct the endpoint-sized store before publishing it in the runtime map. If allocation
      // or map insertion fails, no null/partial entry becomes visible to a later endpoint factory.
      auto store =
          std::make_shared<TensorSidecarStore>(capacity.value(), options.expected_recipient_count);
      tensor_sidecar_stores_.emplace(name, store);
      return store;
    } catch (const std::length_error&) {
      return channel_internal::make_unexpected_nothrow<std::shared_ptr<TensorSidecarStore>>(
          ErrorCode::kResourceExhausted, "Tensor sidecar capacity exceeds allocator limits");
    } catch (const std::bad_alloc&) {
      return channel_internal::make_unexpected_nothrow<std::shared_ptr<TensorSidecarStore>>(
          ErrorCode::kResourceExhausted, "Tensor sidecar allocation failed");
    }
  }

  [[nodiscard]] expected<std::shared_ptr<TensorSidecarStore>, Error>
  register_manual_tensor_sidecar_store(const std::string_view channel_name,
                                       const std::uint32_t capacity) {
    try {
      const std::lock_guard<std::recursive_mutex> lock(tensor_sidecar_store_mutex_);
      const std::string name(channel_name);
      if (manual_tensor_service_names_.contains(name) || tensor_sidecar_stores_.contains(name)) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kDuplicateName,
            "manual Tensor publisher service already exists in this runtime"));
      }

      switch (
          manual_sidecar_fault_.exchange(ManualSidecarFault::kNone, std::memory_order_acq_rel)) {
        case ManualSidecarFault::kBadAlloc:
          throw std::bad_alloc{};
        case ManualSidecarFault::kLengthError:
          throw std::length_error("injected manual Tensor sidecar length failure");
        case ManualSidecarFault::kNone:
          break;
      }

      auto store = std::make_shared<TensorSidecarStore>(capacity, 1U);
      auto [store_it, store_inserted] = tensor_sidecar_stores_.emplace(name, store);
      if (!store_inserted) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kDuplicateName,
            "manual Tensor publisher service already exists in this runtime"));
      }
      try {
        const auto [unused, name_inserted] = manual_tensor_service_names_.emplace(name);
        (void)unused;
        if (!name_inserted) {
          tensor_sidecar_stores_.erase(store_it);
          return make_unexpected(channel_internal::make_channel_error(
              ErrorCode::kDuplicateName,
              "manual Tensor publisher service already exists in this runtime"));
        }
      } catch (...) {
        tensor_sidecar_stores_.erase(store_it);
        throw;
      }
      return store;
    } catch (const std::length_error&) {
      return channel_internal::make_unexpected_nothrow<std::shared_ptr<TensorSidecarStore>>(
          ErrorCode::kResourceExhausted, "manual Tensor sidecar capacity exceeds allocator limits");
    } catch (const std::bad_alloc&) {
      return channel_internal::make_unexpected_nothrow<std::shared_ptr<TensorSidecarStore>>(
          ErrorCode::kResourceExhausted, "manual Tensor sidecar allocation failed");
    }
  }

  void retire_manual_tensor_sidecar_store(
      const std::string& channel_name,
      const std::shared_ptr<TensorSidecarStore>& expected_store) noexcept {
    const std::lock_guard<std::recursive_mutex> lock(tensor_sidecar_store_mutex_);
    const auto it = tensor_sidecar_stores_.find(channel_name);
    if (it != tensor_sidecar_stores_.end() && it->second == expected_store) {
      // Existing subscribers retain the old generation through shared ownership. Remove the
      // capacity-sized store from runtime lookup, but keep the lightweight service-name tombstone
      // so no publisher or subscriber can create a fresh generation under the same identity.
      tensor_sidecar_stores_.erase(it);
    }
  }

  void rollback_manual_tensor_sidecar_store(
      const std::string& channel_name,
      const std::shared_ptr<TensorSidecarStore>& expected_store) noexcept {
    const std::lock_guard<std::recursive_mutex> lock(tensor_sidecar_store_mutex_);
    const auto it = tensor_sidecar_stores_.find(channel_name);
    if (it != tensor_sidecar_stores_.end() && it->second == expected_store) {
      tensor_sidecar_stores_.erase(it);
      manual_tensor_service_names_.erase(channel_name);
    }
  }

  void fail_next_manual_sidecar_with_bad_alloc_for_test() noexcept {
    manual_sidecar_fault_.store(ManualSidecarFault::kBadAlloc, std::memory_order_release);
  }

  void fail_next_manual_sidecar_with_length_error_for_test() noexcept {
    manual_sidecar_fault_.store(ManualSidecarFault::kLengthError, std::memory_order_release);
  }

  void fail_next_manual_backend_with_exception_for_test() noexcept {
    fail_next_manual_backend_with_exception_.store(true, std::memory_order_release);
  }

  void throw_if_manual_backend_failure_requested_for_test() {
    if (fail_next_manual_backend_with_exception_.exchange(false, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected manual Tensor backend setup failure");
    }
  }

  holomq::Context context_;
  // Endpoint construction is cold and may recurse through the public Tensor publisher factory.
  // A manual factory holds this lock from provisional registration through backend construction,
  // so another endpoint cannot retain the provisional store before rollback is decided.
  std::recursive_mutex tensor_sidecar_store_mutex_;
  std::unordered_map<std::string, std::shared_ptr<TensorSidecarStore>> tensor_sidecar_stores_;
  std::unordered_set<std::string> manual_tensor_service_names_;

 private:
  enum class ManualSidecarFault : std::uint8_t {
    kNone,
    kBadAlloc,
    kLengthError,
  };

  std::atomic<ManualSidecarFault> manual_sidecar_fault_{ManualSidecarFault::kNone};
  std::atomic<bool> fail_next_manual_backend_with_exception_{false};
};

}  // namespace holoscan
