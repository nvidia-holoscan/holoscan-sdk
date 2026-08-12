/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/tensor_output_loan.hpp>

#include <utility>

#include "tensor_output_loan_internal.hpp"

namespace holoscan {

TensorOutputWriteGuard::~TensorOutputWriteGuard() noexcept {
  if (state_ != nullptr && ops_ != nullptr && ops_->abandon != nullptr) {
    ops_->abandon(state_);
  }
}

void* TensorOutputWriteGuard::data() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->data != nullptr ? ops_->data(state_)
                                                                       : nullptr;
}

std::size_t TensorOutputWriteGuard::byte_size() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_size != nullptr
             ? ops_->byte_size(state_)
             : 0U;
}

std::optional<::cuda::stream_ref> TensorOutputWriteGuard::stream() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->stream != nullptr ? ops_->stream(state_)
                                                                         : std::nullopt;
}

expected<void, Error> TensorOutputWriteGuard::commit() && {
  if (state_ == nullptr || ops_ == nullptr || ops_->commit == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* const state = std::exchange(state_, nullptr);
  const auto* const ops = std::exchange(ops_, nullptr);
  [[maybe_unused]] auto owner = std::move(state_owner_);
  return ops->commit(state);
}

TensorOutputLoan::~TensorOutputLoan() noexcept {
  if (state_ != nullptr && ops_ != nullptr && ops_->abandon != nullptr) {
    ops_->abandon(state_);
  }
}

TensorShapeView TensorOutputLoan::shape() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->shape != nullptr ? ops_->shape(state_)
                                                                        : TensorShapeView{};
}

TensorByteStridesView TensorOutputLoan::byte_strides() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_strides != nullptr
             ? ops_->byte_strides(state_)
             : TensorByteStridesView{std::nullopt};
}

DLDataType TensorOutputLoan::dtype() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->dtype != nullptr ? ops_->dtype(state_)
                                                                        : DLDataType{};
}

DeviceId TensorOutputLoan::device_id() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->device_id != nullptr
             ? ops_->device_id(state_)
             : DeviceId{};
}

MemoryKind TensorOutputLoan::memory_kind() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->memory_kind != nullptr
             ? ops_->memory_kind(state_)
             : MemoryKind::kUnknown;
}

std::size_t TensorOutputLoan::byte_size() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_size != nullptr
             ? ops_->byte_size(state_)
             : 0U;
}

expected<TensorOutputWriteGuard, Error> TensorOutputLoan::write(
    ::cuda::stream_ref producer_stream) & {
  if (state_ == nullptr || ops_ == nullptr || ops_->write == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return ops_->write(state_, producer_stream);
}

expected<TensorOutputWriteGuard, Error> TensorOutputLoan::write_host() & {
  if (state_ == nullptr || ops_ == nullptr || ops_->write_host == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return ops_->write_host(state_);
}

expected<void, Error> TensorOutputLoan::cancel() && {
  if (state_ == nullptr || ops_ == nullptr || ops_->cancel == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* const state = std::exchange(state_, nullptr);
  const auto* const ops = std::exchange(ops_, nullptr);
  [[maybe_unused]] auto owner = std::move(state_owner_);
  return ops->cancel(state);
}

}  // namespace holoscan
