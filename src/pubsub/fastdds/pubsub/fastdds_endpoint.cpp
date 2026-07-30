/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/fastdds/pubsub/fastdds_endpoint.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan {

FastDdsEndpoint::FastDdsEndpoint(std::vector<uint8_t>* output_buffer)
    : output_buffer_(output_buffer) {
  if (!output_buffer) {
    throw std::invalid_argument("FastDdsEndpoint: output_buffer cannot be nullptr");
  }
}

FastDdsEndpoint::FastDdsEndpoint(uint8_t* buffer, size_t buffer_size)
    : raw_output_buffer_(buffer), raw_buffer_size_(buffer_size) {
  if (!buffer) {
    throw std::invalid_argument("FastDdsEndpoint: buffer cannot be nullptr");
  }
}

FastDdsEndpoint::FastDdsEndpoint(const std::vector<uint8_t>* input_buffer)
    : input_buffer_(input_buffer) {
  if (!input_buffer) {
    throw std::invalid_argument("FastDdsEndpoint: input_buffer cannot be nullptr");
  }
}

bool FastDdsEndpoint::is_write_available() {
  if (output_buffer_) {
    return true;
  }
  if (raw_output_buffer_) {
    return write_position_ < raw_buffer_size_;
  }
  return false;
}

bool FastDdsEndpoint::is_read_available() {
  return input_buffer_ != nullptr && read_position_ < input_buffer_->size();
}

expected<size_t, RuntimeError> FastDdsEndpoint::write(const void* data, size_t size) {
  if (data == nullptr && size > 0) {
    return make_unexpected(
        RuntimeError(ErrorCode::kInvalidArgument, "FastDdsEndpoint::write: data is nullptr"));
  }

  if (size == 0) {
    return 0;
  }

  if (raw_output_buffer_) {
    // Raw buffer mode — fixed size, no growth
    if (size > raw_buffer_size_ - write_position_) {
      return make_unexpected(RuntimeError(
          ErrorCode::kFailure,
          "FastDdsEndpoint::write: write of " + std::to_string(size) +
              " bytes exceeds buffer capacity (position=" + std::to_string(write_position_) +
              ", capacity=" + std::to_string(raw_buffer_size_) + ")"));
    }
    std::memcpy(raw_output_buffer_ + write_position_, data, size);
    write_position_ += size;
    return size;
  }

  if (output_buffer_) {
    // Vector mode — existing behavior (appends to vector)
    if (size > output_buffer_->capacity() - output_buffer_->size()) {
      output_buffer_->reserve(output_buffer_->size() + size);
    }
    const uint8_t* src = static_cast<const uint8_t*>(data);
    output_buffer_->insert(output_buffer_->end(), src, src + size);
    return size;
  }

  return make_unexpected(
      RuntimeError(ErrorCode::kFailure, "FastDdsEndpoint::write: endpoint is in read mode"));
}

expected<size_t, RuntimeError> FastDdsEndpoint::read(void* data, size_t size) {
  if (!input_buffer_) {
    return make_unexpected(
        RuntimeError(ErrorCode::kFailure, "FastDdsEndpoint::read: endpoint is in write mode"));
  }

  if (data == nullptr && size > 0) {
    return make_unexpected(
        RuntimeError(ErrorCode::kInvalidArgument, "FastDdsEndpoint::read: data is nullptr"));
  }

  if (size == 0) {
    return 0;
  }

  // Check if enough data is available
  size_t available = input_buffer_->size() - read_position_;
  if (size > available) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsEndpoint::read: requested {} bytes but only {} available", size, available);
    return make_unexpected(RuntimeError(
        ErrorCode::kFailure,
        "FastDdsEndpoint::read: insufficient data in buffer (requested " + std::to_string(size) +
            " bytes, available " + std::to_string(available) + " bytes)"));
  }

  // Copy data from buffer
  std::memcpy(data, input_buffer_->data() + read_position_, size);
  read_position_ += size;

  return size;
}

expected<void, RuntimeError> FastDdsEndpoint::write_ptr(const void* pointer, size_t size,
                                                        holoscan::MemoryStorageType type) {
  // For DDS transport, GPU device memory cannot be sent directly.
  // The caller must stage GPU data to host memory before serialization.
  //
  // Memory types:
  //   kHost (0)   = CUDA pinned host memory - OK to serialize directly
  //   kDevice (1) = GPU device memory - requires D2H staging
  //   kSystem (2) = Regular CPU memory (malloc/new) - OK to serialize directly
  //   kCudaManaged (3) = CUDA managed memory - CPU-accessible, but may need sync
  //
  if (type == holoscan::MemoryStorageType::kDevice) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsEndpoint::write_ptr: GPU device memory (type={}) cannot be serialized directly. "
        "Stage to host memory first.",
        static_cast<int>(type));
    return make_unexpected(RuntimeError(
        ErrorCode::kFailure,
        "FastDdsEndpoint::write_ptr: GPU device memory cannot be serialized directly for DDS "
        "transport. Use FastDdsSerializer with GPU staging enabled."));
  }

  // For CUDA managed memory, warn that synchronization may be needed
  if (type == holoscan::MemoryStorageType::kCudaManaged) {
    HOLOSCAN_LOG_WARN(
        "FastDdsEndpoint::write_ptr: CUDA managed memory - ensure GPU operations are complete "
        "before serialization to avoid data races.");
  }

  // For host memory, write_ptr is equivalent to write
  auto result = write(pointer, size);
  if (!result) {
    return forward_error(result);
  }

  return expected<void, RuntimeError>();
}

void FastDdsEndpoint::reset_read_position() {
  read_position_ = 0;
}

size_t FastDdsEndpoint::size() const {
  if (raw_output_buffer_) {
    return write_position_;
  }
  if (output_buffer_) {
    return output_buffer_->size();
  }
  if (input_buffer_) {
    return input_buffer_->size();
  }
  return 0;
}

size_t FastDdsEndpoint::bytes_written() const {
  if (raw_output_buffer_) {
    return write_position_;
  }
  if (output_buffer_) {
    return output_buffer_->size();
  }
  return 0;
}

size_t FastDdsEndpoint::bytes_remaining() const {
  if (input_buffer_) {
    return input_buffer_->size() - read_position_;
  }
  return 0;
}

}  // namespace holoscan
