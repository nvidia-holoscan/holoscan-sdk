/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_HPP

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "holoscan/core/expected.hpp"
#include "holoscan/core/resources/gxf/first_fit_allocator_base.hpp"

namespace holoscan {

/**
 * @brief Template memory management class using first-fit allocation strategy.
 *
 * This class pre-allocates memory for a given type and allows acquiring chunks of memory and
 * releasing them efficiently. Only the allocate() function performs memory allocation; other
 * functions use a constant amount of stack memory.
 *
 * The allocator uses the FirstFitAllocatorBase for efficient memory management with logarithmic
 * time complexity for acquire and release operations.
 *
 * @tparam T Type of elements that will be stored in the allocated memory.
 */
template <class T>
class FirstFitAllocator {
 public:
  /// Expected type used by this class.
  template <typename E>
  using expected_t = expected<E, FirstFitAllocatorBase::Error>;
  /// Unexpected type used by this class.
  using unexpected_t = unexpected<FirstFitAllocatorBase::Error>;

  /**
   * @brief Default constructor.
   */
  FirstFitAllocator() = default;

  /**
   * @brief Allocate memory to handle requests for chunks of a given total size.
   *
   * Pre-allocates memory that can be divided into chunks for later acquisition. The total memory
   * allocated will be approximately sizeof(T) * size + 32 * size / chunk_size bytes.
   *
   * @param size Total number of elements of type T to support.
   * @param chunk_size Minimum chunk size. Memory will always be allocated in multiples of this
   * size. This can improve allocation speed but reduces control over exact allocated size.
   * @return Total number of elements that can be managed on success, error on failure.
   *         Error::kAlreadyInUse if already initialized, Error::kInvalidSize for invalid
   * parameters, Error::kOutOfMemory if memory allocation fails.
   */
  expected_t<int32_t> allocate(const int32_t size, const int chunk_size = 1) {
    if (buffer_.get() != nullptr) {
      return unexpected_t(FirstFitAllocatorBase::Error::kAlreadyInUse);
    }
    if (size < 0 || chunk_size <= 0) {
      return unexpected_t(FirstFitAllocatorBase::Error::kInvalidSize);
    }
    chunk_size_ = chunk_size;
    number_of_chunks_ = get_number_of_chunks(size);
    // Allocate memory
    buffer_.reset(new (std::nothrow) T[number_of_chunks_ * chunk_size_]);
    if (buffer_.get() == nullptr) {
      return unexpected_t(FirstFitAllocatorBase::Error::kOutOfMemory);
    }
    // Prepare the memory management.
    auto res = memory_management_.allocate(number_of_chunks_);
    if (!res) {
      return unexpected_t(res.error());
    }
    return size;
  }

  /**
   * @brief Acquire a contiguous block of memory of the specified size.
   *
   * If a suitable contiguous block exists, returns a pointer to that block and the actual size
   * acquired. The actual size will be the smallest multiple of chunk_size that is greater than
   * or equal to the requested size.
   *
   * @param size Number of elements of type T to acquire.
   * @return Pair containing pointer to the allocated block and actual number of elements acquired
   *         on success, Error::kOutOfMemory if no suitable block exists.
   */
  expected_t<std::pair<T*, int32_t>> acquire(const int32_t size) {
    const int32_t number_of_chunks = get_number_of_chunks(size);
    auto res = memory_management_.acquire(number_of_chunks);
    if (!res) {
      return unexpected_t(res.error());
    }
    return std::make_pair(&buffer_.get()[res.value() * chunk_size_],
                          number_of_chunks * chunk_size_);
  }

  /**
   * @brief Release a previously acquired block of memory.
   *
   * The pointer must have been returned by a previous call to acquire(). Once released,
   * the memory becomes available for future acquisitions. A block cannot be released twice.
   *
   * @param ptr Pointer to the memory block to release.
   * @return Success or error status. Error::kBlockNotAllocated if the pointer was not
   *         previously acquired or is invalid.
   */
  expected_t<void> release(const T* ptr) {
    const T* begin = buffer_.get();
    const T* end = begin + (number_of_chunks_ * chunk_size_);
    if (ptr < begin || ptr >= end) {
      return unexpected_t(FirstFitAllocatorBase::Error::kBlockNotAllocated);
    }
    const int32_t index = ptr - begin;
    if (index % chunk_size_ != 0) {
      return unexpected_t(FirstFitAllocatorBase::Error::kBlockNotAllocated);
    }
    return memory_management_.release(index / chunk_size_);
  }

 private:
  /**
   * @brief Calculate the number of chunks needed for a given size.
   *
   * @param size Number of elements requested.
   * @return Number of chunks needed to accommodate at least the requested size.
   */
  int32_t get_number_of_chunks(const int32_t size) const {
    return (size + chunk_size_ - 1) / chunk_size_;
  }

  /// Real memory management implementation.
  FirstFitAllocatorBase memory_management_;
  /// Size of each chunk. Only multiples of this size can be allocated.
  int32_t chunk_size_{};
  /// Total number of chunks available.
  int32_t number_of_chunks_{};
  /// Buffer holding the pre-allocated memory that is provided on demand.
  std::unique_ptr<T[]> buffer_{nullptr};
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_HPP
