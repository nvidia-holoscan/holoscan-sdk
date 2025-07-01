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
#ifndef HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_BASE_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_BASE_HPP

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include "holoscan/core/expected.hpp"

namespace holoscan {

/**
 * @brief Memory management helper class using first-fit allocation strategy.
 *
 * This class keeps track of allocated and free regions within a large memory chunk and can
 * efficiently find the first available block that fits a requested size. It works only with
 * indices and relies on an external class to hold the actual memory.
 *
 * Internally, it uses a binary segment tree to track the largest available block in any given
 * area of memory. All operations have logarithmic time complexity.
 */
class FirstFitAllocatorBase {
 public:
  /**
   * @brief Error codes used by the first-fit allocator classes.
   */
  enum class Error {
    /// Returned when the suballocator is already in use and some memory has not been released yet.
    kAlreadyInUse,
    /// Returned if the size is invalid (negative or too big).
    kInvalidSize,
    /// Returned if the class can't allocate enough memory.
    kOutOfMemory,
    /// Returned if we attempt to release a block of memory not allocated yet.
    kBlockNotAllocated,
    /// This error happens when there is a logical issue during execution. This should never happen.
    kLogicError,
  };

  /// Expected type used by this class.
  template <typename T>
  using expected_t = expected<T, Error>;
  /// Unexpected type used by this class.
  using unexpected_t = unexpected<Error>;

  /**
   * @brief Default constructor.
   */
  FirstFitAllocatorBase();

  /**
   * @brief Allocate internal data structures to handle memory requests for a given size.
   *
   * Requires 32 * 2^ceil(log2(size)) bytes of memory for internal bookkeeping.
   *
   * @param size Maximum size of memory that can be managed.
   * @return Success or error status. Error::kInvalidSize if size is invalid.
   */
  expected_t<void> allocate(int32_t size);

  /**
   * @brief Acquire a contiguous block of memory of the specified size.
   *
   * If such a contiguous block exists, returns the lowest index where the block starts.
   * This block will be marked as allocated until a call to release() is made.
   *
   * @param size Size of the memory block to acquire.
   * @return Index of the allocated block on success, Error::kOutOfMemory if no suitable block
   * exists.
   */
  expected_t<int32_t> acquire(int32_t size);

  /**
   * @brief Release a previously acquired block of memory.
   *
   * The block must have been previously acquired using acquire(). Once released, the memory
   * becomes available for future allocations.
   *
   * @param index Starting index of the block to release.
   * @return Success or error status. Error::kBlockNotAllocated if the block was not previously
   * allocated.
   */
  expected_t<void> release(int32_t index);

 private:
  /**
   * @brief Helper data structure for efficient memory block searching.
   *
   * Represents a node in a binary segment tree used to track free and allocated memory regions.
   */
  struct Memory {
    /**
     * @brief Update this node using information from both children.
     *
     * Updates the node's properties based on its left and right children:
     * - size = left.size + right.size
     * - left = left.left or left.size + right.left iff left.max == left.size
     * - right = right.right or right.right + left.right iff right.max == right.size
     * - max = max(left.max, right.max, left.right + right.left)
     *
     * @param left_child Left child node.
     * @param right_child Right child node.
     */
    void update(const Memory& left_child, const Memory& right_child);

    /**
     * @brief Mark a node as free or allocated.
     *
     * @param free 1 if the node should be marked as free, 0 if allocated.
     */
    void set(int32_t free);

    /// Size of available memory starting from the left side of the subtree.
    int32_t left;
    /// Size of available memory starting from the right side of the subtree.
    int32_t right;
    /// Size of the largest available block in the subtree. Contains -1 when a block starts at this
    /// index.
    int32_t max;
    /// Constant size of the subtree. When a block is acquired, contains the size that was acquired.
    int32_t size;
  };

  /**
   * @brief Update the tree path from a given node to the root.
   *
   * @param idx Index of the node to start propagation from.
   */
  void propagate_to_root(int32_t idx);

  /**
   * @brief Update a segment of the tree and mark it as free or allocated.
   *
   * @param left Left boundary of the segment (inclusive).
   * @param right Right boundary of the segment (exclusive).
   * @param free 1 if marking as free, 0 if marking as allocated.
   */
  void update(int32_t left, int32_t right, int32_t free);

  /// Binary segment tree for memory management.
  std::unique_ptr<Memory[]> tree_;
  /// Total size available at initialization.
  int32_t size_;
  /// Helper index pointing to the first leaf node in the tree.
  int32_t last_layer_first_index_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_RESOURCES_GXF_FIRST_FIT_ALLOCATOR_BASE_HPP
