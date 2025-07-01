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
#include "holoscan/core/resources/gxf/first_fit_allocator_base.hpp"

#include <algorithm>
#include <new>

namespace holoscan {

void FirstFitAllocatorBase::Memory::update(const Memory& left_child, const Memory& right_child) {
  // The memory available from the left is either left_child.left or if left_child is fully empty,
  // it is left_child.size + right_child.left.
  left = left_child.size == left_child.left ? (left_child.size + right_child.left)
                                            : left_child.left;
  // Same for the memory available from the right side.
  right = right_child.size == right_child.right ? (right_child.size + left_child.right)
                                                : right_child.right;
  // The maximum length interval is either the max of one of the children, or is between both
  // children: left_child.right + right_child.left
  max = std::max(right_child.left + left_child.right, std::max(right_child.max, left_child.max));
}

void FirstFitAllocatorBase::Memory::set(const int32_t free) {
  left = size * free;
  right = size * free;
  max = size * free;
}

FirstFitAllocatorBase::FirstFitAllocatorBase() : size_(0), last_layer_first_index_(0) {}

FirstFitAllocatorBase::expected_t<void> FirstFitAllocatorBase::allocate(const int32_t size) {
  // Check that the memory is not already allocated and in use:
  if (tree_.get() != nullptr && tree_.get()[1].max != size_) {
    return unexpected_t(Error::kAlreadyInUse);
  }
  // Quick check size is valid [1, 2^30[
  if (size <= 0 || size >= 1<<30) {
    return unexpected_t(Error::kInvalidSize);
  }
  // Compute the memory used
  size_ = size;
  last_layer_first_index_ = 1;
  // Given size < 2^30, we are guaranteed that last_layer_first_index_ won't overflow.
  while (last_layer_first_index_ < size) last_layer_first_index_ *= 2;
  // Allocate the memory and check it worked
  tree_.reset(new(std::nothrow) Memory[last_layer_first_index_ * 2]);
  if (tree_.get() == nullptr) {
    return unexpected_t(Error::kOutOfMemory);
  }
  // Prepared the binary tree.
  // We first update the leaves. As we reserved potentially more memory to have a power of 2, the
  // leaves < size will be fully free ({1, 1, 1, 1}) and the leaves after will hold no data.
  for (int32_t idx = 0; idx < last_layer_first_index_; idx++) {
    if (idx < size_) {
      tree_.get()[last_layer_first_index_ + idx] = {1, 1, 1, 1};
    } else {
      tree_.get()[last_layer_first_index_ + idx] = {0, 0, 0, 0};
    }
  }
  // We can now update backward the other nodes.
  for (int32_t idx = last_layer_first_index_ - 1; idx > 0; idx--) {
    // First we set the size which is constant over time.
    tree_.get()[idx].size = tree_.get()[2*idx].size + tree_.get()[2*idx + 1].size;
    tree_.get()[idx].update(tree_.get()[2*idx], tree_.get()[2*idx + 1]);
  }
  return expected_t<void>{};
}

FirstFitAllocatorBase::expected_t<void> FirstFitAllocatorBase::release(const int32_t index) {
  if (index < 0 || index >= size_) {
    return unexpected_t(Error::kInvalidSize);
  }
  // Check the block has been allocated before
  if (tree_[last_layer_first_index_ + index].max != -1) {
    return unexpected_t(Error::kBlockNotAllocated);
  }
  const int32_t size = tree_[last_layer_first_index_ + index].size;
  tree_[last_layer_first_index_ + index].size = 1;
  tree_[last_layer_first_index_ + index].max = 1;
  // Make the actual update
  update(index, index + size, /* free = */ 1);
  return expected_t<void>{};
}

FirstFitAllocatorBase::expected_t<int32_t> FirstFitAllocatorBase::acquire(const int32_t size) {
  auto* ptr = tree_.get();
  if (ptr == nullptr) {
    return unexpected_t(Error::kOutOfMemory);
  }
  int32_t idx = 1;
  if (ptr[idx].max < size) {
    // TODO(bbutin): Consider doing garbage collection?
    return unexpected_t(Error::kOutOfMemory);
  }
  // Store the index in the memory, the root start at 0, and each time we jump into the right
  // child we will need to increment index by the left child size.
  int32_t index = 0;
  while (idx < 2 * last_layer_first_index_) {
    if (ptr[idx].left >= size) {
      // The first available spot is the left most index.
      update(index, index + size, /* free = */ 0);
      // Store the size allocated that starts at the given index.
      tree_[last_layer_first_index_ + index].max = -1;
      tree_[last_layer_first_index_ + index].size = size;
      return index;
    } else if (ptr[2*idx].max >= size) {
      // The next available spot is on the left side.
      idx *= 2;
    } else if (ptr[2*idx].right + ptr[2*idx+1].left >= size) {
      // There is an available spot in between both children.
      index += ptr[2*idx].size - ptr[2*idx].right;
      update(index, index + size, /* free = */ 0);
      // Store the size allocated that starts at the given index.
      tree_[last_layer_first_index_ + index].max = -1;
      tree_[last_layer_first_index_ + index].size = size;
      return index;
    } else {
      // The next available spot is on the right side.
      index += ptr[2*idx].size;
      idx = idx * 2 + 1;
    }
  }
  // This should not happen unless the tree has been corrupted, which should be impossible!
  return unexpected_t(Error::kLogicError);
}

void FirstFitAllocatorBase::propagate_to_root(int32_t idx) {
  auto* ptr = tree_.get();
  while (idx > 0) {
    ptr[idx].update(ptr[2*idx], ptr[2*idx + 1]);
    idx /= 2;
  }
}

void FirstFitAllocatorBase::update(int32_t left, int32_t right, const int32_t free) {
  // Update left and right to be inclusive and being the leaf.
  left += last_layer_first_index_;
  right += last_layer_first_index_ - 1;
  auto* ptr = tree_.get();
  while (left <= right) {
    if (left&1) {
      // If left is a right child we need to update it and move to the sibling of the parent.
      ptr[left].set(free);
      propagate_to_root(left / 2);
      left = (left + 1) / 2;
    } else {
      // Otherwise updating the parent is enough.
      left /= 2;
    }
    if (right&1) {
      // If right is a right child, updating the parent is enough
      right /= 2;
    } else {
      // If right is a left child we need to update it and move to the sibling of the parent.
      ptr[right].set(free);
      propagate_to_root(right / 2);
      right = (right - 1) / 2;
    }
  }
}

}  // namespace holoscan
