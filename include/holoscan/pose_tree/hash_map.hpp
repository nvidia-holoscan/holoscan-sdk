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
#ifndef HOLOSCAN_POSE_TREE_HASH_MAP_HPP
#define HOLOSCAN_POSE_TREE_HASH_MAP_HPP

#include <memory>
#include <shared_mutex>
#include <utility>

#include "holoscan/core/expected.hpp"
#include "holoscan/pose_tree/math/pose3.hpp"

namespace holoscan::pose_tree {

/**
 * @brief A hash map implementation using open addressing with linear probing.
 *
 * This class provides a fixed-capacity hash map that uses pre-allocated memory and linear probing
 * for collision resolution. It supports basic operations like insert, get, erase, and has.
 * The hash map is designed for performance-critical scenarios where memory allocation should be
 * minimized.
 *
 * @tparam Key The key type for the hash map.
 * @tparam Value The value type for the hash map.
 */
template <typename Key, typename Value>
class HashMap {
 public:
  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kInvalidArgument is returned when a function is called with argument that does not make
    /// sense such as negative size or capacity smaller than size.
    kInvalidArgument,
    /// kOutOfMemory is returned when a memory allocation fails.
    kOutOfMemory,
    /// kHashMapFull is returned if the hash map is full.
    kHashMapFull,
    /// kKeyNotFound is returned if the key is not found in the hash map.
    kKeyNotFound,
    /// kKeyAlreadyExists is returned if the key already exists in the hash map.
    kKeyAlreadyExists
  };

  /// Expected type used by this class.
  template <typename T>
  using expected_t = expected<T, Error>;
  /// Unexpected type used by this class.
  using unexpected_t = unexpected<Error>;

  /// Hash type used internally.
  using hash_t = uint64_t;

  /**
   * @brief Default constructor used to be able to pre-allocate memory.
   */
  HashMap() = default;

  /**
   * @brief Reserve memory for the hash map.
   *
   * @param size Maximum number of elements the hash map can store.
   * @param capacity Total capacity of the internal array (should be larger than size for
   * efficiency).
   * @return Success or error status.
   */
  expected_t<void> reserve(int32_t size, int32_t capacity) {
    if (size <= 0) {
      return unexpected_t(Error::kInvalidArgument);
    }
    if (capacity < size) {
      return unexpected_t(Error::kInvalidArgument);
    }
    entries_.reset(new Entry[capacity]);
    for (int32_t i = 0; i < capacity; ++i) {
      entries_[i].is_occupied = false;
    }
    max_size_ = size;
    capacity_ = capacity;
    return {};
  }

  /**
   * @brief Check if a key exists in the hash map.
   *
   * @param key The key to search for.
   * @return True if the key exists, false otherwise.
   */
  bool has(const Key& key) const {
    hash_t hash = std::hash<Key>{}(key);
    int32_t index = hash % capacity_;
    while (entries_[index].is_occupied) {
      if (entries_[index].hash == hash && entries_[index].key == key) {
        return true;
      }
      ++index;
      if (index == capacity_) {
        index = 0;
      }
    }
    return false;
  }

  /**
   * @brief Get the value associated with a key.
   *
   * @param key The key to search for.
   * @return Value associated with the key on success, Error::kKeyNotFound if key doesn't exist.
   */
  expected_t<Value> get(const Key& key) const {
    hash_t hash = std::hash<Key>{}(key);
    int32_t index = hash % capacity_;
    while (entries_[index].is_occupied) {
      if (entries_[index].hash == hash && entries_[index].key == key) {
        return entries_[index].value;
      }
      ++index;
      if (index == capacity_) {
        index = 0;
      }
    }
    return unexpected_t(Error::kKeyNotFound);
  }

  /**
   * @brief Insert a key-value pair into the hash map.
   *
   * @param key The key to insert.
   * @param value The value to associate with the key.
   * @return Success or error status. Error::kHashMapFull if the map is full,
   *         Error::kKeyAlreadyExists if the key already exists.
   */
  expected_t<void> insert(const Key& key, Value value) {
    if (size_ >= max_size_) {
      return unexpected_t(Error::kHashMapFull);
    }
    hash_t hash = std::hash<Key>{}(key);
    int32_t index = hash % capacity_;
    while (entries_[index].is_occupied) {
      if (entries_[index].hash == hash && entries_[index].key == key) {
        return unexpected_t(Error::kKeyAlreadyExists);
      }
      ++index;
      if (index == capacity_) {
        index = 0;
      }
    }
    entries_[index].hash = hash;
    entries_[index].key = key;
    entries_[index].value = std::move(value);
    entries_[index].is_occupied = true;
    ++size_;
    return {};
  }

  /**
   * @brief Remove a key-value pair from the hash map.
   *
   * @param key The key to remove.
   * @return Success or error status. Error::kKeyNotFound if the key doesn't exist.
   */
  expected_t<void> erase(const Key& key) {
    hash_t hash = std::hash<Key>{}(key);
    int32_t index = hash % capacity_;
    while (entries_[index].is_occupied) {
      if (entries_[index].hash == hash && entries_[index].key == key) {
        entries_[index].is_occupied = false;
        --size_;
        fill_holes(index);
        return {};
      }
      ++index;
      if (index == capacity_) {
        index = 0;
      }
    }
    return unexpected_t(Error::kKeyNotFound);
  }

  /**
   * @brief Get the current number of elements in the hash map.
   *
   * @return Current size of the hash map.
   */
  int32_t size() const { return size_; }

 private:
  /**
   * @brief Internal structure representing an entry in the hash map.
   */
  struct Entry {
    /// Hash value of the key.
    hash_t hash;
    /// The key.
    Key key;
    /// Whether this entry is occupied.
    bool is_occupied;
    /// The value associated with the key.
    Value value;
  };

  /**
   * @brief Fill holes in the hash table after deletion to maintain linear probing invariants.
   *
   * This method is called after an element is deleted to ensure that subsequent lookups
   * continue to work correctly with linear probing.
   *
   * @param index The index where a deletion occurred.
   */
  void fill_holes(int index) {
    while (true) {
      int32_t start_index = index;
      int32_t swap_index = index;
      while (true) {
        ++index;
        if (index == capacity_) {
          index = 0;
          start_index -= capacity_;
        }
        if (!entries_[index].is_occupied) {
          return;
        }
        int32_t target_index = entries_[index].hash % capacity_;
        if (target_index > index) {
          target_index -= capacity_;
        }
        if (target_index <= start_index) {
          std::swap(entries_[index], entries_[swap_index]);
          break;
        }
      }
    }
  }

  /// Array of entries in the hash map.
  std::unique_ptr<Entry[]> entries_;
  /// Current number of elements in the hash map.
  int32_t size_ = 0;
  /// Maximum number of elements the hash map can store.
  int32_t max_size_ = 0;
  /// Total capacity of the internal array.
  int32_t capacity_ = 0;
};

}  // namespace holoscan::pose_tree

#endif  // HOLOSCAN_POSE_TREE_HASH_MAP_HPP
