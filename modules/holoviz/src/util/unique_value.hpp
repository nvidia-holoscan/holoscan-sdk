/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOVIZ_SRC_UTIL_UNIQUE_VALUE_HPP
#define HOLOVIZ_SRC_UTIL_UNIQUE_VALUE_HPP

#include <algorithm>
#include <utility>

#include "non_copyable.hpp"

namespace holoscan::viz {

/**
 * A RAII-style class holding a value (e.g. handle) and calling a function when control
 * flow leaves the scope.
 * Typically the function to be called is to free the object.
 * Supports std::unique_ptr functionality.
 *
 * @tparam T type to be used
 * @tparam TF signature of the function to be called
 * @tparam F function to be called
 */
template <typename T, typename TF, TF F>
class UniqueValue : public NonCopyable {
 public:
  /**
   * Construct
   */
  UniqueValue() : value_(T()) {}
  /**
   * Construct from value
   *
   * @param value initial value
   */
  explicit UniqueValue(T value) : value_(value) {}

  /**
   * Move constructor
   *
   * @param other  the object to transfer ownership from
   */
  UniqueValue(UniqueValue&& other) noexcept : value_(other.release()) {}

  ~UniqueValue() { reset(); }

  /**
   * Release the value
   *
   * @returns value
   */
  T release() noexcept {
    T value = value_;
    value_ = T();
    return value;
  }

  /**
   * Reset with new value. Previous will be destroyed.
   *
   * @param value new value
   */
  void reset(T value = T()) noexcept {
    T old_value = value_;
    value_ = value;
    if (old_value != T()) { F(old_value); }
  }

  /**
   * Swap
   */
  void swap(UniqueValue& other) noexcept { std::swap(value_, other.value_); }

  /**
   * Move assignment operator
   *
   * @param other  the object to transfer ownership from
   */
  UniqueValue& operator=(UniqueValue&& other) noexcept {
    reset(other.release());
    return *this;
  }

  /**
   * @return the value
   */
  T get() const noexcept { return value_; }

  /**
   * @returns true if the value is set
   */
  explicit operator bool() const noexcept { return (value_ != T()); }

  /**
   * @returns reference to value
   */
  T& operator*() const { return value_; }

  /**
   * @returns value
   */
  T operator->() const noexcept { return value_; }

  /**
   * @returns true if equal
   */
  bool operator==(const UniqueValue& other) const { return (value_ == other.value_); }

  /**
   * @returns true if not equal
   */
  bool operator!=(const UniqueValue& other) const { return !(operator==(other)); }

 private:
  T value_;
};

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_UTIL_UNIQUE_VALUE_HPP */
