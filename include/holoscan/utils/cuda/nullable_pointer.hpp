/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_UTILS_CUDA_NULLABLE_POINTER_HPP
#define HOLOSCAN_UTILS_CUDA_NULLABLE_POINTER_HPP

#include <cstddef>

/**
 * Helper class for using handles with std::unique_ptr which requires that a custom
 * handle type satisfies NullablePointer
 * https://en.cppreference.com/w/cpp/named_req/NullablePointer.
 *
 * @tparam T type to hold
 */
namespace holoscan {

template <typename T>
class Nullable {
 public:
  explicit Nullable(T value = 0) : value_(value) {}
  explicit Nullable(std::nullptr_t) : value_(0) {}
  operator T() const { return value_; }
  explicit operator bool() { return value_ != 0; }

  friend bool operator==(Nullable l, Nullable r) { return l.value_ == r.value_; }
  friend bool operator!=(Nullable l, Nullable r) { return !(l == r); }
  friend bool operator==(Nullable l, std::nullptr_t) { return l.value_ == 0; }
  friend bool operator==(std::nullptr_t, Nullable r) { return r.value_ == 0; }
  friend bool operator!=(Nullable l, std::nullptr_t) { return l.value_ != 0; }
  friend bool operator!=(std::nullptr_t, Nullable r) { return r.value_ != 0; }

  /**
   * Deleter, call the function when the object is deleted.
   *
   * @tparam F function to call
   */
  template <typename RESULT, RESULT func(T)>
  struct Deleter {
    typedef Nullable<T> pointer;
    void operator()(T value) const { func(value); }
  };

 private:
  T value_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_UTILS_CUDA_NULLABLE_POINTER_HPP */
