/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CORE_DUMMY_CLASSES_HPP
#define CORE_DUMMY_CLASSES_HPP

#include <iostream>

namespace holoscan {

class DummyIntClass {
 public:
  explicit DummyIntClass(int val) : val_(val) {
    std::cout << "DummyIntClass " << val_ << std::endl;
  }
  // copy constructor
  DummyIntClass(const DummyIntClass& other) {
    val_ = other.val_;
    std::cout << "DummyIntClass copy " << val_ << std::endl;
  }
  // move constructor
  DummyIntClass(DummyIntClass&& other) {
    val_ = other.val_;
    std::cout << "DummyIntClass move " << val_ << std::endl;
  }
  // copy assignment
  DummyIntClass& operator=(const DummyIntClass& other) {
    val_ = other.val_;
    std::cout << "DummyIntClass copy assignment " << val_ << std::endl;
    return *this;
  }
  // move assignment
  DummyIntClass& operator=(DummyIntClass&& other) {
    val_ = other.val_;
    std::cout << "DummyIntClass move assignment " << val_ << std::endl;
    return *this;
  }
  // equality operator
  bool operator==(const DummyIntClass& other) const { return val_ == other.val_; }

  // inequality operator
  bool operator!=(const DummyIntClass& other) const { return val_ != other.val_; }

  // return value
  int get() const { return val_; }

  // set value
  void set(int val) { val_ = val; }

 private:
  int32_t val_ = 0;
};

}  // namespace holoscan

#endif /* CORE_DUMMY_CLASSES_HPP */
