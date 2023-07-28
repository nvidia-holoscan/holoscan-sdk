/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
