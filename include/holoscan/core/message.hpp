/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_MESSAGE_HPP
#define HOLOSCAN_CORE_MESSAGE_HPP

#include <any>
#include <memory>
#include <utility>

#include "./common.hpp"

namespace holoscan {

/**
 * @brief Class to define a message.
 *
 * A message is a data structure that is used to pass data between operators.
 * It wraps a `std::any` object and provides a type-safe interface to access the data.
 *
 * This class is used by the `holoscan::gxf::GXFWrapper` to support the Holoscan native operator.
 * The `holoscan::gxf::GXFWrapper` will hold the object of this class and delegate the message to
 * the Holoscan native operator.
 */
class Message {
 public:
  /**
   * @brief Construct a new Message object.
   */
  Message() = default;

  /**
   * @brief Construct a new Message object
   *
   * @param value The value to be wrapped by the message.
   */
  template <typename typeT,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<typeT>, Message>>>
  explicit Message(typeT&& value) : value_(std::forward<typeT>(value)) {}

  /**
   * @brief Set the value object.
   *
   * @tparam ValueT The type of the value.
   * @param value The value to be wrapped by the message.
   */
  template <typename ValueT>
  void set_value(ValueT&& value) {
    value_ = std::forward<ValueT>(value);
  }

  /**
   * @brief Get the value object.
   *
   * @return The value wrapped by the message.
   */
  std::any value() const { return value_; }

  /**
   * @brief Get the value object as a specific type.
   *
   * @tparam ValueT The type of the value to be returned.
   * @return The value wrapped by the message.
   */
  template <typename ValueT>
  std::shared_ptr<ValueT> as() const {
    try {
      return std::any_cast<std::shared_ptr<ValueT>>(value_);
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR("The message doesn't have a value of type '{}': {}",
                         typeid(std::decay_t<ValueT>).name(),
                         e.what());
      return nullptr;
    }
  }

 private:
  std::any value_;  ///< The value wrapped by the message.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_MESSAGE_HPP */
