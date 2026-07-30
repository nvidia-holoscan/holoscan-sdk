/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_ENDPOINT_HPP
#define HOLOSCAN_CORE_ENDPOINT_HPP

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "./errors.hpp"
#include "./expected.hpp"
#include "./resource.hpp"

namespace holoscan {

/**
 * @brief Holoscan serialization endpoint.
 *
 */
class Endpoint : public Resource {
 public:
  Endpoint() = default;
  Endpoint(Endpoint&&) = default;
  ~Endpoint() override = default;

  // C++ API wrappers
  virtual bool is_write_available() = 0;
  virtual bool is_read_available() = 0;
  virtual expected<size_t, RuntimeError> write(const void* data, size_t size) = 0;
  virtual expected<size_t, RuntimeError> read(void* data, size_t size) = 0;
  virtual expected<void, RuntimeError> write_ptr(const void* pointer, size_t size,
                                                 holoscan::MemoryStorageType type) = 0;

  // Writes an object of type T to the endpoint
  template <typename T>
  expected<size_t, RuntimeError> write_trivial_type(const T* object) {
    return write(object, sizeof(T));
  }

  // Reads an object of type T from the endpoint
  template <typename T>
  expected<size_t, RuntimeError> read_trivial_type(T* object) {
    return read(object, sizeof(T));
  }
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_ENDPOINT_HPP */
