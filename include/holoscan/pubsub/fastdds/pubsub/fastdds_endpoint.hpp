/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_ENDPOINT_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_ENDPOINT_HPP

#include <cstdint>
#include <vector>

#include <holoscan/core/endpoint.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/**
 * @brief Endpoint implementation for DDS serialization.
 *
 * This class implements the holoscan::Endpoint interface to enable
 * serialization/deserialization to/from a byte buffer, which can then be used
 * with DDS DataWriter/DataReader.
 *
 * Two write-mode storage options are provided:
 * - **Vector mode**: backed by a `std::vector<uint8_t>` that grows automatically.
 * - **Raw-buffer mode**: backed by a caller-supplied fixed-size `uint8_t*`
 *   buffer; writes that would exceed the buffer size fail instead of growing.
 *
 * Read mode always operates over a `const std::vector<uint8_t>*`.
 *
 * ## Usage
 *
 * **Write Mode — vector** (serialization):
 * ```cpp
 * std::vector<uint8_t> buffer;
 * FastDdsEndpoint endpoint(&buffer);
 *
 * int32_t value = 42;
 * endpoint.write_trivial_type(&value);
 * // buffer now contains serialized data
 * ```
 *
 * **Write Mode — raw buffer** (serialization into a fixed-size buffer):
 * ```cpp
 * uint8_t buf[1024];
 * FastDdsEndpoint endpoint(buf, sizeof(buf));
 *
 * int32_t value = 42;
 * endpoint.write_trivial_type(&value);
 * // buf[0..bytes_written()) contains serialized data
 * ```
 *
 * **Read Mode** (deserialization):
 * ```cpp
 * const std::vector<uint8_t>& data = received_from_dds();
 * FastDdsEndpoint endpoint(&data);
 *
 * int32_t value;
 * endpoint.read_trivial_type(&value);
 * ```
 *
 * ## Thread Safety
 *
 * FastDdsEndpoint is NOT thread-safe. Each serialization/deserialization operation
 * should use its own endpoint instance.
 *
 * ## Memory Management
 *
 * FastDdsEndpoint does not own the buffer - it holds a non-owning pointer.
 * The caller must ensure the buffer outlives the endpoint.
 */
class FastDdsEndpoint : public holoscan::Endpoint {
 public:
  /**
   * @brief Construct a FastDdsEndpoint in write mode.
   *
   * Data will be appended to the output buffer.
   *
   * @param output_buffer Non-null pointer to the output buffer.
   *                      Buffer is NOT cleared; data is appended.
   * @throws std::invalid_argument if output_buffer is nullptr
   */
  explicit FastDdsEndpoint(std::vector<uint8_t>* output_buffer);

  /**
   * @brief Construct a FastDdsEndpoint in write mode with a fixed-size raw buffer.
   *
   * Data will be written starting at position 0. Unlike the vector mode,
   * writes that exceed buffer_size will fail (the buffer does not grow).
   *
   * @param buffer Non-null pointer to the output buffer.
   * @param buffer_size Available buffer size in bytes.
   * @throws std::invalid_argument if buffer is nullptr
   */
  FastDdsEndpoint(uint8_t* buffer, size_t buffer_size);

  /**
   * @brief Construct a FastDdsEndpoint in read mode.
   *
   * Data will be read from the input buffer starting at position 0.
   *
   * @param input_buffer Non-null pointer to the input buffer.
   * @throws std::invalid_argument if input_buffer is nullptr
   */
  explicit FastDdsEndpoint(const std::vector<uint8_t>* input_buffer);

  /**
   * @brief Default destructor.
   */
  ~FastDdsEndpoint() override = default;

  // Delete copy operations (non-owning pointer semantics)
  FastDdsEndpoint(const FastDdsEndpoint&) = delete;
  FastDdsEndpoint& operator=(const FastDdsEndpoint&) = delete;

  // Allow move operations
  FastDdsEndpoint(FastDdsEndpoint&&) = default;
  FastDdsEndpoint& operator=(FastDdsEndpoint&&) = default;

  //----------------------------------------------------------------------------
  // holoscan::Endpoint Interface
  //----------------------------------------------------------------------------

  /**
   * @brief Check if write operations are available.
   * @return true if constructed in write mode with valid buffer
   */
  bool is_write_available() override;

  /**
   * @brief Check if read operations are available.
   * @return true if constructed in read mode with data remaining
   */
  bool is_read_available() override;

  /**
   * @brief Write data to the buffer.
   *
   * Appends the specified data to the output buffer.
   *
   * @param data Pointer to data to write
   * @param size Number of bytes to write
   * @return Number of bytes written, or error if in read mode
   */
  expected<size_t, RuntimeError> write(const void* data, size_t size) override;

  /**
   * @brief Read data from the buffer.
   *
   * Reads data from the current position and advances the read position.
   *
   * @param data Pointer to destination buffer
   * @param size Number of bytes to read
   * @return Number of bytes read, or error if insufficient data or in write mode
   */
  expected<size_t, RuntimeError> read(void* data, size_t size) override;

  /**
   * @brief Write a pointer reference for zero-copy support.
   *
   * For DDS, GPU device memory cannot be sent directly. This method fails
   * for `kDevice` memory — the caller must stage device data to host memory
   * before serialization. `kCudaManaged` memory is accepted (with a warning
   * to ensure GPU operations are complete), and `kHost`/`kSystem` memory is
   * written directly via write().
   *
   * @param pointer Pointer to data
   * @param size Size of data in bytes
   * @param type Memory storage type (host, device, system, cuda_managed)
   * @return Success for host/system/managed memory, error for device memory
   */
  expected<void, RuntimeError> write_ptr(const void* pointer, size_t size,
                                         holoscan::MemoryStorageType type) override;

  //----------------------------------------------------------------------------
  // FastDdsEndpoint-specific Methods
  //----------------------------------------------------------------------------

  /**
   * @brief Reset the read position to the beginning.
   *
   * Allows re-reading the buffer from the start.
   * Only valid in read mode.
   */
  void reset_read_position();

  /**
   * @brief Get the current read position.
   * @return Current read position (0 in write mode)
   */
  size_t read_position() const { return read_position_; }

  /**
   * @brief Get the total bytes written (write mode) or buffer size (read mode).
   * @return Total size in bytes
   */
  size_t size() const;

  /**
   * @brief Get remaining bytes available for reading.
   * @return Bytes remaining (0 in write mode)
   */
  size_t bytes_remaining() const;

  /**
   * @brief Get the number of bytes written (raw buffer mode).
   * @return Bytes written in raw buffer mode, or vector size in vector mode, or 0.
   */
  size_t bytes_written() const;

  /**
   * @brief Check if in write mode.
   * @return true if constructed with output buffer (vector or raw)
   */
  bool is_write_mode() const { return output_buffer_ != nullptr || raw_output_buffer_ != nullptr; }

  /**
   * @brief Check if in read mode.
   * @return true if constructed with input buffer
   */
  bool is_read_mode() const { return input_buffer_ != nullptr; }

 private:
  std::vector<uint8_t>* output_buffer_ = nullptr;  ///< Write mode buffer (non-owning, vector)
  uint8_t* raw_output_buffer_ = nullptr;           ///< Write mode buffer (non-owning, raw)
  size_t raw_buffer_size_ = 0;                     ///< Raw buffer capacity
  size_t write_position_ = 0;                      ///< Current write position (raw buffer mode)
  const std::vector<uint8_t>* input_buffer_ = nullptr;  ///< Read mode buffer (non-owning)
  size_t read_position_ = 0;                            ///< Current read position
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_ENDPOINT_HPP
