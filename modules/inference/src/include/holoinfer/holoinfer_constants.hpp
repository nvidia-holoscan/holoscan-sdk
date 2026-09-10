/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <string>

#include <holoinfer/holoinfer_export.h>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace inference {

/// @brief Supported tensor element datatypes across the HoloInfer boundary.
enum class holoinfer_datatype {
  h_Float32 = 0,     ///< 32-bit IEEE float
  h_Int8 = 1,        ///< 8-bit signed integer
  h_Int32 = 2,       ///< 32-bit signed integer
  h_Int64 = 3,       ///< 64-bit signed integer
  h_UInt8 = 4,       ///< 8-bit unsigned integer
  h_Float16 = 5,     ///< IEEE half-precision float
  h_Bool = 6,        ///< Boolean (stored as u8)
  h_Unsupported = 7  ///< Sentinel for unsupported/unknown dtype
};

/// @brief Codes for supported inference backends
enum class holoinfer_backend { h_trt = 0, h_unknown = 1 };

/// @brief Holoscan Inference Toolkit status codes
enum class holoinfer_code { H_SUCCESS, H_ERROR, H_EXCEPTION, H_WARNING };

/// @brief Holoscan Inference toolkit status class
/// contains status code and related message
class HOLOINFER_EXPORT InferStatus {
  holoinfer_code _code;
  std::string _message;

 public:
  /// @brief Return the status code carried by this InferStatus.
  holoinfer_code get_code() const { return _code; }
  /// @brief Return the human-readable message associated with the status.
  std::string get_message() const { return _message; }
  /// @brief Overwrite the status code.
  void set_code(const holoinfer_code& _c) { _code = _c; }
  /// @brief Overwrite the human-readable message.
  void set_message(const std::string& _m) { _message = _m; }
  /// @brief Emit the message at a log level matching the current status code.
  void display_message() const {
    switch (_code) {
      case holoinfer_code::H_WARNING: {
        HOLOSCAN_LOG_WARN(_message);
        break;
      }
      case holoinfer_code::H_SUCCESS:
      default: {
        HOLOSCAN_LOG_INFO(_message);
        break;
      }
      case holoinfer_code::H_ERROR: {
        HOLOSCAN_LOG_ERROR(_message);
        break;
      }
      case holoinfer_code::H_EXCEPTION: {
        HOLOSCAN_LOG_ERROR(_message);
        break;
      }
    }
  }
  /// @brief Construct an InferStatus with the given code and optional message.
  InferStatus(const holoinfer_code& code = holoinfer_code::H_SUCCESS,
              const std::string& message = "")
      : _code(code), _message(message) {}
};

using TimePoint = std::chrono::steady_clock::time_point;
using byte = unsigned char;

}  // namespace inference
}  // namespace holoscan
