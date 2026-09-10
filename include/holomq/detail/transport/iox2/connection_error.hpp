/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/connection_failure.hpp"

namespace holomq::detail::transport {

[[nodiscard]] inline auto map_connection_error_code(const iox2::ConnectionFailure error)
    -> holomq::ErrorCode {
  switch (error) {
    case iox2::ConnectionFailure::FailedToEstablishConnection:
      return holomq::ErrorCode::kNotReady;
    case iox2::ConnectionFailure::UnableToMapSendersDataSegment:
      return holomq::ErrorCode::kStaleEndpoint;
  }
  return holomq::ErrorCode::kFailure;
}

[[nodiscard]] inline auto map_connection_error(const iox2::ConnectionFailure error)
    -> holomq::Error {
  return holomq::Error{map_connection_error_code(error)};
}

}  // namespace holomq::detail::transport
