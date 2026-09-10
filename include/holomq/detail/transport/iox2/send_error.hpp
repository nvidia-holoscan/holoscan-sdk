/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/port_error.hpp"

namespace holomq::detail::transport {

[[nodiscard]] inline auto map_send_error(const iox2::SendError error) -> holomq::Error {
  holomq::ErrorCode code = holomq::ErrorCode::kFailure;
  switch (error) {
    case iox2::SendError::ConnectionBrokenSinceSenderNoLongerExists:
      code = holomq::ErrorCode::kClosed;
      break;
    case iox2::SendError::ConnectionCorrupted:
      code = holomq::ErrorCode::kStaleEndpoint;
      break;
    case iox2::SendError::LoanErrorOutOfMemory:
    case iox2::SendError::LoanErrorExceedsMaxLoans:
    case iox2::SendError::UnableToDeliver:
      code = holomq::ErrorCode::kBackpressured;
      break;
    case iox2::SendError::LoanErrorExceedsMaxLoanSize:
      code = holomq::ErrorCode::kInvalidArgument;
      break;
    case iox2::SendError::ConnectionError:
      code = holomq::ErrorCode::kNotReady;
      break;
    case iox2::SendError::LoanErrorInternalFailure:
    case iox2::SendError::InternalError:
      break;
  }
  return holomq::Error{code};
}

}  // namespace holomq::detail::transport
