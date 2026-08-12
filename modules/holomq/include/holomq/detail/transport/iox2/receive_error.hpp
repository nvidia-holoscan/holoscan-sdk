/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/port_error.hpp"

namespace holomq::detail::transport {

[[nodiscard]] inline auto map_receive_error_code(const iox2::ReceiveError error)
    -> holomq::ErrorCode {
  switch (error) {
    case iox2::ReceiveError::ExceedsMaxBorrows:
      return holomq::ErrorCode::kBackpressured;
    case iox2::ReceiveError::FailedToEstablishConnection:
      return holomq::ErrorCode::kNotReady;
    case iox2::ReceiveError::UnableToMapSendersDataSegment:
      return holomq::ErrorCode::kStaleEndpoint;
  }
  return holomq::ErrorCode::kReceiveError;
}

}  // namespace holomq::detail::transport
