/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/listener_error.hpp"

namespace holomq::detail::transport {

[[nodiscard]] inline auto map_listener_wait_error_code(const iox2::ListenerWaitError error)
    -> holomq::ErrorCode {
  switch (error) {
    case iox2::ListenerWaitError::ContractViolation:
      return holomq::ErrorCode::kProtocolError;
    case iox2::ListenerWaitError::InterruptSignal:
      return holomq::ErrorCode::kInterrupted;
    case iox2::ListenerWaitError::InternalFailure:
      return holomq::ErrorCode::kFailure;
  }
  return holomq::ErrorCode::kFailure;
}

}  // namespace holomq::detail::transport
