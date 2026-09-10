/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>

#include "iox2/port_error.hpp"

namespace holomq::detail::transport {

[[nodiscard]] inline auto map_loan_error(const iox2::LoanError error) -> holomq::Error {
  holomq::ErrorCode code = holomq::ErrorCode::kFailure;
  switch (error) {
    case iox2::LoanError::OutOfMemory:
    case iox2::LoanError::ExceedsMaxLoanedSamples:
      code = holomq::ErrorCode::kBackpressured;
      break;
    case iox2::LoanError::ExceedsMaxLoanSize:
      code = holomq::ErrorCode::kInvalidArgument;
      break;
    case iox2::LoanError::InternalFailure:
      break;
  }
  return holomq::Error{code};
}

}  // namespace holomq::detail::transport
