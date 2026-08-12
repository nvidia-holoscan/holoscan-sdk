/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace holoscan {

/// @brief Identify the Core semantic representation of one logical sample.
///
/// The value classifies the complete logical sample after Core admission. It is
/// deliberately independent of transfer mode, locality, and private Channel or
/// HoloMQ discriminants. Private versioned records must map their values to this
/// enum field by field; this type is not a wire layout or an integer-cast ABI.
enum class PayloadKind : std::uint8_t {
  kInlineTyped = 0,           ///< Complete typed value is represented without an opaque lease.
  kSerializedBytes = 1,       ///< Complete self-contained schema value is represented as bytes.
  kDescriptorBacked = 2,      ///< A private opaque descriptor or lease backs the logical value.
  kCompatibilityPayload = 3,  ///< A compatibility adapter owns the complete payload representation.
  kControlEmpty = 4,          ///< Control sample intentionally carries no data payload.
};

}  // namespace holoscan
