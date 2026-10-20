/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/delivery_attribution.hpp>

#include <iox2/unique_port_id.hpp>

#include <algorithm>
#include <bit>

namespace holomq::detail::transport {

[[nodiscard]] inline holomq::DeliveryRecipientId to_delivery_recipient_id(
    const iox2::RawIdType& id) {
  static_assert(iox2::UNIQUE_PORT_ID_LENGTH == holomq::DeliveryRecipientId::kSize);
  holomq::DeliveryRecipientId recipient_id{};
  std::copy_n(
      id.unchecked_access().data(), holomq::DeliveryRecipientId::kSize, recipient_id.bytes.begin());
  return recipient_id;
}

/// Convert the big-endian raw id returned by iceoryx2 unable-to-deliver callbacks into the same
/// opaque byte order exposed by UniqueSubscriberId::bytes().
[[nodiscard]] inline holomq::DeliveryRecipientId to_delivery_recipient_id_from_big_endian(
    const iox2::RawIdType& id) {
  holomq::DeliveryRecipientId recipient_id = to_delivery_recipient_id(id);
  if constexpr (std::endian::native == std::endian::little) {
    std::reverse(recipient_id.bytes.begin(), recipient_id.bytes.end());
  }
  return recipient_id;
}

}  // namespace holomq::detail::transport
