/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/pubsub_capabilities.hpp>

namespace holomq {

PubSubCapabilities default_pubsub_capabilities() noexcept {
  PubSubCapabilities capabilities{};

#if defined(HOLOMQ_TRANSPORT_IOX2)
  capabilities.waitset = true;
  capabilities.kernel_wait = true;
  capabilities.delivery_attribution = DeliveryAttribution::kPerRecipient;
#endif

  return capabilities;
}

}  // namespace holomq
