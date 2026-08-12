/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include <holoscan/channel/types.hpp>

#include "holomq_tensor.hpp"

namespace holoscan::channel_internal {

/// Transport payload used by direct Tensor channels.
///
/// Same-process Tensor delivery uses the 72-byte local sidecar descriptor. Cross-process Tensor
/// delivery uses the HoloMQ Tensor descriptor, which carries a HoloMQ BufferHandle plus Tensor view
/// metadata. Only one reference is meaningful for a sample, selected by payload_kind.
union TensorTransportReference {
  ChannelSampleDescriptor local{};
  HoloMQTensorDescriptor holomq;
};

enum class TensorTransportOperationKind : std::uint8_t {
  kData = 1U,
  kEndOfStream = 2U,
};

struct TensorTransportPayload {
  SampleMetadata metadata{};
  std::uint64_t producer_generation{};
  TensorTransportOperationKind operation_kind{TensorTransportOperationKind::kData};
  ChannelPayloadKind payload_kind{ChannelPayloadKind::kLocalBufferDescriptor};
  TensorTransportReference reference{};
};

static_assert(std::is_standard_layout_v<TensorTransportPayload>);
static_assert(std::is_trivially_copyable_v<TensorTransportPayload>);

}  // namespace holoscan::channel_internal
