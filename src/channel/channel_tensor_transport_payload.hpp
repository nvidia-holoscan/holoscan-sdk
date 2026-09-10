/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
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
  std::uint64_t health_report_sequence{};
  TensorTransportOperationKind operation_kind{TensorTransportOperationKind::kData};
  ChannelPayloadKind payload_kind{ChannelPayloadKind::kLocalBufferDescriptor};
  TensorTransportReference reference{};
};

static_assert(std::is_standard_layout_v<TensorTransportPayload>);
static_assert(std::is_trivially_copyable_v<TensorTransportPayload>);
static_assert(offsetof(TensorTransportPayload, metadata) == 0U);
static_assert(offsetof(TensorTransportPayload, producer_generation) == 80U);
static_assert(offsetof(TensorTransportPayload, health_report_sequence) == 88U);
static_assert(offsetof(TensorTransportPayload, operation_kind) == 96U);
static_assert(offsetof(TensorTransportPayload, payload_kind) == 98U);
static_assert(offsetof(TensorTransportPayload, reference) == 104U);
static_assert(sizeof(TensorTransportPayload) == 472U);

}  // namespace holoscan::channel_internal
