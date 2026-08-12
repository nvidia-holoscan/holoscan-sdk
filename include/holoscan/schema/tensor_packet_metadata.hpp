/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"

#include "tensor_packet_generated.h"

namespace holoscan::schema {

/// @brief Extract the descriptor-backed Tensor view from TensorPacket metadata.
template <>
struct DescriptorFlatBufferTensorViewTraits<TensorPacket>
    : DescriptorFlatBufferTensorViewAdapter<TensorPacket, &TensorPacket::data> {};

}  // namespace holoscan::schema
