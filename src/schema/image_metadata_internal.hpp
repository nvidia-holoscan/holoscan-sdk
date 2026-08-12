/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <span>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

namespace holoscan::schema {

struct ImageT;

namespace detail {

/**
 * Encode descriptor-backed Image metadata without serializing embedded Tensor bytes.
 *
 * The caller supplies the complete bounded FlatBuffers arena and final destination. The encoded
 * Image preserves its optional Header, Time, frame identifier, dimensions, encoding, and Tensor
 * layout metadata, but deliberately omits the Tensor `data` vector regardless of ambient Tensor
 * materialization or gather scopes.
 *
 * @param value Owning Image value containing the Tensor whose storage is carried separately.
 * @param destination Bounded storage aligned for the FlatBuffers largest scalar type.
 * @return Number of serialized bytes at the beginning of `destination`, or an error without a
 *         partial usable payload.
 */
[[nodiscard]] expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, std::span<std::byte> destination) noexcept;

/// @brief Encode Image metadata using separately owned authoritative Tensor storage.
/// @param value Owning Image value supplying all non-Tensor metadata fields.
/// @param tensor Provider-backed Tensor whose layout replaces `value.data`.
/// @param destination Bounded storage aligned for the FlatBuffers largest scalar type.
/// @return Number of serialized bytes at the beginning of `destination`, or an error without a
/// partial usable payload.
[[nodiscard]] expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, const holoscan::Tensor& tensor, std::span<std::byte> destination) noexcept;

}  // namespace detail
}  // namespace holoscan::schema
