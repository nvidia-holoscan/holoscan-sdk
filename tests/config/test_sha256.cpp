/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config/sha256.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace holoscan::config::detail {
namespace {

// Adapts a string_view to a const-byte span for Sha256Builder::add() input.
// Avoids verbose std::as_bytes conversions in every test call site.
std::span<const std::byte> as_bytes(std::string_view text) {
  return std::as_bytes(std::span<const char>{text.data(), text.size()});
}

// Renders a Sha256Digest as a lowercase hex string for comparison against
// published NIST known-answer test (KAT) vectors.
std::string to_hex(const Sha256Digest& digest) {
  static constexpr std::string_view kHexDigits = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2U);
  for (const std::byte byte : digest) {
    const unsigned int value = std::to_integer<unsigned int>(byte);
    result.push_back(kHexDigits[value >> 4U]);
    result.push_back(kHexDigits[value & 0x0FU]);
  }
  return result;
}

TEST(Sha256Builder, KnownAnswerVectorsCoverPaddingAndChunkedInput) {
  const Sha256Builder empty;
  EXPECT_EQ(to_hex(empty.finish()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

  Sha256Builder single;
  single.add(as_bytes("abc"));
  EXPECT_EQ(to_hex(single.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

  // Exactly 56 payload bytes force finish() to emit a second padding block. Split the input so
  // the regression also covers buffered updates immediately before that boundary.
  static constexpr std::string_view kTwoPaddingBlockInput =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  Sha256Builder two_padding_blocks;
  two_padding_blocks.add(as_bytes(kTwoPaddingBlockInput.substr(0U, 13U)));
  two_padding_blocks.add(as_bytes(kTwoPaddingBlockInput.substr(13U)));
  EXPECT_EQ(to_hex(two_padding_blocks.finish()),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");

  static constexpr std::string_view kMultiBlockInput =
      "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
      "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
  static constexpr std::array<std::size_t, 5> kChunkSizes{1U, 7U, 55U, 2U, 47U};
  // Key setup: feed the 112-byte NIST vector in five uneven chunks to stress buffered
  //            updates immediately before and across the 512-bit block boundary.
  Sha256Builder chunked;
  std::size_t offset = 0U;
  for (const std::size_t chunk_size : kChunkSizes) {
    chunked.add(as_bytes(kMultiBlockInput.substr(offset, chunk_size)));
    offset += chunk_size;
  }
  ASSERT_EQ(offset, kMultiBlockInput.size());
  // Key assertion: chunked output matches the published 112-byte NIST KAT vector.
  EXPECT_EQ(to_hex(chunked.finish()),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
}

}  // namespace
}  // namespace holoscan::config::detail
