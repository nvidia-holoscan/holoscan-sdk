/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_config_export.h>
#include <holoscan/config/document.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace holoscan::config::detail {

constexpr std::uint8_t kSourceFormatValueMask{0x7FU};
constexpr std::uint8_t kYamlAliasSourceFlag{0x80U};

using ConfigArray = std::vector<std::shared_ptr<const ConfigNode>>;
using ConfigObject = std::vector<std::pair<std::string, std::shared_ptr<const ConfigNode>>>;

struct ConfigNode {
  using Storage = std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double,
                               std::string, ConfigArray, ConfigObject>;
  Storage storage;
};

struct LosslessScalarSpan {
  std::string path;
  ConfigValueKind kind{ConfigValueKind::kNull};
  std::size_t begin{0};
  std::size_t end{0};
};

struct DocumentState {
  std::shared_ptr<const ConfigNode> root;
  ConfigDigest digest;
  std::optional<std::uint8_t> source_format;
  std::shared_ptr<const std::string> raw_source;
  std::vector<LosslessScalarSpan> scalar_spans;
  bool requires_lossless{false};
  bool lossless_available{false};
  bool contains_yaml_alias{false};
};

[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigResult<SemanticDigest> semantic_digest(
    const std::shared_ptr<const ConfigNode>& root) noexcept;
[[nodiscard]] HOLOSCAN_CONFIG_EXPORT ConfigResult<RawSourceDigest> raw_digest(
    std::string_view bytes) noexcept;

}  // namespace holoscan::config::detail

namespace holoscan::config {

class HOLOSCAN_CONFIG_EXPORT ConfigIoAccess {
 public:
  [[nodiscard]] static ConfigResult<Document> from_parsed(
      ConfigValue root, RawSourceDigest raw_digest, std::uint8_t source_format,
      std::shared_ptr<const std::string> raw_source, std::vector<detail::LosslessScalarSpan> spans,
      bool requires_lossless) noexcept;
  [[nodiscard]] static const detail::DocumentState& state(const Document& document) noexcept;
};

}  // namespace holoscan::config
