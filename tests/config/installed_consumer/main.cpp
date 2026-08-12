/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <map>
#include <ranges>
#include <string>
#include <utility>

#include <holoscan/config/codec.hpp>
#if defined(HOLOSCAN_TEST_USE_CONFIG_IO)
#include <holoscan/config/io.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_PARAMS)
#include <holoscan/params/rebind.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_PARAMS_IO)
#include <holoscan/params/persistence.hpp>
#endif
#if defined(HOLOSCAN_TEST_USE_TUNER)
#include <holoscan/experimental/params/web_tuner.hpp>
#endif

int main() {
#if defined(HOLOSCAN_TEST_USE_PARAMS)
  // REQ: PARAM-024
  // REQ: PARAM-035
  const holoscan::params::ParamRebindOptions options{};
  auto path = holoscan::OperatorPath::parse("camera");
  if (!path) {
    return 4;
  }
  const auto scope = holoscan::params::ParamRebindTargetScope::operator_(*path);
  return options.mode == holoscan::ParamRebindMode::kSparseContributions &&
                 scope.kind() == holoscan::ParamRebindTargetScopeKind::kOperator &&
                 scope.paths().size() == 1U &&
                 std::ranges::equal(scope.paths().front(), path->segments())
             ? 0
             : 4;
#elif defined(HOLOSCAN_TEST_USE_PARAMS_IO)
  const holoscan::params::ParamSaveOptions options{};
  return options.safe_projection == holoscan::params::ParamSafeProjection::kPersistedPublicOnly ? 0
                                                                                                : 4;
#elif defined(HOLOSCAN_TEST_USE_TUNER)
  const holoscan::experimental::params::WebTunerOptions options{};
  return options.max_clients == 1U && options.max_connections == 4U ? 0 : 4;
#else
  const std::map<std::string, std::int32_t> source{{"camera", 7}};
  auto encoded = holoscan::config::encode_config(source);
  if (!encoded) {
    return 1;
  }
  auto document = holoscan::config::Document::from_value(*encoded);
  if (!document) {
    return 2;
  }
#if defined(HOLOSCAN_TEST_USE_CONFIG_IO)
  auto text = holoscan::config_io::serialize(*document, holoscan::config_io::Format::kJson);
  return text && text->find("camera") != std::string::npos ? 0 : 3;
#else
  auto decoded = document->get<std::map<std::string, std::int32_t> >("");
  return decoded && *decoded == source ? 0 : 3;
#endif
#endif
}
