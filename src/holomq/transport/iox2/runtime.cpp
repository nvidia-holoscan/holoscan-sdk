/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/detail/transport/iox2/runtime.hpp>
#include <holomq/detail/transport/iox2/setup_error.hpp>

#include <holomq/expected.hpp>

#include "iox2/node.hpp"

#include <utility>

namespace holomq::detail::transport {

holomq::expected<Runtime, holomq::Error> create_runtime() {
  detail::maybe_install_transport_log_bridge();

  auto node_or = std::move(iox2::NodeBuilder()).create<iox2::ServiceType::Ipc>();
  if (!node_or.has_value()) {
    return holomq::make_unexpected(map_node_creation_error(node_or.error()));
  }
  return Runtime(std::move(node_or).value());
}

}  // namespace holomq::detail::transport
