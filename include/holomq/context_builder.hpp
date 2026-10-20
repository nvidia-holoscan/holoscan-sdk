/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/context.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <string>

namespace holomq {

/// Fluent builder for \ref Context; call \c create() to open the node and optional VMM FD-pass
/// server.
class ContextBuilder {
 public:
  ContextBuilder() = default;
  ContextBuilder(const ContextBuilder&) = delete;
  ContextBuilder& operator=(const ContextBuilder&) = delete;
  ContextBuilder(ContextBuilder&&) noexcept = default;
  ContextBuilder& operator=(ContextBuilder&&) noexcept = default;

  /// Enable the internal AF_UNIX FD-pass server (see \c with_fd_pass_socket).
  /// Publishers exporting \ref BufferHandle cross-process should enable this. Required for device
  /// pools, pageable host pools, and pinned host without \c CU_MEMHOSTREGISTER_DEVICEMAP. Pinned
  /// host pools with \c DEVICEMAP also use device-resident sync words when it is enabled, and fall
  /// back to the devicemap interleaved layout without it. Default \c true; subscribers that do not
  /// export buffers may pass \c false.
  HOLOMQ_EXPORT ContextBuilder& with_vmm_fd_export(bool enable);

  /// AF_UNIX listen path for the FD-pass server; when empty, a default path under \c /tmp is used
  /// at \c create(). Non-empty path with export disabled is an error at \c create().
  HOLOMQ_EXPORT ContextBuilder& with_fd_pass_socket(std::string path);

  /// When \c true, route iceoryx2 transport diagnostics through the HoloMQ logger.
  HOLOMQ_EXPORT ContextBuilder& with_transport_logging(bool enable);

  /**
   * @brief Open a \ref Context and its optional VMM FD-pass server.
   * @return A context on success, or a typed configuration, resource, permission, or transport
   * setup error on failure.
   * @throws std::exception if backend setup or resource allocation propagates an exception.
   * @note Failures are logged with setup detail.
   */
  [[nodiscard]] HOLOMQ_EXPORT auto create() const -> expected<Context, Error>;

 private:
  bool vmm_fd_export_enabled_{true};
  bool transport_logging_enabled_{false};
  std::string fd_pass_socket_path_;
};

}  // namespace holomq
