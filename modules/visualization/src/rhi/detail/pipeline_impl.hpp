/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <memory>
#include <string>

#include <holoviz/rhi/pipeline.hpp>

namespace holoviz::rhi::detail {

class DescriptorImpl;

/// Backend interface for Pipeline; see the Pipeline handle for API docs.
class PipelineImpl {
 public:
  PipelineImpl(const PipelineImpl&) = delete;
  PipelineImpl& operator=(const PipelineImpl&) = delete;
  PipelineImpl(PipelineImpl&&) = delete;
  PipelineImpl& operator=(PipelineImpl&&) = delete;
  virtual ~PipelineImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  [[nodiscard]] virtual expected<std::unique_ptr<DescriptorImpl>> create_descriptor() = 0;
  [[nodiscard]] virtual expected<uint32_t> binding_name_to_index(const std::string& name) const = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  PipelineImpl() = default;
};

}  // namespace holoviz::rhi::detail
