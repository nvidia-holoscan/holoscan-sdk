/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstddef>

#include <flatbuffers/flatbuffer_builder.h>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/holoscan_schema_export.h"

#include "tensor_generated.h"

namespace holoscan::schema {

/**
 * @brief Add a metadata-only Tensor table to a FlatBuffer builder.
 *
 * The returned table contains shape, dtype, device, rank, and optional DLPack strides. Its
 * `data` vector is always absent, including while a \ref WithTensorMaterialization or
 * \ref WithTensorGather scope is active. The caller owns the builder and therefore controls its
 * storage bound and enclosing FlatBuffer root.
 *
 * @param builder FlatBuffer builder that will own the returned table.
 * @param tensor Tensor whose read-only metadata is serialized.
 * @return Tensor table offset on success, or Error for invalid metadata or exhausted builder
 * storage.
 */
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT expected<flatbuffers::Offset<Tensor>, Error>
create_tensor_metadata(flatbuffers::FlatBufferBuilder& builder,
                       const holoscan::Tensor& tensor) noexcept;

/**
 * @brief Decode and validate a metadata-only generated Tensor table.
 *
 * @param tensor Generated Tensor table borrowed from an already verifier-valid enclosing root.
 * @return Core Tensor view metadata, or Error for invalid fields, arithmetic overflow, or a
 * nonempty embedded `data` vector.
 */
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT expected<TensorViewMetadata, Error> tensor_view_metadata(
    const schema::Tensor& tensor) noexcept;

/**
 * @brief Thread-local scope that forces Tensor bytes into FlatBuffer Tensor.data.
 *
 * The default schema path keeps Tensor data descriptor/sidecar-backed for local channels and only
 * serializes metadata. This opt-in scope is for cases that intentionally need byte
 * materialization, such as replay snapshots, network fallback payloads, or tests that verify
 * schema-only Tensor unpacking can reconstruct owned storage.
 */
class HOLOSCAN_SCHEMA_EXPORT WithTensorMaterialization {
 public:
  WithTensorMaterialization() {
    assert(!enabled_);
    enabled_ = true;
  }

  ~WithTensorMaterialization() { enabled_ = false; }

  static bool enabled() noexcept { return enabled_; }

  WithTensorMaterialization(const WithTensorMaterialization&) = delete;
  WithTensorMaterialization& operator=(const WithTensorMaterialization&) = delete;
  WithTensorMaterialization(WithTensorMaterialization&&) = delete;
  WithTensorMaterialization& operator=(WithTensorMaterialization&&) = delete;

 private:
  inline static constinit thread_local bool enabled_ = false;
};

/**
 * @brief Thread-local scope that materializes Tensor bytes only below a size threshold.
 *
 * `WithTensorGather(0)` is equivalent to metadata-only/reference mode. A positive threshold
 * materializes small tensors and leaves larger tensors metadata-only.
 */
class HOLOSCAN_SCHEMA_EXPORT WithTensorGather {
 public:
  explicit WithTensorGather(std::size_t size_threshold_bytes = 1024) {
    ++nesting_level_;
    enabled_ = true;
    size_threshold_bytes_ = size_threshold_bytes;
  }

  ~WithTensorGather() {
    --nesting_level_;
    if (nesting_level_ == 0) {
      enabled_ = false;
      size_threshold_bytes_ = 0;
    }
  }

  static bool enabled() noexcept { return enabled_; }
  static std::size_t size_threshold_bytes() noexcept { return size_threshold_bytes_; }

  WithTensorGather(const WithTensorGather&) = delete;
  WithTensorGather& operator=(const WithTensorGather&) = delete;
  WithTensorGather(WithTensorGather&&) = delete;
  WithTensorGather& operator=(WithTensorGather&&) = delete;

 private:
  inline static constinit thread_local bool enabled_ = false;
  inline static constinit thread_local std::size_t size_threshold_bytes_ = 0;
  inline static constinit thread_local int nesting_level_ = 0;
};

}  // namespace holoscan::schema
