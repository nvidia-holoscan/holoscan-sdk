/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_UCX_COMPONENT_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_UCX_COMPONENT_SERIALIZER_HPP

#include <memory>
#include <vector>

#include <gxf/ucx/ucx_component_serializer.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief UCX-based component serializer.
 *
 * Used by UcxEntitySerializer to serialize and deserialize GXF components such as
 * nvidia::gxf::Tensor, nvidia::gxf::VideoBuffer, nvidia::gxf::AudioBuffer,
 * nvidia::gxf::Timestamp and nvidia::gxf::EndOfStream.
 *
 * ==Parameters==
 *
 * - **allocator** (std::shared_ptr<holoscan::Allocator>, optional): The allocator used for
 * deserialization of `Tensor`, `VideoBuffer` or `AudioBuffer` components. Defaults to an
 * `UnboundedAllocator` if none is provided.
 */
class UcxComponentSerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UcxComponentSerializer, GXFResource)
  UcxComponentSerializer() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::UcxComponentSerializer"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::UcxComponentSerializer* get() const;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_UCX_COMPONENT_SERIALIZER_HPP */
