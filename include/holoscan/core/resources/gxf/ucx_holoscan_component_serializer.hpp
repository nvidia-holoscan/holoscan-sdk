/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP

#include <memory>
#include <vector>

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief UCX-based Holoscan component serializer.
 *
 * Used by UcxEntitySerializer to serialize and deserialize Holoscan SDK class holoscan::Message.
 * See the CodecRegistry class for adding serialization codecs for additional holoscan::Message
 * types.
 *
 * ==Parameters==
 *
 * - **allocator** (std::shared_ptr<holoscan::Allocator>): The allocator used for deserialization
 * of Tensor, VideoBuffer or AudioBuffer components. Defaults to an `UnboundedAllocator` if none is
 * provided.
 *
 */
class UcxHoloscanComponentSerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UcxHoloscanComponentSerializer, GXFResource)
  UcxHoloscanComponentSerializer() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::UcxHoloscanComponentSerializer";
  }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_UCX_HOLOSCAN_COMPONENT_SERIALIZER_HPP */
