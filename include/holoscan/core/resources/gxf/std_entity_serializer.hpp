/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_STD_ENTITY_SERIALIZER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_STD_ENTITY_SERIALIZER_HPP

#include <memory>
#include <vector>

#include <gxf/serialization/std_entity_serializer.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Standard GXF entity serializer.
 *
 * This class is capable of serializing and deserialing an `nvidia::gxf::Entity` (this is the
 * underlying GXF type that Holoscan uses to send data between Operators).
 *
 * ==Parameters==
 *
 * - **component_serializers** (std::vector<std::shared_ptr<holoscan::Resource>>): The component
 * serializers available for serialization/deserialization of components in the message entity. By
 * default, Holoscan uses only `StdComponentSerializer`.
 * - **verbose_warning** (bool): If true, more verbose warnings are logged by the serializer.
 */
class StdEntitySerializer : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(StdEntitySerializer, GXFResource)
  StdEntitySerializer() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::StdEntitySerializer"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::StdEntitySerializer* get() const;

 private:
  Parameter<std::vector<std::shared_ptr<holoscan::Resource>>> component_serializers_;
  Parameter<bool> verbose_warning_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_STD_ENTITY_SERIALIZER_HPP */
