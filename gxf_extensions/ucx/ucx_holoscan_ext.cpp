/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gxf/std/extension_factory_helper.hpp>
#include "ucx_holoscan_component_serializer.hpp"

// Helper macros to convert macro value to string
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define HOLOSCAN_BUILD_VERSION_STR TOSTRING(HOLOSCAN_BUILD_VERSION)

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xe549f7ce9ecf4d53, 0x8156418727c176df, "UcxHoloscanExtension",
                         "Extension for Unified Communication X framework with Holoscan", "NVIDIA",
                         HOLOSCAN_BUILD_VERSION_STR, "LICENSE");
GXF_EXT_FACTORY_ADD(0xb8de0c9d54c64a2d, 0x88b6b642ad1bb268,
                    nvidia::gxf::UcxHoloscanComponentSerializer, nvidia::gxf::ComponentSerializer,
                    "Holoscan component serializer for UCX.");
GXF_EXT_FACTORY_END()
