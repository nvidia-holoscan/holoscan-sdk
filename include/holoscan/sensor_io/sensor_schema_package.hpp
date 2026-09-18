/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "holoscan/holoscan_sensor_io_export.h"
#include "holoscan/sensor_io/sensor_schema_traits.hpp"

namespace holoscan::detail {

/// @brief Return the sensor-specific descriptor Runtime operations for generated Image values.
/// @return Process-local operations whose address never enters a plan or transport record.
template <>
[[nodiscard]] HOLOSCAN_SENSOR_IO_EXPORT const RuntimePortTypeOps*
runtime_port_type_ops<schema::ImageT>() noexcept;

}  // namespace holoscan::detail
