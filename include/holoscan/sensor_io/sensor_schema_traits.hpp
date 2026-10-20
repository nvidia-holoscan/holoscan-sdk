/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file
/// @brief Generated identities and traits for the sensor payload schemas this library owns.
///
/// Image and Imu remain in the `holoscan::schema` FlatBuffers namespace because that namespace is
/// part of their wire family name. Their generated traits live with `holoscan::sensor_io`, so Core
/// stays independent of sensor-domain payloads while both SDK components use the same installed
/// schema-generation path.

#include <holoscan/sensor_io/schemas/image_schema_traits.hpp>
#include <holoscan/sensor_io/schemas/imu_schema_traits.hpp>
