/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// @file
/// @brief Aggregate the sensor I/O composition surface.
///
/// The pieces here are collaborators an acquisition operator holds as members, not a base class it
/// inherits from. Each one owns a single decision that is easy to get subtly wrong and whose
/// failure mode is silence rather than a crash: sequencing lifecycle stage bodies behind their own
/// preconditions, turning a device sample counter into an explicit dropped-sample count, and
/// projecting device capture timestamps into a declared clock domain. Registering the stage hooks
/// themselves is Core's job, through the lifecycle builder its operator spec already exposes.
///
/// \ref holoscan::sensor_io::derive_from and \ref holoscan::sensor_io::ProvenanceAccumulator are
/// the downstream half of the same concern: an acquisition operator's careful capture timestamp is
/// worth nothing if the first transform after it publishes a fresh metadata record and drops it.
///
/// An operator keeps deriving from \ref holoscan::Operator, keeps declaring its own ports, and
/// keeps writing its own `compute()`. Nothing here takes over those responsibilities.

#include <holoscan/sensor_io/clock_discipline.hpp>
#include <holoscan/sensor_io/lifecycle_guards.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sequence_tracker.hpp>
