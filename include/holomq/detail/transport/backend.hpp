/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// Compile-time transport backend selector. See \c docs/design-notes/transport-abstraction.md.
///
/// CMake sets \c HOLOMQ_TRANSPORT and defines \c HOLOMQ_TRANSPORT_IOX2; this header includes
/// the iceoryx2 backend's \c backend.hpp.

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <holomq/detail/transport/iox2/backend.hpp>
#else
#error "Unknown HOLOMQ_TRANSPORT backend"
#error "Define HOLOMQ_TRANSPORT_IOX2"
#endif
