/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The HoloMQ CMake target publishes the authoritative value for configured consumers. Preserve a
// source-only fallback so a public header remains self-contained outside the target usage
// requirements; the current native descriptor implementation is Linux epoll/eventfd only.
#ifndef HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
#if defined(__linux__)
#define HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS 1
#else
#define HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS 0
#endif
#endif
