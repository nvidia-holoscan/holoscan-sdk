/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

namespace holoscan::profiler {

struct holoscan_operator_domain {
  static constexpr char const* name{"holoscan"};
};

/**
 * @brief Return the process-wide NVTX domain used for Holoscan operator profiling annotations.
 *
 * @return NVTX domain handle for operator profiling events.
 * @throws Does not throw.
 */
inline nvtxDomainHandle_t operator_profile_domain() noexcept {
  return nvtx3::domain::get<holoscan_operator_domain>();
}

/// A `<port>:emit` mark means the runtime accepted the output or delivery may already have
/// started. A rejected attempt produces no mark. Match emit and receive marks by this payload,
/// because a concurrent receiver can record its mark first.
struct EmitReceiveNVTXPayload {
  nvtxStringHandle_t source_identity{};
  std::uint64_t sequence{};
};

}  // namespace holoscan::profiler

// NVTX's schema macro materializes its required C descriptor table internally.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
NVTX3_DEFINE_SCHEMA_GET(holoscan::profiler::holoscan_operator_domain,
                        holoscan::profiler::EmitReceiveNVTXPayload, "Holoscan OCL Message Payload",
                        NVTX_PAYLOAD_ENTRIES((source_identity, TYPE_NVTX_REGISTERED_STRING_HANDLE,
                                              "source_identity"),
                                             (sequence, TYPE_UINT64, "sequence")))
