/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <holoscan/channel/endpoint_catalog_builder.hpp>

#include "channel/endpoint_catalog_state.hpp"

namespace {

[[nodiscard]] std::size_t item_count(const benchmark::State& state) noexcept {
  return static_cast<std::size_t>(state.range(0));
}

[[nodiscard]] holoscan::detail::ChannelEndpointRecord make_pod_endpoint(std::size_t field_count) {
  holoscan::PodIdentityRecord pod{
      .canonical_name = "holoscan.benchmark.Pod",
      .version = 1U,
      .object_size = static_cast<std::uint32_t>(field_count),
      .object_alignment = 1U,
  };
  pod.fields.reserve(field_count);
  for (std::size_t index = 0U; index < field_count; ++index) {
    pod.fields.push_back(holoscan::PodFieldIdentityRecord{
        .name = "field-" + std::to_string(index),
        .byte_offset = static_cast<std::uint32_t>(index),
        .scalar_kind = holoscan::PodScalarKind::kUInt8,
        .element_count = 1U,
    });
  }
  return holoscan::detail::ChannelEndpointRecord{
      .name = "pod-endpoint",
      .uri = "holomq://pod-endpoint",
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kPod,
              .payload_kind = holoscan::PayloadKind::kInlineTyped,
              .pod = std::move(pod),
          },
  };
}

void BM_EndpointCatalogBuilder(benchmark::State& state) {
  const std::size_t count = item_count(state);
  std::vector<std::pair<std::string, std::string>> endpoints;
  endpoints.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    const std::string suffix = std::to_string(index);
    endpoints.emplace_back("endpoint-" + suffix, "holomq://endpoint-" + suffix);
  }

  for (auto _ : state) {
    auto builder = holoscan::ChannelEndpointCatalog::builder("benchmark");
    for (const auto& [name, uri] : endpoints) {
      auto added = builder.import_topic<std::uint32_t>(name, uri);
      if (!added) {
        state.SkipWithError(added.error().message);
        return;
      }
    }
    auto built = std::move(builder).build();
    if (!built) {
      state.SkipWithError(built.error().message);
      return;
    }
    benchmark::DoNotOptimize(built->digest());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

void BM_EndpointCatalogPodFields(benchmark::State& state) {
  const std::size_t count = item_count(state);
  const auto endpoint = make_pod_endpoint(count);
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<holoscan::detail::ChannelEndpointRecord> draft{endpoint};
    state.ResumeTiming();
    auto normalized = holoscan::detail::ChannelEndpointCatalogAccess::normalize("pod-benchmark",
                                                                                std::move(draft));
    if (!normalized) {
      state.SkipWithError(normalized.error().message);
      return;
    }
    benchmark::DoNotOptimize(normalized->digest());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
  state.SetComplexityN(static_cast<std::int64_t>(count));
}

BENCHMARK(BM_EndpointCatalogBuilder)
    ->ArgName("endpoints")
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(4096)
    ->Complexity(benchmark::oAuto);
BENCHMARK(BM_EndpointCatalogPodFields)
    ->ArgName("fields")
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->Arg(1024)
    ->Complexity(benchmark::oAuto);

}  // namespace
