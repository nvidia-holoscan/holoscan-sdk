/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Python.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/channel/pod_type.hpp>

namespace {

template <typename T, std::size_t Size>
struct Record {
  std::uint64_t sequence;
  std::array<T, Size> data;
};
using SmallBytes = Record<std::byte, 8>;
using LargeBytes = Record<std::byte, 16 * 1024>;
using SmallUInt8 = Record<std::uint8_t, 8>;
using LargeUInt8 = Record<std::uint8_t, 16 * 1024>;
using SmallNumeric = Record<double, 3>;
using LargeNumeric = Record<double, 4096>;

PyModuleDef kModule{PyModuleDef_HEAD_INIT,
                    "_pod_fixtures",
                    nullptr,
                    -1,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr};

}  // namespace

namespace holoscan {

template <typename T, std::size_t Size>
struct ChannelPodLayout<Record<T, Size>> {
  using Value = Record<T, Size>;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &Value::sequence, offsetof(Value, sequence)),
                 pod_field("data", &Value::data, offsetof(Value, data)));
};

template <>
struct ChannelTypeIdentity<SmallBytes> {
  static constexpr std::string_view name = "holoscan.benchmark.SmallBytes";
  static constexpr std::uint32_t version = 1;
};
template <>
struct ChannelTypeIdentity<LargeBytes> {
  static constexpr std::string_view name = "holoscan.benchmark.LargeBytes";
  static constexpr std::uint32_t version = 1;
};
template <>
struct ChannelTypeIdentity<SmallUInt8> {
  static constexpr std::string_view name = "holoscan.benchmark.SmallUInt8";
  static constexpr std::uint32_t version = 1;
};
template <>
struct ChannelTypeIdentity<LargeUInt8> {
  static constexpr std::string_view name = "holoscan.benchmark.LargeUInt8";
  static constexpr std::uint32_t version = 1;
};
template <>
struct ChannelTypeIdentity<SmallNumeric> {
  static constexpr std::string_view name = "holoscan.benchmark.SmallNumeric";
  static constexpr std::uint32_t version = 1;
};
template <>
struct ChannelTypeIdentity<LargeNumeric> {
  static constexpr std::string_view name = "holoscan.benchmark.LargeNumeric";
  static constexpr std::uint32_t version = 1;
};

}  // namespace holoscan

// Python's extension-module ABI requires the reserved PyInit_ spelling.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
PyMODINIT_FUNC PyInit__pod_fixtures() {
  auto& registry = holoscan::authoring::authoring_registry();
  for (auto row : {holoscan::authoring::detail::make_registered_payload<SmallBytes>(
                       holoscan::ChannelTypeIdentity<SmallBytes>::name),
                   holoscan::authoring::detail::make_registered_payload<LargeBytes>(
                       holoscan::ChannelTypeIdentity<LargeBytes>::name),
                   holoscan::authoring::detail::make_registered_payload<SmallUInt8>(
                       holoscan::ChannelTypeIdentity<SmallUInt8>::name),
                   holoscan::authoring::detail::make_registered_payload<LargeUInt8>(
                       holoscan::ChannelTypeIdentity<LargeUInt8>::name),
                   holoscan::authoring::detail::make_registered_payload<SmallNumeric>(
                       holoscan::ChannelTypeIdentity<SmallNumeric>::name),
                   holoscan::authoring::detail::make_registered_payload<LargeNumeric>(
                       holoscan::ChannelTypeIdentity<LargeNumeric>::name)}) {
    if (auto installed = registry.register_payload(std::move(row)); !installed) {
      const auto message = holoscan::runtime_error_message(installed.error());
      PyErr_SetString(PyExc_RuntimeError, message.c_str());
      return nullptr;
    }
  }
  return PyModule_Create(&kModule);
}
