/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_IPC_DETAIL_POINTER_DESCRIPTOR_HPP
#define HOLOSCAN_IPC_DETAIL_POINTER_DESCRIPTOR_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan {
namespace ipc {

/** How to interpret opaque handle bytes; ordinals match DDS/IDL `HandleType`. */
enum class HandleType : int32_t { CUDA_IPC };

/**
 * Transport-agnostic pointer descriptor for the IPC API.
 * Mirrors the logical fields of the DDS gen::PointerDescriptor (version, key, handle_type,
 * handle, reply_to_topic_name). Distinct from gen::PointerDescriptor (the DDS-generated type).
 */
struct PointerDescriptor {
  std::string version;
  std::vector<uint8_t> key;
  HandleType handle_type = HandleType::CUDA_IPC;
  std::vector<uint8_t> handle;
  std::string reply_to_topic_name;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_POINTER_DESCRIPTOR_HPP
