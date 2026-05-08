/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
