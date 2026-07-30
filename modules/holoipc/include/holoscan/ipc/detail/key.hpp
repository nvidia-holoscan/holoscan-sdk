/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_IPC_DETAIL_KEY_HPP
#define HOLOSCAN_IPC_DETAIL_KEY_HPP

#include <fmt/format.h>

#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <holoscan/ipc/detail/pointer_descriptor.hpp>

namespace holoscan {
namespace ipc {

/** Key type: handle_type + ptr; converts to/from wire format (`PointerDescriptor::key`).
 *
 * Serialized as **tight packed** octets—only the object representations of `handle_type_` and
 * `ptr_`, so **no ABI struct padding** is sent (avoids nondeterministic padding bytes from
 * `memcpy(sizeof(Key))`). Layout: `sizeof(HandleType)` octets, then
 * `sizeof(void*)` octets (**native endian**). Length is `wire_byte_count()` (e.g. **12** on typical
 * LP64, **8** on ILP32). For a given pair `(handle_type_, ptr_)` on a fixed ABI, the octet sequence
 * is stable.
 *
 * This is the **Holoscan IPC** reference encoding; the protocol spec treats `key` as **opaque** and
 * **implementation-defined**—subscribers echo bytes verbatim (see `PROTOCOL_SPEC.md`).
 *
 * Pointer bits are a **process-local** correlation id for the publisher; **not** meaningful across
 * machines. Interoperating peers must use the **same pointer width and endianness** (typical: same
 * binary or same platform build).
 */
struct Key {
  HandleType handle_type_;
  void* ptr_;

  /** Octet length on the wire: handle type + pointer representation (platform-dependent). */
  static constexpr size_t wire_byte_count() noexcept { return sizeof(HandleType) + sizeof(void*); }

  Key(HandleType handle_type, void* ptr) : handle_type_(handle_type), ptr_(ptr) {}

  explicit Key(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != wire_byte_count()) {
      throw std::invalid_argument("Key: bytes.size() != wire_byte_count()");
    }
    std::memcpy(&handle_type_, bytes.data(), sizeof(handle_type_));
    std::memcpy(&ptr_, bytes.data() + sizeof(handle_type_), sizeof(ptr_));
  }

  /** Packed wire bytes for `PointerDescriptor::key` (native endian; see class comment). */
  std::vector<uint8_t> to_bytes() const {
    std::vector<uint8_t> out(wire_byte_count());
    std::memcpy(out.data(), &handle_type_, sizeof(handle_type_));
    std::memcpy(out.data() + sizeof(handle_type_), &ptr_, sizeof(ptr_));
    return out;
  }

  bool operator<(const Key& other) const {
    return handle_type_ < other.handle_type_ ||
           (handle_type_ == other.handle_type_ && ptr_ < other.ptr_);
  }

  bool operator==(const Key& other) const {
    return handle_type_ == other.handle_type_ && ptr_ == other.ptr_;
  }
};

inline std::ostream& operator<<(std::ostream& os, const Key& k) {
  const std::vector<uint8_t> bytes = k.to_bytes();
  if (bytes.empty()) {
    return os << "empty";
  }
  os << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    os << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return os << std::dec;
}

}  // namespace ipc
}  // namespace holoscan

namespace fmt {

template <>
struct formatter<holoscan::ipc::Key> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.begin(); }
  template <typename FormatContext>
  auto format(const holoscan::ipc::Key& k, FormatContext& ctx) const -> decltype(ctx.out()) {
    std::ostringstream oss;
    oss << k;
    return fmt::format_to(ctx.out(), "{}", oss.str());
  }
};

}  // namespace fmt

#endif  // HOLOSCAN_IPC_DETAIL_KEY_HPP
