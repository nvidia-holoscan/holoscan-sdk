/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef _HOLOSCAN_INFER_TENSOR_LOOKUP_H
#define _HOLOSCAN_INFER_TENSOR_LOOKUP_H

#include <string>

namespace holoscan {
namespace inference {

/**
 * @brief Look up a tensor by name, first in `primary` and then in `fallback`.
 *
 * Each iterator is compared against its own container's end() — comparing
 * iterators from different std::map instances is undefined behaviour and
 * aborts under debug-STL builds.
 *
 * @tparam Map A std::map-like container whose value_type is (string, T).
 * @param primary  First map to consult.
 * @param fallback Map consulted on primary miss.
 * @param name     Tensor name to look up.
 * @param out      Populated with a pointer to the value on hit; unchanged on miss.
 * @return true if `name` was found in either map, false otherwise.
 */
template <typename Map>
inline bool lookup_tensor_buffer(const Map& primary, const Map& fallback, const std::string& name,
                                 const typename Map::mapped_type** out) {
  auto in_it = primary.find(name);
  if (in_it != primary.end()) {
    *out = &in_it->second;
    return true;
  }
  auto out_it = fallback.find(name);
  if (out_it != fallback.end()) {
    *out = &out_it->second;
    return true;
  }
  return false;
}

}  // namespace inference
}  // namespace holoscan

#endif
