/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#include <array>
#include <string>
#include <vector>

#include "./inference.hpp"

#include "holoscan/core/codecs.hpp"
#include "holoscan/core/endpoint.hpp"
#include "holoscan/core/expected.hpp"

// Define codec for serialization of ops::InferenceOp::ActivationSpec
namespace holoscan {

template <>
struct codec<ops::InferenceOp::ActivationSpec> {
  static expected<size_t, RuntimeError> serialize(const ops::InferenceOp::ActivationSpec& spec,
                                                  Endpoint* endpoint) {
    size_t total_size = 0;
    auto maybe_size = codec<std::string>::serialize(spec.model_name_, endpoint);
    if (!maybe_size) { return forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<bool>(spec.active_, endpoint);
    if (!maybe_size) { return forward_error(maybe_size); }
    total_size += maybe_size.value();

    return total_size;
  }

  static expected<ops::InferenceOp::ActivationSpec, RuntimeError> deserialize(Endpoint* endpoint) {
    ops::InferenceOp::ActivationSpec out;
    auto model_name = codec<std::string>::deserialize(endpoint);
    if (!model_name) { return forward_error(model_name); }
    out.model_name_ = model_name.value();

    auto active = deserialize_trivial_type<bool>(endpoint);
    if (!active) { return forward_error(active); }
    out.active_ = active.value();

    return out;
  }
};

// Define codec for serialization of std::vector<ops::InferenceOp::ActivationSpec>
template <>
struct codec<std::vector<ops::InferenceOp::ActivationSpec>> {
  static expected<size_t, RuntimeError> serialize(
      const std::vector<ops::InferenceOp::ActivationSpec>& specs, Endpoint* endpoint) {
    size_t total_size = 0;

    // header is just the total number of specs
    size_t num_specs = specs.size();
    auto size = endpoint->write_trivial_type<size_t>(&num_specs);
    if (!size) { return forward_error(size); }
    total_size += size.value();

    // now transmit each individual spec
    for (const auto& spec : specs) {
      size = codec<ops::InferenceOp::ActivationSpec>::serialize(spec, endpoint);
      if (!size) { return forward_error(size); }
      total_size += size.value();
    }
    return total_size;
  }
  static expected<std::vector<ops::InferenceOp::ActivationSpec>, RuntimeError> deserialize(
      Endpoint* endpoint) {
    size_t num_specs;
    auto size = endpoint->read_trivial_type<size_t>(&num_specs);
    if (!size) { return forward_error(size); }

    std::vector<ops::InferenceOp::ActivationSpec> data;
    data.reserve(num_specs);
    for (size_t i = 0; i < num_specs; i++) {
      auto spec = codec<ops::InferenceOp::ActivationSpec>::deserialize(endpoint);
      if (!spec) { return forward_error(spec); }
      data.push_back(spec.value());
    }
    return data;
  }
};

}  // namespace holoscan
