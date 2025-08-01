/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_PROFILER_NVTX3_HPP
#define HOLOSCAN_PROFILER_NVTX3_HPP

#include <nvtx3/nvtx3.hpp>

// Copy of nvtx3::v1::scoped_range_in which checks whether tracing is enabled or not
namespace nvtx3::v1 {

template <class D = domain::global>
class holoscan_scoped_range_in {
  bool enabled_ = false;

 public:
  explicit holoscan_scoped_range_in(event_attributes const& attr) noexcept {
    enabled_ = ::holoscan::profiler::trace_enabled();
    if (enabled_) {
      nvtxDomainRangePushEx(domain::get<D>(), attr.get());
    }
  }

  template <typename... Args>
  explicit holoscan_scoped_range_in(Args const&... args) noexcept
      : holoscan_scoped_range_in{event_attributes{args...}} {}

  holoscan_scoped_range_in() noexcept : holoscan_scoped_range_in{event_attributes{}} {}

  void* operator new(std::size_t) = delete;

  holoscan_scoped_range_in(holoscan_scoped_range_in const&) = delete;
  holoscan_scoped_range_in& operator=(holoscan_scoped_range_in const&) = delete;
  holoscan_scoped_range_in(holoscan_scoped_range_in&&) = delete;
  holoscan_scoped_range_in& operator=(holoscan_scoped_range_in&&) = delete;

  ~holoscan_scoped_range_in() noexcept {
    if (enabled_) {
      nvtxDomainRangePop(domain::get<D>());
    }
  }
};

}  // namespace nvtx3::v1

/**
 * Registers an integer ID with a profiler category. Should be called before
 * any events are profiled using that category.
 */
#define PROF_REGISTER_CATEGORY(id_, name_) holoscan::profiler::named_category(id_, name_)

/**
 * Helper to provide the category handle from its ID.
 */
#define PROF_CATEGORY(category_) holoscan::profiler::category(category_)

/**
 * Defines an event type to be used for profiling.
 */
#define PROF_DEFINE_EVENT(name_, message_, r_, g_, b_)           \
  struct name_ {                                                 \
    static constexpr char const* message{message_};              \
    static constexpr nvtx3::color color{nvtx3::rgb{r_, g_, b_}}; \
  }

/**
 * Helper to provide the event arguments from its type name.
 */
#define PROF_EVENT(name_) holoscan::profiler::message::get<name_>(), name_::color

/**
 * Declares a scoped profiling event given a category ID and event name.
 */
#define PROF_SCOPED_EVENT(category_, name_)     \
  holoscan::profiler::scoped_range p {          \
    PROF_CATEGORY(category_), PROF_EVENT(name_) \
  }

/**
 * Declares a scoped profiling event with pre-formatted port-specific message.
 * Uses operator-level category but port-specific message for better granularity.
 * No string formatting overhead - message should be pre-formatted (e.g., IOSpec::unique_id()).
 */
#define PROF_SCOPED_PORT_EVENT(category_, message_, color_)    \
  holoscan::profiler::scoped_range p {                         \
    PROF_CATEGORY(category_), nvtx3::message{message_}, color_ \
  }

namespace holoscan::profiler {

// Domain
struct domain {
  static constexpr char const* name{"Holoscan"};
};

// Aliases
using scoped_range = nvtx3::holoscan_scoped_range_in<domain>;
using category = nvtx3::category;
using named_category = nvtx3::named_category_in<domain>;
using message = nvtx3::registered_string_in<domain>;

}  // namespace holoscan::profiler

#endif  // HOLOSCAN_PROFILER_NVTX3_HPP
