/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct ThrowingDestructorField {
  ThrowingDestructorField() = default;
  ThrowingDestructorField(const ThrowingDestructorField&) = default;
  ~ThrowingDestructorField() noexcept(false) {}
};

struct Params {
  ThrowingDestructorField value{};

  // Deliberately mask the member's throwing destructor at aggregate level. The schema must inspect
  // the exact declared field because its erased staging thunk destroys that field directly.
  ~Params() noexcept {}
};

// REQ: PARAM-012
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("holoscan.test.ThrowingFieldDestructor", 1),
                      holoscan::field<&Params::value>("value"));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
