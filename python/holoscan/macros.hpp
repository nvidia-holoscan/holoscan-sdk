/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_MACRO_HPP
#define PYHOLOSCAN_MACRO_HPP

#include <fmt/format.h>

#include <string>

constexpr const char* remove_leading_spaces(const char* str) {
  return *str == '\0' ? str
                      : ((*str == ' ' || *str == '\n') ? remove_leading_spaces(str + 1) : str);
}

#define PYDOC(method, doc) static constexpr const char* doc_##method = remove_leading_spaces(doc);

#endif  // PYHOLOSCAN_MACRO_HPP
