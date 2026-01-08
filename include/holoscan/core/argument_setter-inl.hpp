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

#ifndef HOLOSCAN_CORE_ARGUMENT_SETTER_INL_HPP
#define HOLOSCAN_CORE_ARGUMENT_SETTER_INL_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "./condition.hpp"
#include "./resource.hpp"

namespace holoscan {

// Helper function to convert signed integer with range checking
// ValueT is the target value type (e.g., int32_t), ParamT is the parameter wrapper type
template <typename ValueT, typename SourceT, typename ParamT>
inline bool convert_signed_integer_with_check(SourceT arg_value, ParamT& param,
                                              const std::string& arg_name) {
  static_assert(std::is_integral_v<SourceT> && std::is_signed_v<SourceT>,
                "SourceT must be a signed integer type");

  // Check for out-of-range values during narrowing conversion
  if constexpr (std::is_floating_point_v<ValueT>) {
    // Integer to floating-point conversion is always safe (no range check needed)
  } else if constexpr (std::is_signed_v<ValueT>) {
    // For signed targets, only check if target is smaller than source
    if constexpr (sizeof(ValueT) < sizeof(SourceT)) {
      if (arg_value < static_cast<SourceT>(std::numeric_limits<ValueT>::min()) ||
          arg_value > static_cast<SourceT>(std::numeric_limits<ValueT>::max())) {
        HOLOSCAN_LOG_ERROR(
            "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
            static_cast<int64_t>(arg_value),
            typeid(ValueT).name(),
            static_cast<int64_t>(std::numeric_limits<ValueT>::min()),
            static_cast<int64_t>(std::numeric_limits<ValueT>::max()),
            arg_name);
        return false;
      }
    }
  } else {
    // Target is unsigned - check negative first, then check max if narrowing
    if (arg_value < 0) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is negative and cannot be converted to unsigned type '{}' for '{}'",
          static_cast<int64_t>(arg_value),
          typeid(ValueT).name(),
          arg_name);
      return false;
    }
    if constexpr (sizeof(ValueT) < sizeof(SourceT)) {
      using UnsignedSourceT = std::make_unsigned_t<SourceT>;
      // Cast the target max to unsigned to avoid sign-compare warnings
      if (static_cast<UnsignedSourceT>(arg_value) >
          static_cast<UnsignedSourceT>(std::numeric_limits<ValueT>::max())) {
        HOLOSCAN_LOG_ERROR(
            "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
            static_cast<int64_t>(arg_value),
            typeid(ValueT).name(),
            static_cast<uint64_t>(std::numeric_limits<ValueT>::min()),
            static_cast<uint64_t>(std::numeric_limits<ValueT>::max()),
            arg_name);
        return false;
      }
    }
  }
  param = static_cast<ValueT>(arg_value);
  return true;
}

// Helper function to convert unsigned integer with range checking
// ValueT is the target value type (e.g., uint32_t), ParamT is the parameter wrapper type
template <typename ValueT, typename SourceT, typename ParamT>
inline bool convert_unsigned_integer_with_check(SourceT arg_value, ParamT& param,
                                                const std::string& arg_name) {
  static_assert(std::is_integral_v<SourceT> && std::is_unsigned_v<SourceT>,
                "SourceT must be an unsigned integer type");

  // Check for out-of-range values during narrowing conversion
  if constexpr (std::is_floating_point_v<ValueT>) {
    // Integer to floating-point conversion is always safe (no range check needed)
  } else if constexpr (std::is_signed_v<ValueT>) {
    // Converting unsigned to signed - check against signed max
    // Cast the signed max to unsigned to avoid sign-compare warning
    using UnsignedValueT = std::make_unsigned_t<ValueT>;
    if (arg_value >
        static_cast<SourceT>(static_cast<UnsignedValueT>(std::numeric_limits<ValueT>::max()))) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
          arg_value,
          typeid(ValueT).name(),
          static_cast<int64_t>(std::numeric_limits<ValueT>::min()),
          static_cast<int64_t>(std::numeric_limits<ValueT>::max()),
          arg_name);
      return false;
    }
  } else {
    // Both unsigned - only check max (min is always 0 for both)
    // Cast to SourceT to ensure same type comparison and avoid sign-compare warnings
    if (arg_value > static_cast<SourceT>(std::numeric_limits<ValueT>::max())) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
          arg_value,
          typeid(ValueT).name(),
          static_cast<uint64_t>(std::numeric_limits<ValueT>::min()),
          static_cast<uint64_t>(std::numeric_limits<ValueT>::max()),
          arg_name);
      return false;
    }
  }
  param = static_cast<ValueT>(arg_value);
  return true;
}

// Helper function to convert floating-point to integer with range and truncation checking
// ValueT is the target value type (e.g., int32_t), ParamT is the parameter wrapper type
template <typename ValueT, typename SourceT, typename ParamT>
inline bool convert_float_to_integer_with_check(SourceT arg_value, ParamT& param,
                                                 const std::string& arg_name) {
  static_assert(std::is_floating_point_v<SourceT>, "SourceT must be a floating-point type");
  static_assert(std::is_integral_v<ValueT>, "ValueT must be an integral type");

  // Check for NaN or infinity
  if (!std::isfinite(arg_value)) {
    HOLOSCAN_LOG_ERROR(
        "Cannot convert non-finite floating-point value {} to integer type '{}' for '{}'",
        arg_value,
        typeid(ValueT).name(),
        arg_name);
    return false;
  }

  // Check for fractional part (warn about truncation)
  SourceT int_part;
  SourceT frac_part = std::modf(arg_value, &int_part);
  if (frac_part != 0.0) {
    HOLOSCAN_LOG_WARN(
        "Truncating fractional part {} when converting {} to integer type '{}' for '{}'",
        frac_part,
        arg_value,
        typeid(ValueT).name(),
        arg_name);
  }

  // Check for negative values when target is unsigned
  if constexpr (std::is_unsigned_v<ValueT>) {
    if (arg_value < 0.0) {
      HOLOSCAN_LOG_ERROR(
          "Cannot convert negative floating-point value {} to unsigned integer type '{}' for '{}'",
          arg_value,
          typeid(ValueT).name(),
          arg_name);
      return false;
    }
  }

  // Check range for integer overflow
  // Use the integer part for comparison to avoid issues with fractional values
  if constexpr (std::is_signed_v<ValueT>) {
    if (int_part < static_cast<SourceT>(std::numeric_limits<ValueT>::min()) ||
        int_part > static_cast<SourceT>(std::numeric_limits<ValueT>::max())) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
          arg_value,
          typeid(ValueT).name(),
          static_cast<int64_t>(std::numeric_limits<ValueT>::min()),
          static_cast<int64_t>(std::numeric_limits<ValueT>::max()),
          arg_name);
      return false;
    }
  } else {
    // Unsigned target
    if (int_part > static_cast<SourceT>(std::numeric_limits<ValueT>::max())) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is out of range for parameter type '{}' (valid range: {} to {}) for '{}'",
          arg_value,
          typeid(ValueT).name(),
          static_cast<uint64_t>(std::numeric_limits<ValueT>::min()),
          static_cast<uint64_t>(std::numeric_limits<ValueT>::max()),
          arg_name);
      return false;
    }
  }

  param = static_cast<ValueT>(arg_value);
  return true;
}

// Helper function to convert float with range and truncation checking
// ValueT is the target value type, ParamT is the parameter wrapper type
template <typename ValueT, typename ParamT>
inline bool convert_float_with_check(float arg_value, ParamT& param, const std::string& arg_name) {
  if constexpr (std::is_integral_v<ValueT>) {
    // Use the specialized function for float-to-integer conversion
    return convert_float_to_integer_with_check<ValueT>(arg_value, param, arg_name);
  }
  // float to double is always safe, float to float is identity
  param = static_cast<ValueT>(arg_value);
  return true;
}

// Helper function to convert double with range and truncation checking
// ValueT is the target value type, ParamT is the parameter wrapper type
template <typename ValueT, typename ParamT>
inline bool convert_double_with_check(double arg_value, ParamT& param,
                                       const std::string& arg_name) {
  // Check for out-of-range values when narrowing double to float
  if constexpr (std::is_same_v<ValueT, float>) {
    if (std::isfinite(arg_value) && (arg_value < std::numeric_limits<float>::lowest() ||
                                      arg_value > std::numeric_limits<float>::max())) {
      HOLOSCAN_LOG_ERROR(
          "Value {} is out of range for parameter type 'float' (valid range: {} to {}) for '{}'",
          arg_value,
          std::numeric_limits<float>::lowest(),
          std::numeric_limits<float>::max(),
          arg_name);
      return false;
    }
  } else if constexpr (std::is_integral_v<ValueT>) {
    // Use the specialized function for float-to-integer conversion
    return convert_float_to_integer_with_check<ValueT>(arg_value, param, arg_name);
  }
  param = static_cast<ValueT>(arg_value);
  return true;
}

template <typename typeT>
void ArgumentSetter::add_argument_setter() {
  function_map_.try_emplace(
      std::type_index(typeid(typeT)), [](ParameterWrapper& param_wrap, Arg& arg) -> bool {
        HOLOSCAN_LOG_TRACE(
            "add_argument_setter<{}>: setting arg '{}' with element_type={}, container_type={}",
            typeid(typeT).name(),
            arg.name(),
            static_cast<int>(arg.arg_type().element_type()),
            static_cast<int>(arg.arg_type().container_type()));

        std::any& any_param = param_wrap.value();
        // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
        auto& param = *std::any_cast<Parameter<typeT>*>(any_param);

        // If arg has no name and value, that indicates that we want to set the default value for
        // the native operator if it is not specified.
        if (arg.name().empty() && !arg.has_value()) {
          param.set_default_value();
          return true;
        }

        std::any& any_arg = arg.value();
        const auto& arg_type = arg.arg_type();

        auto element_type = arg_type.element_type();
        auto container_type = arg_type.container_type();

        try {
          switch (container_type) {
            case ArgContainerType::kNative: {
              switch (element_type) {
                // Handle the argument with 'kInt64' type differently because the argument might
                // come from Python, and Python only has 'int' type ('int64_t' in C++).
                case ArgElementType::kInt64: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // Try to cast as int64_t, long, or long long (depending on platform)
                    int64_t arg_value;
                    if (any_arg.type() == typeid(int64_t)) {
                      arg_value = std::any_cast<int64_t>(any_arg);
                    } else if (any_arg.type() == typeid(long)) {
                      arg_value = static_cast<int64_t>(std::any_cast<long>(any_arg));
                    } else if (any_arg.type() == typeid(long long)) {
                      arg_value = static_cast<int64_t>(std::any_cast<long long>(any_arg));
                    } else {
                      HOLOSCAN_LOG_ERROR("Unable to convert argument type '{}' to int64_t for '{}'",
                                         any_arg.type().name(),
                                         arg.name());
                      return false;
                    }
                    if (!convert_signed_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                // Handle the argument with 'kFloat64' type differently because the argument might
                // come from Python, and Python only has 'float' type ('double' in C++).
                case ArgElementType::kFloat64: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    auto& arg_value = std::any_cast<double&>(any_arg);
                    if (!convert_double_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kBoolean: {
                  if constexpr (std::is_same_v<typeT, bool>) {
                    auto& arg_value = std::any_cast<bool&>(any_arg);
                    param = arg_value;
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kInt8: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // int8_t is always equivalent to signed char
                    auto& arg_value = std::any_cast<int8_t&>(any_arg);
                    if (!convert_signed_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kInt16: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // int16_t is always equivalent to short
                    auto& arg_value = std::any_cast<int16_t&>(any_arg);
                    if (!convert_signed_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kInt32: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // int32_t is always equivalent to int
                    auto& arg_value = std::any_cast<int32_t&>(any_arg);
                    if (!convert_signed_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kUnsigned8: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // uint8_t is always equivalent to unsigned char
                    auto& arg_value = std::any_cast<uint8_t&>(any_arg);
                    if (!convert_unsigned_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kUnsigned16: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // uint16_t is always equivalent to unsigned short
                    auto& arg_value = std::any_cast<uint16_t&>(any_arg);
                    if (!convert_unsigned_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kUnsigned32: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // uint32_t is always equivalent to unsigned int
                    auto& arg_value = std::any_cast<uint32_t&>(any_arg);
                    if (!convert_unsigned_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kUnsigned64: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    // Try to cast as uint64_t, unsigned long, or unsigned long long (depending on
                    // platform)
                    uint64_t arg_value;
                    if (any_arg.type() == typeid(uint64_t)) {
                      arg_value = std::any_cast<uint64_t>(any_arg);
                    } else if (any_arg.type() == typeid(unsigned long)) {
                      arg_value = static_cast<uint64_t>(std::any_cast<unsigned long>(any_arg));
                    } else if (any_arg.type() == typeid(unsigned long long)) {
                      arg_value = static_cast<uint64_t>(std::any_cast<unsigned long long>(any_arg));
                    } else {
                      HOLOSCAN_LOG_ERROR(
                          "Unable to convert argument type '{}' to uint64_t for '{}'",
                          any_arg.type().name(),
                          arg.name());
                      return false;
                    }
                    if (!convert_unsigned_integer_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kFloat32: {
                  if constexpr (holoscan::is_one_of_v<typeT,
                                                      int8_t,
                                                      int16_t,
                                                      int32_t,
                                                      int64_t,
                                                      uint8_t,
                                                      uint16_t,
                                                      uint32_t,
                                                      uint64_t,
                                                      long,
                                                      unsigned long,
                                                      long long,
                                                      unsigned long long,
                                                      float,
                                                      double>) {
                    auto& arg_value = std::any_cast<float&>(any_arg);
                    if (!convert_float_with_check<typeT>(arg_value, param, arg.name())) {
                      return false;
                    }
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kComplex64: {
                  if constexpr (holoscan::
                                    is_one_of_v<typeT, std::complex<float>, std::complex<double>>) {
                    auto& arg_value = std::any_cast<std::complex<float>&>(any_arg);
                    param = static_cast<typeT>(arg_value);
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kComplex128: {
                  if constexpr (holoscan::
                                    is_one_of_v<typeT, std::complex<float>, std::complex<double>>) {
                    auto& arg_value = std::any_cast<std::complex<double>&>(any_arg);
                    // Check for out-of-range values when narrowing complex<double> to
                    // complex<float>
                    if constexpr (std::is_same_v<typeT, std::complex<float>>) {
                      double real_part = arg_value.real();
                      double imag_part = arg_value.imag();
                      if ((std::isfinite(real_part) &&
                           (real_part < std::numeric_limits<float>::lowest() ||
                            real_part > std::numeric_limits<float>::max())) ||
                          (std::isfinite(imag_part) &&
                           (imag_part < std::numeric_limits<float>::lowest() ||
                            imag_part > std::numeric_limits<float>::max()))) {
                        HOLOSCAN_LOG_ERROR(
                            "Value ({}, {}) is out of range for parameter type "
                            "'std::complex<float>' "
                            "(valid range for each component: {} to {}) for '{}'",
                            real_part,
                            imag_part,
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::max(),
                            arg.name());
                        return false;
                      }
                    }
                    param = static_cast<typeT>(arg_value);
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kString: {
                  if constexpr (std::is_same_v<typeT, std::string>) {
                    auto& arg_value = std::any_cast<std::string&>(any_arg);
                    param = arg_value;
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kIOSpec: {
                  if constexpr (std::is_same_v<typeT, IOSpec*>) {
                    auto& arg_value = std::any_cast<IOSpec*&>(any_arg);
                    param = arg_value;
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kCondition: {
                  if constexpr (std::is_same_v<typename holoscan::type_info<typeT>::element_type,
                                               std::shared_ptr<Condition>> &&
                                holoscan::type_info<typeT>::dimension == 0) {
                    auto& arg_value = std::any_cast<std::shared_ptr<Condition>&>(any_arg);
                    auto converted_value = std::dynamic_pointer_cast<
                        typename holoscan::type_info<typeT>::derived_type>(arg_value);
                    // Initialize the condition in case the condition created by
                    // Fragment::make_condition<T>() is added to the operator as an argument.
                    // TODO(unknown): would like this to be assigned to the same entity as the
                    // operator
                    if (converted_value) {
                      converted_value->initialize();
                    }

                    param = converted_value;
                  }
                  break;
                }
                case ArgElementType::kResource: {
                  if constexpr (std::is_same_v<typename holoscan::type_info<typeT>::element_type,
                                               std::shared_ptr<Resource>> &&
                                holoscan::type_info<typeT>::dimension == 0) {
                    auto& arg_value = std::any_cast<std::shared_ptr<Resource>&>(any_arg);
                    auto converted_value = std::dynamic_pointer_cast<
                        typename holoscan::type_info<typeT>::derived_type>(arg_value);
                    // Initialize the resource in case the resource created by
                    // Fragment::make_resource<T>() is added to the operator as an argument.
                    // TODO(unknown): would like this to be assigned to the same entity as the
                    // operator
                    if (converted_value) {
                      converted_value->initialize();
                    }

                    param = converted_value;
                  }
                  break;
                }
                case ArgElementType::kYAMLNode: {
                  if constexpr (!holoscan::is_yaml_convertable_v<typeT>) {
                    HOLOSCAN_LOG_ERROR(
                        "YAML conversion for key '{}' is not supported for type '{}'",
                        arg.name(),
                        typeid(typeT).name());
                    return false;
                  } else {
                    auto node = std::any_cast<YAML::Node>(any_arg);
                    typeT value = YAMLNodeParser<typeT>::parse(node);
                    param = value;
                  }
                  break;
                }
                case ArgElementType::kHandle:
                  break;
                case ArgElementType::kCustom: {
                  // Attempt to directly bind if the Arg already holds the exact parameter type
                  try {
                    auto& casted = std::any_cast<typeT&>(any_arg);
                    param = casted;
                  } catch (const std::bad_any_cast&) {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
              }
              break;
            }
            case ArgContainerType::kVector: {
              switch (element_type) {
                case ArgElementType::kBoolean:
                case ArgElementType::kInt8:
                case ArgElementType::kInt16:
                case ArgElementType::kInt32:
                case ArgElementType::kInt64:
                case ArgElementType::kUnsigned8:
                case ArgElementType::kUnsigned16:
                case ArgElementType::kUnsigned32:
                case ArgElementType::kUnsigned64:
                case ArgElementType::kFloat32:
                case ArgElementType::kFloat64:
                case ArgElementType::kComplex64:
                case ArgElementType::kComplex128:
                case ArgElementType::kString:
                case ArgElementType::kIOSpec: {
                  if constexpr (holoscan::is_one_of_v<
                                    typeT,
                                    std::vector<bool>,
                                    std::vector<int8_t>,
                                    std::vector<int16_t>,
                                    std::vector<int32_t>,
                                    std::vector<int64_t>,
                                    std::vector<uint8_t>,
                                    std::vector<uint16_t>,
                                    std::vector<uint32_t>,
                                    std::vector<uint64_t>,
                                    std::vector<float>,
                                    std::vector<double>,
                                    std::vector<std::complex<float>>,
                                    std::vector<std::complex<double>>,
                                    std::vector<std::string>,
                                    std::vector<std::vector<bool>>,
                                    std::vector<std::vector<int8_t>>,
                                    std::vector<std::vector<int16_t>>,
                                    std::vector<std::vector<int32_t>>,
                                    std::vector<std::vector<int64_t>>,
                                    std::vector<std::vector<uint8_t>>,
                                    std::vector<std::vector<uint16_t>>,
                                    std::vector<std::vector<uint32_t>>,
                                    std::vector<std::vector<uint64_t>>,
                                    std::vector<std::vector<float>>,
                                    std::vector<std::vector<double>>,
                                    std::vector<std::vector<std::complex<float>>>,
                                    std::vector<std::vector<std::complex<double>>>,
                                    std::vector<std::vector<std::string>>,
                                    std::vector<IOSpec*>>) {
                    auto& arg_value = std::any_cast<typeT&>(any_arg);
                    param = arg_value;
                  } else {
                    HOLOSCAN_LOG_ERROR(
                        "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                        any_arg.type().name(),
                        typeid(typeT).name(),
                        arg.name());
                    return false;
                  }
                  break;
                }
                case ArgElementType::kHandle:
                case ArgElementType::kYAMLNode:
                  break;
                case ArgElementType::kCondition: {
                  if constexpr (std::is_same_v<typename holoscan::type_info<typeT>::element_type,
                                               std::shared_ptr<Condition>> &&
                                holoscan::type_info<typeT>::dimension == 1) {
                    auto& arg_value =
                        std::any_cast<std::vector<std::shared_ptr<Condition>>&>(any_arg);
                    typeT converted_value;
                    converted_value.reserve(arg_value.size());
                    for (auto& arg_value_item : arg_value) {
                      auto&& condition = std::dynamic_pointer_cast<
                          typename holoscan::type_info<typeT>::derived_type>(arg_value_item);

                      // Initialize the condition in case the condition created by
                      // Fragment::make_condition<T>() is added to the operator as an argument.
                      // TODO(unknown): would like this to be assigned to the same entity as the
                      // operator
                      if (condition) {
                        condition->initialize();
                      }

                      converted_value.push_back(condition);
                    }
                    param = converted_value;
                  }
                  break;
                }
                case ArgElementType::kResource: {
                  if constexpr (std::is_same_v<typename holoscan::type_info<typeT>::element_type,
                                               std::shared_ptr<Resource>> &&
                                holoscan::type_info<typeT>::dimension == 1) {
                    auto& arg_value =
                        std::any_cast<std::vector<std::shared_ptr<Resource>>&>(any_arg);
                    typeT converted_value;
                    converted_value.reserve(arg_value.size());
                    for (auto& arg_value_item : arg_value) {
                      auto&& resource = std::dynamic_pointer_cast<
                          typename holoscan::type_info<typeT>::derived_type>(arg_value_item);

                      // Initialize the resource in case the resource created by
                      // Fragment::make_resource<T>() is added to the operator as an argument.
                      // TODO(unknown): would like this to be assigned to the same entity as the
                      // operator
                      if (resource) {
                        resource->initialize();
                      }

                      converted_value.push_back(resource);
                    }
                    param = converted_value;
                  }
                  break;
                }
                case ArgElementType::kCustom: {
                  HOLOSCAN_LOG_ERROR(
                      "Unable to convert argument type '{}' to parameter type '{}' for '{}'",
                      any_arg.type().name(),
                      typeid(typeT).name(),
                      arg.name());
                  return false;
                }
              }
              break;
            }
            case ArgContainerType::kArray: {
              HOLOSCAN_LOG_ERROR("Unable to handle ArgContainerType::kArray type for '{}'",
                                 arg.name());
              return false;
            }
          }
          return true;
        } catch (std::bad_any_cast const& e) {
          // Capture type information for detailed error reporting
          const char* expected = typeid(typeT).name();
          const std::type_info& actual_type = any_arg.type();
          const char* actual = actual_type == typeid(void) ? "<empty>" : actual_type.name();
          std::string error_message =
              fmt::format("Bad any cast while setting argument '{}': expected '{}', got '{}'. {}",
                          arg.name(),
                          expected,
                          actual,
                          e.what());
          HOLOSCAN_LOG_ERROR(error_message);
          return false;
        }
      });
}
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_ARGUMENT_SETTER_INL_HPP */
