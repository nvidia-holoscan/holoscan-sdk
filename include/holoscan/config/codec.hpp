/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/config/document.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::config {

namespace detail {

template <class T>
[[nodiscard]] constexpr bool integer_exactly_representable(std::uint64_t magnitude) noexcept {
  static_assert(std::is_floating_point_v<T>);
  if (magnitude == 0) {
    return true;
  }
  const auto significant_bits = std::bit_width(magnitude) - std::countr_zero(magnitude);
  return std::cmp_less_equal(significant_bits, std::numeric_limits<T>::digits);
}
struct CodecContextFactory;
struct CodecCallState {
  std::size_t unfinished_decoders{0};
  std::uint32_t remaining_nodes{1'000'000};
  std::size_t remaining_allocation_bytes{64U * 1024U * 1024U};
};
}  // namespace detail

/// @brief Carry package-owned monotonic state for one decode call tree.
///
/// Applications receive this non-copyable object only inside ConfigCodec callbacks. A context and
/// its children borrow top-level call state and must not escape the callback. It is not
/// thread-safe; one top-level decode owns one independent context tree.
class DecodeContext {
 public:
  /// @brief Disable copying because the context borrows one call's budget state.
  DecodeContext(const DecodeContext&) = delete;
  /// @brief Disable copy assignment because the context borrows one call's budget state.
  DecodeContext& operator=(const DecodeContext&) = delete;
  /// @brief Return the owned RFC 6901 path for the value currently being decoded.
  [[nodiscard]] std::string_view path() const noexcept { return path_; }

  /// @brief Decode one nested value while preserving path and monotonic call budgets.
  ///
  /// @tparam T Nested owning type with a visible complete codec specialization.
  /// @param token Unescaped object key or decimal array index for the child.
  /// @param value Borrowed child semantic value.
  /// @param options Immutable policy shared by the top-level invocation.
  /// @return The owning decoded child or its structured failure.
  template <class T>
  [[nodiscard]] ConfigResult<T> decode_child(std::string_view token, ConfigView value,
                                             const DecodeOptions& options);

  /// @brief Charge an owning result allocation against this call tree before allocating it.
  /// @param bytes Number of bytes the codec is about to allocate.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED` without changing the budget on failure.
  [[nodiscard]] ConfigResult<void> charge_allocation(std::size_t bytes) noexcept;

  /// @brief Validate one string and charge its owning result bytes before allocating it.
  /// @param bytes Decoded UTF-8 bytes the codec is about to own.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] ConfigResult<void> charge_string(std::size_t bytes) noexcept;

  /// @brief Validate one container and charge its element storage before allocating it.
  /// @param entries Number of immediate elements.
  /// @param element_bytes Owning storage bytes charged per element.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED`, including multiplication overflow.
  [[nodiscard]] ConfigResult<void> charge_container(std::size_t entries,
                                                    std::size_t element_bytes) noexcept;

 private:
  DecodeContext(std::string path, detail::CodecCallState& state, std::uint32_t depth)
      : path_(std::move(path)), state_(state), depth_(depth) {}
  [[nodiscard]] DecodeContext child(std::string_view token) const;
  [[nodiscard]] ConfigResult<void> charge(const ConfigView& value) noexcept;
  void begin_decoder() noexcept { ++state_.unfinished_decoders; }
  void finish_decoder() noexcept {
    if (state_.unfinished_decoders != 0) {
      --state_.unfinished_decoders;
    }
  }
  std::string path_;
  detail::CodecCallState& state_;
  std::uint32_t depth_{1};
  friend struct detail::CodecContextFactory;
  friend class ObjectDecoder;
  friend class ArrayDecoder;
};

/// @brief Carry package-owned monotonic state for one encode call tree.
///
/// Applications receive this non-copyable object only inside ConfigCodec callbacks. A context and
/// its children borrow top-level call state and must not escape the callback. It is not
/// thread-safe; one top-level encode owns one independent context tree.
class EncodeContext {
 public:
  /// @brief Disable copying because the context borrows one call's budget state.
  EncodeContext(const EncodeContext&) = delete;
  /// @brief Disable copy assignment because the context borrows one call's budget state.
  EncodeContext& operator=(const EncodeContext&) = delete;
  /// @brief Return the owned RFC 6901 path for the value currently being encoded.
  [[nodiscard]] std::string_view path() const noexcept { return path_; }

  /// @brief Encode one nested value under an owned child path.
  ///
  /// @tparam T Nested owning type with a visible complete codec specialization.
  /// @param token Unescaped object key or decimal array index for the child.
  /// @param value Borrowed typed child value.
  /// @param options Immutable policy shared by the top-level invocation.
  /// @return The owning encoded semantic child or its structured failure.
  template <class T>
  [[nodiscard]] ConfigResult<ConfigValue> encode_child(std::string_view token, const T& value,
                                                       const EncodeOptions& options);

  /// @brief Charge an owning result allocation against this call tree before allocating it.
  /// @param bytes Number of bytes the codec is about to allocate.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED` without changing the budget on failure.
  [[nodiscard]] ConfigResult<void> charge_allocation(std::size_t bytes) noexcept;

  /// @brief Validate one string and charge its owning result bytes before allocating it.
  /// @param bytes UTF-8 bytes the codec is about to copy.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED`.
  [[nodiscard]] ConfigResult<void> charge_string(std::size_t bytes) noexcept;

  /// @brief Validate one container and charge its element storage before allocating it.
  /// @param entries Number of immediate elements.
  /// @param element_bytes Owning storage bytes charged per element.
  /// @return Success or `CONFIG_LIMIT_EXCEEDED`, including multiplication overflow.
  [[nodiscard]] ConfigResult<void> charge_container(std::size_t entries,
                                                    std::size_t element_bytes) noexcept;

 private:
  EncodeContext(std::string path, detail::CodecCallState& state, std::uint32_t depth)
      : path_(std::move(path)), state_(state), depth_(depth) {}
  [[nodiscard]] ConfigResult<void> charge_node() noexcept;
  std::string path_;
  detail::CodecCallState& state_;
  std::uint32_t depth_{1};
  friend struct detail::CodecContextFactory;
};

/// @brief Declare the sole application customization point for typed configuration values.
///
/// A user specialization must provide whole-value `decode` and `encode` with the exact
/// ConfigCodecSignature signatures. It may implement them directly or inherit ObjectCodec for
/// ordinary data members. It should route every child through the supplied context and charge
/// owning allocations before performing them. No reflection or field-discovery fallback is
/// selected when a specialization is absent.
///
/// @note The decode callback intentionally accepts ConfigView by value so it retains shared
/// ownership of the immutable node for the callback duration. Specializations must preserve that
/// public signature.
///
/// @tparam T Owning application type represented by the specialization.
template <class T>
struct ConfigCodec;

/// @brief Require a visible complete codec with the exact package call signatures.
/// @tparam T Candidate owning application type.
template <class T>
concept ConfigCodecSignature = requires(
    ConfigView view, const T& value, const DecodeOptions& decode, const EncodeOptions& encode,
    DecodeContext& decode_context, EncodeContext& encode_context) {
  { ConfigCodec<T>::decode(view, decode, decode_context) } -> std::same_as<ConfigResult<T>>;
  {
    ConfigCodec<T>::encode(value, encode, encode_context)
  } -> std::same_as<ConfigResult<ConfigValue>>;
};

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
/// @brief Require a complete codec; exception-enabled helpers translate escaping exceptions.
/// @tparam T Candidate owning application type.
template <class T>
concept ConfigCodable = ConfigCodecSignature<T>;
#else
/// @brief Require a complete non-throwing codec in an exception-disabled translation unit.
/// @tparam T Candidate owning application type.
template <class T>
concept ConfigCodable = ConfigCodecSignature<T> &&
                        requires(ConfigView view, const T& value, const DecodeOptions& decode,
                                 const EncodeOptions& encode, DecodeContext& decode_context,
                                 EncodeContext& encode_context) {
                          requires noexcept(ConfigCodec<T>::decode(view, decode, decode_context));
                          requires noexcept(ConfigCodec<T>::encode(value, encode, encode_context));
                        };
#endif

namespace detail {
struct CodecContextFactory {
  [[nodiscard]] static DecodeContext decode_root(CodecCallState& state) {
    return DecodeContext{"", state, 1U};
  }
  [[nodiscard]] static ConfigResult<void> charge_root(DecodeContext& context,
                                                      const ConfigView& value) noexcept {
    return context.charge(value);
  }
  [[nodiscard]] static EncodeContext encode_root(CodecCallState& state) {
    return EncodeContext{"", state, 1U};
  }
  [[nodiscard]] static ConfigResult<void> charge_root(EncodeContext& context) noexcept {
    return context.charge_node();
  }
};

inline std::string append_path_token(std::string_view path, std::string_view token) {
  std::string result{path};
  result.push_back('/');
  for (const char value : token) {
    if (value == '~') {
      result += "~0";
    } else if (value == '/') {
      result += "~1";
    } else {
      result.push_back(value);
    }
  }
  return result;
}
}  // namespace detail

inline DecodeContext DecodeContext::child(std::string_view token) const {
  return DecodeContext{detail::append_path_token(path_, token), state_, depth_ + 1U};
}

inline ConfigResult<void> DecodeContext::charge(const ConfigView& value) noexcept {
  if (state_.remaining_nodes == 0 || depth_ > 64U || path_.size() > ConfigError::kMaxPathBytes ||
      value.size() > 1'000'000U) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec budget exhausted", path_));
  }
  --state_.remaining_nodes;
  if (value.kind() == ConfigValueKind::kString) {
    const auto text = value.try_as_string();
    if (!text) {
      return tl::make_unexpected(text.error());
    }
    if (text->size() > 4U * 1024U * 1024U) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kLimitExceeded, "configuration codec string budget exhausted", path_));
    }
  }
  return {};
}

inline ConfigResult<void> DecodeContext::charge_allocation(std::size_t bytes) noexcept {
  if (bytes > state_.remaining_allocation_bytes) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec allocation budget exhausted", path_));
  }
  state_.remaining_allocation_bytes -= bytes;
  return {};
}

inline ConfigResult<void> DecodeContext::charge_string(std::size_t bytes) noexcept {
  if (bytes > 4U * 1024U * 1024U) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec string budget exhausted", path_));
  }
  return charge_allocation(bytes);
}

inline ConfigResult<void> DecodeContext::charge_container(std::size_t entries,
                                                          std::size_t element_bytes) noexcept {
  if (entries > 1'000'000U ||
      (entries != 0 && element_bytes > std::numeric_limits<std::size_t>::max() / entries)) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec container budget exhausted", path_));
  }
  return charge_allocation(entries * element_bytes);
}

inline ConfigResult<void> EncodeContext::charge_node() noexcept {
  if (state_.remaining_nodes == 0 || depth_ > 64U || path_.size() > ConfigError::kMaxPathBytes) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec budget exhausted", path_));
  }
  --state_.remaining_nodes;
  return {};
}

inline ConfigResult<void> EncodeContext::charge_allocation(std::size_t bytes) noexcept {
  if (bytes > state_.remaining_allocation_bytes) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec allocation budget exhausted", path_));
  }
  state_.remaining_allocation_bytes -= bytes;
  return {};
}

inline ConfigResult<void> EncodeContext::charge_string(std::size_t bytes) noexcept {
  if (bytes > 4U * 1024U * 1024U) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec string budget exhausted", path_));
  }
  return charge_allocation(bytes);
}

inline ConfigResult<void> EncodeContext::charge_container(std::size_t entries,
                                                          std::size_t element_bytes) noexcept {
  if (entries > 1'000'000U ||
      (entries != 0 && element_bytes > std::numeric_limits<std::size_t>::max() / entries)) {
    return tl::make_unexpected(ConfigError::make(
        ConfigErrorCode::kLimitExceeded, "configuration codec container budget exhausted", path_));
  }
  return charge_allocation(entries * element_bytes);
}

template <class T>
ConfigResult<T> DecodeContext::decode_child(std::string_view token, ConfigView value,
                                            const DecodeOptions& options) {
  auto child_context = child(token);
  auto charged = child_context.charge(value);
  if (!charged) {
    return tl::make_unexpected(charged.error());
  }
  auto decoded = ConfigCodec<T>::decode(value, options, child_context);
  if (!decoded) {
    if (decoded.error().path().empty()) {
      return tl::make_unexpected(
          ConfigError::nest(ConfigError::make(decoded.error().code(),
                                              decoded.error().message(),
                                              child_context.path(),
                                              decoded.error().source(),
                                              decoded.error().excerpt(),
                                              decoded.error().operation_stage(),
                                              decoded.error().destination_state()),
                            decoded.error()));
    }
    return tl::make_unexpected(decoded.error());
  }
  return ConfigResult<T>{std::move(*decoded)};
}

template <class T>
ConfigResult<ConfigValue> EncodeContext::encode_child(std::string_view token, const T& value,
                                                      const EncodeOptions& options) {
  EncodeContext child_context{detail::append_path_token(path_, token), state_, depth_ + 1U};
  auto charged = child_context.charge_node();
  if (!charged) {
    return tl::make_unexpected(charged.error());
  }
  auto encoded = ConfigCodec<T>::encode(value, options, child_context);
  if (!encoded && encoded.error().path().empty()) {
    return tl::make_unexpected(
        ConfigError::nest(ConfigError::make(encoded.error().code(),
                                            encoded.error().message(),
                                            child_context.path(),
                                            encoded.error().source(),
                                            encoded.error().excerpt(),
                                            encoded.error().operation_stage(),
                                            encoded.error().destination_state()),
                          encoded.error()));
  }
  return encoded;
}

/// @brief Try to decode one complete semantic value through its explicit ConfigCodec
/// specialization.
///
/// The helper owns one monotonic context tree. A successful codec must close every ObjectDecoder
/// or ArrayDecoder scope it opened. With exceptions enabled, an escaping ConfigException retains
/// its structured error; other exceptions become `CONFIG_CODEC_FAILURE`. Without exceptions,
/// ConfigCodable requires non-throwing callbacks.
///
/// @tparam T Owning destination type.
/// @param view Valid semantic value.
/// @param options Immutable typed-decoding policy.
/// @return The owning decoded value or a bounded structured failure.
template <ConfigCodable T>
[[nodiscard]] ConfigResult<T> try_decode_config(ConfigView view,
                                                const DecodeOptions& options = DecodeOptions{}) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    detail::CodecCallState state;
    auto context = detail::CodecContextFactory::decode_root(state);
    auto charged = detail::CodecContextFactory::charge_root(context, view);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    auto result = ConfigCodec<T>::decode(view, options, context);
    if (!result) {
      return tl::make_unexpected(result.error());
    }
    if (state.unfinished_decoders != 0) {
      return tl::make_unexpected(ConfigError::make(ConfigErrorCode::kCodecFailure,
                                                   "configuration decoder did not call finish"));
    }
    return ConfigResult<T>{std::move(*result)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const ConfigException& exception) {
    return tl::make_unexpected(exception.error());
  } catch (...) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kCodecFailure, "configuration codec threw"));
  }
#endif
}

/// @brief Decode one complete semantic value through its explicit ConfigCodec specialization.
/// @tparam T Owning destination type.
/// @param view Valid semantic value.
/// @param options Immutable typed-decoding policy.
/// @return The owning decoded value.
/// @throws ConfigException on a structured decode or codec failure.
template <ConfigCodable T>
[[nodiscard]] T decode_config(ConfigView view, const DecodeOptions& options = DecodeOptions{}) {
  return detail::value_or_throw(try_decode_config<T>(std::move(view), options));
}

/// @brief Try to encode one complete typed value through its explicit ConfigCodec specialization.
///
/// The helper owns one monotonic context tree. With exceptions enabled, an escaping
/// ConfigException retains its structured error; other exceptions become `CONFIG_CODEC_FAILURE`.
///
/// @tparam T Owning source type.
/// @param value Typed value borrowed for the duration of this call.
/// @param options Immutable typed-encoding policy.
/// @return The immutable semantic value or a bounded structured failure.
template <ConfigCodable T>
[[nodiscard]] ConfigResult<ConfigValue> try_encode_config(const T& value,
                                                          const EncodeOptions& options = {}) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    detail::CodecCallState state;
    auto context = detail::CodecContextFactory::encode_root(state);
    auto charged = detail::CodecContextFactory::charge_root(context);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    auto result = ConfigCodec<T>::encode(value, options, context);
    return result;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const ConfigException& exception) {
    return tl::make_unexpected(exception.error());
  } catch (...) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kCodecFailure, "configuration codec threw"));
  }
#endif
}

/// @brief Encode one complete typed value through its explicit ConfigCodec specialization.
/// @tparam T Owning source type.
/// @param value Typed value borrowed for the duration of this call.
/// @param options Immutable typed-encoding policy.
/// @return The immutable semantic value.
/// @throws ConfigException on a structured encode or codec failure.
template <ConfigCodable T>
[[nodiscard]] ConfigValue encode_config(const T& value, const EncodeOptions& options = {}) {
  return detail::value_or_throw(try_encode_config(value, options));
}

/// @brief Traverse an object while recording consumed members for strict finish validation.
class ObjectDecoder {
 public:
  /// @brief Begin consuming one object value through the package-owned decode context.
  ///
  /// Call \ref finish exactly once before returning a successful codec result.
  ///
  /// @param view Object value to traverse. A non-object is reported by the first operation.
  /// @param options Immutable policy controlling unknown members and child conversion.
  /// @param context Borrowed context that must outlive this decoder.
  ObjectDecoder(ConfigView view, const DecodeOptions& options, DecodeContext& context)
      : view_(std::move(view)), options_(options), context_(context) {
    context_.begin_decoder();
    auto charged = context_.charge_container(view_.size(), sizeof(bool));
    if (!charged) {
      initialization_error_ = charged.error();
      return;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      consumed_.resize(view_.size(), false);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      initialization_error_ = ConfigError::make(ConfigErrorCode::kLimitExceeded,
                                                "configuration decoder allocation failed",
                                                context_.path());
    }
#endif
  }

  /// @brief Disable copying because the decoder owns consumption state and borrows a context.
  ObjectDecoder(const ObjectDecoder&) = delete;
  /// @brief Disable copy assignment because the decoder owns consumption state.
  ObjectDecoder& operator=(const ObjectDecoder&) = delete;
  /// @brief Destroy local consumption state without implicitly closing the context scope.
  ///
  /// A codec that returns success without calling \ref finish is rejected by try_decode_config.
  ~ObjectDecoder() = default;

  /// @brief Decode and mark one required object member as consumed.
  /// @tparam T Owning member type satisfying ConfigCodable.
  /// @param key Exact unescaped UTF-8 object key.
  /// @return The decoded value, `CONFIG_PATH_NOT_FOUND`, or a structured child failure.
  template <ConfigCodable T>
  [[nodiscard]] ConfigResult<T> required(std::string_view key) {
    if (initialization_error_) {
      return tl::make_unexpected(*initialization_error_);
    }
    if (view_.kind() != ConfigValueKind::kObject) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kTypeMismatch, "configuration value is not an object", context_.path()));
    }
    for (std::size_t index = 0; index < view_.size(); ++index) {
      auto candidate = view_.try_key(index);
      if (candidate && *candidate == key) {
        consumed_[index] = true;
        auto child = view_.try_element(index);
        if (!child) {
          return tl::make_unexpected(child.error());
        }
        return context_.decode_child<T>(key, *child, options_);
      }
    }
    return tl::make_unexpected(ConfigError::make(ConfigErrorCode::kPathNotFound,
                                                 "required object member is missing",
                                                 detail::append_path_token(context_.path(), key)));
  }

  /// @brief Decode and mark one optional object member as consumed when present.
  /// @tparam T Owning member type satisfying ConfigCodable.
  /// @param key Exact unescaped UTF-8 object key.
  /// @return Empty optional when absent, the decoded value when present, or a structured failure.
  template <ConfigCodable T>
  [[nodiscard]] ConfigResult<std::optional<T>> optional(std::string_view key) {
    if (initialization_error_) {
      return tl::make_unexpected(*initialization_error_);
    }
    if (view_.kind() != ConfigValueKind::kObject) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kTypeMismatch, "configuration value is not an object", context_.path()));
    }
    for (std::size_t index = 0; index < view_.size(); ++index) {
      auto candidate = view_.try_key(index);
      if (candidate && *candidate == key) {
        consumed_[index] = true;
        auto child = view_.try_element(index);
        if (!child) {
          return tl::make_unexpected(child.error());
        }
        auto value = context_.decode_child<T>(key, *child, options_);
        if (!value) {
          return tl::make_unexpected(value.error());
        }
        return std::optional<T>{std::move(*value)};
      }
    }
    return std::optional<T>{};
  }

  /// @brief Validate complete object consumption and close this decoder scope.
  /// @return Success, the construction failure, or `CONFIG_CODEC_FAILURE` for an unconsumed member
  /// under UnknownFieldPolicy::kReject.
  /// @note Calling finish again after success is idempotent.
  [[nodiscard]] ConfigResult<void> finish() {
    if (initialization_error_) {
      return tl::make_unexpected(*initialization_error_);
    }
    if (finished_) {
      return {};
    }
    if (options_.unknown_fields() == UnknownFieldPolicy::kReject) {
      for (const bool consumed : consumed_) {
        if (!consumed) {
          return tl::make_unexpected(ConfigError::make(ConfigErrorCode::kCodecFailure,
                                                       "object contains an unconsumed member",
                                                       context_.path()));
        }
      }
    }
    finished_ = true;
    context_.finish_decoder();
    return {};
  }

 private:
  ConfigView view_;
  DecodeOptions options_;
  DecodeContext& context_;
  std::vector<bool> consumed_;
  std::optional<ConfigError> initialization_error_;
  bool finished_{false};
};

namespace detail {

template <class Owner, class Member>
struct ObjectField {
  using owner_type = Owner;
  using member_type = Member;

  std::string_view name;
  Member Owner::* member;
  bool required;
};

template <class T, class Field>
struct ObjectFieldValid : std::false_type {};

template <class T, class Owner, class Member>
struct ObjectFieldValid<T, ObjectField<Owner, Member>>
    : std::bool_constant<std::same_as<T, Owner> && ConfigCodable<Member> &&
                         std::is_constructible_v<Member, Member&&> &&
                         std::is_assignable_v<Member&, Member&&>> {};

template <class T, class Fields>
[[nodiscard]] constexpr bool validate_object_fields(const Fields&) {
  return false;
}

template <class T, class... Fields>
[[nodiscard]] constexpr bool validate_object_fields(const std::tuple<Fields...>& fields) {
  if constexpr (!std::default_initializable<T> || !std::is_constructible_v<T, T&&> ||
                !(ObjectFieldValid<T, Fields>::value && ...)) {
    return false;
  } else {
    const auto members_are_valid =
        std::apply([](const auto&... field) { return ((field.member != nullptr) && ...); }, fields);
    if (!members_are_valid) {
      return false;
    }
    const auto names = std::apply(
        [](const auto&... field) {
          return std::array<std::string_view, sizeof...(Fields)>{field.name...};
        },
        fields);
    for (std::size_t first = 0; first < names.size(); ++first) {
      for (std::size_t second = first + 1; second < names.size(); ++second) {
        if (names[first] == names[second]) {
          return false;
        }
      }
    }
    return true;
  }
}

template <class T, class = void>
struct ObjectCodecSchemaValid : std::false_type {};

template <class T>
struct ObjectCodecSchemaValid<
    T,
    std::void_t<decltype(std::bool_constant<validate_object_fields<T>(ConfigCodec<T>::fields)>{})>>
    : std::bool_constant<validate_object_fields<T>(ConfigCodec<T>::fields)> {};

template <class T>
concept ValidObjectCodecSchema = ObjectCodecSchemaValid<T>::value;

template <std::size_t I = 0, class T, class Fields>
ConfigResult<void> decode_object_fields(ObjectDecoder& decoder, T& value, const Fields& fields) {
  if constexpr (I == std::tuple_size_v<Fields>) {
    return {};
  } else {
    const auto& field = std::get<I>(fields);
    using Field = std::remove_cvref_t<decltype(field)>;
    using Member = typename Field::member_type;
    if (field.required) {
      auto decoded = decoder.required<Member>(field.name);
      if (!decoded) {
        return tl::make_unexpected(decoded.error());
      }
      value.*(field.member) = std::move(*decoded);
    } else {
      auto decoded = decoder.optional<Member>(field.name);
      if (!decoded) {
        return tl::make_unexpected(decoded.error());
      }
      if (*decoded) {
        value.*(field.member) = std::move(**decoded);
      }
    }
    return decode_object_fields<I + 1>(decoder, value, fields);
  }
}

template <std::size_t I = 0, class T, class Fields, std::size_t N>
ConfigResult<void> encode_object_fields(const T& value, const Fields& fields,
                                        const EncodeOptions& options, EncodeContext& context,
                                        std::array<ConfigObjectEntry, N>& entries) {
  if constexpr (I == N) {
    return {};
  } else {
    const auto& field = std::get<I>(fields);
    using Field = std::remove_cvref_t<decltype(field)>;
    using Member = typename Field::member_type;
    auto key_charged = context.charge_string(field.name.size());
    if (!key_charged) {
      return tl::make_unexpected(key_charged.error());
    }
    auto encoded = context.encode_child<Member>(field.name, value.*(field.member), options);
    if (!encoded) {
      return tl::make_unexpected(encoded.error());
    }
    entries[I] = ConfigObjectEntry{field.name, std::move(*encoded)};
    return encode_object_fields<I + 1>(value, fields, options, context, entries);
  }
}

}  // namespace detail

/// @brief Describe one required member in an ObjectCodec field list.
/// @tparam Owner Object type containing the member.
/// @tparam Member Member value type with a visible ConfigCodec specialization.
/// @param name Configuration object key.
/// @param member Pointer to the object member populated by decoding.
/// @return A constexpr field descriptor consumed by ObjectCodec.
template <class Owner, class Member>
[[nodiscard]] constexpr auto object_field(std::string_view name, Member Owner::* member) noexcept {
  return detail::ObjectField<Owner, Member>{name, member, true};
}

/// @brief Describe a member that retains its default value when its object key is absent.
/// @tparam Owner Object type containing the member.
/// @tparam Member Member value type with a visible ConfigCodec specialization.
/// @param name Configuration object key.
/// @param member Pointer to the object member populated when the key is present.
/// @return A constexpr field descriptor consumed by ObjectCodec.
template <class Owner, class Member>
[[nodiscard]] constexpr auto defaulted_object_field(std::string_view name,
                                                    Member Owner::* member) noexcept {
  return detail::ObjectField<Owner, Member>{name, member, false};
}

/// @brief Implement ConfigCodec for a default-constructible object from a declared field tuple.
///
/// A ConfigCodec specialization can inherit this type and declare a static constexpr `fields`
/// tuple containing object_field and defaulted_object_field descriptors. Field names must be
/// unique, every descriptor must contain a non-null pointer to a move-constructible,
/// move-assignable ConfigCodable member of T, and T must be default-initializable and
/// move-constructible. The helper preserves child paths and codec budgets, rejects unknown keys
/// under the selected DecodeOptions, and encodes every declared field in tuple order.
///
/// @note Object traversal can allocate and invoke potentially throwing operations. The inherited
/// callbacks are therefore not admitted by ConfigCodable when exceptions are disabled; provide a
/// manually proven non-throwing ConfigCodec for that environment.
///
/// @tparam T Default-initializable object type owned by the ConfigCodec specialization.
template <class T>
struct ObjectCodec {
  /// @brief Decode declared object fields through their member codecs.
  static ConfigResult<T> decode(ConfigView view, const DecodeOptions& options,
                                DecodeContext& context)
    requires detail::ValidObjectCodecSchema<T>
  {  // NOLINT(whitespace/braces)
    if (view.kind() != ConfigValueKind::kObject) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kTypeMismatch, "configuration value is not an object", context.path()));
    }
    constexpr const auto& fields = ConfigCodec<T>::fields;
    T result{};
    ObjectDecoder object{std::move(view), options, context};
    auto decoded = detail::decode_object_fields(object, result, fields);
    if (!decoded) {
      return tl::make_unexpected(decoded.error());
    }
    auto finished = object.finish();
    if (!finished) {
      return tl::make_unexpected(finished.error());
    }
    return ConfigResult<T>{std::move(result)};
  }

  /// @brief Encode every declared object field through its member codec.
  static ConfigResult<ConfigValue> encode(const T& value, const EncodeOptions& options,
                                          EncodeContext& context)
    requires detail::ValidObjectCodecSchema<T>
  {  // NOLINT(whitespace/braces)
    constexpr const auto& fields = ConfigCodec<T>::fields;
    constexpr auto field_count = std::tuple_size_v<std::remove_cvref_t<decltype(fields)>>;
    constexpr auto field_storage_bytes =
        sizeof(std::string) + sizeof(ConfigValue) + sizeof(ConfigObjectEntry);
    auto charged = context.charge_container(field_count, field_storage_bytes);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::array<ConfigObjectEntry, field_count> entries{};
    auto encoded = detail::encode_object_fields(value, fields, options, context, entries);
    if (!encoded) {
      return tl::make_unexpected(encoded.error());
    }
    return ConfigValue::try_object(entries);
  }
};

/// @brief Traverse an array in order and enforce complete consumption.
class ArrayDecoder {
 public:
  /// @brief Begin consuming one array value through the package-owned decode context.
  ///
  /// Call \ref finish exactly once before returning a successful codec result.
  ///
  /// @param view Array value to traverse. A non-array is reported by the first operation.
  /// @param options Immutable child-conversion policy.
  /// @param context Borrowed context that must outlive this decoder.
  ArrayDecoder(ConfigView view, const DecodeOptions& options, DecodeContext& context)
      : view_(std::move(view)), options_(options), context_(context) {
    context_.begin_decoder();
  }

  /// @brief Disable copying because the decoder owns a cursor and borrows a context.
  ArrayDecoder(const ArrayDecoder&) = delete;
  /// @brief Disable copy assignment because the decoder owns a cursor.
  ArrayDecoder& operator=(const ArrayDecoder&) = delete;
  /// @brief Destroy local cursor state without implicitly closing the context scope.
  ///
  /// A codec that returns success without calling \ref finish is rejected by try_decode_config.
  ~ArrayDecoder() = default;

  /// @brief Decode the next array element in order.
  /// @tparam T Owning element type satisfying ConfigCodable.
  /// @return Empty optional at end, the decoded next element, or a structured failure.
  template <ConfigCodable T>
  [[nodiscard]] ConfigResult<std::optional<T>> next() {
    if (view_.kind() != ConfigValueKind::kArray) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kTypeMismatch, "configuration value is not an array", context_.path()));
    }
    if (index_ == view_.size()) {
      return std::optional<T>{};
    }
    const auto child_index = index_++;
    auto child = view_.try_element(child_index);
    if (!child) {
      return tl::make_unexpected(child.error());
    }
    auto value = context_.decode_child<T>(std::to_string(child_index), *child, options_);
    if (!value) {
      return tl::make_unexpected(value.error());
    }
    return std::optional<T>{std::move(*value)};
  }

  /// @brief Validate complete array consumption and close this decoder scope.
  /// @return Success or `CONFIG_CODEC_FAILURE` when elements remain.
  /// @note Calling finish again after success is idempotent.
  [[nodiscard]] ConfigResult<void> finish() {
    if (finished_) {
      return {};
    }
    if (index_ != view_.size()) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kCodecFailure, "array was not fully consumed"));
    }
    finished_ = true;
    context_.finish_decoder();
    return {};
  }

 private:
  ConfigView view_;
  DecodeOptions options_;
  DecodeContext& context_;
  std::size_t index_{0};
  bool finished_{false};
};

namespace detail {

template <class T>
[[nodiscard]] ConfigResult<T> decode_signed_integer(const ConfigView& view) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  auto value = view.try_as_signed_integer();
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  if constexpr (std::is_signed_v<T>) {
    if (*value < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
        *value > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kOutOfRange, "signed integer is out of range"));
    }
  } else {
    if (*value < 0 || static_cast<std::uint64_t>(*value) > std::numeric_limits<T>::max()) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kOutOfRange, "signed integer is out of range"));
    }
  }
  return static_cast<T>(*value);
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_unsigned_integer(const ConfigView& view) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  auto value = view.try_as_unsigned_integer();
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  if constexpr (std::is_signed_v<T>) {
    if (*value > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kOutOfRange, "unsigned integer is out of range"));
    }
  } else if (*value > std::numeric_limits<T>::max()) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "unsigned integer is out of range"));
  }
  return static_cast<T>(*value);
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_floating_integer(const ConfigView& view) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  auto value = view.try_as_floating();
  constexpr int value_bits = std::numeric_limits<T>::digits;
  const double upper_exclusive = std::ldexp(1.0, value_bits);
  const double lower_inclusive = std::is_signed_v<T> ? -upper_exclusive : 0.0;
  if (!value || !std::isfinite(*value) || std::trunc(*value) != *value ||
      *value < lower_inclusive || *value >= upper_exclusive) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "floating value is not an exact integer"));
  }
  return static_cast<T>(*value);
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_integer(const ConfigView& view, const DecodeOptions& options) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  switch (view.kind()) {
    case ConfigValueKind::kSignedInteger:
      return decode_signed_integer<T>(view);
    case ConfigValueKind::kUnsignedInteger:
      return decode_unsigned_integer<T>(view);
    case ConfigValueKind::kFloating:
      if (options.numeric_conversion() == NumericConversionPolicy::kAllowCheckedLossy) {
        return decode_floating_integer<T>(view);
      }
      break;
    default:
      break;
  }
  return tl::make_unexpected(
      ConfigError::make(ConfigErrorCode::kTypeMismatch, "configuration value is not an integer"));
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_native_floating(const ConfigView& view,
                                                     const DecodeOptions& options) {
  static_assert(std::is_floating_point_v<T>);
  auto value = view.try_as_floating();
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  const T converted = static_cast<T>(*value);
  if (!std::isfinite(converted)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric value is out of range"));
  }
  if (options.numeric_conversion() == NumericConversionPolicy::kLossless &&
      static_cast<double>(converted) != *value) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric conversion would be lossy"));
  }
  if (*value != 0.0 && converted == static_cast<T>(0)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric conversion underflowed"));
  }
  return converted;
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_signed_floating(const ConfigView& view,
                                                     const DecodeOptions& options) {
  static_assert(std::is_floating_point_v<T>);
  auto value = view.try_as_signed_integer();
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  const std::uint64_t magnitude = *value < 0 ? static_cast<std::uint64_t>(-(*value + 1)) + 1U
                                             : static_cast<std::uint64_t>(*value);
  if (options.numeric_conversion() == NumericConversionPolicy::kLossless &&
      !integer_exactly_representable<T>(magnitude)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric conversion would be lossy"));
  }
  const T converted = static_cast<T>(*value);
  if (!std::isfinite(converted)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric value is out of range"));
  }
  return converted;
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_unsigned_floating(const ConfigView& view,
                                                       const DecodeOptions& options) {
  static_assert(std::is_floating_point_v<T>);
  auto value = view.try_as_unsigned_integer();
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  if (options.numeric_conversion() == NumericConversionPolicy::kLossless &&
      !integer_exactly_representable<T>(*value)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric conversion would be lossy"));
  }
  const T converted = static_cast<T>(*value);
  if (!std::isfinite(converted)) {
    return tl::make_unexpected(
        ConfigError::make(ConfigErrorCode::kOutOfRange, "numeric value is out of range"));
  }
  return converted;
}

template <class T>
[[nodiscard]] ConfigResult<T> decode_floating(const ConfigView& view,
                                              const DecodeOptions& options) {
  static_assert(std::is_floating_point_v<T>);
  switch (view.kind()) {
    case ConfigValueKind::kFloating:
      return decode_native_floating<T>(view, options);
    case ConfigValueKind::kSignedInteger:
      return decode_signed_floating<T>(view, options);
    case ConfigValueKind::kUnsignedInteger:
      return decode_unsigned_floating<T>(view, options);
    default:
      break;
  }
  return tl::make_unexpected(
      ConfigError::make(ConfigErrorCode::kTypeMismatch, "configuration value is not numeric"));
}

template <class T>
[[nodiscard]] ConfigResult<ConfigValue> encode_integer(T value, EncodeContext& context) {
  auto charged = context.charge_allocation(64U);
  if (!charged) {
    return tl::make_unexpected(charged.error());
  }
  if constexpr (std::is_signed_v<T>) {
    return ConfigValue::try_signed_integer(static_cast<std::int64_t>(value));
  } else {
    return ConfigValue::try_unsigned_integer(static_cast<std::uint64_t>(value));
  }
}

}  // namespace detail

#define HOLOSCAN_CONFIG_INTEGER_CODEC(Type)                                         \
  template <>                                                                       \
  struct ConfigCodec<Type> {                                                        \
    /* Preserve ConfigCodec's intentional by-value ConfigView signature. */         \
    static ConfigResult<Type> decode(ConfigView view, const DecodeOptions& options, \
                                     DecodeContext&) {                              \
      return detail::decode_integer<Type>(view, options);                           \
    }                                                                               \
    static ConfigResult<ConfigValue> encode(Type value, const EncodeOptions&,       \
                                            EncodeContext& context) {               \
      return detail::encode_integer(value, context);                                \
    }                                                                               \
  }

/// @brief Provide the built-in signed 8-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::int8_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in unsigned 8-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::uint8_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in signed 16-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::int16_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in unsigned 16-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::uint16_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in signed 32-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::int32_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in unsigned 32-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::uint32_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in signed 64-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::int64_t);  // NOLINT(performance-unnecessary-value-param)
/// @brief Provide the built-in unsigned 64-bit integer codec.
HOLOSCAN_CONFIG_INTEGER_CODEC(std::uint64_t);  // NOLINT(performance-unnecessary-value-param)
#undef HOLOSCAN_CONFIG_INTEGER_CODEC

/// @brief Provide the built-in Boolean codec.
template <>
struct ConfigCodec<bool> {
  /// @brief Decode an exact Boolean semantic value.
  /// @param view Semantic value to inspect.
  /// @return The Boolean or `CONFIG_TYPE_MISMATCH`.
  static ConfigResult<bool> decode(ConfigView view,  // NOLINT(performance-unnecessary-value-param)
                                   const DecodeOptions&, DecodeContext&) {
    return view.try_as_bool();
  }
  /// @brief Encode a Boolean after charging its owning node allocation.
  /// @param value Boolean to encode.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable Boolean value or a structured budget/allocation failure.
  static ConfigResult<ConfigValue> encode(bool value, const EncodeOptions&,
                                          EncodeContext& context) {
    auto charged = context.charge_allocation(64U);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return ConfigValue::try_boolean(value);
  }
};

/// @brief Provide the built-in binary32 codec with explicit numeric conversion policy.
template <>
struct ConfigCodec<float> {
  /// @brief Decode a number under the supplied lossless or checked-lossy policy.
  /// @param view Semantic number to inspect.
  /// @param options Numeric conversion policy.
  /// @return The binary32 value or a structured type/range/inexact failure.
  static ConfigResult<float> decode(ConfigView view,  // NOLINT(performance-unnecessary-value-param)
                                    const DecodeOptions& options, DecodeContext&) {
    return detail::decode_floating<float>(view, options);
  }
  /// @brief Encode a finite binary32 value as the floating semantic kind.
  /// @param value Number to encode.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable floating value or a structured budget/value/allocation failure.
  static ConfigResult<ConfigValue> encode(float value, const EncodeOptions&,
                                          EncodeContext& context) {
    auto charged = context.charge_allocation(64U);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return ConfigValue::try_floating(static_cast<double>(value));
  }
};

/// @brief Provide the built-in binary64 codec with explicit numeric conversion policy.
template <>
struct ConfigCodec<double> {
  /// @brief Decode a number under the supplied lossless or checked-lossy policy.
  /// @param view Semantic number to inspect.
  /// @param options Numeric conversion policy.
  /// @return The binary64 value or a structured type/range/inexact failure.
  static ConfigResult<double> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions& options, DecodeContext&) {
    return detail::decode_floating<double>(view, options);
  }
  /// @brief Encode a finite binary64 value as the floating semantic kind.
  /// @param value Number to encode.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable floating value or a structured budget/value/allocation failure.
  static ConfigResult<ConfigValue> encode(double value, const EncodeOptions&,
                                          EncodeContext& context) {
    auto charged = context.charge_allocation(64U);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return ConfigValue::try_floating(value);
  }
};

/// @brief Provide the built-in owning UTF-8 string codec.
template <>
struct ConfigCodec<std::string> {
  /// @brief Decode and copy a UTF-8 string under the monotonic allocation budget.
  /// @param view Semantic string to inspect.
  /// @param context Borrowed monotonic decode context.
  /// @return The owning string or a structured type/budget/allocation failure.
  static ConfigResult<std::string> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext& context) {
    auto value = view.try_as_string();
    if (!value) {
      return tl::make_unexpected(value.error());
    }
    auto charged = context.charge_string(value->size());
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return std::string{*value};
  }
  /// @brief Encode and copy an owning UTF-8 string under the monotonic allocation budget.
  /// @param value UTF-8 string to encode.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable string value or a structured UTF-8/budget/allocation failure.
  static ConfigResult<ConfigValue> encode(const std::string& value, const EncodeOptions&,
                                          EncodeContext& context) {
    auto charged = context.charge_string(value.size());
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    charged = context.charge_allocation(64U);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return ConfigValue::try_string(value);
  }
};

/// @brief Provide the identity codec for immutable semantic values.
template <>
struct ConfigCodec<ConfigValue> {
  /// @brief Share the decoded semantic node as an owning ConfigValue.
  /// @param view Semantic value to share.
  /// @return The immutable value without changing its kind.
  static ConfigResult<ConfigValue> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) {
    return view.to_value();
  }
  /// @brief Share the encoded ConfigValue without changing its semantic kind.
  /// @param value Immutable value to share.
  /// @return The same immutable semantic value.
  static ConfigResult<ConfigValue> encode(const ConfigValue& value, const EncodeOptions&,
                                          EncodeContext&) {
    return value;
  }
};

/// @brief Provide the built-in null-or-value codec for std::optional.
/// @tparam T Owning contained type satisfying ConfigCodable.
template <ConfigCodable T>
struct ConfigCodec<std::optional<T>> {
  /// @brief Decode null as empty and every other kind through ConfigCodec<T>.
  /// @param view Null or a semantic value for T.
  /// @param options Typed child conversion policy.
  /// @param context Borrowed monotonic decode context.
  /// @return Empty, a decoded T, or a structured child failure.
  static ConfigResult<std::optional<T>> decode(ConfigView view, const DecodeOptions& options,
                                               DecodeContext& context) {
    if (view.kind() == ConfigValueKind::kNull) {
      return std::optional<T>{};
    }
    auto value = ConfigCodec<T>::decode(view, options, context);
    if (!value) {
      return tl::make_unexpected(value.error());
    }
    return std::optional<T>{std::move(*value)};
  }
  /// @brief Encode empty as null and a contained value through ConfigCodec<T>.
  /// @param value Optional value to encode.
  /// @param options Typed child encoding policy.
  /// @param context Borrowed monotonic encode context.
  /// @return Null, an encoded T, or a structured child failure.
  static ConfigResult<ConfigValue> encode(const std::optional<T>& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return value ? ConfigCodec<T>::encode(*value, options, context)
                 : ConfigResult<ConfigValue>{ConfigValue::null()};
  }
};

/// @brief Provide the built-in ordered variable-length array codec for std::vector.
/// @tparam T Owning element type satisfying ConfigCodable.
template <ConfigCodable T>
struct ConfigCodec<std::vector<T>> {
  /// @brief Decode every array element in order under shared monotonic budgets.
  /// @param view Semantic array to decode.
  /// @param options Typed element conversion policy.
  /// @param context Borrowed monotonic decode context.
  /// @return The owning ordered vector or a structured type/budget/child failure.
  static ConfigResult<std::vector<T>> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions& options, DecodeContext& context) {
    if (view.kind() != ConfigValueKind::kArray) {
      return tl::make_unexpected(
          ConfigError::make(ConfigErrorCode::kTypeMismatch, "configuration value is not an array"));
    }
    auto charged = context.charge_container(view.size(), sizeof(T));
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::vector<T> result;
    result.reserve(view.size());
    for (std::size_t index = 0; index < view.size(); ++index) {
      auto child = view.try_element(index);
      if (!child) {
        return tl::make_unexpected(child.error());
      }
      auto value = context.decode_child<T>(std::to_string(index), *child, options);
      if (!value) {
        return tl::make_unexpected(value.error());
      }
      result.push_back(std::move(*value));
    }
    return result;
  }
  /// @brief Encode every vector element in order under shared monotonic budgets.
  /// @param values Ordered elements to encode.
  /// @param options Typed element encoding policy.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable semantic array or a structured budget/child failure.
  static ConfigResult<ConfigValue> encode(const std::vector<T>& values,
                                          const EncodeOptions& options, EncodeContext& context) {
    auto charged = context.charge_container(
        values.size(), sizeof(ConfigValue) + sizeof(std::shared_ptr<const detail::ConfigNode>));
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::vector<ConfigValue> encoded;
    encoded.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      auto item = context.encode_child<T>(std::to_string(index), values[index], options);
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      encoded.push_back(std::move(*item));
    }
    return ConfigValue::try_array(encoded);
  }
};

/// @brief Provide the built-in exact-cardinality array codec for std::array.
/// @tparam T Owning element type satisfying ConfigCodable.
/// @tparam N Required semantic array cardinality.
template <ConfigCodable T, std::size_t N>
struct ConfigCodec<std::array<T, N>> {
  /// @brief Decode exactly N elements and reject every other cardinality.
  /// @param view Semantic array to decode.
  /// @param options Typed element conversion policy.
  /// @param context Borrowed monotonic decode context.
  /// @return The exact-cardinality array or a structured type/budget/child failure.
  static ConfigResult<std::array<T, N>> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions& options, DecodeContext& context) {
    if (view.kind() != ConfigValueKind::kArray || view.size() != N) {
      return tl::make_unexpected(ConfigError::make(
          ConfigErrorCode::kTypeMismatch, "configuration array has the wrong cardinality"));
    }
    auto charged = context.charge_container(N, 0U);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::array<T, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
      auto child = view.try_element(index);
      if (!child) {
        return tl::make_unexpected(child.error());
      }
      auto value = context.decode_child<T>(std::to_string(index), *child, options);
      if (!value) {
        return tl::make_unexpected(value.error());
      }
      result[index] = std::move(*value);
    }
    return result;
  }
  /// @brief Encode all N elements in index order.
  /// @param values Ordered elements to encode.
  /// @param options Typed element encoding policy.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable semantic array or a structured budget/child failure.
  static ConfigResult<ConfigValue> encode(const std::array<T, N>& values,
                                          const EncodeOptions& options, EncodeContext& context) {
    auto charged = context.charge_container(
        N, sizeof(ConfigValue) + sizeof(std::shared_ptr<const detail::ConfigNode>));
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::vector<ConfigValue> encoded;
    encoded.reserve(N);
    for (std::size_t index = 0; index < N; ++index) {
      auto item = context.encode_child<T>(std::to_string(index), values[index], options);
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      encoded.push_back(std::move(*item));
    }
    return ConfigValue::try_array(encoded);
  }
};

/// @brief Provide the built-in string-keyed ordered-map codec.
///
/// Only `std::less<std::string>` and transparent `std::less<>` comparators are admitted so encoded
/// object behavior does not depend on an application-defined comparator.
///
/// @tparam T Owning mapped type satisfying ConfigCodable.
/// @tparam Compare Admitted standard lexicographic comparator.
/// @tparam Allocator Map allocator retained by the decoded result.
template <ConfigCodable T, class Compare, class Allocator>
  requires(std::same_as<Compare, std::less<std::string>> || std::same_as<Compare, std::less<>>)
struct ConfigCodec<std::map<std::string, T, Compare, Allocator>> {
  /// Owning decoded map type.
  using Map = std::map<std::string, T, Compare, Allocator>;
  /// @brief Decode each unique object entry into the resulting ordered map.
  /// @param view Semantic object to decode.
  /// @param options Typed mapped-value conversion policy.
  /// @param context Borrowed monotonic decode context.
  /// @return The owning ordered map or a structured type/budget/child failure.
  static ConfigResult<Map> decode(ConfigView view,  // NOLINT(performance-unnecessary-value-param)
                                  const DecodeOptions& options, DecodeContext& context) {
    if (view.kind() != ConfigValueKind::kObject) {
      return tl::make_unexpected(ConfigError::make(ConfigErrorCode::kTypeMismatch,
                                                   "configuration value is not an object"));
    }
    auto charged = context.charge_container(view.size(), sizeof(typename Map::value_type));
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    Map result;
    for (std::size_t index = 0; index < view.size(); ++index) {
      auto key = view.try_key(index);
      auto child = view.try_element(index);
      if (!key) {
        return tl::make_unexpected(key.error());
      }
      if (!child) {
        return tl::make_unexpected(child.error());
      }
      auto key_charge = context.charge_string(key->size());
      if (!key_charge) {
        return tl::make_unexpected(key_charge.error());
      }
      auto value = context.decode_child<T>(*key, *child, options);
      if (!value) {
        return tl::make_unexpected(value.error());
      }
      result.emplace(std::string{*key}, std::move(*value));
    }
    return result;
  }
  /// @brief Encode map entries as an object under shared monotonic budgets.
  /// @param values String-keyed values to encode.
  /// @param options Typed mapped-value encoding policy.
  /// @param context Borrowed monotonic encode context.
  /// @return The immutable semantic object or a structured UTF-8/budget/child failure.
  static ConfigResult<ConfigValue> encode(const Map& values, const EncodeOptions& options,
                                          EncodeContext& context) {
    auto charged = context.charge_container(
        values.size(), sizeof(std::string) + sizeof(ConfigValue) + sizeof(ConfigObjectEntry));
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    std::vector<std::string> keys;
    std::vector<ConfigValue> encoded;
    keys.reserve(values.size());
    encoded.reserve(values.size());
    for (const auto& [key, value] : values) {
      auto key_charge = context.charge_string(key.size());
      if (!key_charge) {
        return tl::make_unexpected(key_charge.error());
      }
      auto item = context.encode_child<T>(key, value, options);
      if (!item) {
        return tl::make_unexpected(item.error());
      }
      keys.push_back(key);
      encoded.push_back(std::move(*item));
    }
    std::vector<ConfigObjectEntry> entries;
    entries.reserve(values.size());
    for (std::size_t index = 0; index < keys.size(); ++index) {
      entries.push_back({keys[index], encoded[index]});
    }
    return ConfigValue::try_object(entries);
  }
};

template <class T>
T Document::get(const Path& path, DecodeOptions options) const {
  return detail::value_or_throw(try_get<T>(path, options));
}

template <class T>
ConfigResult<T> Document::try_get(const Path& path, DecodeOptions options) const {
  auto value = try_at(path);
  if (!value) {
    return tl::make_unexpected(value.error());
  }
  return try_decode_config<T>(*value, options);
}

template <class T>
T Document::get(std::string_view path, DecodeOptions options) const {
  return detail::value_or_throw(try_get<T>(path, options));
}

template <class T>
ConfigResult<T> Document::try_get(std::string_view path, DecodeOptions options) const {
  auto parsed = Path::try_parse(path);
  if (!parsed) {
    return tl::make_unexpected(parsed.error());
  }
  return try_get<T>(*parsed, options);
}

template <class T>
T Document::patch(const Path& path, T baseline, DecodeOptions options) const {
  return detail::value_or_throw(try_patch(path, T(std::move(baseline)), options));
}

template <class T>
ConfigResult<T> Document::try_patch(const Path& path, T baseline, DecodeOptions options) const {
  auto patch_view = try_at(path);
  if (!patch_view) {
    return tl::make_unexpected(patch_view.error());
  }
  auto encoded = try_encode_config<T>(baseline);
  if (!encoded) {
    return tl::make_unexpected(encoded.error());
  }
  if (encoded->kind() != ConfigValueKind::kObject ||
      patch_view->kind() != ConfigValueKind::kObject) {
    return tl::make_unexpected(detail::make_path_error(
        ConfigErrorCode::kTypeMismatch, "typed configuration patch requires object values", path));
  }
  auto merged = try_merge_config_values(*encoded, patch_view->to_value());
  if (!merged) {
    return tl::make_unexpected(merged.error());
  }
  return try_decode_config<T>(merged->view(), options);
}

template <class T>
Document Document::replace(const Path& path, T&& value) const {
  return detail::value_or_throw(try_replace(path, std::forward<T>(value)));
}

template <class T>
ConfigResult<Document> Document::try_replace(const Path& path, T&& value) const {
  using Value = std::remove_cvref_t<T>;
  auto encoded = try_encode_config<Value>(std::forward<T>(value));
  if (!encoded) {
    return tl::make_unexpected(encoded.error());
  }
  return try_patched(Edit::replace(path, std::move(*encoded)));
}

template <class T>
Document Document::replace(std::string_view path, T&& value) const {
  return detail::value_or_throw(try_replace(path, std::forward<T>(value)));
}

template <class T>
ConfigResult<Document> Document::try_replace(std::string_view path, T&& value) const {
  auto parsed = Path::try_parse(path);
  if (!parsed) {
    return tl::make_unexpected(parsed.error());
  }
  return try_replace(*parsed, std::forward<T>(value));
}

}  // namespace holoscan::config
