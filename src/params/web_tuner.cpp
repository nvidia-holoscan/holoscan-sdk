/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/experimental/params/web_tuner.hpp>

#include <yyjson.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>

#include "runtime/runtime_callback_context.hpp"
#include "web_tuner_socket.hpp"

#if defined(__linux__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace holoscan::experimental::params {
namespace detail {

struct WebTunerResultState {
  ParamSnapshot snapshot;
  ParamUpdateSequence terminalized_through{};
  std::uint32_t accepted_edits{};
  std::uint32_t published_edits{};
};

struct WebTunerResultAccess {
  [[nodiscard]] static WebTunerResult make(std::shared_ptr<const WebTunerResultState> state) {
    return WebTunerResult{std::move(state)};
  }
};

struct WebTunerEditRecord {
  std::uint64_t request_ordinal{};
  std::uint32_t edit_ordinal{};
  ParamUpdateSequence sequence{};
  ParamGeneration target_generation{};
  std::uint64_t lifecycle_transaction{};
  std::uint64_t coalesced_into_transaction{};
  ParamAddress address{};
  std::string operator_path;
  std::string field_path;
  std::string kind;
  ParamValue desired_value{};
  std::optional<ParamReadResult> effective_value;
  ParamUpdateState state{ParamUpdateState::kSubmitted};
  bool effective_proven{};
};

struct WebTunerConnection {
  WebTunerConnection(int socket, ParamControlHandle base_control) noexcept
      : descriptor(socket), control(std::move(base_control)) {}

  int descriptor{-1};
  ParamControlHandle control;
  bool completed{};
  std::thread worker;
};

struct WebTunerState {
  std::mutex mutex;
  std::condition_variable completed_cv;
  WebTunerEndpoint endpoint;
  std::optional<RunSession> session;
  std::optional<ParamControlHandle> control;
  std::thread server;
  std::thread::id server_thread_id;
  std::atomic<int> listener{-1};
  std::atomic<bool> stop_requested{false};
  std::vector<std::shared_ptr<WebTunerConnection>> connections;
  std::uint32_t active_connections{};
  std::string cookie_name;
  std::string cookie_secret;
  std::string catalog_json;
  std::uint64_t request_ordinal{};
  std::uint32_t accepted_edits{};
  std::uint32_t published_edits{};
  std::uint32_t reserved_edits{};
  std::vector<WebTunerEditRecord> edits;
  bool finalizing{};
  bool completed{};
  bool server_completed{};
  std::optional<WebTunerResult> result;
  std::optional<holoscan::params::ParamToolError> error;
  WebTunerOptions options;
};

}  // namespace detail
namespace {

using holoscan::params::ParamResult;
using holoscan::params::ParamToolError;
using holoscan::params::ParamToolErrorCode;

constexpr std::uint32_t kHardMaxClients = 1U;
constexpr std::uint32_t kHardMaxConnections = 64U;
constexpr std::uint32_t kHardMaxPendingRecords = 4'096U;
constexpr std::uint32_t kHardMaxEvidenceRecords = 65'536U;
constexpr std::uint32_t kHardMaxMessageBytes = 16U << 20U;
constexpr std::uint32_t kMinimumRequestBytes = 256U;
constexpr std::uint32_t kMinimumResponseBytes = 256U;
constexpr auto kHardMaxShutdownTimeout = std::chrono::seconds{60};

[[nodiscard]] ParamToolError tuner_error(ParamToolErrorCode code, std::string_view token,
                                         std::optional<Error> cause = std::nullopt) {
  return ParamToolError{code, token, std::move(cause)};
}

template <typename T>
[[nodiscard]] T& require_optional(std::optional<T>& value) noexcept {
  // These optionals are populated before the server thread starts and are never reset. Preserve
  // that invariant in one checked seam instead of scattering unchecked accesses through workers.
  if (!value.has_value()) {
    std::terminate();
  }
  return value.value();
}

template <typename T>
[[nodiscard]] const T& require_optional(const std::optional<T>& value) noexcept {
  if (!value.has_value()) {
    std::terminate();
  }
  return value.value();
}

[[nodiscard]] bool options_valid(const WebTunerOptions& options) noexcept {
  const bool capacities = options.max_clients != 0U && options.max_connections != 0U &&
                          options.max_pending_requests != 0U &&
                          options.max_pending_responses != 0U && options.max_pending_edits != 0U &&
                          options.max_event_backlog != 0U && options.max_log_records != 0U &&
                          options.max_audit_records != 0U && options.max_request_bytes != 0U &&
                          options.max_response_bytes != 0U;
  return capacities && options.max_clients <= options.max_connections &&
         options.max_clients <= kHardMaxClients && options.max_connections <= kHardMaxConnections &&
         options.max_pending_requests <= kHardMaxPendingRecords &&
         options.max_pending_responses <= kHardMaxPendingRecords &&
         options.max_pending_edits <= kHardMaxPendingRecords &&
         options.max_event_backlog <= kHardMaxEvidenceRecords &&
         options.max_log_records <= kHardMaxEvidenceRecords &&
         options.max_audit_records <= kHardMaxEvidenceRecords &&
         options.max_request_bytes >= kMinimumRequestBytes &&
         options.max_request_bytes <= kHardMaxMessageBytes &&
         options.max_response_bytes >= kMinimumResponseBytes &&
         options.max_response_bytes <= kHardMaxMessageBytes &&
         options.adapter_shutdown_timeout > std::chrono::seconds::zero() &&
         options.adapter_shutdown_timeout <= kHardMaxShutdownTimeout;
}

[[nodiscard]] std::string scope_hex(TunerScopeId scope) {
  constexpr std::string_view kDigits = "0123456789abcdef";
  std::string result;
  result.reserve(scope.bytes.size() * 2U);
  for (const std::uint8_t value : scope.bytes) {
    result.push_back(kDigits[value >> 4U]);
    result.push_back(kDigits[value & 0x0FU]);
  }
  return result;
}

[[nodiscard]] std::string percent_encode_path(std::span<const std::string> segments) {
  constexpr std::string_view kHex = "0123456789ABCDEF";
  std::string result;
  for (const auto& segment : segments) {
    if (!result.empty()) {
      result.push_back('/');
    }
    for (const unsigned char byte : segment) {
      const bool safe = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                        (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' || byte == '.' ||
                        byte == '~';
      if (safe) {
        result.push_back(static_cast<char>(byte));
      } else {
        result.push_back('%');
        result.push_back(kHex[(byte >> 4U) & 0x0FU]);
        result.push_back(kHex[byte & 0x0FU]);
      }
    }
  }
  return result;
}

[[nodiscard]] std::string value_kind(ParamLogicalType type, std::uint16_t width) {
  switch (type) {
    case ParamLogicalType::kBool:
      return "bool";
    case ParamLogicalType::kSignedInteger:
      return "i" + std::to_string(width);
    case ParamLogicalType::kUnsignedInteger:
      return "u" + std::to_string(width);
    case ParamLogicalType::kFloatingPoint:
      return "f" + std::to_string(width);
    case ParamLogicalType::kDuration:
      return "duration_ns";
    case ParamLogicalType::kString:
      return "string";
    case ParamLogicalType::kSequence:
      return "sequence";
    case ParamLogicalType::kFilesystemPath:
      return "path";
    case ParamLogicalType::kObject:
      return "object";
  }
  return "object";
}

template <typename Value>
[[nodiscard]] std::string integral_text(Value value) {
  std::array<char, 96> text{};
  const auto encoded = std::to_chars(text.data(), text.data() + text.size(), value);
  return encoded.ec == std::errc() ? std::string{text.data(), encoded.ptr} : std::string{};
}

template <typename Value>
[[nodiscard]] std::string floating_text(Value value) {
  std::array<char, 96> text{};
  const auto encoded = std::to_chars(text.data(),
                                     text.data() + text.size(),
                                     value,
                                     std::chars_format::general,
                                     std::numeric_limits<Value>::max_digits10);
  return encoded.ec == std::errc() ? std::string{text.data(), encoded.ptr} : std::string{};
}

[[nodiscard]] std::optional<std::int64_t> decode_signed_scalar(ParamScalarValue scalar) noexcept {
  if (scalar.bit_width == 64U) {
    return std::bit_cast<std::int64_t>(scalar.raw_bits);
  }
  if (scalar.bit_width == 0U || scalar.bit_width >= 64U) {
    return std::nullopt;
  }
  const std::uint64_t mask = (std::uint64_t{1} << scalar.bit_width) - 1U;
  const std::uint64_t sign = std::uint64_t{1} << (scalar.bit_width - 1U);
  const std::uint64_t bits = scalar.raw_bits & mask;
  return static_cast<std::int64_t>((bits & sign) == 0U ? bits : (bits | ~mask));
}

[[nodiscard]] std::string scalar_text(ParamValue value) {
  const auto scalar = value.scalar();
  if (scalar.kind == ParamScalarKind::kBool) {
    return scalar.raw_bits == 0U ? "false" : "true";
  }
  if (scalar.kind == ParamScalarKind::kSignedInteger ||
      scalar.kind == ParamScalarKind::kDurationNanoseconds) {
    const auto decoded = decode_signed_scalar(scalar);
    return decoded.has_value() ? integral_text(decoded.value()) : std::string{};
  }
  if (scalar.kind == ParamScalarKind::kUnsignedInteger) {
    return integral_text(scalar.raw_bits);
  }
  if (scalar.kind == ParamScalarKind::kFloatingPoint && scalar.bit_width == 32U) {
    return floating_text(std::bit_cast<float>(static_cast<std::uint32_t>(scalar.raw_bits)));
  }
  if (scalar.kind == ParamScalarKind::kFloatingPoint && scalar.bit_width == 64U) {
    return floating_text(std::bit_cast<double>(scalar.raw_bits));
  }
  return {};
}

[[nodiscard]] std::string scalar_json(ParamValue value, std::string_view kind) {
  return "{\"kind\":\"" + std::string(kind) + "\",\"value\":\"" + scalar_text(value) + "\"}";
}

#if defined(__linux__)

[[nodiscard]] bool fill_os_entropy(std::span<std::uint8_t> output) noexcept {
  std::size_t filled{};
  std::uint32_t attempts{};
  while (filled < output.size() && attempts < 16U) {
    ++attempts;
    const auto count = ::getrandom(output.data() + filled, output.size() - filled, GRND_NONBLOCK);
    if (count > 0) {
      filled += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return filled == output.size();
}

[[nodiscard]] ParamResult<TunerScopeId> make_scope() {
  TunerScopeId result;
  if (!fill_os_entropy(result.bytes)) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_SCOPE_ENTROPY_FAILED"));
  }
  return result;
}

[[nodiscard]] ParamResult<std::string> make_secret() {
  std::array<std::uint8_t, 32> bytes{};
  if (!fill_os_entropy(bytes)) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_TOKEN_ENTROPY_FAILED"));
  }
  constexpr std::string_view kDigits = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2U);
  for (const auto value : bytes) {
    result.push_back(kDigits[value >> 4U]);
    result.push_back(kDigits[value & 0x0FU]);
  }
  return result;
}

struct SocketBinding {
  int descriptor{-1};
  std::uint16_t port{};
  std::string origin;
};

void close_socket(int descriptor) noexcept {
  if (descriptor >= 0) {
    ::close(descriptor);
  }
}

void shutdown_active_connections(const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  try {
    const std::lock_guard lock{state->mutex};
    for (const auto& connection : state->connections) {
      if (connection->descriptor >= 0) {
        ::shutdown(connection->descriptor, SHUT_RDWR);
      }
    }
  } catch (...) {
    std::terminate();
  }
}

void reap_completed_connections(const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  for (;;) {
    std::thread completed;
    {
      const std::lock_guard lock{state->mutex};
      const auto found = std::ranges::find_if(
          state->connections, [](const auto& connection) { return connection->completed; });
      if (found == state->connections.end()) {
        return;
      }
      completed = std::move((*found)->worker);
      state->connections.erase(found);
    }
    try {
      if (completed.joinable()) {
        completed.join();
      }
    } catch (...) {
      std::terminate();
    }
  }
}

void join_all_connections(const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  for (;;) {
    std::thread worker;
    {
      const std::lock_guard lock{state->mutex};
      if (state->connections.empty()) {
        return;
      }
      worker = std::move(state->connections.back()->worker);
      state->connections.pop_back();
    }
    try {
      if (worker.joinable()) {
        worker.join();
      }
    } catch (...) {
      std::terminate();
    }
  }
}

class SocketOwner {
 public:
  explicit SocketOwner(int descriptor) noexcept : descriptor_(descriptor) {}
  SocketOwner(const SocketOwner&) = delete;
  SocketOwner& operator=(const SocketOwner&) = delete;
  ~SocketOwner() { close_socket(descriptor_); }

  void release() noexcept { descriptor_ = -1; }

 private:
  int descriptor_;
};

[[nodiscard]] ParamResult<SocketBinding> bind_loopback(const WebTunerOptions& options) {
  const int family = options.address_family == WebTunerAddressFamily::kIpv4 ? AF_INET : AF_INET6;
  const int descriptor = ::socket(family, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (descriptor < 0) {
    return make_unexpected(tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_SOCKET_FAILED"));
  }
  const int enabled = 1;
  if (::setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) {
    close_socket(descriptor);
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_SOCKET_OPTION_FAILED"));
  }
  int bound{};
  if (family == AF_INET) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(options.port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bound = ::bind(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
  } else {
    if (::setsockopt(descriptor, IPPROTO_IPV6, IPV6_V6ONLY, &enabled, sizeof(enabled)) != 0) {
      close_socket(descriptor);
      return make_unexpected(
          tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_IPV6_ONLY_FAILED"));
    }
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_port = htons(options.port);
    address.sin6_addr = in6addr_loopback;
    bound = ::bind(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
  }
  if (bound != 0 || ::listen(descriptor, static_cast<int>(options.max_connections)) != 0) {
    close_socket(descriptor);
    return make_unexpected(tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_BIND_FAILED"));
  }
  sockaddr_storage address{};
  socklen_t address_size = sizeof(address);
  if (::getsockname(descriptor, reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
    close_socket(descriptor);
    return make_unexpected(tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_ENDPOINT_FAILED"));
  }
  std::uint16_t port{};
  std::string origin;
  if (family == AF_INET) {
    port = ntohs(reinterpret_cast<const sockaddr_in*>(&address)->sin_port);
    origin = "http://127.0.0.1:" + std::to_string(port);
  } else {
    port = ntohs(reinterpret_cast<const sockaddr_in6*>(&address)->sin6_port);
    origin = "http://[::1]:" + std::to_string(port);
  }
  return SocketBinding{.descriptor = descriptor, .port = port, .origin = std::move(origin)};
}

[[nodiscard]] bool send_bytes(const std::shared_ptr<detail::WebTunerState>& state, int connection,
                              std::string_view bytes) noexcept {
  return detail::send_web_tuner_bytes(state->stop_requested, connection, bytes);
}

[[nodiscard]] bool send_response(const std::shared_ptr<detail::WebTunerState>& state,
                                 int connection, std::string_view status, std::string_view body,
                                 std::string_view extra_headers = {}) {
  if (body.size() > state->options.max_response_bytes) {
    return false;
  }
  std::string response = "HTTP/1.1 ";
  response += status;
  response += "\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: ";
  response += std::to_string(body.size());
  response += "\r\nCache-Control: no-store\r\n\r\n";
  if (!extra_headers.empty()) {
    const std::size_t delimiter = response.size() - 2U;
    response.insert(delimiter, extra_headers);
  }
  response += body;
  if (response.size() <= state->options.max_response_bytes) {
    return send_bytes(state, connection, response);
  }
  return false;
}

[[nodiscard]] constexpr char ascii_lower(char value) noexcept {
  return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A')) : value;
}

[[nodiscard]] bool ascii_iequals(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (ascii_lower(lhs[index]) != ascii_lower(rhs[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string_view trim_http_whitespace(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1U);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1U);
  }
  return value;
}

struct HttpHeaderView {
  std::string_view name;
  std::string_view value;
};

[[nodiscard]] std::optional<HttpHeaderView> split_http_header(std::string_view header) noexcept {
  const std::size_t colon = header.find(':');
  if (colon == std::string_view::npos || colon == 0U) {
    return std::nullopt;
  }
  return HttpHeaderView{.name = header.substr(0U, colon),
                        .value = trim_http_whitespace(header.substr(colon + 1U))};
}

[[nodiscard]] bool http_header_has_token(std::string_view value,
                                         std::string_view expected) noexcept {
  std::size_t begin = 0U;
  while (begin <= value.size()) {
    const std::size_t end = value.find(',', begin);
    if (ascii_iequals(trim_http_whitespace(value.substr(begin, end - begin)), expected)) {
      return true;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return false;
}

enum class RequestReadStatus : std::uint8_t {
  kOk,
  kClosed,
  kStopped,
  kMalformed,
  kTooLarge,
};

struct RequestReadResult {
  RequestReadStatus status{RequestReadStatus::kClosed};
  std::string request;
};

struct RequestFrameInspection {
  RequestReadStatus status{RequestReadStatus::kOk};
  std::optional<std::size_t> expected_size;
};

[[nodiscard]] RequestFrameInspection inspect_request_frame(std::string_view request,
                                                           std::size_t max_request_bytes) {
  const std::size_t delimiter = request.find("\r\n\r\n");
  if (delimiter == std::string_view::npos) {
    return {.status = request.size() == max_request_bytes ? RequestReadStatus::kTooLarge
                                                          : RequestReadStatus::kOk};
  }

  std::optional<std::size_t> content_length;
  std::size_t line = request.find("\r\n") + 2U;
  while (line < delimiter) {
    const std::size_t end = request.find("\r\n", line);
    if (end == std::string_view::npos || end > delimiter || end == line) {
      return {.status = RequestReadStatus::kMalformed};
    }
    const auto parsed_header = split_http_header(request.substr(line, end - line));
    if (!parsed_header) {
      return {.status = RequestReadStatus::kMalformed};
    }
    if (ascii_iequals(parsed_header->name, "Content-Length")) {
      if (content_length) {
        return {.status = RequestReadStatus::kMalformed};
      }
      std::size_t parsed{};
      const std::string_view text = parsed_header->value;
      const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
      if (text.empty() || result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        return {.status = RequestReadStatus::kMalformed};
      }
      content_length = parsed;
    }
    if (ascii_iequals(parsed_header->name, "Transfer-Encoding")) {
      return {.status = RequestReadStatus::kMalformed};
    }
    line = end + 2U;
  }

  const std::size_t header_size = delimiter + 4U;
  const std::size_t body_size = content_length.value_or(0U);
  if (header_size > max_request_bytes || body_size > max_request_bytes - header_size) {
    return {.status = RequestReadStatus::kTooLarge};
  }
  return {.status = RequestReadStatus::kOk, .expected_size = header_size + body_size};
}

enum class RequestChunkStatus : std::uint8_t {
  kData,
  kRetry,
  kClosed,
  kMalformed,
  kTooLarge,
};

[[nodiscard]] RequestChunkStatus receive_request_chunk(int connection,
                                                       std::size_t max_request_bytes,
                                                       std::string& request) {
  pollfd descriptor{.fd = connection, .events = POLLIN, .revents = 0};
  const int ready = ::poll(&descriptor, 1, 100);
  if (ready < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
    return RequestChunkStatus::kClosed;
  }
  if (ready == 0) {
    return RequestChunkStatus::kRetry;
  }
  std::array<char, 4'096> buffer{};
  const auto received = ::recv(connection, buffer.data(), buffer.size(), 0);
  if (received <= 0) {
    return RequestChunkStatus::kClosed;
  }
  const std::size_t count = static_cast<std::size_t>(received);
  if (request.size() > max_request_bytes || count > max_request_bytes - request.size()) {
    return RequestChunkStatus::kTooLarge;
  }
  request.append(buffer.data(), count);
  return request.find('\0') == std::string::npos ? RequestChunkStatus::kData
                                                 : RequestChunkStatus::kMalformed;
}

[[nodiscard]] RequestReadStatus request_status(RequestChunkStatus status) noexcept {
  switch (status) {
    case RequestChunkStatus::kData:
    case RequestChunkStatus::kRetry:
      return RequestReadStatus::kOk;
    case RequestChunkStatus::kClosed:
      return RequestReadStatus::kClosed;
    case RequestChunkStatus::kMalformed:
      return RequestReadStatus::kMalformed;
    case RequestChunkStatus::kTooLarge:
      return RequestReadStatus::kTooLarge;
  }
  return RequestReadStatus::kClosed;
}

enum class RequestSizeStatus : std::uint8_t { kIncomplete, kComplete, kMalformed };

[[nodiscard]] RequestSizeStatus inspect_expected_request_size(
    std::size_t request_size, const std::optional<std::size_t>& expected_size) noexcept {
  if (!expected_size.has_value()) {
    return RequestSizeStatus::kIncomplete;
  }
  if (request_size == expected_size.value()) {
    return RequestSizeStatus::kComplete;
  }
  if (request_size > expected_size.value()) {
    return RequestSizeStatus::kMalformed;
  }
  return RequestSizeStatus::kIncomplete;
}

[[nodiscard]] RequestReadResult read_request(const std::shared_ptr<detail::WebTunerState>& state,
                                             int connection) {
  std::string request;
  request.reserve(std::min<std::uint32_t>(state->options.max_request_bytes, 4'096U));
  std::optional<std::size_t> expected_size;
  while (!state->stop_requested.load(std::memory_order_acquire)) {
    if (inspect_expected_request_size(request.size(), expected_size) ==
        RequestSizeStatus::kComplete) {
      return {RequestReadStatus::kOk, std::move(request)};
    }
    const auto chunk = receive_request_chunk(connection, state->options.max_request_bytes, request);
    if (chunk == RequestChunkStatus::kRetry) {
      continue;
    }
    if (chunk != RequestChunkStatus::kData) {
      return {request_status(chunk), {}};
    }
    if (expected_size.has_value()) {
      if (inspect_expected_request_size(request.size(), expected_size) ==
          RequestSizeStatus::kMalformed) {
        return {RequestReadStatus::kMalformed, {}};
      }
      continue;
    }
    const auto inspection = inspect_request_frame(request, state->options.max_request_bytes);
    if (inspection.status != RequestReadStatus::kOk) {
      return {inspection.status, {}};
    }
    if (!inspection.expected_size) {
      continue;
    }
    expected_size = inspection.expected_size;
    const auto size_status = inspect_expected_request_size(request.size(), expected_size);
    if (size_status == RequestSizeStatus::kComplete) {
      return {RequestReadStatus::kOk, std::move(request)};
    }
    if (size_status == RequestSizeStatus::kMalformed) {
      return {RequestReadStatus::kMalformed, {}};
    }
  }
  return {RequestReadStatus::kStopped, {}};
}

[[nodiscard]] std::optional<std::string_view> request_header(std::string_view request,
                                                             std::string_view name) {
  std::size_t line = request.find("\r\n") + 2U;
  while (line != std::string_view::npos && line < request.size()) {
    const std::size_t end = request.find("\r\n", line);
    if (end == std::string_view::npos || end == line) {
      break;
    }
    const auto text = request.substr(line, end - line);
    const auto header = split_http_header(text);
    if (header && ascii_iequals(header->name, name)) {
      return header->value;
    }
    line = end + 2U;
  }
  return std::nullopt;
}

[[nodiscard]] std::array<std::uint8_t, 20> sha1(std::string_view input) {
  std::vector<std::uint8_t> message(input.begin(), input.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8U;
  message.push_back(0x80U);
  while ((message.size() % 64U) != 56U) {
    message.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  }
  std::uint32_t h0 = 0x67452301U;
  std::uint32_t h1 = 0xEFCDAB89U;
  std::uint32_t h2 = 0x98BADCFEU;
  std::uint32_t h3 = 0x10325476U;
  std::uint32_t h4 = 0xC3D2E1F0U;
  for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
    std::array<std::uint32_t, 80> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t position = offset + index * 4U;
      words[index] = (static_cast<std::uint32_t>(message[position]) << 24U) |
                     (static_cast<std::uint32_t>(message[position + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(message[position + 2U]) << 8U) |
                     static_cast<std::uint32_t>(message[position + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      words[index] = std::rotl(
          words[index - 3U] ^ words[index - 8U] ^ words[index - 14U] ^ words[index - 16U], 1);
    }
    std::uint32_t a = h0;
    std::uint32_t b = h1;
    std::uint32_t c = h2;
    std::uint32_t d = h3;
    std::uint32_t e = h4;
    for (std::size_t index = 0; index < words.size(); ++index) {
      std::uint32_t function{};
      std::uint32_t constant{};
      if (index < 20U) {
        function = (b & c) | ((~b) & d);
        constant = 0x5A827999U;
      } else if (index < 40U) {
        function = b ^ c ^ d;
        constant = 0x6ED9EBA1U;
      } else if (index < 60U) {
        function = (b & c) | (b & d) | (c & d);
        constant = 0x8F1BBCDCU;
      } else {
        function = b ^ c ^ d;
        constant = 0xCA62C1D6U;
      }
      const std::uint32_t temporary = std::rotl(a, 5) + function + e + constant + words[index];
      e = d;
      d = c;
      c = std::rotl(b, 30);
      b = a;
      a = temporary;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }
  const std::array words{h0, h1, h2, h3, h4};
  std::array<std::uint8_t, 20> digest{};
  for (std::size_t index = 0; index < words.size(); ++index) {
    digest[index * 4U] = static_cast<std::uint8_t>(words[index] >> 24U);
    digest[index * 4U + 1U] = static_cast<std::uint8_t>(words[index] >> 16U);
    digest[index * 4U + 2U] = static_cast<std::uint8_t>(words[index] >> 8U);
    digest[index * 4U + 3U] = static_cast<std::uint8_t>(words[index]);
  }
  return digest;
}

[[nodiscard]] std::string base64(std::span<const std::uint8_t> input) {
  constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((input.size() + 2U) / 3U) * 4U);
  for (std::size_t index = 0; index < input.size(); index += 3U) {
    const std::uint32_t first = input[index];
    const std::uint32_t second = index + 1U < input.size() ? input[index + 1U] : 0U;
    const std::uint32_t third = index + 2U < input.size() ? input[index + 2U] : 0U;
    const std::uint32_t value = (first << 16U) | (second << 8U) | third;
    output.push_back(kAlphabet[(value >> 18U) & 0x3FU]);
    output.push_back(kAlphabet[(value >> 12U) & 0x3FU]);
    output.push_back(index + 1U < input.size() ? kAlphabet[(value >> 6U) & 0x3FU] : '=');
    output.push_back(index + 2U < input.size() ? kAlphabet[value & 0x3FU] : '=');
  }
  return output;
}

[[nodiscard]] std::string websocket_accept(std::string_view key) {
  std::string material{key};
  material += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  const auto digest = sha1(material);
  return base64(digest);
}

[[nodiscard]] bool websocket_key_valid(std::string_view key) noexcept {
  if (key.size() != 24U || !key.ends_with("==")) {
    return false;
  }
  const auto base64_value = [](char value) -> int {
    if (value >= 'A' && value <= 'Z') {
      return value - 'A';
    }
    if (value >= 'a' && value <= 'z') {
      return value - 'a' + 26;
    }
    if (value >= '0' && value <= '9') {
      return value - '0' + 52;
    }
    if (value == '+') {
      return 62;
    }
    return value == '/' ? 63 : -1;
  };
  const bool alphabet_valid = std::ranges::all_of(key.substr(0U, 22U), [](char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
           (value >= '0' && value <= '9') || value == '+' || value == '/';
  });
  // A 16-byte nonce leaves only two payload bits in the final Base64 character. Reject aliases
  // whose nonzero pad bits decode to the same bytes but are not the RFC 4648 canonical spelling.
  return alphabet_valid && (base64_value(key[21U]) & 0x0F) == 0;
}

[[nodiscard]] std::string websocket_text_frame(std::string_view payload) {
  std::string frame;
  frame.push_back(static_cast<char>(0x81U));
  if (payload.size() <= 125U) {
    frame.push_back(static_cast<char>(payload.size()));
  } else if (payload.size() <= std::numeric_limits<std::uint16_t>::max()) {
    frame.push_back(static_cast<char>(126U));
    frame.push_back(static_cast<char>((payload.size() >> 8U) & 0xFFU));
    frame.push_back(static_cast<char>(payload.size() & 0xFFU));
  } else {
    frame.push_back(static_cast<char>(127U));
    const std::uint64_t size = payload.size();
    for (int shift = 56; shift >= 0; shift -= 8) {
      frame.push_back(static_cast<char>((size >> shift) & 0xFFU));
    }
  }
  frame.append(payload);
  return frame;
}

[[nodiscard]] std::string expected_host(const WebTunerEndpoint& endpoint) {
  return endpoint.origin.substr(std::string_view{"http://"}.size());
}

[[nodiscard]] bool host_valid(const detail::WebTunerState& state, std::string_view request) {
  const auto host = request_header(request, "Host");
  return host && *host == expected_host(state.endpoint);
}

[[nodiscard]] bool constant_time_equal(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  volatile std::uint8_t difference{};
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    const auto byte_difference = static_cast<std::uint8_t>(static_cast<unsigned char>(lhs[index]) ^
                                                           static_cast<unsigned char>(rhs[index]));
    difference = static_cast<std::uint8_t>(difference | byte_difference);
  }
  return difference == 0U;
}

[[nodiscard]] bool cookie_valid(const detail::WebTunerState& state, std::string_view request) {
  const auto cookie = request_header(request, "Cookie");
  if (!cookie) {
    return false;
  }
  const std::string expected = state.cookie_name + "=" + state.cookie_secret;
  std::size_t begin = 0U;
  while (begin <= cookie->size()) {
    const std::size_t end = cookie->find(';', begin);
    auto item = cookie->substr(begin, end - begin);
    while (!item.empty() && item.front() == ' ') {
      item.remove_prefix(1U);
    }
    while (!item.empty() && item.back() == ' ') {
      item.remove_suffix(1U);
    }
    if (constant_time_equal(item, expected)) {
      return true;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return false;
}

[[nodiscard]] bool state_change_authorized(const detail::WebTunerState& state,
                                           std::string_view request) {
  const auto origin = request_header(request, "Origin");
  return host_valid(state, request) && cookie_valid(state, request) && origin &&
         *origin == state.endpoint.origin;
}

[[nodiscard]] std::string_view request_body(std::string_view request) noexcept {
  const std::size_t delimiter = request.find("\r\n\r\n");
  return delimiter == std::string_view::npos ? std::string_view() : request.substr(delimiter + 4U);
}

struct TunerEditPayload {
  std::string operator_path;
  std::string field_path;
  std::string kind;
  std::string value;
};

struct TunerEditMemberState {
  TunerEditPayload payload;
  bool operator_seen{};
  bool field_seen{};
  bool kind_seen{};
  bool value_seen{};
};

[[nodiscard]] bool assign_tuner_edit_member(TunerEditMemberState& state, std::string_view key,
                                            std::string value) {
  if (key == "operator" && !state.operator_seen) {
    state.operator_seen = true;
    state.payload.operator_path = std::move(value);
    return true;
  }
  if (key == "field" && !state.field_seen) {
    state.field_seen = true;
    state.payload.field_path = std::move(value);
    return true;
  }
  if (key == "kind" && !state.kind_seen) {
    state.kind_seen = true;
    state.payload.kind = std::move(value);
    return true;
  }
  if (key == "value" && !state.value_seen) {
    state.value_seen = true;
    state.payload.value = std::move(value);
    return true;
  }
  return false;
}

struct YyjsonDocumentDeleter {
  void operator()(yyjson_doc* document) const noexcept { yyjson_doc_free(document); }
};

[[nodiscard]] std::optional<TunerEditPayload> parse_tuner_edit(std::string_view body) {
  yyjson_read_err error{};
  const std::unique_ptr<yyjson_doc, YyjsonDocumentDeleter> document{yyjson_read_opts(
      const_cast<char*>(body.data()), body.size(), YYJSON_READ_NOFLAG, nullptr, &error)};
  if (!document) {
    return std::nullopt;
  }
  yyjson_val* root = yyjson_doc_get_root(document.get());
  if (root == nullptr || !yyjson_is_obj(root) || yyjson_obj_size(root) != 4U) {
    return std::nullopt;
  }

  TunerEditMemberState state;
  yyjson_obj_iter iterator = yyjson_obj_iter_with(root);
  while (yyjson_val* key_value = yyjson_obj_iter_next(&iterator)) {
    yyjson_val* member = yyjson_obj_iter_get_val(key_value);
    if (member == nullptr || !yyjson_is_str(member)) {
      return std::nullopt;
    }
    const char* key_data = yyjson_get_str(key_value);
    const char* value_data = yyjson_get_str(member);
    if (key_data == nullptr || value_data == nullptr) {
      return std::nullopt;
    }
    const std::string_view key{key_data, yyjson_get_len(key_value)};
    if (!assign_tuner_edit_member(
            state,
            key,
            std::string{value_data, static_cast<std::size_t>(yyjson_get_len(member))})) {
      return std::nullopt;
    }
  }
  if (!state.operator_seen || !state.field_seen || !state.kind_seen || !state.value_seen) {
    return std::nullopt;
  }
  return std::move(state.payload);
}

template <typename Integer>
[[nodiscard]] std::optional<ParamValue> parse_tuner_integer(std::string_view text) {
  Integer value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size()
             ? std::optional<ParamValue>{ParamValue::from(value)}
             : std::nullopt;
}

[[nodiscard]] std::optional<ParamValue> parse_tuner_duration(std::string_view text) {
  std::int64_t value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size()
             ? std::optional<ParamValue>{ParamValue::from(std::chrono::nanoseconds{value})}
             : std::nullopt;
}

[[nodiscard]] std::optional<ParamValue> parse_tuner_float(std::string_view text, bool narrow) {
  double value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::general);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value)) {
    return std::nullopt;
  }
  if (!narrow) {
    return ParamValue::from(value);
  }
  const float narrowed = static_cast<float>(value);
  return std::isfinite(narrowed) ? std::optional<ParamValue>{ParamValue::from(narrowed)}
                                 : std::nullopt;
}

[[nodiscard]] std::optional<ParamValue> parse_tuner_value(std::string_view kind,
                                                          std::string_view text) {
  if (kind == "bool") {
    return text == "true"    ? std::optional<ParamValue>{ParamValue::from(true)}
           : text == "false" ? std::optional<ParamValue>{ParamValue::from(false)}
                             : std::nullopt;
  }
  if (kind == "duration_ns") {
    return parse_tuner_duration(text);
  }
  if (kind == "i8") {
    return parse_tuner_integer<std::int8_t>(text);
  }
  if (kind == "i16") {
    return parse_tuner_integer<std::int16_t>(text);
  }
  if (kind == "i32") {
    return parse_tuner_integer<std::int32_t>(text);
  }
  if (kind == "i64") {
    return parse_tuner_integer<std::int64_t>(text);
  }
  if (kind == "u8") {
    return parse_tuner_integer<std::uint8_t>(text);
  }
  if (kind == "u16") {
    return parse_tuner_integer<std::uint16_t>(text);
  }
  if (kind == "u32") {
    return parse_tuner_integer<std::uint32_t>(text);
  }
  if (kind == "u64") {
    return parse_tuner_integer<std::uint64_t>(text);
  }
  if (kind == "f32" || kind == "f64") {
    return parse_tuner_float(text, kind == "f32");
  }
  return std::nullopt;
}

[[nodiscard]] std::string edit_state_token(ParamUpdateState state) {
  switch (state) {
    case ParamUpdateState::kSubmitted:
      return "submitted";
    case ParamUpdateState::kRejected:
      return "rejected";
    case ParamUpdateState::kAccepted:
      return "accepted";
    case ParamUpdateState::kQuiescing:
      return "quiescing";
    case ParamUpdateState::kRestarting:
      return "restarting";
    case ParamUpdateState::kAcceptedNoChange:
      return "accepted-no-change";
    case ParamUpdateState::kPublished:
      return "published";
    case ParamUpdateState::kObserved:
      return "observed";
    case ParamUpdateState::kSuperseded:
      return "superseded";
    case ParamUpdateState::kCanceled:
      return "canceled";
    case ParamUpdateState::kRolledBack:
      return "rolled-back";
    case ParamUpdateState::kFailedStopped:
      return "failed-stopped";
  }
  return "rejected";
}

[[nodiscard]] ParamResult<std::string> catalog_body(const ParameterCatalogView& catalog,
                                                    const ParamControlHandle& control,
                                                    std::uint32_t max_response_bytes) {
  std::string result{"["};
  bool first = true;
  for (const auto& field : catalog.fields()) {
    const auto owner = std::ranges::find(catalog.operators(),
                                         field.address.operator_instance,
                                         &OperatorParamRecord::operator_instance);
    if (owner == catalog.operators().end()) {
      return make_unexpected(
          tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_CATALOG_OWNER_MISSING"));
    }
    const bool hidden = field.projection != holoscan::params::ParamProjectionStatus::kIncluded ||
                        field.visibility != ParamVisibility::kPublic ||
                        field.sensitivity == ParamSensitivity::kSensitive;
    const bool read_only = hidden || field.external_access != ParamExternalAccess::kTuning ||
                           field.update_disposition != ParameterFieldUpdateDisposition::kAdmitted ||
                           !field.adapter_eligible;
    const std::string kind = value_kind(field.logical_type, field.type_shape.bit_width);
    if (!first) {
      result.push_back(',');
    }
    first = false;
    result += "{\"operator\":\"" + percent_encode_path(owner->path.segments()) + "\",\"field\":\"" +
              percent_encode_path(field.path.segments()) + "\",\"kind\":\"" + kind +
              "\",\"hidden\":" + (hidden ? "true" : "false") +
              ",\"read_only\":" + (read_only ? "true" : "false") + ",\"effective\":";
    if (hidden) {
      result += "null";
    } else {
      auto current = control.read(field.address, ParamReadView::kPublished);
      if (!current) {
        result += "null";
      } else {
        result += "{\"value\":" + scalar_json(current->value, kind) +
                  ",\"generation\":" + std::to_string(current->generation) +
                  ",\"sequence\":" + std::to_string(current->update_sequence) + ",\"state\":\"" +
                  edit_state_token(current->state) + "\"}";
      }
    }
    result.push_back('}');
    if (result.size() > max_response_bytes) {
      return make_unexpected(
          tuner_error(ParamToolErrorCode::kResourceExhausted, "WEB_TUNER_CATALOG_RESPONSE_LIMIT"));
    }
  }
  result.push_back(']');
  return result;
}

[[nodiscard]] std::string state_body(detail::WebTunerState& state) {
  const std::lock_guard lock{state.mutex};
  const ParamUpdateSequence last_sequence =
      state.edits.empty() ? ParamUpdateSequence{} : state.edits.back().sequence;
  const ParamUpdateState last_state =
      state.edits.empty() ? ParamUpdateState::kRejected : state.edits.back().state;
  std::string result = "{\"controls\":" + state.catalog_json +
                       ",\"accepted_edits\":" + std::to_string(state.accepted_edits) +
                       ",\"published_edits\":" + std::to_string(state.published_edits) +
                       ",\"last_sequence\":" + std::to_string(last_sequence) +
                       ",\"last_state\":\"" + edit_state_token(last_state) + "\",\"last_edit\":";
  if (state.edits.empty()) {
    result += "null";
  } else {
    const auto& edit = state.edits.back();
    result += "{\"request_ordinal\":" + std::to_string(edit.request_ordinal) +
              ",\"edit_ordinal\":" + std::to_string(edit.edit_ordinal) + ",\"operator\":\"" +
              edit.operator_path + "\",\"field\":\"" + edit.field_path +
              "\",\"desired\":" + scalar_json(edit.desired_value, edit.kind) +
              ",\"sequence\":" + std::to_string(edit.sequence) +
              ",\"target_generation\":" + std::to_string(edit.target_generation) +
              ",\"lifecycle_transaction\":" + std::to_string(edit.lifecycle_transaction) +
              ",\"coalesced_into_transaction\":" + std::to_string(edit.coalesced_into_transaction) +
              ",\"state\":\"" + edit_state_token(edit.state) +
              "\",\"effective_proven\":" + (edit.effective_proven ? "true" : "false") +
              ",\"effective\":";
    if (edit.effective_value) {
      result += "{\"value\":" + scalar_json(edit.effective_value->value, edit.kind) +
                ",\"generation\":" + std::to_string(edit.effective_value->generation) +
                ",\"sequence\":" + std::to_string(edit.effective_value->update_sequence) +
                ",\"state\":\"" + edit_state_token(edit.effective_value->state) + "\"}";
    } else {
      result += "null";
    }
    result.push_back('}');
  }
  result.push_back('}');
  return result;
}

[[nodiscard]] std::optional<ParamReadResult> effective_terminal(
    const ParamControlHandle& control, const detail::WebTunerEditRecord& edit,
    ParamUpdateState terminal_state) {
  if ((terminal_state != ParamUpdateState::kPublished &&
       terminal_state != ParamUpdateState::kObserved &&
       terminal_state != ParamUpdateState::kAcceptedNoChange)) {
    return std::nullopt;
  }
  auto published = control.read(edit.address, ParamReadView::kPublished);
  if (!published || published->value != edit.desired_value) {
    return std::nullopt;
  }
  return *published;
}

[[nodiscard]] bool terminal_state(ParamUpdateState state) noexcept {
  return state != ParamUpdateState::kSubmitted && state != ParamUpdateState::kAccepted &&
         state != ParamUpdateState::kQuiescing && state != ParamUpdateState::kRestarting;
}

[[nodiscard]] std::optional<std::uint64_t> reserve_request_ordinal(
    const std::shared_ptr<detail::WebTunerState>& state) {
  const std::lock_guard lock{state->mutex};
  if (state->request_ordinal == std::numeric_limits<std::uint64_t>::max()) {
    return std::nullopt;
  }
  return ++state->request_ordinal;
}

void handle_bootstrap(const std::shared_ptr<detail::WebTunerState>& state, std::string_view request,
                      int connection) {
  if (!host_valid(*state, request)) {
    (void)send_response(state, connection, "403 Forbidden", "{\"error\":\"host\"}");
    return;
  }
  const std::string body = "{\"service\":\"holoscan.experimental.params\",\"scope\":\"" +
                           scope_hex(state->endpoint.scope_id) +
                           "\",\"state\":\"state\",\"edit\":\"edit\",\"events\":\"events\"}";
  if (body.size() > state->options.max_response_bytes) {
    (void)send_response(state, connection, "507 Insufficient Storage", "{\"error\":\"capacity\"}");
    return;
  }
  const std::string cookie = "Set-Cookie: " + state->cookie_name + "=" + state->cookie_secret +
                             "; HttpOnly; SameSite=Strict; Path=" + state->endpoint.bootstrap_path +
                             "\r\n";
  if (!send_response(state, connection, "200 OK", body, cookie)) {
    (void)send_response(state, connection, "507 Insufficient Storage", "{\"error\":\"capacity\"}");
  }
}

void handle_events(const std::shared_ptr<detail::WebTunerState>& state, std::string_view request,
                   int connection) {
  if (!state_change_authorized(*state, request)) {
    (void)send_response(state, connection, "403 Forbidden", "{\"error\":\"unauthorized\"}");
    return;
  }
  const auto upgrade = request_header(request, "Upgrade");
  const auto connection_header = request_header(request, "Connection");
  const auto version = request_header(request, "Sec-WebSocket-Version");
  const auto key = request_header(request, "Sec-WebSocket-Key");
  const bool valid = upgrade && ascii_iequals(*upgrade, "websocket") && connection_header &&
                     http_header_has_token(*connection_header, "upgrade") && version &&
                     *version == "13" && key && websocket_key_valid(*key);
  if (!valid) {
    (void)send_response(
        state, connection, "400 Bad Request", "{\"error\":\"websocket_handshake\"}");
    return;
  }
  const std::string frame = websocket_text_frame(state_body(*state));
  const std::string handshake =
      "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
      "Sec-WebSocket-Accept: " +
      websocket_accept(*key) + "\r\nCache-Control: no-store\r\n\r\n";
  if (handshake.size() + frame.size() > state->options.max_response_bytes) {
    (void)send_response(state, connection, "507 Insufficient Storage", "{\"error\":\"capacity\"}");
    return;
  }
  if (send_bytes(state, connection, handshake)) {
    (void)send_bytes(state, connection, frame);
  }
}

void handle_state(const std::shared_ptr<detail::WebTunerState>& state, std::string_view request,
                  int connection) {
  if (!host_valid(*state, request) || !cookie_valid(*state, request)) {
    (void)send_response(state, connection, "403 Forbidden", "{\"error\":\"unauthorized\"}");
    return;
  }
  if (!send_response(state, connection, "200 OK", state_body(*state))) {
    (void)send_response(state, connection, "507 Insufficient Storage", "{\"error\":\"capacity\"}");
  }
}

struct ResolvedTunerEdit {
  TunerEditPayload payload;
  ParamAddress address;
  ParamValue value;
};

[[nodiscard]] std::optional<ResolvedTunerEdit> resolve_tuner_edit(const ParamControlHandle& control,
                                                                  std::string_view body) {
  auto payload = parse_tuner_edit(body);
  if (!payload) {
    return std::nullopt;
  }
  auto operator_path = OperatorPath::parse(payload->operator_path);
  auto field_path = ParamPath::parse(payload->field_path);
  const auto value = parse_tuner_value(payload->kind, payload->value);
  if (!operator_path || !field_path || !value) {
    return std::nullopt;
  }
  auto address = control.catalog().resolve(*operator_path, *field_path);
  if (!address) {
    return std::nullopt;
  }
  payload->operator_path = percent_encode_path(operator_path->segments());
  payload->field_path = percent_encode_path(field_path->segments());
  return ResolvedTunerEdit{.payload = std::move(*payload), .address = *address, .value = *value};
}

[[nodiscard]] bool reserve_edit_capacity(const std::shared_ptr<detail::WebTunerState>& state) {
  const std::lock_guard lock{state->mutex};
  const auto pending = static_cast<std::uint32_t>(std::ranges::count_if(
      state->edits, [](const auto& edit) { return !terminal_state(edit.state); }));
  const bool edit_ordinal_available =
      state->reserved_edits < std::numeric_limits<std::uint32_t>::max() - state->accepted_edits;
  const bool audit_available =
      state->reserved_edits < state->options.max_audit_records &&
      state->edits.size() < state->options.max_audit_records - state->reserved_edits;
  const bool pending_available = state->reserved_edits < state->options.max_pending_edits &&
                                 pending < state->options.max_pending_edits - state->reserved_edits;
  if (state->finalizing || !edit_ordinal_available || !audit_available || !pending_available) {
    return false;
  }
  ++state->reserved_edits;
  return true;
}

void release_edit_reservation(const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  const std::lock_guard lock{state->mutex};
  if (state->reserved_edits == 0U) {
    std::terminate();
  }
  --state->reserved_edits;
}

[[nodiscard]] std::size_t append_accepted_edit(const std::shared_ptr<detail::WebTunerState>& state,
                                               std::uint64_t request_ordinal,
                                               ResolvedTunerEdit edit,
                                               const ParamBatchUpdateResult& submitted) {
  const std::lock_guard lock{state->mutex};
  const std::size_t edit_index = state->edits.size();
  try {
    state->edits.push_back(detail::WebTunerEditRecord{
        .request_ordinal = request_ordinal,
        .edit_ordinal = state->accepted_edits + 1U,
        .sequence = submitted.update_sequence,
        .target_generation = submitted.target_generation,
        .address = edit.address,
        .operator_path = std::move(edit.payload.operator_path),
        .field_path = std::move(edit.payload.field_path),
        .kind = std::move(edit.payload.kind),
        .desired_value = edit.value,
        .state = ParamUpdateState::kAccepted,
    });
  } catch (...) {
    --state->reserved_edits;
    throw;
  }
  --state->reserved_edits;
  ++state->accepted_edits;
  return edit_index;
}

void record_terminal_edit(const std::shared_ptr<detail::WebTunerState>& state,
                          const ParamControlHandle& control, std::size_t edit_index,
                          const ParamUpdateResult& terminal) {
  detail::WebTunerEditRecord edit;
  {
    const std::lock_guard lock{state->mutex};
    edit = state->edits[edit_index];
  }
  auto effective = effective_terminal(control, edit, terminal.state);
  const std::lock_guard lock{state->mutex};
  auto& stored = state->edits[edit_index];
  stored.state = terminal.state;
  stored.target_generation = terminal.target_generation;
  stored.lifecycle_transaction = terminal.lifecycle_transaction_id;
  stored.coalesced_into_transaction = terminal.coalesced_into_transaction;
  stored.effective_value = effective;
  stored.effective_proven = effective.has_value();
  if (effective && state->published_edits != std::numeric_limits<std::uint32_t>::max()) {
    ++state->published_edits;
  }
}

void handle_edit(const std::shared_ptr<detail::WebTunerState>& state, ParamControlHandle& control,
                 std::string_view request, int connection, std::uint64_t request_ordinal) {
  if (!state_change_authorized(*state, request)) {
    (void)send_response(state, connection, "403 Forbidden", "{\"error\":\"unauthorized\"}");
    return;
  }
  auto edit = resolve_tuner_edit(control, request_body(request));
  if (!edit) {
    (void)send_response(state, connection, "400 Bad Request", "{\"error\":\"invalid_edit\"}");
    return;
  }
  if (!reserve_edit_capacity(state)) {
    (void)send_response(
        state, connection, "429 Too Many Requests", "{\"error\":\"edit_capacity\"}");
    return;
  }
  ParamBatchUpdateResult submitted;
  try {
    submitted = control.submit(ParamPatch{.target = edit->address, .value = edit->value});
  } catch (...) {
    release_edit_reservation(state);
    throw;
  }
  if (!submitted.committed || submitted.update_sequence == 0U) {
    release_edit_reservation(state);
    (void)send_response(state, connection, "422 Unprocessable Content", "{\"state\":\"rejected\"}");
    return;
  }
  const std::size_t edit_index =
      append_accepted_edit(state, request_ordinal, std::move(*edit), submitted);
  auto terminal = control.await(submitted.update_sequence, std::chrono::seconds{5});
  if (!terminal) {
    (void)send_response(state, connection, "202 Accepted", "{\"state\":\"accepted\"}");
    return;
  }
  record_terminal_edit(state, control, edit_index, *terminal);
  const std::string response = "{\"sequence\":" + std::to_string(submitted.update_sequence) +
                               ",\"state\":\"" + edit_state_token(terminal->state) + "\"}";
  (void)send_response(state, connection, "200 OK", response);
}

void handle_stop(const std::shared_ptr<detail::WebTunerState>& state, std::string_view request,
                 int connection) {
  if (!state_change_authorized(*state, request)) {
    (void)send_response(state, connection, "403 Forbidden", "{\"error\":\"unauthorized\"}");
    return;
  }
  (void)send_response(state, connection, "202 Accepted", "{\"stopping\":true}");
  state->stop_requested.store(true, std::memory_order_release);
  state->completed_cv.notify_all();
#if defined(__linux__)
  const int listener = state->listener.load(std::memory_order_acquire);
  if (listener >= 0) {
    ::shutdown(listener, SHUT_RDWR);
  }
  shutdown_active_connections(state);
#endif
}

void service_connection(const std::shared_ptr<detail::WebTunerState>& state,
                        ParamControlHandle& control, int connection) {
  const auto request_ordinal = reserve_request_ordinal(state);
  if (!request_ordinal) {
    (void)send_response(
        state, connection, "429 Too Many Requests", "{\"error\":\"request_ordinal_overflow\"}");
    return;
  }
  auto decoded = read_request(state, connection);
  if (decoded.status == RequestReadStatus::kTooLarge) {
    (void)send_response(
        state, connection, "413 Content Too Large", "{\"error\":\"request_limit\"}");
    return;
  }
  if (decoded.status == RequestReadStatus::kMalformed) {
    (void)send_response(state, connection, "400 Bad Request", "{\"error\":\"malformed_request\"}");
    return;
  }
  if (decoded.status != RequestReadStatus::kOk) {
    return;
  }
  const std::string& request = decoded.request;
  const std::size_t request_line_end = request.find("\r\n");
  if (request_line_end == std::string::npos) {
    (void)send_response(state, connection, "400 Bad Request", "{\"error\":\"malformed_request\"}");
    return;
  }
  const std::string_view line{request.data(), request_line_end};
  const std::string prefix = state->endpoint.bootstrap_path;
  if (line == "GET " + prefix + " HTTP/1.1") {
    handle_bootstrap(state, request, connection);
  } else if (line == "GET " + prefix + "events HTTP/1.1") {
    handle_events(state, request, connection);
  } else if (line == "GET " + prefix + "state HTTP/1.1") {
    handle_state(state, request, connection);
  } else if (line == "POST " + prefix + "edit HTTP/1.1") {
    handle_edit(state, control, request, connection, *request_ordinal);
  } else if (line == "POST " + prefix + "stop HTTP/1.1") {
    handle_stop(state, request, connection);
  } else {
    (void)send_response(state, connection, "404 Not Found", "{\"error\":\"not_found\"}");
  }
}

void connection_main(const std::shared_ptr<detail::WebTunerState>& state,
                     const std::shared_ptr<detail::WebTunerConnection>& connection) noexcept {
  const int descriptor = connection->descriptor;
  try {
    service_connection(state, connection->control, descriptor);
  } catch (const std::bad_alloc&) {
    {
      const std::lock_guard lock{state->mutex};
      if (!state->error) {
        state->error =
            tuner_error(ParamToolErrorCode::kResourceExhausted, "WEB_TUNER_RESOURCE_EXHAUSTED");
      }
    }
    constexpr std::string_view kResponse =
        "HTTP/1.1 507 Insufficient Storage\r\nContent-Type: application/json\r\n"
        "Connection: close\r\nContent-Length: 30\r\nCache-Control: no-store\r\n\r\n"
        "{\"error\":\"resource_exhausted\"}";
    (void)send_bytes(state, descriptor, kResponse);
  } catch (...) {
    {
      const std::lock_guard lock{state->mutex};
      if (!state->error) {
        state->error = tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_INTERNAL_FAILURE");
      }
    }
    constexpr std::string_view kResponse =
        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
        "Connection: close\r\nContent-Length: 20\r\nCache-Control: no-store\r\n\r\n"
        "{\"error\":\"internal\"}";
    (void)send_bytes(state, descriptor, kResponse);
  }
  {
    const std::lock_guard lock{state->mutex};
    close_socket(connection->descriptor);
    connection->descriptor = -1;
    connection->completed = true;
    if (state->active_connections == 0U) {
      std::terminate();
    }
    --state->active_connections;
  }
  state->completed_cv.notify_all();
}

[[nodiscard]] bool start_connection_worker(const std::shared_ptr<detail::WebTunerState>& state,
                                           int descriptor) {
  // ParamControlHandle permits concurrent calls only through distinct wrappers. The listener is
  // the sole worker creator, so copy the base capability before launch and reserve the original
  // wrapper for finalization after every connection worker has joined.
  auto connection =
      std::make_shared<detail::WebTunerConnection>(descriptor, require_optional(state->control));
  const std::lock_guard lock{state->mutex};
  if (state->active_connections >= state->options.max_connections ||
      state->stop_requested.load(std::memory_order_acquire)) {
    return false;
  }
  state->connections.push_back(connection);
  ++state->active_connections;
  try {
    connection->worker = std::thread{connection_main, state, connection};
  } catch (...) {
    const auto found = std::ranges::find(state->connections, connection);
    if (found != state->connections.end()) {
      state->connections.erase(found);
    }
    --state->active_connections;
    throw;
  }
  return true;
}

[[nodiscard]] bool is_adapter_owned_thread(
    const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  const std::thread::id current = std::this_thread::get_id();
  // This write-once identity is published before start() returns and remains stable while the
  // std::thread object may concurrently transition through join().
  if (state->server_thread_id == current) {
    return true;
  }
  const std::lock_guard lock{state->mutex};
  return std::ranges::any_of(state->connections, [&](const auto& connection) {
    return connection->worker.joinable() && connection->worker.get_id() == current;
  });
}

void server_main(const std::shared_ptr<detail::WebTunerState>& state) noexcept {
  const int listener = state->listener.load(std::memory_order_acquire);
  while (!state->stop_requested.load(std::memory_order_acquire)) {
    reap_completed_connections(state);
    // RunSession has no blocking completion callback. The bounded listener poll
    // is also the adapter's completion observer, so a naturally terminal run
    // wakes wait() without requiring an otherwise unnecessary external stop.
    if (!require_optional(state->session).is_running()) {
      state->stop_requested.store(true, std::memory_order_release);
      break;
    }
    pollfd descriptor{.fd = listener, .events = POLLIN, .revents = 0};
    const int ready = ::poll(&descriptor, 1, 100);
    if (ready < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      break;
    }
    if (ready == 0 || (descriptor.revents & POLLIN) == 0) {
      continue;
    }
    const int connection = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (connection < 0) {
      continue;
    }
    reap_completed_connections(state);
    try {
      if (!start_connection_worker(state, connection)) {
        (void)send_response(
            state, connection, "503 Service Unavailable", "{\"error\":\"connection_capacity\"}");
        close_socket(connection);
      }
    } catch (const std::bad_alloc&) {
      {
        const std::lock_guard lock{state->mutex};
        if (!state->error) {
          state->error =
              tuner_error(ParamToolErrorCode::kResourceExhausted, "WEB_TUNER_RESOURCE_EXHAUSTED");
        }
      }
      close_socket(connection);
      state->stop_requested.store(true, std::memory_order_release);
      break;
    } catch (const std::system_error&) {
      {
        const std::lock_guard lock{state->mutex};
        if (!state->error) {
          state->error = tuner_error(ParamToolErrorCode::kResourceExhausted,
                                     "WEB_TUNER_CONNECTION_THREAD_FAILED");
        }
      }
      close_socket(connection);
      state->stop_requested.store(true, std::memory_order_release);
      break;
    }
  }
  state->stop_requested.store(true, std::memory_order_release);
  close_socket(listener);
  state->listener.store(-1, std::memory_order_release);
  shutdown_active_connections(state);
  join_all_connections(state);
  {
    const std::lock_guard lock{state->mutex};
    state->server_completed = true;
  }
  state->completed_cv.notify_all();
}

#endif

[[nodiscard]] ParamResult<WebTunerResult> completed_tuner_result(
    const detail::WebTunerState& state) {
  if (state.error.has_value()) {
    return make_unexpected(state.error.value());
  }
  if (state.result.has_value()) {
    return state.result.value();
  }
  return make_unexpected(tuner_error(ParamToolErrorCode::kClosed, "WEB_TUNER_SESSION_MISSING"));
}

[[nodiscard]] bool stop_adapter(const std::shared_ptr<detail::WebTunerState>& state,
                                std::optional<ParamToolError>& error) {
  state->stop_requested.store(true, std::memory_order_release);
#if defined(__linux__)
  const int listener = state->listener.load(std::memory_order_acquire);
  if (listener >= 0) {
    ::shutdown(listener, SHUT_RDWR);
  }
  shutdown_active_connections(state);
#endif
  std::unique_lock lock{state->mutex};
  const bool completed = state->completed_cv.wait_for(
      lock, state->options.adapter_shutdown_timeout, [&] { return state->server_completed; });
  if (!completed) {
    error = tuner_error(ParamToolErrorCode::kTimeout, "WEB_TUNER_ADAPTER_SHUTDOWN_TIMEOUT");
    require_optional(state->session).request_stop();
  }
  return completed;
}

void join_adapter(const std::shared_ptr<detail::WebTunerState>& state, bool adapter_completed,
                  std::optional<ParamToolError>& error) {
#if defined(__linux__)
  if (!adapter_completed) {
    shutdown_active_connections(state);
  }
#else
  (void)adapter_completed;
#endif
  if (state->server.joinable()) {
    state->server.join();
  }
  const std::lock_guard lock{state->mutex};
  if (state->error.has_value() && !error.has_value()) {
    error = state->error;
  }
}

[[nodiscard]] ParamUpdateSequence close_parameter_ingress(
    const std::shared_ptr<detail::WebTunerState>& state, std::optional<ParamToolError>& error) {
  auto fence = require_optional(state->session).close_parameter_update_ingress();
  if (fence.has_value()) {
    return fence.value().terminal_update_sequence;
  }
  if (fence.error().code != ErrorCode::kClosed && !error.has_value()) {
    error = tuner_error(
        ParamToolErrorCode::kClosed, "WEB_TUNER_INGRESS_CLOSE_FAILED", std::move(fence.error()));
  }
  return {};
}

[[nodiscard]] bool terminalize_one_edit(const ParamControlHandle& control,
                                        detail::WebTunerEditRecord& edit,
                                        std::chrono::steady_clock::time_point deadline,
                                        std::optional<ParamToolError>& error) {
  if (terminal_state(edit.state)) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    if (!error.has_value()) {
      error = tuner_error(ParamToolErrorCode::kTimeout, "WEB_TUNER_EDIT_TERMINALIZATION_TIMEOUT");
    }
    return false;
  }
  auto terminal = control.await(
      edit.sequence, std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now));
  if (!terminal.has_value()) {
    if (!error.has_value()) {
      const ParamToolErrorCode code = terminal.error().code == ErrorCode::kTimeout
                                          ? ParamToolErrorCode::kTimeout
                                          : ParamToolErrorCode::kClosed;
      error =
          tuner_error(code, "WEB_TUNER_EDIT_TERMINALIZATION_FAILED", std::move(terminal.error()));
    }
    return false;
  }
  const auto& result = terminal.value();
  edit.state = result.state;
  edit.target_generation = result.target_generation;
  edit.lifecycle_transaction = result.lifecycle_transaction_id;
  edit.coalesced_into_transaction = result.coalesced_into_transaction;
  edit.effective_value = effective_terminal(control, edit, result.state);
  edit.effective_proven = edit.effective_value.has_value();
  return true;
}

void terminalize_edits(const std::shared_ptr<detail::WebTunerState>& state,
                       std::optional<ParamToolError>& error) {
  const ParamControlHandle& control = require_optional(state->control);
  const auto deadline = std::chrono::steady_clock::now() + state->options.adapter_shutdown_timeout;
  std::uint32_t effective_count{};
  for (auto& edit : state->edits) {
    if (!terminalize_one_edit(control, edit, deadline, error)) {
      break;
    }
    if (edit.effective_proven) {
      ++effective_count;
    }
  }
  const std::lock_guard lock{state->mutex};
  state->published_edits = effective_count;
  if (!state->edits.empty() && !state->edits.back().effective_proven && !error.has_value()) {
    error = tuner_error(ParamToolErrorCode::kClosed, "WEB_TUNER_LAST_EDIT_NOT_EFFECTIVE");
  }
}

void stop_and_join_session(const std::shared_ptr<detail::WebTunerState>& state,
                           std::optional<ParamToolError>& error) {
  RunSession& session = require_optional(state->session);
  session.request_stop();
  try {
    session.wait();
  } catch (const std::system_error&) {
    error = tuner_error(ParamToolErrorCode::kClosed, "WEB_TUNER_SESSION_JOIN_FAILED");
  }
}

[[nodiscard]] std::optional<WebTunerResult> capture_tuner_result(
    const std::shared_ptr<detail::WebTunerState>& state, ParamUpdateSequence terminalized_through,
    std::optional<ParamToolError>& error) {
  if (error.has_value()) {
    return std::nullopt;
  }
  auto snapshot = require_optional(state->session)
                      .capture_params(ParamCaptureOptions{
                          .view = ParamReadView::kPublished,
                          .consistency = ParamCaptureConsistency::kRequireStopped,
                          .after_update_sequence =
                              terminalized_through == 0U
                                  ? std::optional<ParamUpdateSequence>{}
                                  : std::optional<ParamUpdateSequence>{terminalized_through},
                      });
  if (!snapshot.has_value()) {
    error = tuner_error(
        ParamToolErrorCode::kClosed, "WEB_TUNER_CAPTURE_FAILED", std::move(snapshot.error()));
    return std::nullopt;
  }
  auto result_state = std::make_shared<detail::WebTunerResultState>();
  result_state->snapshot = std::move(snapshot.value());
  result_state->terminalized_through = terminalized_through;
  {
    const std::lock_guard lock{state->mutex};
    result_state->accepted_edits = state->accepted_edits;
    result_state->published_edits = state->published_edits;
  }
  return detail::WebTunerResultAccess::make(std::move(result_state));
}

[[nodiscard]] ParamResult<WebTunerResult> finalize(
    const std::shared_ptr<detail::WebTunerState>& state) {
  {
    std::unique_lock lock{state->mutex};
    if (state->completed) {
      return completed_tuner_result(*state);
    }
    if (state->finalizing) {
      state->completed_cv.wait(lock, [&] { return state->completed; });
      return completed_tuner_result(*state);
    }
    state->finalizing = true;
  }

  std::optional<ParamToolError> error;
  const bool adapter_completed = stop_adapter(state, error);
  join_adapter(state, adapter_completed, error);
  const ParamUpdateSequence terminalized_through = close_parameter_ingress(state, error);
  terminalize_edits(state, error);
  stop_and_join_session(state, error);
  std::optional<WebTunerResult> result = capture_tuner_result(state, terminalized_through, error);
  if (!result.has_value() && !error.has_value()) {
    error = tuner_error(ParamToolErrorCode::kClosed, "WEB_TUNER_SESSION_MISSING");
  }
  {
    const std::lock_guard lock{state->mutex};
    state->result = result;
    state->error = error;
    state->completed = true;
  }
  state->completed_cv.notify_all();
  if (error.has_value()) {
    return make_unexpected(std::move(error.value()));
  }
  return std::move(result.value());
}

}  // namespace

const ParamSnapshot& WebTunerResult::stopped_snapshot() const noexcept {
  return state_->snapshot;
}

ParamUpdateSequence WebTunerResult::terminalized_through() const noexcept {
  return state_->terminalized_through;
}

std::uint32_t WebTunerResult::accepted_edits() const noexcept {
  return state_->accepted_edits;
}

std::uint32_t WebTunerResult::published_edits() const noexcept {
  return state_->published_edits;
}

ParamResult<WebTuner> WebTuner::start(const ExecutionPlan& plan, WebTunerOptions options) {
  if (!options_valid(options)) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kInvalidArgument, "WEB_TUNER_OPTIONS_INVALID"));
  }
#if !defined(__linux__)
  (void)plan;
  return make_unexpected(
      tuner_error(ParamToolErrorCode::kNotSupported, "WEB_TUNER_PLATFORM_UNSUPPORTED"));
#else
  try {
    auto socket = bind_loopback(options);
    if (!socket) {
      return make_unexpected(std::move(socket.error()));
    }
    SocketBinding binding = std::move(socket.value());
    SocketOwner socket_owner{binding.descriptor};
    auto state = std::make_shared<detail::WebTunerState>();
    state->options = options;
    // A worker can finish after the listener's reap pass but before the next worker is recorded.
    // Keep one bounded slot for that completed-but-not-yet-reaped ownership record.
    state->connections.reserve(static_cast<std::size_t>(options.max_connections) + 1U);
    state->edits.reserve(options.max_audit_records);
    auto scope = make_scope();
    if (!scope) {
      state->listener.store(-1, std::memory_order_release);
      return make_unexpected(std::move(scope.error()));
    }
    auto secret = make_secret();
    if (!secret) {
      state->listener.store(-1, std::memory_order_release);
      return make_unexpected(std::move(secret.error()));
    }
    state->endpoint.scope_id = scope.value();
    state->endpoint.origin = std::move(binding.origin);
    state->endpoint.bootstrap_path =
        "/__holoscan/tuner/v1/" + scope_hex(state->endpoint.scope_id) + "/";
    state->cookie_name = "holoscan_tuner_" + scope_hex(state->endpoint.scope_id);
    state->cookie_secret = std::move(secret.value());
    state->listener.store(binding.descriptor, std::memory_order_release);
    RunSession& session = state->session.emplace(run_async(plan));
    auto control = session.parameter_control(ParamAccessPolicy{
        .origin = ParamControlOrigin::kExternalAdapter,
        .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait |
                       ParamPermission::kCapture | ParamPermission::kObserve,
        .allowed_scope = ParamScopeFilter::run(),
        .max_event_cursors = 1U,
    });
    if (!control) {
      state->listener.store(-1, std::memory_order_release);
      session.request_stop();
      session.wait();
      return make_unexpected(tuner_error(ParamToolErrorCode::kPermissionDenied,
                                         "WEB_TUNER_CONTROL_UNAVAILABLE",
                                         std::move(control.error())));
    }
    const ParamControlHandle& tuner_control = state->control.emplace(std::move(control.value()));
    auto catalog = catalog_body(tuner_control.catalog(), tuner_control, options.max_response_bytes);
    if (!catalog) {
      state->listener.store(-1, std::memory_order_release);
      session.request_stop();
      session.wait();
      return make_unexpected(std::move(catalog.error()));
    }
    if (catalog.value().size() + 1'024U > options.max_response_bytes) {
      state->listener.store(-1, std::memory_order_release);
      session.request_stop();
      session.wait();
      return make_unexpected(
          tuner_error(ParamToolErrorCode::kResourceExhausted, "WEB_TUNER_STATE_RESPONSE_LIMIT"));
    }
    state->catalog_json = std::move(catalog.value());
    state->server = std::thread{server_main, state};
    state->server_thread_id = state->server.get_id();
    socket_owner.release();
    return WebTuner{std::move(state)};
  } catch (const CompileError& error) {
    return make_unexpected(tuner_error(
        ParamToolErrorCode::kInvalidArgument, "WEB_TUNER_PLAN_REJECTED", make_error(error)));
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kResourceExhausted, "WEB_TUNER_ALLOC_FAILED"));
  } catch (const std::system_error&) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kProtocol, "WEB_TUNER_THREAD_START_FAILED"));
  }
#endif
}

WebTuner::WebTuner(WebTuner&& other) noexcept : state_(std::move(other.state_)) {}

WebTuner::~WebTuner() noexcept {
  if (state_) {
    request_stop();
    try {
      (void)wait();
    } catch (...) {
      std::terminate();
    }
  }
}

WebTunerEndpoint WebTuner::endpoint() const {
  return state_ ? state_->endpoint : WebTunerEndpoint{};
}

void WebTuner::request_stop() noexcept {
  if (!state_) {
    return;
  }
  state_->stop_requested.store(true, std::memory_order_release);
  state_->completed_cv.notify_all();
#if defined(__linux__)
  const int listener = state_->listener.load(std::memory_order_acquire);
  if (listener >= 0) {
    ::shutdown(listener, SHUT_RDWR);
  }
  shutdown_active_connections(state_);
#endif
}

ParamResult<WebTunerResult> WebTuner::wait() {
  if (!state_) {
    return make_unexpected(tuner_error(ParamToolErrorCode::kClosed, "WEB_TUNER_MOVED_FROM"));
  }
#if defined(__linux__)
  if (is_adapter_owned_thread(state_) || holoscan::detail::runtime_callback_context_restricted()) {
    HOLOSCAN_LOG_ERROR("WEB_TUNER_SELF_JOIN_FORBIDDEN: wait attempted from owned work");
    std::terminate();
  }
#endif
  {
    std::unique_lock lock{state_->mutex};
    state_->completed_cv.wait(lock, [&] {
      return state_->stop_requested.load(std::memory_order_acquire) || state_->completed;
    });
  }
  return finalize(state_);
}

ParamResult<WebTunerResult> run_with_tuner(const ExecutionPlan& plan, WebTunerOptions options) {
  if (options.endpoint_announcement == WebTunerEndpointAnnouncement::kSilent &&
      options.port == 0U) {
    return make_unexpected(
        tuner_error(ParamToolErrorCode::kInvalidArgument, "WEB_TUNER_OPTIONS_INVALID"));
  }
  auto tuner = WebTuner::start(plan, options);
  if (!tuner) {
    return make_unexpected(std::move(tuner.error()));
  }
  if (options.endpoint_announcement == WebTunerEndpointAnnouncement::kLogOrigin) {
    const WebTunerEndpoint endpoint = tuner->endpoint();
    HOLOSCAN_LOG_INFO(
        "Experimental Parameter tuner listening at {}{}", endpoint.origin, endpoint.bootstrap_path);
  }
  return tuner->wait();
}

}  // namespace holoscan::experimental::params
