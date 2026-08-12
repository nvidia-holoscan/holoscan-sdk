/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/experimental/params/web_tuner.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <gtest/gtest.h>

#include "params/web_tuner_socket.hpp"
#include "runtime/runtime_test_access.hpp"

#if defined(__linux__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace holoscan::experimental::params {
namespace {

using namespace std::chrono_literals;

// TunerProbeParams — standard two-field tunable parameter schema.
// Exposes `value` (i32) and `interval` (duration_ns) as runtime-mutable externally-tunable
// fields. Used by TunerProbeOperator and FiniteTunerProbeOperator in most suite tests.
struct TunerProbeParams {
  std::int32_t value{7};
  std::chrono::nanoseconds interval{1ms};
};

HOLOSCAN_PARAM_SCHEMA(
    TunerProbeParams, param_schema("nvidia.holoscan.test.TunerProbeParams", 1),
    field<&TunerProbeParams::value>("value").runtime_mutable().external_tuning(),
    field<&TunerProbeParams::interval>("interval").runtime_mutable().external_tuning());

// TunerProbeOperator — clock-driven operator that records each observed `value` tick.
// Supports cooperative blocking via static atomics (block/entered/release) so tests can
// hold the operator inside compute() while concurrent HTTP edit requests race against it.
// Representative tests: PublishesAuthorizedScalarEditWithDesiredEffectiveLineage,
//                       ConcurrentEditsReservePendingCapacityBeforeSubmission
class TunerProbeOperator final : public Operator<TunerProbeParams> {
 public:
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (block.load(std::memory_order_acquire)) {
      entered.store(true, std::memory_order_release);
      while (!release.load(std::memory_order_acquire) && !context.cancellation().stop_requested()) {
        std::this_thread::yield();
      }
    }
    last_value.store(context.params<TunerProbeParams>().value, std::memory_order_release);
    return {};
  }

  static void reset_blocking() noexcept {
    block.store(false, std::memory_order_release);
    entered.store(false, std::memory_order_release);
    release.store(false, std::memory_order_release);
  }

  static inline std::atomic<std::int32_t> last_value{7};
  static inline std::atomic_bool block{};
  static inline std::atomic_bool entered{};
  static inline std::atomic_bool release{};
};

// EscapedPathParams — schema whose field name contains JSON-special characters
// (double-quote and backslash). Used to verify percent-encoding in tuner state paths.
// Representative test: StateUsesCanonicalJsonSafeEditPaths
struct EscapedPathParams {
  std::int32_t value{7};
};

HOLOSCAN_PARAM_SCHEMA(
    EscapedPathParams, param_schema("nvidia.holoscan.test.EscapedPathParams", 1),
    field<&EscapedPathParams::value>(R"(value"\path)").runtime_mutable().external_tuning());

// EscapedPathOperator — minimal clock-driven operator whose operator name and field name
// both contain characters requiring percent-encoding ('"' → %22, '\' → %5C).
// Representative test: StateUsesCanonicalJsonSafeEditPaths
class EscapedPathOperator final : public Operator<EscapedPathParams> {
 public:
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// FiniteTunerProbeOperator — spin-triggered operator that runs exactly once then exits.
// Blocks in compute() until `release` is set, enabling the test to observe the operator
// in-flight before letting it complete and terminate the run naturally.
// Representative test: WaitReturnsWhenRunCompletesNaturally
class FiniteTunerProbeOperator final : public Operator<TunerProbeParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(spin_, "activation").capacity(1); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    entered.store(true, std::memory_order_release);
    while (!release.load(std::memory_order_acquire) && !context.cancellation().stop_requested()) {
      std::this_thread::yield();
    }
    return {};
  }

  static void reset() noexcept {
    entered.store(false, std::memory_order_release);
    release.store(false, std::memory_order_release);
  }

  SpinSource spin_;
  static inline std::atomic_bool entered{};
  static inline std::atomic_bool release{};
};

// Builds a single-operator execution plan used by most WebTuner tests.
// Runtime parameter updates are enabled so the tuner can publish edits into the plan.
[[nodiscard]] ExecutionPlan tuner_probe_plan() {
  Graph graph{"web-tuner-probe"};
  (void)graph.op<TunerProbeOperator>("tunable");
  const ClockKey clock = graph.add_clock<RealtimeClock>("clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

// Builds a spin-triggered, direct-once plan so the operator runs exactly once.
// Used to test that WebTuner::wait() returns when the run completes without a stop call.
[[nodiscard]] ExecutionPlan finite_tuner_probe_plan() {
  Graph graph{"finite-web-tuner-probe"};
  const auto tunable = graph.op<FiniteTunerProbeOperator>("tunable");
  graph.partition("runtime-spin-worker").add(tunable);
  const ClockKey clock = graph.add_clock<RealtimeClock>("clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return holoscan::detail::RuntimeTestPlanAccess::enable_direct_once(compile(graph, options));
}

// Builds a plan whose operator and field names contain characters that must be
// percent-encoded in the tuner's canonical JSON-safe edit paths.
[[nodiscard]] ExecutionPlan escaped_path_probe_plan() {
  Graph graph{"web-tuner-escaped-path"};
  (void)graph.op<EscapedPathOperator>(R"(tunable"\path)");
  const ClockKey clock = graph.add_clock<RealtimeClock>("clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

#if defined(__linux__)
// Opens a loopback TCP socket to the WebTuner endpoint and returns the file descriptor.
// Caller is responsible for closing the descriptor.
[[nodiscard]] int connect_endpoint(const WebTunerEndpoint& endpoint) {
  const auto port_separator = endpoint.origin.rfind(':');
  EXPECT_NE(port_separator, std::string::npos);
  const auto port =
      static_cast<std::uint16_t>(std::stoul(endpoint.origin.substr(port_separator + 1U)));
  const int descriptor = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  EXPECT_GE(descriptor, 0);
  if (descriptor < 0) {
    return -1;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  EXPECT_EQ(::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);
  return descriptor;
}

// Reads all available bytes from `descriptor` until EOF and returns them as a string.
// Used to collect a complete HTTP response after the server closes the connection.
[[nodiscard]] std::string receive_all(int descriptor) {
  std::string response;
  std::array<char, 4096> buffer{};
  for (;;) {
    const auto count = ::recv(descriptor, buffer.data(), buffer.size(), 0);
    if (count <= 0) {
      break;
    }
    response.append(buffer.data(), static_cast<std::size_t>(count));
  }
  return response;
}

// Connects to `endpoint`, sends `wire` bytes verbatim, and returns the full response.
// Suitable for raw or intentionally malformed HTTP round-trips.
[[nodiscard]] std::string send_request(const WebTunerEndpoint& endpoint, std::string_view wire) {
  const int descriptor = connect_endpoint(endpoint);
  if (descriptor < 0) {
    return {};
  }
  EXPECT_EQ(::send(descriptor, wire.data(), wire.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(wire.size()));
  std::string response = receive_all(descriptor);
  ::close(descriptor);
  return response;
}

// Constructs and sends a well-formed HTTP/1.1 request, then returns the raw response.
// Always sets Content-Length; `headers` may add Cookie, Origin, Upgrade, etc.
[[nodiscard]] std::string request(const WebTunerEndpoint& endpoint, std::string_view method,
                                  std::string_view suffix, std::string_view headers = {},
                                  std::string_view body = {}) {
  const std::string host = endpoint.origin.substr(std::string_view{"http://"}.size());
  std::string wire = std::string(method) + " " + endpoint.bootstrap_path + std::string(suffix) +
                     " HTTP/1.1\r\nHost: " + host + "\r\n";
  wire += headers;
  wire += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
  wire += body;
  return send_request(endpoint, wire);
}

// Extracts the cookie token (up to the first ';') from the Set-Cookie header in
// `response`. The bootstrap cookie is required for all authenticated tuner routes.
[[nodiscard]] std::string bootstrap_cookie(const std::string& response) {
  constexpr std::string_view prefix{"Set-Cookie: "};
  const auto begin = response.find(prefix);
  EXPECT_NE(begin, std::string::npos);
  if (begin == std::string::npos) {
    return {};
  }
  const auto value = begin + prefix.size();
  const auto end = response.find(';', value);
  EXPECT_NE(end, std::string::npos);
  return response.substr(value, end - value);
}

// Returns the digit string following `"member":` in `json`, starting search at `from`.
// Handles only non-negative integers; used for sequence numbers and edit counters.
[[nodiscard]] std::string json_number_member(std::string_view json, std::string_view member,
                                             std::size_t from = 0U) {
  const std::string needle = "\"" + std::string(member) + "\":";
  const auto begin = json.find(needle, from);
  EXPECT_NE(begin, std::string_view::npos) << json;
  if (begin == std::string_view::npos) {
    return {};
  }
  const auto value = begin + needle.size();
  const auto end = json.find_first_not_of("0123456789", value);
  return std::string(json.substr(value, end - value));
}

// Returns the string value following `"member":"` in `json`, starting search at `from`.
// Assumes the value does not contain an unescaped double-quote.
[[nodiscard]] std::string json_string_member(std::string_view json, std::string_view member,
                                             std::size_t from = 0U) {
  const std::string needle = "\"" + std::string(member) + "\":\"";
  const auto begin = json.find(needle, from);
  EXPECT_NE(begin, std::string_view::npos) << json;
  if (begin == std::string_view::npos) {
    return {};
  }
  const auto value = begin + needle.size();
  const auto end = json.find('"', value);
  EXPECT_NE(end, std::string_view::npos) << json;
  return end == std::string_view::npos ? std::string{}
                                       : std::string(json.substr(value, end - value));
}
#endif

// REQ: PARAM-030
TEST(WebTuner, RejectsZeroCapacityBeforeStartingRun) {
  const ExecutionPlan empty_plan;
  WebTunerOptions options;
  options.max_clients = 0U;
  auto tuner = WebTuner::start(empty_plan, options);
  ASSERT_FALSE(tuner.has_value());
  EXPECT_EQ(tuner.error().code(), holoscan::params::ParamToolErrorCode::kInvalidArgument);
}

// REQ: PARAM-030
TEST(WebTuner, SilentEphemeralEndpointIsAmbiguousAndRejected) {
  const ExecutionPlan empty_plan;
  WebTunerOptions options;
  options.endpoint_announcement = WebTunerEndpointAnnouncement::kSilent;
  auto result = run_with_tuner(empty_plan, options);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().token(), "WEB_TUNER_OPTIONS_INVALID");
}

// REQ: PARAM-030
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, AdvancedFacadeDiscoversSilentEphemeralEndpoint) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.endpoint_announcement = WebTunerEndpointAnnouncement::kSilent;
  // Key setup: kSilent suppresses stdout announcement; WebTuner::start still binds an
  //            ephemeral port and the advanced facade exposes it via endpoint().
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  // Key assertion: endpoint() is non-empty even though nothing was printed to stdout.
  EXPECT_FALSE(tuner->endpoint().empty());
  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-030
TEST(WebTuner, RejectsPackageHardMaximumViolations) {
  const ExecutionPlan empty_plan;
  WebTunerOptions options;
  options.max_clients = 2U;
  auto multi_client_tuner = WebTuner::start(empty_plan, options);
  ASSERT_FALSE(multi_client_tuner.has_value());
  EXPECT_EQ(multi_client_tuner.error().token(), "WEB_TUNER_OPTIONS_INVALID");

  options = {};
  options.max_connections = 65U;
  auto tuner = WebTuner::start(empty_plan, options);
  ASSERT_FALSE(tuner.has_value());
  EXPECT_EQ(tuner.error().token(), "WEB_TUNER_OPTIONS_INVALID");

  options = {};
  options.adapter_shutdown_timeout = 61s;
  auto timeout_tuner = WebTuner::start(empty_plan, options);
  ASSERT_FALSE(timeout_tuner.has_value());
  EXPECT_EQ(timeout_tuner.error().token(), "WEB_TUNER_OPTIONS_INVALID");
}

// REQ: PARAM-030
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, ActivePlanOwnsLoopbackEndpointUntilExplicitStop) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  EXPECT_TRUE(tuner->endpoint().origin.starts_with("http://127.0.0.1:"));
  EXPECT_TRUE(tuner->endpoint().bootstrap_path.starts_with("/__holoscan/tuner/v1/"));

  tuner->request_stop();
  auto result = tuner->wait();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->stopped_snapshot().operators().size(), 1U);
}

// REQ: PARAM-030
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [FiniteTunerProbeOperator "tunable"] (single-fire spin)
TEST(WebTuner, WaitReturnsWhenRunCompletesNaturally) {
  const ExecutionPlan plan = finite_tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  FiniteTunerProbeOperator::reset();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  // Key setup: wait() runs on a separate thread; we block the operator until the test
  //            confirms it has entered compute(), then release it to let the run end.
  auto result = std::async(std::launch::async, [&tuner] { return tuner->wait(); });
  const auto entered_deadline = std::chrono::steady_clock::now() + 2s;
  while (!FiniteTunerProbeOperator::entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < entered_deadline) {
    std::this_thread::yield();
  }
  const bool entered = FiniteTunerProbeOperator::entered.load(std::memory_order_acquire);
  FiniteTunerProbeOperator::release.store(true, std::memory_order_release);
  ASSERT_TRUE(entered);

  const auto status = result.wait_for(2s);
  if (status != std::future_status::ready) {
    tuner->request_stop();
    result.wait();
  }
  // Key assertion: wait() becomes ready without an explicit stop() call because the
  //                operator's single-fire spin source exhausts after one execution.
  ASSERT_EQ(status, std::future_status::ready);
  const auto completed = result.get();
  ASSERT_TRUE(completed.has_value()) << completed.error().token();
  EXPECT_EQ(completed->stopped_snapshot().operators().size(), 1U);
}

#if defined(__linux__)
// REQ: PARAM-031
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, AuthenticatesStateAndMutationRoutesWithScopedCookie) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();

  // Key setup: the bootstrap GET issues a scoped HttpOnly SameSite=Strict cookie tied to
  //            the bootstrap path; Domain= must be absent to stay loopback-only.
  const auto bootstrap = request(endpoint, "GET", "");
  EXPECT_TRUE(bootstrap.starts_with("HTTP/1.1 200 OK"));
  EXPECT_NE(bootstrap.find("HttpOnly; SameSite=Strict; Path=" + endpoint.bootstrap_path),
            std::string::npos);
  EXPECT_EQ(bootstrap.find("Domain="), std::string::npos);
  const std::string cookie = bootstrap_cookie(bootstrap);
  ASSERT_FALSE(cookie.empty());

  const auto unauthorized_edit = request(endpoint, "POST", "edit", {}, "{}");
  EXPECT_TRUE(unauthorized_edit.starts_with("HTTP/1.1 403 Forbidden"));

  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";
  const auto invalid_edit = request(endpoint, "POST", "edit", authorization, "{}");
  EXPECT_TRUE(invalid_edit.starts_with("HTTP/1.1 400 Bad Request"));

  const auto state = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_TRUE(state.starts_with("HTTP/1.1 200 OK"));
  EXPECT_NE(state.find("\"accepted_edits\":0"), std::string::npos);
  EXPECT_NE(state.find("\"published_edits\":0"), std::string::npos);

  // Key assertion: a single-character mutation anywhere in the cookie must yield 403,
  //                confirming the token is cryptographically verified end-to-end.
  const std::size_t secret = cookie.find('=') + 1U;
  ASSERT_GT(secret, 0U);
  ASSERT_LT(secret, cookie.size());
  std::string wrong_first = cookie;
  wrong_first[secret] = wrong_first[secret] == '0' ? '1' : '0';
  std::string wrong_last = cookie;
  wrong_last.back() = wrong_last.back() == '0' ? '1' : '0';
  EXPECT_TRUE(request(endpoint, "GET", "state", "Cookie: " + wrong_first + "\r\n")
                  .starts_with("HTTP/1.1 403 Forbidden"));
  EXPECT_TRUE(request(endpoint, "GET", "state", "Cookie: " + wrong_last + "\r\n")
                  .starts_with("HTTP/1.1 403 Forbidden"));

  const auto stopped = request(endpoint, "POST", "stop", authorization, "{}");
  EXPECT_TRUE(stopped.starts_with("HTTP/1.1 202 Accepted"));
  auto result = tuner->wait();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 0U);
  EXPECT_EQ(result->published_edits(), 0U);
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [EscapedPathOperator "tunable\"\\path"]
TEST(WebTuner, StateUsesCanonicalJsonSafeEditPaths) {
  const ExecutionPlan plan = escaped_path_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";
  // Key setup: the edit body uses the raw (unescaped) operator and field names containing
  //            '"' and '\'; the state response must render them percent-encoded.
  constexpr std::string_view body =
      R"json({"operator":"tunable\"\\path","field":"value\"\\path","kind":"i32","value":"41"})json";

  const std::string edited = request(endpoint, "POST", "edit", authorization, body);
  EXPECT_TRUE(edited.starts_with("HTTP/1.1 200 OK") || edited.starts_with("HTTP/1.1 202 Accepted"))
      << edited;
  // Key assertion: state JSON shows percent-encoded paths (%22 for '"', %5C for '\').
  const std::string state = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_NE(state.find("\"operator\":\"tunable%22%5Cpath\""), std::string::npos) << state;
  EXPECT_NE(state.find("\"field\":\"value%22%5Cpath\""), std::string::npos) << state;

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, RejectsMalformedDuplicateAndExtendedEditDocuments) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";
  // Key setup: each body violates a distinct structural rule — duplicate key, unicode-escaped
  //            duplicate, extra field, trailing bytes, nested value, non-string value,
  //            trailing comma, and a lone surrogate escape sequence.
  const std::vector<std::string> rejected{
      std::string(R"json({"operator":"tunable","operator":"tunable","field":)json") +
          R"json("value","kind":"i32","value":"41"})json",
      std::string(R"json({"operator":"tunable","\u006fperator":"tunable","field":)json") +
          R"json("value","kind":"i32","value":"41"})json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":"41","extra":"x"})json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":"41"} trailing)json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":{"text":"41"}})json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":41})json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":"41",})json",
      R"json({"operator":"tunable","field":"value","kind":"i32","value":"\uD800"})json",
  };
  for (const auto& body : rejected) {
    const std::string response = request(endpoint, "POST", "edit", authorization, body);
    EXPECT_TRUE(response.starts_with("HTTP/1.1 400 Bad Request")) << body << '\n' << response;
  }
  // Key assertion: no edit was accepted despite multiple POST attempts.
  const std::string current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_NE(current.find("\"accepted_edits\":0"), std::string::npos);

  tuner->request_stop();
  const auto result = tuner->wait();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 0U);
}

// REQ: PARAM-031
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, PerformsRealWebSocketUpgradeAndPublishesStateFrame) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());

  // Key setup: a known-bad Sec-WebSocket-Key (wrong padding) must yield 400 with a
  //            descriptive websocket_handshake token before the valid upgrade is tried.
  const std::string invalid_headers =
      "Origin: " + endpoint.origin + "\r\nCookie: " + cookie +
      "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZR==\r\n";
  const std::string invalid_response = request(endpoint, "GET", "events", invalid_headers);
  EXPECT_TRUE(invalid_response.starts_with("HTTP/1.1 400 Bad Request")) << invalid_response;
  EXPECT_NE(invalid_response.find("websocket_handshake"), std::string::npos);

  const std::string headers =
      "Origin: " + endpoint.origin + "\r\nCookie: " + cookie +
      "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
  const std::string response = request(endpoint, "GET", "events", headers);
  // Key assertion: the server computes the RFC 6455 accept hash and pushes an initial
  //                state frame containing the operator name inside the same response.
  EXPECT_TRUE(response.starts_with("HTTP/1.1 101 Switching Protocols"));
  EXPECT_NE(response.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="), std::string::npos);
  EXPECT_NE(response.find("\"operator\":\"tunable\""), std::string::npos);

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, TreatsHttpFieldNamesAndConnectionTokensCaseInsensitively) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string host = endpoint.origin.substr(std::string_view{"http://"}.size());

  // Key setup: headers deliberately use mixed case (hOsT, oRiGiN, cOoKiE, etc.) and
  //            tab-separated OWS around values; transfer-encoding: chunked must be rejected.
  const std::string body = R"({"operator":"tunable","field":"value","kind":"i32","value":"41"})";
  const std::string edit = "POST " + endpoint.bootstrap_path + "edit HTTP/1.1\r\nhOsT:\t" + host +
                           " \r\noRiGiN: " + endpoint.origin + "\r\ncOoKiE: " + cookie +
                           "\r\ncontent-length:\t" + std::to_string(body.size()) +
                           "\r\nconnection: close\r\n\r\n" + body;
  const std::string edit_response = send_request(endpoint, edit);
  EXPECT_TRUE(edit_response.starts_with("HTTP/1.1 200 OK") ||
              edit_response.starts_with("HTTP/1.1 202 Accepted"))
      << edit_response;

  const std::string events =
      "GET " + endpoint.bootstrap_path + "events HTTP/1.1\r\nhost: " + host +
      "\r\norigin: " + endpoint.origin + "\r\ncookie: " + cookie +
      "\r\nupgrade: WebSocket\r\nconnection: keep-alive, upgrade\r\n"
      "sec-websocket-version: 13\r\n"
      "sec-websocket-key: dGhlIHNhbXBsZSBub25jZQ==\r\ncontent-length: 0\r\n\r\n";
  const std::string events_response = send_request(endpoint, events);
  EXPECT_TRUE(events_response.starts_with("HTTP/1.1 101 Switching Protocols")) << events_response;

  const std::string transfer_encoded = "GET " + endpoint.bootstrap_path +
                                       " HTTP/1.1\r\nhost: " + host +
                                       "\r\ntransfer-encoding: chunked\r\n\r\n";
  const std::string rejected = send_request(endpoint, transfer_encoded);
  EXPECT_TRUE(rejected.starts_with("HTTP/1.1 400 Bad Request")) << rejected;

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, AcceptsFragmentedBodiesAndRejectsOversizedAggregates) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.max_request_bytes = 512U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());

  // Key setup: send header and body as two separate ::send() calls to exercise the
  //            server's incremental read path; body is "{}" which is a valid JSON object
  //            but not a valid edit document, so the response must be 400.
  const int descriptor = connect_endpoint(endpoint);
  ASSERT_GE(descriptor, 0);
  const std::string host = endpoint.origin.substr(std::string_view{"http://"}.size());
  const std::string header = "POST " + endpoint.bootstrap_path + "edit HTTP/1.1\r\nHost: " + host +
                             "\r\nOrigin: " + endpoint.origin + "\r\nCookie: " + cookie +
                             "\r\nContent-Length: 2\r\nConnection: close\r\n\r\n";
  ASSERT_EQ(::send(descriptor, header.data(), header.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(header.size()));
  ASSERT_EQ(::send(descriptor, "{", 1, MSG_NOSIGNAL), 1);
  ASSERT_EQ(::send(descriptor, "}", 1, MSG_NOSIGNAL), 1);
  const std::string fragmented = receive_all(descriptor);
  ::close(descriptor);
  EXPECT_TRUE(fragmented.starts_with("HTTP/1.1 400 Bad Request"));

  // Key assertion: a body exceeding max_request_bytes is rejected with 413 Content Too Large
  //                before any JSON parsing occurs.
  const std::string oversized(800U, 'x');
  const std::string rejected = request(endpoint, "POST", "edit", {}, oversized);
  EXPECT_TRUE(rejected.starts_with("HTTP/1.1 413 Content Too Large"));

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, IncompleteClientDoesNotBlockConcurrentRequest) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.max_connections = 2U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  // Key setup: a slow client sends only the request line with no CRLF terminator,
  //            occupying one of the two connection slots without completing.
  const int slow = connect_endpoint(endpoint);
  ASSERT_GE(slow, 0);
  constexpr std::string_view partial{"GET /incomplete HTTP/1.1\r\n"};
  ASSERT_EQ(::send(slow, partial.data(), partial.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(partial.size()));
  std::this_thread::sleep_for(20ms);

  // Key assertion: the second connection slot is still available for a normal request.
  const std::string bootstrap = request(endpoint, "GET", "");
  EXPECT_TRUE(bootstrap.starts_with("HTTP/1.1 200 OK")) << bootstrap;

  ::close(slow);
  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-031
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, RejectsConnectionAboveCapacityAndRecoversAfterWorkerExit) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.max_connections = 1U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  // Key setup: the single allowed slot is occupied by a slow partial-request client.
  const int slow = connect_endpoint(endpoint);
  ASSERT_GE(slow, 0);
  constexpr std::string_view partial{"GET /incomplete HTTP/1.1\r\n"};
  ASSERT_EQ(::send(slow, partial.data(), partial.size(), MSG_NOSIGNAL),
            static_cast<ssize_t>(partial.size()));
  std::this_thread::sleep_for(20ms);

  const std::string rejected = request(endpoint, "GET", "");
  EXPECT_TRUE(rejected.starts_with("HTTP/1.1 503 Service Unavailable")) << rejected;

  // Key assertion: after the slow client is shut down, the slot is reclaimed and a fresh
  //                request succeeds within the recovery deadline.
  ::shutdown(slow, SHUT_RDWR);
  ::close(slow);
  const auto recovery_deadline = std::chrono::steady_clock::now() + 1s;
  std::string recovered;
  do {
    recovered = request(endpoint, "GET", "");
    if (recovered.starts_with("HTTP/1.1 200 OK")) {
      break;
    }
    std::this_thread::sleep_for(5ms);
  } while (std::chrono::steady_clock::now() < recovery_deadline);
  EXPECT_TRUE(recovered.starts_with("HTTP/1.1 200 OK")) << recovered;

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, CompletedWorkersDoNotConsumeActiveConnectionCapacity) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.max_connections = 1U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();

  // An EOF can reach this client just before the listener observes the completed worker record.
  // Capacity is therefore based on live workers, not on records awaiting the next reap pass.
  for (std::uint32_t request_ordinal = 0U; request_ordinal < 128U; ++request_ordinal) {
    const std::string response = request(endpoint, "GET", "");
    ASSERT_TRUE(response.starts_with("HTTP/1.1 200 OK")) << "request " << request_ordinal << '\n'
                                                         << response;
  }

  tuner->request_stop();
  EXPECT_TRUE(tuner->wait().has_value());
}

#if defined(__linux__)

// REQ: PARAM-032
TEST(WebTuner, FatalPeerStateTerminatesBlockedResponseSend) {
  // Key setup: fill the socket send buffer to force the next send() to block, then close
  //            the peer end so the kernel delivers an error to the blocked write.
  std::array<int, 2> sockets{};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()), 0);

  std::array<char, 4'096> fill{};
  for (;;) {
    const auto count = ::send(sockets[0], fill.data(), fill.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
    if (count > 0) {
      continue;
    }
    ASSERT_EQ(count, -1);
    ASSERT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
    break;
  }

  std::atomic_bool stop_requested{};
  auto blocked_send = std::async(std::launch::async, [&] {
    return detail::send_web_tuner_bytes(stop_requested, sockets[0], "response");
  });
  std::this_thread::sleep_for(20ms);
  ::close(sockets[1]);

  // Key assertion: the blocked send returns within 500 ms after the peer socket closes,
  //                confirming that fatal peer state interrupts the write rather than hanging.
  const auto status = blocked_send.wait_for(500ms);
  stop_requested.store(true, std::memory_order_release);
  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_FALSE(blocked_send.get());
  ::close(sockets[0]);
}

#endif

// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, StopInterruptsAndJoinsAllIncompleteClients) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  WebTunerOptions options;
  options.max_connections = 3U;
  options.adapter_shutdown_timeout = 1s;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  // Key setup: three partial-request clients each hold an open socket without completing;
  //            the tuner must interrupt all three when request_stop() is called.
  std::array<int, 3> descriptors{};
  constexpr std::string_view partial{"GET /incomplete HTTP/1.1\r\n"};
  for (int& descriptor : descriptors) {
    descriptor = connect_endpoint(tuner->endpoint());
    ASSERT_GE(descriptor, 0);
    ASSERT_EQ(::send(descriptor, partial.data(), partial.size(), MSG_NOSIGNAL),
              static_cast<ssize_t>(partial.size()));
  }
  std::this_thread::sleep_for(20ms);

  const auto started = std::chrono::steady_clock::now();
  tuner->request_stop();
  const auto result = tuner->wait();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  for (const int descriptor : descriptors) {
    ::close(descriptor);
  }
  ASSERT_TRUE(result.has_value()) << result.error().token();
  // Key assertion: shutdown completes well within the adapter_shutdown_timeout (1 s).
  EXPECT_LT(elapsed, 1s);
}

// REQ: PARAM-030
// REQ: PARAM-031
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, PublishesAuthorizedScalarEditWithDesiredEffectiveLineage) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  TunerProbeOperator::last_value.store(7, std::memory_order_release);
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";

  // Key setup: verify initial state metadata before issuing any edit.
  const std::string initial = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_NE(initial.find("\"operator\":\"tunable\""), std::string::npos);
  EXPECT_NE(initial.find("\"field\":\"value\""), std::string::npos);
  EXPECT_NE(initial.find("\"hidden\":false"), std::string::npos);
  EXPECT_NE(initial.find("\"read_only\":false"), std::string::npos);

  const std::string body =
      "{\"operator\":\"tunable\",\"field\":\"value\",\"kind\":\"i32\",\"value\":\"42\"}";
  const std::string edit = request(endpoint, "POST", "edit", authorization, body);
  EXPECT_TRUE(edit.starts_with("HTTP/1.1 200 OK")) << edit;
  EXPECT_TRUE(edit.find("\"state\":\"published\"") != std::string::npos ||
              edit.find("\"state\":\"observed\"") != std::string::npos)
      << edit;

  // Key assertion: state shows desired value, effective_proven, and the operator's
  //                last_value atomically reflects the new value within the observation window.
  const std::string current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_NE(current.find("\"accepted_edits\":1"), std::string::npos);
  EXPECT_NE(current.find("\"published_edits\":1"), std::string::npos);
  EXPECT_NE(current.find("\"desired\":{\"kind\":\"i32\",\"value\":\"42\"}"), std::string::npos);
  EXPECT_NE(current.find("\"effective_proven\":true"), std::string::npos);

  const auto observation_deadline = std::chrono::steady_clock::now() + 1s;
  while (TunerProbeOperator::last_value.load(std::memory_order_acquire) != 42 &&
         std::chrono::steady_clock::now() < observation_deadline) {
    std::this_thread::sleep_for(1ms);
  }

  const std::string stopped = request(endpoint, "POST", "stop", authorization, "{}");
  EXPECT_TRUE(stopped.starts_with("HTTP/1.1 202 Accepted"));
  auto result = tuner->wait();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 1U);
  EXPECT_EQ(result->published_edits(), 1U);
  EXPECT_EQ(TunerProbeOperator::last_value.load(std::memory_order_acquire), 42);
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, ConcurrentEditsReservePendingCapacityBeforeSubmission) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  TunerProbeOperator::reset_blocking();
  TunerProbeOperator::block.store(true, std::memory_order_release);
  WebTunerOptions options;
  options.max_connections = 3U;
  options.max_pending_edits = 1U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";
  // Key setup: operator is blocked inside compute(); wait until it enters the spin loop so
  //            the pending-edit slot is provably consumed before the second edit arrives.
  const auto entered_deadline = std::chrono::steady_clock::now() + 1s;
  while (!TunerProbeOperator::entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < entered_deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(TunerProbeOperator::entered.load(std::memory_order_acquire));

  const std::string first_body =
      "{\"operator\":\"tunable\",\"field\":\"value\",\"kind\":\"i32\",\"value\":\"42\"}";
  auto first = std::async(std::launch::async, [&] {
    return request(endpoint, "POST", "edit", authorization, first_body);
  });
  const auto accepted_deadline = std::chrono::steady_clock::now() + 1s;
  std::string current;
  do {
    current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
    if (current.find("\"accepted_edits\":1") != std::string::npos) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < accepted_deadline);
  ASSERT_NE(current.find("\"accepted_edits\":1"), std::string::npos) << current;

  // Key assertion: the second edit is rejected with 429 because the pending-edit capacity
  //                (max_pending_edits=1) is already consumed by the first in-flight edit.
  const std::string second_body =
      "{\"operator\":\"tunable\",\"field\":\"value\",\"kind\":\"i32\",\"value\":\"43\"}";
  const std::string second = request(endpoint, "POST", "edit", authorization, second_body);
  EXPECT_TRUE(second.starts_with("HTTP/1.1 429 Too Many Requests")) << second;

  TunerProbeOperator::release.store(true, std::memory_order_release);
  TunerProbeOperator::block.store(false, std::memory_order_release);
  const std::string first_response = first.get();
  EXPECT_TRUE(first_response.starts_with("HTTP/1.1 200 OK")) << first_response;
  tuner->request_stop();
  const auto result = tuner->wait();
  TunerProbeOperator::reset_blocking();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 1U);
}

// REQ: PARAM-031
// REQ: PARAM-032
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, ConcurrentEditsKeepLastSummaryCoherent) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  TunerProbeOperator::reset_blocking();
  TunerProbeOperator::block.store(true, std::memory_order_release);
  WebTunerOptions options;
  options.max_connections = 4U;
  options.max_pending_edits = 2U;
  auto tuner = WebTuner::start(plan, options);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";
  // Key setup: operator is blocked; two edits are submitted concurrently. The second edit
  //            targets the `interval` field at its current value so it terminates immediately
  //            as AcceptedNoChange while the first edit remains pending in compute().
  const auto entered_deadline = std::chrono::steady_clock::now() + 1s;
  while (!TunerProbeOperator::entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < entered_deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(TunerProbeOperator::entered.load(std::memory_order_acquire));

  const auto submit_edit = [&](const std::string& body) {
    return std::async(std::launch::async,
                      [&, body] { return request(endpoint, "POST", "edit", authorization, body); });
  };
  auto first = submit_edit(
      "{\"operator\":\"tunable\",\"field\":\"value\",\"kind\":\"i32\",\"value\":\"42\"}");
  const auto first_accepted_deadline = std::chrono::steady_clock::now() + 1s;
  std::string current;
  do {
    current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
    if (current.find("\"accepted_edits\":1") != std::string::npos) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  } while (std::chrono::steady_clock::now() < first_accepted_deadline);
  ASSERT_NE(current.find("\"accepted_edits\":1"), std::string::npos) << current;

  // The second edit writes interval's existing 1 ms value. It is the latest accepted edit and
  // terminalizes as AcceptedNoChange while the older value edit remains blocked in compute.
  auto second = submit_edit(
      "{\"operator\":\"tunable\",\"field\":\"interval\",\"kind\":\"duration_ns\","
      "\"value\":\"1000000\"}");
  const std::string second_response = second.get();
  ASSERT_TRUE(second_response.starts_with("HTTP/1.1 200 OK")) << second_response;
  ASSERT_NE(second_response.find("\"state\":\"accepted-no-change\""), std::string::npos)
      << second_response;

  // Key assertion: last_edit/last_sequence/last_state in the state document track the
  //                highest-sequence accepted edit, not the most recently published one.
  current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  ASSERT_NE(current.find("\"accepted_edits\":2"), std::string::npos) << current;
  auto last_edit = current.find("\"last_edit\":{");
  ASSERT_NE(last_edit, std::string::npos) << current;
  const std::string latest_sequence = json_number_member(current, "last_sequence");
  EXPECT_EQ(latest_sequence, json_number_member(current, "sequence", last_edit));
  EXPECT_EQ(json_string_member(current, "last_state"), "accepted-no-change");
  EXPECT_EQ(json_string_member(current, "last_state"),
            json_string_member(current, "state", last_edit));

  TunerProbeOperator::release.store(true, std::memory_order_release);
  TunerProbeOperator::block.store(false, std::memory_order_release);
  const std::string first_response = first.get();
  EXPECT_TRUE(first_response.starts_with("HTTP/1.1 200 OK")) << first_response;

  current = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  last_edit = current.find("\"last_edit\":{");
  ASSERT_NE(last_edit, std::string::npos) << current;
  EXPECT_EQ(json_number_member(current, "last_sequence"), latest_sequence);
  EXPECT_EQ(json_number_member(current, "sequence", last_edit), latest_sequence);
  EXPECT_EQ(json_string_member(current, "last_state"), "accepted-no-change");
  EXPECT_EQ(json_string_member(current, "state", last_edit), "accepted-no-change");

  tuner->request_stop();
  const auto result = tuner->wait();
  TunerProbeOperator::reset_blocking();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 2U);
}

// REQ: PARAM-031
// Graph:  [RealtimeClock] ──► [TunerProbeOperator "tunable"]
TEST(WebTuner, PublishesAdvertisedDurationNanosecondsEdit) {
  const ExecutionPlan plan = tuner_probe_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto tuner = WebTuner::start(plan);
  ASSERT_TRUE(tuner.has_value()) << tuner.error().token();
  const auto endpoint = tuner->endpoint();
  const std::string cookie = bootstrap_cookie(request(endpoint, "GET", ""));
  ASSERT_FALSE(cookie.empty());
  const std::string authorization = "Origin: " + endpoint.origin + "\r\nCookie: " + cookie + "\r\n";

  // Key setup: confirm the state advertises the interval field with kind "duration_ns"
  //            before submitting an edit so the kind round-trip is verified end-to-end.
  const std::string initial = request(endpoint, "GET", "state", "Cookie: " + cookie + "\r\n");
  EXPECT_NE(initial.find("\"field\":\"interval\""), std::string::npos);
  EXPECT_NE(initial.find("\"kind\":\"duration_ns\""), std::string::npos);

  const std::string body =
      "{\"operator\":\"tunable\",\"field\":\"interval\",\"kind\":\"duration_ns\","
      "\"value\":\"2000000\"}";
  const std::string edit = request(endpoint, "POST", "edit", authorization, body);
  EXPECT_TRUE(edit.starts_with("HTTP/1.1 200 OK")) << edit;
  EXPECT_TRUE(edit.find("\"state\":\"published\"") != std::string::npos ||
              edit.find("\"state\":\"observed\"") != std::string::npos)
      << edit;

  const std::string stopped = request(endpoint, "POST", "stop", authorization, "{}");
  EXPECT_TRUE(stopped.starts_with("HTTP/1.1 202 Accepted"));
  auto result = tuner->wait();
  ASSERT_TRUE(result.has_value()) << result.error().token();
  EXPECT_EQ(result->accepted_edits(), 1U);
  EXPECT_EQ(result->published_edits(), 1U);
}
#endif

}  // namespace
}  // namespace holoscan::experimental::params
