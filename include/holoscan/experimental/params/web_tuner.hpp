/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/holoscan_experimental_params_web_tuner_export.h>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/params/error.hpp>

namespace holoscan::experimental::params {

namespace detail {
struct WebTunerResultAccess;
struct WebTunerResultState;
struct WebTunerState;
}  // namespace detail

/// @brief Select how the blocking tuner facade announces its local endpoint.
enum class WebTunerEndpointAnnouncement : std::uint8_t {
  kLogOrigin = 0,  ///< Emit one bounded INFO record containing only the non-secret origin.
  kSilent = 1,     ///< Emit no endpoint record.
};

/// @brief Select the sole loopback address family bound by one tuner run.
enum class WebTunerAddressFamily : std::uint8_t {
  kIpv4 = 0,  ///< Bind exactly IPv4 address 127.0.0.1.
  kIpv6 = 1,  ///< Bind exactly IPv6-only address ::1.
};

/// @brief Identify one tuner run's public cookie and route scope.
///
/// The bytes are public routing identity, not authorization material. A live
/// tuner mints a fresh value from operating-system entropy and renders it as
/// exactly 32 lowercase hexadecimal characters in its route and cookie name.
struct TunerScopeId {
  /// Public, non-secret per-run identity bytes.
  std::array<std::uint8_t, 16> bytes{};
  /// @brief Compare two public scope identities byte-for-byte.
  friend constexpr bool operator==(TunerScopeId, TunerScopeId) noexcept = default;
};

/// @brief Configure one finite experimental tuner run.
///
/// Except for the port field, every numeric value must be positive. Startup checks
/// the complete immutable option set against package hard maxima before
/// binding a socket or starting the owned run session. The options select
/// tooling resources only; they do not modify the execution plan or a
/// Runtime profile.
struct WebTunerOptions {
  /// Exact TCP port, or zero to request an ephemeral port.
  std::uint16_t port{};
  /// Sole loopback family to bind; startup never falls back to the other family.
  WebTunerAddressFamily address_family{WebTunerAddressFamily::kIpv4};
  /// Maximum authenticated logical clients.
  ///
  /// baseline mints one per-run credential representing exactly one logical client,
  /// so the only supported value is one. That client may use up to
  /// \ref max_connections simultaneous HTTP or WebSocket connections.
  std::uint32_t max_clients{1};
  /// Maximum simultaneous HTTP and WebSocket connections.
  std::uint32_t max_connections{4};
  /// Maximum complete requests awaiting service.
  std::uint32_t max_pending_requests{64};
  /// Maximum complete responses awaiting transmission.
  std::uint32_t max_pending_responses{64};
  /// Maximum accepted edits awaiting terminalization.
  std::uint32_t max_pending_edits{64};
  /// Maximum retained tuner event-lineage records.
  std::uint32_t max_event_backlog{256};
  /// Maximum retained tuner diagnostic records.
  std::uint32_t max_log_records{256};
  /// Maximum retained tuner audit records.
  std::uint32_t max_audit_records{256};
  /// Maximum bytes in one complete inbound aggregate.
  std::uint32_t max_request_bytes{1U << 20U};
  /// Maximum bytes in one complete outbound aggregate.
  std::uint32_t max_response_bytes{1U << 20U};
  /// Grace period for tuner-owned network/frontend I/O. This does not bound
  /// the subsequent physical join or the owned RunSession shutdown.
  std::chrono::seconds adapter_shutdown_timeout{5};
  /// Blocking-facade origin announcement policy. The advanced facade always
  /// exposes the non-secret endpoint through WebTuner::endpoint().
  WebTunerEndpointAnnouncement endpoint_announcement{WebTunerEndpointAnnouncement::kLogOrigin};
};

/// @brief Describe the public, token-free endpoint of one live tuner.
struct WebTunerEndpoint {
  /// Public identity used to scope routes and the cookie name.
  TunerScopeId scope_id;
  /// Canonical loopback HTTP origin without a trailing slash.
  std::string origin;
  /// Exact per-run route and cookie path; contains no token.
  std::string bootstrap_path;

  /// @brief Return whether this is the inert moved-from endpoint value.
  [[nodiscard]] bool empty() const noexcept { return origin.empty() && bootstrap_path.empty(); }
};

/// @brief Retain one immutable successful terminal tuner result.
///
/// Copies share immutable result state. The reference returned by
/// stopped_snapshot() remains valid while any result copy retaining that
/// state exists. Accessors do not recapture or mutate Runtime state.
class HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT WebTunerResult {
 public:
  /// @brief Copy one immutable shared result view.
  WebTunerResult(const WebTunerResult&) noexcept = default;
  /// @brief Move one immutable shared result view.
  WebTunerResult(WebTunerResult&&) noexcept = default;
  /// @brief Replace this view with a copy of another immutable result.
  WebTunerResult& operator=(const WebTunerResult&) noexcept = default;
  /// @brief Replace this view by moving another immutable result.
  WebTunerResult& operator=(WebTunerResult&&) noexcept = default;
  /// @brief Return the coherent stopped published-state snapshot.
  /// @return Reference retained by the shared immutable result state.
  /// @note Safe for concurrent reads while at least one result view remains alive.
  [[nodiscard]] const ParamSnapshot& stopped_snapshot() const noexcept;

  /// @brief Return the closed parameter-ingress terminal frontier.
  /// @note Safe for concurrent reads.
  [[nodiscard]] ParamUpdateSequence terminalized_through() const noexcept;

  /// @brief Return the number of tuner edits accepted by Core.
  /// @note Safe for concurrent reads.
  [[nodiscard]] std::uint32_t accepted_edits() const noexcept;

  /// @brief Return the accepted edits proven effective in published state.
  ///
  /// Core accepted-no-change edits contribute only after an authoritative
  /// published-view semantic-equality proof; they remain accepted-no-change in
  /// Core history. This accessor is safe for concurrent reads.
  [[nodiscard]] std::uint32_t published_edits() const noexcept;

 private:
  explicit WebTunerResult(std::shared_ptr<const detail::WebTunerResultState> state)
      : state_(std::move(state)) {}
  std::shared_ptr<const detail::WebTunerResultState> state_;
  friend struct detail::WebTunerResultAccess;
  friend class WebTuner;
};

/// @brief Own one advanced local tuner, its RunSession, and all helper work.
///
/// The owner is move-only. Destruction performs stop, drain, and join and can
/// therefore block indefinitely when user code ignores cooperative
/// cancellation. No tuner-owned thread is detached. endpoint(), request_stop(),
/// and wait() support concurrent external callers; moving or destroying the
/// wrapper requires exclusive ownership.
class HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT WebTuner {
 public:
  /// @brief Start one tuner after complete option, package, and plan checks.
  /// @param plan Accepted immutable plan permitting bounded external Parameter control.
  /// @param options Complete immutable tuner configuration.
  /// @return A live owner, or a typed error after partial-start cleanup.
  /// @note On Linux the socket binds only the selected loopback family. A
  /// present target on an unsupported platform returns a typed not-supported error.
  /// @note This experimental tuner is built only when the project is configured with
  /// `-DHOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER=ON` (the default). This header and its
  /// target are absent otherwise.
  [[nodiscard]] static holoscan::params::ParamResult<WebTuner> start(const ExecutionPlan& plan,
                                                                     WebTunerOptions options = {});
  /// @brief Move the complete live ownership state from another tuner.
  /// @param other Source made inert after the move.
  WebTuner(WebTuner&&) noexcept;
  /// @brief Prohibit ownership replacement after construction.
  WebTuner& operator=(WebTuner&&) = delete;
  /// @brief Prohibit copying a unique tuner owner.
  WebTuner(const WebTuner&) = delete;
  /// @brief Prohibit copying a unique tuner owner.
  WebTuner& operator=(const WebTuner&) = delete;
  /// @brief Request stop, drain, and join all owned work.
  /// @warning Destruction can block indefinitely if user code ignores
  /// cancellation. Destruction from tuner- or RunSession-owned work
  /// unconditionally terminates the process to avoid a self-join.
  ~WebTuner() noexcept;

  /// @brief Return the public endpoint after successful startup.
  /// @return An empty endpoint for a moved-from wrapper.
  /// @note The returned value is an owning copy and contains no authorization token.
  [[nodiscard]] WebTunerEndpoint endpoint() const;

  /// @brief Publish a cooperative, nonblocking, idempotent stop request.
  ///
  /// This operation is safe from tuner and Runtime callback threads. It does
  /// not itself claim accepted edits are effective or capture a result. It is
  /// thread-safe and may race endpoint() or wait().
  void request_stop() noexcept;

  /// @brief Join and return the one cached immutable terminal result.
  /// @return The same successful result or error on every completed live call;
  /// a moved-from wrapper returns a bounded closed error.
  /// @note Concurrent external callers serialize on one completion owner. This
  /// operation may allocate and blocks until adapter and run work are joined.
  /// @warning Calling from tuner- or RunSession-owned work unconditionally
  /// terminates when joining would create a self-join or join cycle.
  [[nodiscard]] holoscan::params::ParamResult<WebTunerResult> wait();

 private:
  explicit WebTuner(std::shared_ptr<detail::WebTunerState> state) : state_(std::move(state)) {}
  std::shared_ptr<detail::WebTunerState> state_;
};

/// @brief Run one local tuner and block until its strict stop predicate holds.
/// @param plan Accepted immutable plan permitting bounded external Parameter control.
/// @param options Complete immutable tuner configuration.
/// @return One immutable stopped result, or a bounded typed tooling error.
/// @note This convenience performs no load, save, archive, or filesystem
/// action. Silent announcement with an ephemeral port is invalid because this
/// blocking form exposes no endpoint object. The call blocks until an external
/// client requests stop or the run otherwise terminates, then joins all owned work.
[[nodiscard]] HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT
    holoscan::params::ParamResult<WebTunerResult>
    run_with_tuner(const ExecutionPlan& plan, WebTunerOptions options = {});

}  // namespace holoscan::experimental::params
