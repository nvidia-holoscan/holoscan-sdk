/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standalone child process for the EA1 two-process runtime smoke. It loads the
// exact catalog written by the parent Graph process and drives one endpoint
// side through the supported installed Channel surface only. Exit code zero
// means every validation stage passed; any other value identifies the failed
// stage for the parent's diagnostics.
//
// Coordination protocol (all state via env vars + a pair of anonymous pipes):
//   HOLOSCAN_IPR_SMOKE_ROLE          — which run_* function to execute
//   HOLOSCAN_IPR_SMOKE_CATALOG       — path to the parent's serialized catalog JSON
//   HOLOSCAN_IPR_SMOKE_ENDPOINT      — primary endpoint name within that catalog
//   HOLOSCAN_IPR_SMOKE_SECOND_ENDPOINT — secondary endpoint for dual-port roles
//   HOLOSCAN_IPR_SMOKE_RESULT_FD     — write-end fd: child writes kReady then result byte
//   HOLOSCAN_IPR_SMOKE_START_FD      — read-end fd: child blocks here until parent signals

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::uint8_t kReady = 1U;

[[nodiscard]] const char* required_env(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

[[nodiscard]] bool write_code(const int fd, const std::uint8_t code) {
  return ::write(fd, &code, sizeof(code)) == static_cast<ssize_t>(sizeof(code));
}

[[nodiscard]] bool wait_for_start(const int fd) {
  std::uint8_t token = 0U;
  return ::read(fd, &token, sizeof(token)) == static_cast<ssize_t>(sizeof(token));
}

[[nodiscard]] int wait_for_publisher_subscriber(
    holoscan::ChannelPublisher<std::uint32_t>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (true) {
    if (publisher.has_subscribers()) {
      // Refresh once more after observing the peer so delivery covers the subscriber that flipped
      // the advisory snapshot.
      return publisher.update_connections() ? 0 : 12;
    }
    if (!publisher.update_connections()) {
      return 12;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return 13;
    }
    std::this_thread::sleep_for(5ms);
  }
}

[[nodiscard]] int emit_publisher_value(holoscan::ChannelPublisher<std::uint32_t>& publisher,
                                       const std::uint32_t value) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  const std::int64_t capture_ns = 100 + static_cast<std::int64_t>(value);
  while (true) {
    auto emitted = publisher.emit_captured(value, capture_ns);
    if (emitted) {
      return 0;
    }
    const auto code = emitted.error().code;
    const bool retryable =
        code == holoscan::ErrorCode::kNotReady || code == holoscan::ErrorCode::kBackpressured;
    if (!retryable || std::chrono::steady_clock::now() >= deadline) {
      return 100 + static_cast<int>(code);
    }
    if (!publisher.update_connections()) {
      return 12;
    }
    std::this_thread::sleep_for(2ms);
  }
}

// publisher role: creates one ChannelPublisher, waits for at least one subscriber,
// then emits uint32_t values 41–43 with monotone capture timestamps.
[[nodiscard]] int run_publisher(holoscan::ChannelRuntime& runtime,
                                const holoscan::ChannelEndpointCatalog& catalog,
                                const std::string& endpoint_name, const int result_fd) {
  auto publisher = runtime.create_publisher<std::uint32_t>(catalog, endpoint_name);
  if (!publisher) {
    return 10;
  }
  if (!write_code(result_fd, kReady)) {
    return 11;
  }
  const int connection_result = wait_for_publisher_subscriber(*publisher);
  if (connection_result != 0) {
    return connection_result;
  }
  for (std::uint32_t value = 41U; value <= 43U; ++value) {
    const int emit_result = emit_publisher_value(*publisher, value);
    if (emit_result != 0) {
      return emit_result;
    }
  }
  return 0;
}

// subscriber role: creates one ChannelSubscriber, drains at least two sidecar samples,
// and verifies each carries the sentinel value 0xdecafbad with a strictly increasing
// metadata sequence number.
[[nodiscard]] int run_subscriber(holoscan::ChannelRuntime& runtime,
                                 const holoscan::ChannelEndpointCatalog& catalog,
                                 const std::string& endpoint_name, const int result_fd) {
  auto subscriber = runtime.create_subscriber<std::uint32_t>(catalog, endpoint_name);
  if (!subscriber) {
    return 20;
  }
  if (!write_code(result_fd, kReady)) {
    return 21;
  }
  std::uint64_t last_sequence = 0U;
  std::size_t received = 0U;
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (received < 2U) {
    // Poll through receive_sample so the subscriber side also establishes its
    // transport connections; a checked empty result is retried until the
    // deadline while a backend failure stays terminal.
    auto sample = subscriber->receive_sample();
    if (!sample) {
      if (sample.error().code != holoscan::ErrorCode::kNotReady) {
        return 24;
      }
      auto pending = subscriber->has_samples();
      if (!pending) {
        return 22;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return subscriber->has_publishers() ? 26 : 27;
      }
      std::this_thread::sleep_for(2ms);
      continue;
    }
    if (sample->data != 0xdecafbadU || sample->metadata.sequence <= last_sequence) {
      return 25;
    }
    last_sequence = sample->metadata.sequence;
    ++received;
  }
  return 0;
}

// monitored-publisher role: exercises clock-health reporting across the process boundary.
// Emits two values under a healthy report, then one more after declaring
// kSynchronizationLockLost so the receiving Runtime applies the H1→S1→S2→H2→S3
// health-gate transition sequence.
[[nodiscard]] int run_monitored_publisher(holoscan::ChannelRuntime& runtime,
                                          const holoscan::ChannelEndpointCatalog& catalog,
                                          const std::string& endpoint_name, const int result_fd) {
  auto publisher = runtime.create_publisher<std::uint32_t>(catalog, endpoint_name);
  if (!publisher) {
    return 30;
  }
  if (!write_code(result_fd, kReady)) {
    return 31;
  }
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!publisher->has_subscribers()) {
    if (!publisher->update_connections()) {
      return 32;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return 33;
    }
    std::this_thread::sleep_for(5ms);
  }
  if (!publisher->update_connections()) {
    return 32;
  }
  const auto emit_value = [&](const std::uint32_t value, const std::int64_t capture_ns) -> bool {
    const auto emit_deadline = std::chrono::steady_clock::now() + 10s;
    while (true) {
      auto emitted = publisher->emit_captured(value, capture_ns);
      if (emitted) {
        return true;
      }
      const auto code = emitted.error().code;
      if ((code != holoscan::ErrorCode::kNotReady && code != holoscan::ErrorCode::kBackpressured) ||
          std::chrono::steady_clock::now() >= emit_deadline) {
        return false;
      }
      static_cast<void>(publisher->update_connections());
      std::this_thread::sleep_for(2ms);
    }
  };
  // H1 -> S1 -> S2 -> H2 -> S3: the receiving Runtime must apply H1 before S1
  // eligibility and H2 before S3 across the real process boundary.
  if (!publisher->report_clock_health(holoscan::ClockHealthReport::healthy())) {
    return 34;
  }
  if (!emit_value(41U, 101) || !emit_value(42U, 102)) {
    return 35;
  }
  if (!publisher->report_clock_health(holoscan::ClockHealthReport::unhealthy(
          holoscan::ClockHealthReason::kSynchronizationLockLost))) {
    return 36;
  }
  if (!emit_value(43U, 103)) {
    return 37;
  }
  return 0;
}

// abrupt-publisher role: emits one sample then calls _exit(0) without destroying
// any endpoint or runtime, verifying that the peer graph reclaims the publisher's
// resource claims within a finite bound rather than hanging.
[[nodiscard]] int run_abrupt_publisher(holoscan::ChannelRuntime& runtime,
                                       const holoscan::ChannelEndpointCatalog& catalog,
                                       const std::string& endpoint_name, const int result_fd) {
  auto publisher = runtime.create_publisher<std::uint32_t>(catalog, endpoint_name);
  if (!publisher) {
    return 40;
  }
  if (!write_code(result_fd, kReady)) {
    return 41;
  }
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!publisher->has_subscribers()) {
    if (!publisher->update_connections()) {
      return 42;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return 43;
    }
    std::this_thread::sleep_for(5ms);
  }
  if (!publisher->update_connections()) {
    return 42;
  }
  const auto emit_deadline = std::chrono::steady_clock::now() + 10s;
  while (true) {
    auto emitted = publisher->emit_captured(77U, 150);
    if (emitted) {
      break;
    }
    const auto code = emitted.error().code;
    if ((code != holoscan::ErrorCode::kNotReady && code != holoscan::ErrorCode::kBackpressured) ||
        std::chrono::steady_clock::now() >= emit_deadline) {
      return 44;
    }
    static_cast<void>(publisher->update_connections());
    std::this_thread::sleep_for(2ms);
  }
  static_cast<void>(write_code(result_fd, 0U));
  // Abandon every endpoint and the runtime without any teardown: the parent
  // graph must reclaim the peer's claims within a finite bound.
  ::_exit(0);
}

[[nodiscard]] bool wait_for_subscriber(holoscan::ChannelPublisher<std::uint32_t>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!publisher.has_subscribers()) {
    if (!publisher.update_connections() || std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(2ms);
  }
  return publisher.update_connections().has_value();
}

[[nodiscard]] bool emit_captured_until(holoscan::ChannelPublisher<std::uint32_t>& publisher,
                                       const std::uint32_t value, const std::int64_t capture_ns) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (true) {
    auto emitted = publisher.emit_captured(value, capture_ns);
    if (emitted) {
      return true;
    }
    if ((emitted.error().code != holoscan::ErrorCode::kNotReady &&
         emitted.error().code != holoscan::ErrorCode::kBackpressured) ||
        std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(2ms);
  }
}

// temporal-drain-publisher role: publishes values {1,2,3} at t={10,20,30} on the
// values endpoint, then a single driver sample at t=40 on the driver endpoint,
// giving the receiving Runtime enough evidence to drain the matched temporal batch.
[[nodiscard]] int run_temporal_drain_publisher(holoscan::ChannelRuntime& runtime,
                                               const holoscan::ChannelEndpointCatalog& catalog,
                                               const std::string& values_endpoint,
                                               const std::string& driver_endpoint,
                                               const int result_fd) {
  auto values = runtime.create_publisher<std::uint32_t>(catalog, values_endpoint);
  auto driver = runtime.create_publisher<std::uint32_t>(catalog, driver_endpoint);
  if (!values || !driver) {
    return 50;
  }
  if (!write_code(result_fd, kReady)) {
    return 51;
  }
  if (!wait_for_subscriber(*values) || !wait_for_subscriber(*driver)) {
    return 52;
  }
  for (std::uint32_t value = 1U; value <= 3U; ++value) {
    if (!emit_captured_until(*values, value, static_cast<std::int64_t>(value) * 10)) {
      return 53;
    }
  }
  if (!emit_captured_until(*driver, 99U, 40)) {
    return 54;
  }
  return 0;
}

// temporal-window-publisher role: sends anchor and window samples in an interleaved
// order so the parent Runtime can close the bracketed window defined by the second
// anchor [10, 11] once the strict successor at t=15 arrives.
[[nodiscard]] int run_temporal_window_publisher(holoscan::ChannelRuntime& runtime,
                                                const holoscan::ChannelEndpointCatalog& catalog,
                                                const std::string& anchor_endpoint,
                                                const std::string& window_endpoint,
                                                const int result_fd) {
  auto anchor = runtime.create_publisher<std::uint32_t>(catalog, anchor_endpoint);
  auto window = runtime.create_publisher<std::uint32_t>(catalog, window_endpoint);
  if (!anchor || !window) {
    return 60;
  }
  if (!write_code(result_fd, kReady)) {
    return 61;
  }
  if (!wait_for_subscriber(*anchor) || !wait_for_subscriber(*window)) {
    return 62;
  }
  // The second anchor defines [10, 11]; the strict successor at t=15 closes
  // that interval only after the complete [5, 10, 15] bracket is visible.
  if (!emit_captured_until(*window, 5U, 5) || !emit_captured_until(*anchor, 10U, 10) ||
      !emit_captured_until(*window, 10U, 10) || !emit_captured_until(*anchor, 11U, 11) ||
      !emit_captured_until(*window, 15U, 15)) {
    return 63;
  }
  return 0;
}

// temporal-predicate-publisher role: emits the additional input before the driver
// witness so the parent Runtime observes ordered input presence when evaluating
// the OnPredicate trigger across the process boundary.
[[nodiscard]] int run_temporal_predicate_publisher(holoscan::ChannelRuntime& runtime,
                                                   const holoscan::ChannelEndpointCatalog& catalog,
                                                   const std::string& driver_endpoint,
                                                   const std::string& additional_endpoint,
                                                   const int result_fd) {
  auto driver = runtime.create_publisher<std::uint32_t>(catalog, driver_endpoint);
  auto additional = runtime.create_publisher<std::uint32_t>(catalog, additional_endpoint);
  if (!driver || !additional) {
    return 70;
  }
  if (!write_code(result_fd, kReady)) {
    return 71;
  }
  if (!wait_for_subscriber(*driver) || !wait_for_subscriber(*additional)) {
    return 72;
  }
  if (!emit_captured_until(*additional, 70U, 70)) {
    return 73;
  }
  // The services are independent. Leave a finite observation interval so the
  // predicate's ordered additional input is present before its driver witness.
  std::this_thread::sleep_for(50ms);
  if (!emit_captured_until(*driver, 7U, 70)) {
    return 74;
  }
  return 0;
}

}  // namespace

// Parent/child handshake sequence:
//
//   Parent process                         Child process (this binary)
//   ─────────────────────────────────────  ─────────────────────────────────────────────
//   serialize catalog → temp file          read HOLOSCAN_IPR_SMOKE_* env vars
//   fork() + set env vars                    missing/empty var       → exit 2
//   build runtime + graph endpoint           load catalog from path  → exit 3 on error
//   open START_FD / RESULT_FD pipes          create ChannelRuntime   → exit 4 on error
//   write start token on START_FD ────────>  wait_for_start()        → exit 5 on error
//                                            run_<role>():
//                                              create endpoint
//                                              write kReady ─────────────────────────>
//   <── read kReady; run graph side            run role body
//   <── read result byte                     write result code (0 = pass, N = stage)
int main() {
  const char* role = required_env("HOLOSCAN_IPR_SMOKE_ROLE");
  const char* catalog_path = required_env("HOLOSCAN_IPR_SMOKE_CATALOG");
  const char* endpoint = required_env("HOLOSCAN_IPR_SMOKE_ENDPOINT");
  const char* second_endpoint = required_env("HOLOSCAN_IPR_SMOKE_SECOND_ENDPOINT");
  const char* result_fd_text = required_env("HOLOSCAN_IPR_SMOKE_RESULT_FD");
  const char* start_fd_text = required_env("HOLOSCAN_IPR_SMOKE_START_FD");
  if (role == nullptr || catalog_path == nullptr || endpoint == nullptr ||
      result_fd_text == nullptr || start_fd_text == nullptr) {
    return 2;  // exit 2: required env var absent or empty
  }
  const int result_fd = std::atoi(result_fd_text);
  const int start_fd = std::atoi(start_fd_text);

  // The child loads the parent's exact serialized catalog: no C++ object,
  // compile state, or transport handle crosses the process boundary.
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(catalog_path);
  if (!catalog) {
    static_cast<void>(write_code(result_fd, 3U));
    return 3;  // exit 3: catalog JSON could not be loaded from the supplied path
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    static_cast<void>(write_code(result_fd, 4U));
    return 4;  // exit 4: ChannelRuntime could not be initialized in this process
  }
  if (!wait_for_start(start_fd)) {
    return 5;  // exit 5: start token not received from parent before pipe was closed
  }

  const std::string endpoint_name{endpoint};
  const std::string role_name{role};
  int result = 6;  // default: role string did not match any known role
  if (role_name == "publisher") {
    result = run_publisher(*runtime, *catalog, endpoint_name, result_fd);
  } else if (role_name == "subscriber") {
    result = run_subscriber(*runtime, *catalog, endpoint_name, result_fd);
  } else if (role_name == "monitored-publisher") {
    result = run_monitored_publisher(*runtime, *catalog, endpoint_name, result_fd);
  } else if (role_name == "abrupt-publisher") {
    result = run_abrupt_publisher(*runtime, *catalog, endpoint_name, result_fd);
  } else if (role_name == "temporal-drain-publisher" && second_endpoint != nullptr) {
    result =
        run_temporal_drain_publisher(*runtime, *catalog, endpoint_name, second_endpoint, result_fd);
  } else if (role_name == "temporal-window-publisher" && second_endpoint != nullptr) {
    result = run_temporal_window_publisher(
        *runtime, *catalog, endpoint_name, second_endpoint, result_fd);
  } else if (role_name == "temporal-predicate-publisher" && second_endpoint != nullptr) {
    result = run_temporal_predicate_publisher(
        *runtime, *catalog, endpoint_name, second_endpoint, result_fd);
  }
  static_cast<void>(write_code(result_fd, static_cast<std::uint8_t>(result)));
  return result;
}
