/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <holomq/buffer_pool.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/limits.hpp>
#include <holomq/wait_strategy.hpp>
#include "cuda_raii.hpp"
#include "frame_payload.hpp"
#include "holoargs.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace {

enum class LogLevel {
  Error,
  Warn,
  Info,
};

void log_message(const LogLevel level, const std::string& msg) {
  const char* tag = "info";
  if (level == LogLevel::Error) {
    tag = "error";
  } else if (level == LogLevel::Warn) {
    tag = "warn";
  }
  std::fprintf(stderr, "holomq_main: %s: %s\n", tag, msg.c_str());
}

void log_message(const LogLevel level, const char* msg) {
  const char* tag = "info";
  if (level == LogLevel::Error) {
    tag = "error";
  } else if (level == LogLevel::Warn) {
    tag = "warn";
  }
  std::fprintf(stderr, "holomq_main: %s: %s\n", tag, msg);
}

void log_cu_failure(const char* step, const CUresult rc) {
  const char* err_name = nullptr;
  if (cuGetErrorString(rc, &err_name) == CUDA_SUCCESS && err_name != nullptr) {
    const std::string msg = std::string(step) + ": " + err_name;
    log_message(LogLevel::Error, msg);
  } else {
    const std::string msg =
        std::string(step) + ": CUDA error code " + std::to_string(static_cast<int>(rc));
    log_message(LogLevel::Error, msg);
  }
}

constexpr const char* k_samples_option = "samples";
constexpr const char* k_fd_pass_socket_option = "fd-pass-socket";
constexpr const char* k_role_option = "role";
constexpr const char* k_publisher_hold_ms_option = "publisher-hold-ms";
constexpr const char* k_publish_pace_ms_option = "publish-pace-ms";
constexpr const char* k_pool_type_option = "pool-type";
constexpr const char* k_host_register_flags_option = "host-register-flags";

enum class DemoPoolType {
  Device,
  Host,
};

enum class DemoHostRegisterFlags {
  Pageable,
  Portable,
  Devicemap,
  PortableDevicemap,
};

[[nodiscard]] std::optional<DemoPoolType> parse_pool_type(const std::string& value) {
  if (value == "device") {
    return DemoPoolType::Device;
  }
  if (value == "host" || value == "host-pinned") {
    return DemoPoolType::Host;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<DemoHostRegisterFlags> parse_host_register_flags(
    const std::string& value) {
  if (value == "pageable") {
    return DemoHostRegisterFlags::Pageable;
  }
  if (value == "portable") {
    return DemoHostRegisterFlags::Portable;
  }
  if (value == "devicemap") {
    return DemoHostRegisterFlags::Devicemap;
  }
  if (value == "portable-devicemap") {
    return DemoHostRegisterFlags::PortableDevicemap;
  }
  return std::nullopt;
}

[[nodiscard]] unsigned int demo_cuda_host_register_flags(const DemoHostRegisterFlags flags) {
  switch (flags) {
    case DemoHostRegisterFlags::Portable:
      return CU_MEMHOSTREGISTER_PORTABLE;
    case DemoHostRegisterFlags::Devicemap:
      return CU_MEMHOSTREGISTER_DEVICEMAP;
    case DemoHostRegisterFlags::PortableDevicemap:
      return CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
    case DemoHostRegisterFlags::Pageable:
      break;
  }
  return 0U;
}

[[nodiscard]] bool host_pool_uses_split_generation_vmm(const DemoHostRegisterFlags flags) noexcept {
  switch (flags) {
    case DemoHostRegisterFlags::Pageable:
    case DemoHostRegisterFlags::Portable:
      return true;
    case DemoHostRegisterFlags::Devicemap:
    case DemoHostRegisterFlags::PortableDevicemap:
      return false;
  }
  return false;
}

[[nodiscard]] bool fill_buffer_pattern(holomq::Buffer& buffer, CUstream stream,
                                       std::string* error_out) {
  if (holomq::is_device_buffer_memory_type(buffer.memory_type())) {
    const auto device_ptr =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(buffer.ptr()));
    const CUresult memset_rc =
        cuMemsetD8Async(device_ptr, 0, static_cast<std::size_t>(buffer.size_bytes()), stream);
    if (memset_rc != CUDA_SUCCESS) {
      if (error_out != nullptr) {
        log_cu_failure("cuMemsetD8Async", memset_rc);
      }
      return false;
    }
    return true;
  }

  if (holomq::is_host_buffer_memory_type(buffer.memory_type())) {
    std::memset(buffer.ptr(), 0, buffer.size_bytes());
    return true;
  }

  if (error_out != nullptr) {
    *error_out = "unsupported buffer memory type for fill";
  }
  return false;
}

constexpr const char* k_demo_service_name = "holomq_main_v4";

using DemoPublisher = holomq::Publisher<holomq_main::FramePayload>;

using DemoOutboundMessage = holomq::OutboundMessage<holomq_main::FramePayload>;

using DemoSubscriber = holomq::Subscriber<holomq_main::FramePayload>;

/// Polls until \ref DemoPublisher::has_subscribers reports a connected subscriber or \p timeout
/// elapses.
[[nodiscard]] bool wait_for_subscribers(DemoPublisher& publisher,
                                        std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!publisher.update_connections().has_value()) {
      return false;
    }
    if (publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return publisher.update_connections().has_value() && publisher.has_subscribers();
}

/// Retries \ref DemoPublisher::loan while slots are exhausted; returns empty optional on timeout.
[[nodiscard]] std::optional<DemoOutboundMessage> try_loan_sample_with_wait(
    DemoPublisher& publisher, const std::uint64_t sequence) {
  int loan_wait_retries = 0;
  bool logged_loan_wait_diag = false;
  constexpr int k_max_loan_wait_retries = 100000;
  for (;;) {
    auto loaned = publisher.loan();
    if (loaned.has_value()) {
      return std::move(loaned).value();
    }
    if (loaned.error().code != holomq::ErrorCode::kBackpressured) {
      log_message(LogLevel::Error, "publisher.loan failed");
      return std::nullopt;
    }
    if (!logged_loan_wait_diag) {
      const std::string msg =
          std::string("[diag] publisher: loan() failed for sequence ") + std::to_string(sequence) +
          " — see holomq::Publisher logs; also check external BufferPool acquire() slots";
      log_message(LogLevel::Warn, msg);
      logged_loan_wait_diag = true;
    }
    if (++loan_wait_retries > k_max_loan_wait_retries) {
      log_message(
          LogLevel::Error,
          "publisher: timed out waiting for a loan slot (max_loaned or BufferPool exhausted)");
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

/// Retries \ref holomq::BufferPool::acquire while all slots are leased; returns empty optional on
/// timeout.
[[nodiscard]] std::optional<holomq::Buffer> try_acquire_buffer_with_wait(
    holomq::BufferPool& device_pool) {
  int acquire_retries = 0;
  constexpr int k_max_acquire_retries = 100000;
  for (;;) {
    auto buffer = device_pool.acquire();
    if (buffer.has_value()) {
      return std::move(buffer).value();
    }
    if (buffer.error().code != holomq::ErrorCode::kBackpressured) {
      log_message(LogLevel::Error, "publisher: BufferPool::acquire failed");
      return std::nullopt;
    }
    if (++acquire_retries > k_max_acquire_retries) {
      log_message(LogLevel::Error,
                  "publisher: BufferPool::acquire() timed out (no free slot with ref_count==0)");
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

/// Loans, fills, and sends one sample; retries until at least one subscriber accepts or a timeout.
/// \c true on success.
///
/// Loan before acquire: the transport releases owner \c Buffer leases when a sample slot is
/// reused on the next \c loan(), so acquire-first deadlocks when \c buffer_count equals queue
/// depth.
[[nodiscard]] bool try_publish_sample_with_wait(holomq::Context& context, DemoPublisher& publisher,
                                                holomq::BufferPool& device_pool, CUstream stream,
                                                const std::uint64_t sequence) {
  auto fill_and_send = [&](DemoOutboundMessage message, holomq::Buffer device_buffer) -> bool {
    log_message(LogLevel::Info,
                std::string("publisher: loaned sample for sequence ") + std::to_string(sequence));

    auto& payload = message.payload();
    payload.width = holomq_main::k_frame_width;
    payload.height = holomq_main::k_frame_height;
    payload.sequence = sequence;
    std::string fill_error;
    if (!fill_buffer_pattern(device_buffer, stream, &fill_error)) {
      if (!fill_error.empty()) {
        log_message(LogLevel::Error, fill_error);
      }
      return false;
    }
    const auto exported = device_buffer.export_handle(stream);
    if (!exported.has_value()) {
      log_message(LogLevel::Error, "Buffer::export_handle failed");
      return false;
    }
    payload.buffer_handle = exported.value();
    const auto sent =
        publisher.send(std::move(message), [device_buffer = std::move(device_buffer)]() {});
    if (!sent.has_value() || sent.value() == 0) {
      log_message(LogLevel::Error, "publisher.send failed");
      return false;
    }
    log_message(LogLevel::Info,
                std::string("publisher: sent sample sequence ") + std::to_string(sequence) +
                    " to " + std::to_string(sent.value()) + " subscriber(s)");
    return true;
  };

  int wait_for_subscriber_retries = 0;
  constexpr int k_max_wait_for_subscriber_retries = 60000;
  for (;;) {
    auto message = try_loan_sample_with_wait(publisher, sequence);
    if (!message.has_value()) {
      return false;
    }
    auto device_buffer = try_acquire_buffer_with_wait(device_pool);
    if (!device_buffer.has_value()) {
      return false;
    }
    if (fill_and_send(std::move(message).value(), std::move(device_buffer).value())) {
      return true;
    }

    if (++wait_for_subscriber_retries > k_max_wait_for_subscriber_retries) {
      log_message(LogLevel::Error,
                  "publisher: timed out waiting for a connected subscriber to accept samples");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int run_publisher(holomq::Context& context, std::uint64_t sample_count,
                  std::uint32_t publisher_hold_ms, std::uint32_t publish_pace_ms,
                  DemoPoolType pool_type, DemoHostRegisterFlags host_register_flags) {
  if (sample_count == 0U) {
    log_message(LogLevel::Error, "sample count must be non-zero");
    return EXIT_FAILURE;
  }
  if (sample_count > holomq::k_max_in_flight && publish_pace_ms == 0U) {
    log_message(LogLevel::Warn,
                "samples exceeds k_max_in_flight with no publish pacing; pool acquire may block "
                "indefinitely or fail — use --publish-pace-ms or reduce --samples");
  }

  const holomq_main::DemoCudaStream publish_stream(CU_STREAM_DEFAULT);
  if (!publish_stream) {
    log_message(LogLevel::Error, "cuStreamCreate failed");
    return EXIT_FAILURE;
  }

  auto device_pool_result = [&] {
    if (pool_type == DemoPoolType::Device) {
      return context.create_device_buffer_pool(holomq_main::k_frame_buffer_bytes,
                                               holomq::k_max_in_flight);
    }
    if (host_register_flags == DemoHostRegisterFlags::Pageable) {
      return context.create_host_buffer_pool(holomq_main::k_frame_buffer_bytes,
                                             holomq::k_max_in_flight);
    }
    return context.create_host_pinned_buffer_pool(
        holomq_main::k_frame_buffer_bytes,
        holomq::k_max_in_flight,
        demo_cuda_host_register_flags(host_register_flags));
  }();

  if (!device_pool_result.has_value()) {
    log_message(LogLevel::Error, "Context buffer pool create failed");
    return EXIT_FAILURE;
  }
  holomq::BufferPool device_pool = std::move(device_pool_result).value();

  auto publisher = context.publisher_builder(k_demo_service_name)
                       .payload_type<holomq_main::FramePayload>()
                       .create();
  if (!publisher.has_value()) {
    log_message(LogLevel::Error, "PublisherBuilder::create failed");
    return EXIT_FAILURE;
  }

  log_message(LogLevel::Info,
              std::string("[diag] k_max_in_flight=") + std::to_string(holomq::k_max_in_flight) +
                  " => BufferPool buffer_count=" + std::to_string(holomq::k_max_in_flight) +
                  " (external pool; SHM ref-count tracks Buffer leases)");

  if (!wait_for_subscribers(publisher.value(), std::chrono::seconds(5))) {
    log_message(LogLevel::Error, "publisher: timed out waiting for a connected subscriber");
    return EXIT_FAILURE;
  }

  for (std::uint64_t i = 0; i < sample_count; ++i) {
    if (!try_publish_sample_with_wait(
            context, publisher.value(), device_pool, publish_stream.get(), i)) {
      return EXIT_FAILURE;
    }
    if (publish_pace_ms > 0U && i + 1U < sample_count) {
      std::this_thread::sleep_for(std::chrono::milliseconds(publish_pace_ms));
    }
  }

  log_message(LogLevel::Info,
              std::string("publisher transmitted ") + std::to_string(sample_count) + " sample(s)");

  // Per-sample owner Buffer lifetime is extended via send(..., callback); hold is only for process
  // teardown.
  if (publisher_hold_ms > 0U) {
    const std::string msg = std::string("publisher sleeping ") + std::to_string(publisher_hold_ms) +
                            " ms after all sends, then exiting";
    log_message(LogLevel::Info, msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(publisher_hold_ms));
  }

  log_message(LogLevel::Info, "publisher finished; exiting");
  return EXIT_SUCCESS;
}

int run_subscriber(holomq::Context& context, std::uint64_t sample_count) {
  if (sample_count == 0U) {
    log_message(LogLevel::Error, "sample count must be non-zero");
    return EXIT_FAILURE;
  }

  auto subscriber = context.subscriber_builder(k_demo_service_name)
                        .payload_type<holomq_main::FramePayload>()
                        .create();
  if (!subscriber.has_value()) {
    log_message(LogLevel::Error, "SubscriberBuilder::create failed");
    return EXIT_FAILURE;
  }

  const holomq_main::DemoCudaStream subscribe_stream(CU_STREAM_DEFAULT);
  if (!subscribe_stream) {
    log_message(LogLevel::Error, "cuStreamCreate failed");
    return EXIT_FAILURE;
  }

  std::uint64_t next_sequence = 0;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(30 + static_cast<std::int64_t>(sample_count));

  holomq::WaitStrategy wait_strategy{};
  wait_strategy.mode = holomq::WaitMode::KernelWait;

  while (next_sequence < sample_count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      log_message(LogLevel::Error, "subscriber timed out waiting for samples");
      return EXIT_FAILURE;
    }

    auto received = subscriber->wait_and_take(wait_strategy);
    if (!received.has_value()) {
      if (received.error().code == holomq::ErrorCode::kTimeout) {
        continue;
      }
      log_message(LogLevel::Error, "Subscriber::wait_and_take failed");
      return EXIT_FAILURE;
    }

    const holomq_main::FramePayload& payload = received.value().payload();
    if (payload.sequence != next_sequence) {
      const std::string msg = std::string("unexpected sample sequence: expected ") +
                              std::to_string(next_sequence) + " got " +
                              std::to_string(payload.sequence);
      log_message(LogLevel::Error, msg);
      return EXIT_FAILURE;
    }
    log_message(LogLevel::Info,
                std::string("subscriber: received sample (sequence=") +
                    std::to_string(payload.sequence) + ")");

    auto device_buffer = context.import_buffer(payload.buffer_handle, subscribe_stream.get());
    if (!device_buffer.has_value()) {
      log_message(LogLevel::Error,
                  "Context::import_buffer(handle, stream) failed on received payload");
      return EXIT_FAILURE;
    }

    const auto device_ptr =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_buffer->ptr()));
    (void)device_ptr;
    ++next_sequence;
    log_message(LogLevel::Info,
                std::string("[diag] subscriber: destroying sample sequence ") +
                    std::to_string(payload.sequence) +
                    " — iceoryx2 loan released; device_buffer view may outlive sample until scope "
                    "end");
  }

  log_message(LogLevel::Info,
              std::string("subscriber received all ") + std::to_string(sample_count) +
                  " sample(s); exiting");
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
  holomq_examples::args::OptionsDescription desc("holomq");
  desc.add_options()(k_role_option,
                     holomq_examples::args::value<std::string>()->value_name("NAME")->default_value(
                         std::string{"publisher"}),
                     "publisher | subscriber")(
      k_samples_option,
      holomq_examples::args::value<std::uint64_t>()->value_name("N")->default_value(
          static_cast<std::uint64_t>(holomq::k_max_in_flight)),
      "Publish/receive this many samples (strict host payload sequence 0..N-1; default matches "
      "holomq::k_max_in_flight)")(
      k_fd_pass_socket_option,
      holomq_examples::args::value<std::string>()->value_name("PATH")->default_value(std::string{}),
      "Publisher: AF_UNIX listen path for FD-pass; empty uses a unique /tmp path (copied into each "
      "sample "
      "descriptor). Subscribers use the path from the received descriptor")(
      k_publisher_hold_ms_option,
      holomq_examples::args::value<std::uint32_t>()->value_name("MS")->default_value(
          std::uint32_t{2000}),
      "After all samples are sent, sleep this many ms before the publisher exits (default 2000 = 2 "
      "s)")(k_publish_pace_ms_option,
            holomq_examples::args::value<std::uint32_t>()->value_name("MS")->default_value(
                std::uint32_t{0}),
            "Publisher: optional sleep between sends (0 = no pacing; lossless mode blocks on "
            "loan/send when the "
            "subscriber is slow)")(
      k_pool_type_option,
      holomq_examples::args::value<std::string>()->value_name("NAME")->default_value(
          std::string{"device"}),
      "device | host (alias: host-pinned)")(
      k_host_register_flags_option,
      holomq_examples::args::value<std::string>()->value_name("FLAGS")->default_value(
          std::string{"devicemap"}),
      "host pools only: pageable | portable | devicemap | portable-devicemap");

  const holomq_examples::args::Parser parser;
  if (argc <= 1) {
    std::array<char*, 2> help_argv = {argv[0], const_cast<char*>("--help")};
    parser.parse_command_line(2, help_argv.data(), desc);  // argp prints help and exits
  }
  holomq_examples::args::VariablesMap vars;
  try {
    vars = parser.parse_command_line(argc, argv, desc);
  } catch (const std::exception& ex) {
    log_message(LogLevel::Error, ex.what());
    return EXIT_FAILURE;
  }
  const std::string role = vars.at(k_role_option).as<std::string>();
  const std::uint64_t samples = vars.at(k_samples_option).as<std::uint64_t>();
  const std::string fd_pass_arg = vars.at(k_fd_pass_socket_option).as<std::string>();
  const std::uint32_t publisher_hold_ms = vars.at(k_publisher_hold_ms_option).as<std::uint32_t>();
  const std::uint32_t publish_pace_ms = vars.at(k_publish_pace_ms_option).as<std::uint32_t>();
  const auto pool_type_or = parse_pool_type(vars.at(k_pool_type_option).as<std::string>());
  if (!pool_type_or.has_value()) {
    log_message(LogLevel::Error, "pool-type must be \"device\", \"host\", or \"host-pinned\"");
    return EXIT_FAILURE;
  }
  const DemoPoolType pool_type = *pool_type_or;
  const std::string host_register_flags_arg =
      vars.at(k_host_register_flags_option).as<std::string>();
  const auto host_register_flags_or = parse_host_register_flags(host_register_flags_arg);
  if (pool_type == DemoPoolType::Host && !host_register_flags_or.has_value()) {
    log_message(LogLevel::Error,
                "host-register-flags must be pageable, portable, devicemap, or portable-devicemap "
                "for host pools");
    return EXIT_FAILURE;
  }
  const DemoHostRegisterFlags host_register_flags =
      host_register_flags_or.value_or(DemoHostRegisterFlags::Devicemap);

  const bool publisher_needs_fd_pass =
      role == "publisher" && (pool_type == DemoPoolType::Device ||
                              (pool_type == DemoPoolType::Host &&
                               host_pool_uses_split_generation_vmm(host_register_flags)));

  // Some Cuda initialization boilerplate to make sure the application is using the correct CUDA
  // context.
  const CUresult init_rc = cuInit(0);
  if (init_rc != CUDA_SUCCESS) {
    log_cu_failure("cuInit", init_rc);
    return EXIT_FAILURE;
  }
  CUdevice cuda_dev{};
  const CUresult get_dev_rc = cuDeviceGet(&cuda_dev, 0);
  if (get_dev_rc != CUDA_SUCCESS) {
    log_cu_failure("cuDeviceGet", get_dev_rc);
    return EXIT_FAILURE;
  }
  const holomq_main::DemoCudaContext app_cuda_ctx(cuda_dev, CU_CTX_SCHED_AUTO);
  if (!app_cuda_ctx) {
    return EXIT_FAILURE;
  }
  const CUresult set_rc = cuCtxSetCurrent(app_cuda_ctx.get());
  if (set_rc != CUDA_SUCCESS) {
    log_cu_failure("cuCtxSetCurrent", set_rc);
    return EXIT_FAILURE;
  }

  // Publisher: FD-pass when the pool layout requires export (default true otherwise).
  // Subscriber: explicitly disable local FD-pass.
  auto context = holomq::ContextBuilder()
                     .with_vmm_fd_export(role == "publisher" ? publisher_needs_fd_pass : false)
                     .with_fd_pass_socket(publisher_needs_fd_pass ? fd_pass_arg : std::string{})
                     .create();
  if (!context.has_value()) {
    log_message(LogLevel::Error, "ContextBuilder::create failed");
    return EXIT_FAILURE;
  }

  if (role == "subscriber") {
    return run_subscriber(context.value(), samples);
  }
  if (role == "publisher") {
    return run_publisher(context.value(),
                         samples,
                         publisher_hold_ms,
                         publish_pace_ms,
                         pool_type,
                         host_register_flags);
  }

  log_message(LogLevel::Error, "role must be \"publisher\" or \"subscriber\"");
  return EXIT_FAILURE;
}
