/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/fastdds/pubsub/fastdds_transport.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/log/Log.hpp>
#include <fastdds/dds/log/OStreamConsumer.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/common/Guid.hpp>

#include <gxf/core/gxf.h>

#include <holoscan/logger/logger.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_qos_profiles.hpp>
#include <holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp>

namespace holoscan {

using namespace eprosima::fastdds::dds;
using nvidia::gxf::Expected;
using nvidia::gxf::Gid;
using nvidia::gxf::MessageMetadata;
using nvidia::gxf::Unexpected;

namespace {

//==============================================================================
// HoloscanLogConsumer - Bridges FastDDS logs into Holoscan's spdlog logger
//==============================================================================
//
// FastDDS has its own logging system (eprosima::fastdds::dds::Log) that by
// default writes to stdout/stderr via StdoutErrConsumer.  Registering this
// consumer routes FastDDS messages through HOLOSCAN_LOG_* so they appear in
// the same stream, with the same formatting and level filtering, as all other
// Holoscan log output.
//
// NOTE: FastDDS Info logging is only available when the library was compiled
// with FASTDDS_ENFORCE_LOG_INFO or in Debug builds.  In release builds, only
// Warning and Error messages are emitted.  To enable Info logs, rebuild
// FastDDS with:
//   cmake -DLOG_NO_INFO=OFF -DFASTDDS_ENFORCE_LOG_INFO=ON ...
// or use the Holoscan option:
//   cmake -DHOLOSCAN_FASTDDS_BUILD_FROM_SOURCE=ON ...
// which now passes those flags automatically.
//
class HoloscanLogConsumer : public eprosima::fastdds::dds::LogConsumer {
 public:
  void Consume(const eprosima::fastdds::dds::Log::Entry& entry) override {
    const char* category = entry.context.category ? entry.context.category : "?";
    const char* function = entry.context.function ? entry.context.function : "";
    const char* filename = entry.context.filename ? entry.context.filename : "";

    switch (entry.kind) {
      case eprosima::fastdds::dds::Log::Kind::Error:
        HOLOSCAN_LOG_ERROR("[FastDDS/{}] {} ({}:{} {})",
                           category,
                           entry.message,
                           filename,
                           entry.context.line,
                           function);
        break;
      case eprosima::fastdds::dds::Log::Kind::Warning:
        HOLOSCAN_LOG_WARN("[FastDDS/{}] {} ({}:{} {})",
                          category,
                          entry.message,
                          filename,
                          entry.context.line,
                          function);
        break;
      case eprosima::fastdds::dds::Log::Kind::Info:
        // FastDDS Info messages are very chatty (RTPS_PDP heartbeats, RTPS_MSG_IN
        // submessage traces, etc.).  Route them based on Holoscan log level:
        //   TRACE  -> HOLOSCAN_LOG_TRACE  (no filtering, full detail)
        //   DEBUG  -> HOLOSCAN_LOG_DEBUG  (visible at debug, hidden at info+)
        //   INFO+  -> suppressed (FastDDS verbosity set to Warning, see below)
        if (holoscan::log_level() <= holoscan::LogLevel::TRACE) {
          HOLOSCAN_LOG_TRACE("[FastDDS/{}] {} ({}:{} {})",
                             category,
                             entry.message,
                             filename,
                             entry.context.line,
                             function);
        } else {
          HOLOSCAN_LOG_DEBUG("[FastDDS/{}] {} ({}:{} {})",
                             category,
                             entry.message,
                             filename,
                             entry.context.line,
                             function);
        }
        break;
      default:
        HOLOSCAN_LOG_DEBUG("[FastDDS/{}] {}", category, entry.message);
        break;
    }
  }
};

/// Convert FastDDS GUID_t (16 bytes) to GXF Gid (16 bytes)
/// GUID_t layout: guidPrefix (12 bytes) + entityId (4 bytes)
Gid guid_to_gid(const eprosima::fastdds::rtps::GUID_t& guid) {
  std::array<uint8_t, nvidia::gxf::kGidSize> bytes{};
  // Copy guidPrefix (12 bytes)
  std::memcpy(bytes.data(), guid.guidPrefix.value, 12);
  // Copy entityId (4 bytes)
  std::memcpy(bytes.data() + 12, guid.entityId.value, 4);
  return Gid(bytes);
}

}  // namespace

//==============================================================================
// ReaderListener - Handles incoming DDS messages
//==============================================================================

class FastDdsTransport::ReaderListener : public DataReaderListener {
 public:
  explicit ReaderListener(FastDdsTransport* transport) : transport_(transport) {}

  void on_data_available(DataReader* reader) override {
    HOLOSCAN_LOG_DEBUG("FastDdsTransport::ReaderListener::on_data_available called");

    // Read all available samples
    HoloscanEntityData data;
    SampleInfo info;

    int sample_count = 0;
    while (reader->take_next_sample(&data, &info) == RETCODE_OK) {
      sample_count++;
      HOLOSCAN_LOG_DEBUG(
          "FastDdsTransport: took sample {}, valid_data={}", sample_count, info.valid_data);

      if (info.valid_data) {
        // Find the application-side subscriber GID associated with this local
        // DataReader instance. This is NOT the same identifier space as the
        // transport-native DDS GUID carried in SampleInfo.
        // TODO(grelee): Replace linear scan with a reverse map (DataReader* -> Gid)
        // for O(1) lookup if the number of subscribers becomes large.
        Gid subscriber_gid;
        {
          std::lock_guard<std::mutex> lock(transport_->endpoints_mutex_);
          for (const auto& [gid, reader_info] : transport_->readers_) {
            if (reader_info.reader == reader) {
              subscriber_gid = gid;
              break;
            }
          }
        }

        // Extract the transport-native DDS writer GUID from SampleInfo and map it
        // into the framework-level Gid space. This GUID-derived publisher_gid is
        // used for routing and metadata, but it should not be confused with any
        // application-assigned local publisher Gid passed to create_publisher_endpoint().
        Gid publisher_gid = guid_to_gid(info.sample_identity.writer_guid());

        HOLOSCAN_LOG_DEBUG("FastDdsTransport: received {} bytes from publisher {}, timestamp={}",
                           data.serialized_data.size(),
                           publisher_gid.to_string(),
                           data.timestamp_ns);

        // Build metadata
        MessageMetadata metadata;
        metadata.source_timestamp_ns = static_cast<uint64_t>(data.timestamp_ns);
        metadata.publisher_gid = publisher_gid;
        metadata.destination_gid = subscriber_gid;

        // Copy callback under lock, invoke outside to avoid holding it during dispatch
        ReceiveCallback cb;
        {
          std::lock_guard<std::mutex> cb_lock(transport_->callback_mutex_);
          cb = transport_->receive_callback_;
        }
        if (cb) {
          HOLOSCAN_LOG_DEBUG("FastDdsTransport: invoking receive callback");
          cb(publisher_gid, std::move(data.serialized_data), metadata);
          HOLOSCAN_LOG_DEBUG("FastDdsTransport: receive callback completed");
        } else {
          HOLOSCAN_LOG_WARN("FastDdsTransport: no receive callback set!");
        }
      }
    }
    HOLOSCAN_LOG_DEBUG("FastDdsTransport: on_data_available processed {} samples", sample_count);
  }

  void on_subscription_matched([[maybe_unused]] DataReader* reader,
                               const SubscriptionMatchedStatus& status) override {
    HOLOSCAN_LOG_DEBUG("FastDdsTransport: Reader subscription_matched: current_count={}, change={}",
                       status.current_count,
                       status.current_count_change);
    if (status.current_count_change > 0) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport: Reader matched {} new publisher(s), total={}",
                         status.current_count_change,
                         status.current_count);
    } else if (status.current_count_change < 0) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport: Reader lost {} publisher(s), total={}",
                         -status.current_count_change,
                         status.current_count);
    }
  }

  void on_liveliness_changed([[maybe_unused]] DataReader* reader,
                             const LivelinessChangedStatus& status) override {
    if (status.alive_count_change > 0) {
      HOLOSCAN_LOG_INFO("FastDdsTransport: Writer reconnected: alive_count={}, alive_change=+{}",
                        status.alive_count,
                        status.alive_count_change);
    }
    if (status.not_alive_count_change > 0) {
      HOLOSCAN_LOG_WARN(
          "FastDdsTransport: Writer lost liveliness: not_alive_count={}, not_alive_change=+{}",
          status.not_alive_count,
          status.not_alive_count_change);
    }
  }

  void on_requested_deadline_missed([[maybe_unused]] DataReader* reader,
                                    const RequestedDeadlineMissedStatus& status) override {
    HOLOSCAN_LOG_WARN(
        "FastDdsTransport: Requested deadline missed: total_count={}, total_count_change={}",
        status.total_count,
        status.total_count_change);
  }

  void on_requested_incompatible_qos([[maybe_unused]] DataReader* reader,
                                     const RequestedIncompatibleQosStatus& status) override {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport: Requested incompatible QoS: total_count={}, last_policy_id={}",
        status.total_count,
        static_cast<int>(status.last_policy_id));
  }

  void on_sample_lost([[maybe_unused]] DataReader* reader,
                      const SampleLostStatus& status) override {
    HOLOSCAN_LOG_WARN("FastDdsTransport: Sample lost: total_count={}, total_count_change={}",
                      status.total_count,
                      status.total_count_change);
  }

  void on_sample_rejected([[maybe_unused]] DataReader* reader,
                          const SampleRejectedStatus& status) override {
    HOLOSCAN_LOG_WARN(
        "FastDdsTransport: Sample rejected: total_count={}, total_count_change={}, reason={}",
        status.total_count,
        status.total_count_change,
        static_cast<int>(status.last_reason));
  }

 private:
  FastDdsTransport* transport_;
};

//==============================================================================
// WriterListener - Handles DataWriter events (optional, for tracking)
//==============================================================================

class FastDdsTransport::WriterListener : public DataWriterListener {
 public:
  explicit WriterListener(FastDdsTransport* transport) : transport_(transport) {}

  void on_publication_matched([[maybe_unused]] DataWriter* writer,
                              const PublicationMatchedStatus& status) override {
    HOLOSCAN_LOG_DEBUG("FastDdsTransport: Writer publication_matched: current_count={}, change={}",
                       status.current_count,
                       status.current_count_change);
    if (status.current_count_change > 0) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport: Writer matched {} new subscriber(s), total={}",
                         status.current_count_change,
                         status.current_count);
    } else if (status.current_count_change < 0) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport: Writer lost {} subscriber(s), total={}",
                         -status.current_count_change,
                         status.current_count);
    }
  }

  void on_offered_incompatible_qos([[maybe_unused]] DataWriter* writer,
                                   const OfferedIncompatibleQosStatus& status) override {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport: Offered incompatible QoS: total_count={}, last_policy_id={}",
        status.total_count,
        static_cast<int>(status.last_policy_id));
  }

  void on_offered_deadline_missed([[maybe_unused]] DataWriter* writer,
                                  const OfferedDeadlineMissedStatus& status) override {
    HOLOSCAN_LOG_WARN(
        "FastDdsTransport: Offered deadline missed: total_count={}, total_count_change={}",
        status.total_count,
        status.total_count_change);
  }

  void on_liveliness_lost([[maybe_unused]] DataWriter* writer,
                          const LivelinessLostStatus& status) override {
    HOLOSCAN_LOG_WARN("FastDdsTransport: Liveliness lost: total_count={}, total_count_change={}",
                      status.total_count,
                      status.total_count_change);
  }

  void on_unacknowledged_sample_removed(
      [[maybe_unused]] DataWriter* writer,
      [[maybe_unused]] const InstanceHandle_t& instance) override {
    HOLOSCAN_LOG_WARN("FastDdsTransport: Unacknowledged sample removed from writer");
  }

 private:
  FastDdsTransport* transport_;
};

//==============================================================================
// SidecarReaderListener - Handles incoming native descriptor messages
//==============================================================================

class FastDdsTransport::SidecarReaderListener : public DataReaderListener {
 public:
  explicit SidecarReaderListener(FastDdsTransport* transport) : transport_(transport) {}

  void on_data_available(DataReader* reader) override {
    HoloscanEntityData data;
    SampleInfo info;

    while (reader->take_next_sample(&data, &info) == RETCODE_OK) {
      if (!info.valid_data)
        continue;

      // The main-topic publisher GID is carried in the dedicated publisher_gid
      // field because the sidecar DDS writer has a different GUID than the
      // main-topic writer.
      Gid publisher_gid;
      if (!data.publisher_gid.empty()) {
        auto parsed = Gid::from_string(data.publisher_gid);
        if (parsed) {
          publisher_gid = parsed.value();
        } else {
          HOLOSCAN_LOG_WARN(
              "FastDdsTransport::SidecarReaderListener: failed to parse publisher_gid '{}'",
              data.publisher_gid);
          publisher_gid = guid_to_gid(info.sample_identity.writer_guid());
        }
      } else {
        publisher_gid = guid_to_gid(info.sample_identity.writer_guid());
      }

      MessageMetadata metadata;
      metadata.source_timestamp_ns = static_cast<uint64_t>(data.timestamp_ns);
      metadata.publisher_gid = publisher_gid;
      metadata.payload_mode = nvidia::gxf::PayloadMode::kNativeHandleDescriptor;
      metadata.descriptor_format_version = data.descriptor_format_version;
      metadata.protocol_name = data.protocol_name;

      // Derive the main topic name by stripping the sidecar suffix
      auto topic_name = reader->get_topicdescription()->get_name();
      const std::string suffix = "/_native_desc";
      if (topic_name.size() > suffix.size() &&
          topic_name.compare(topic_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        metadata.topic_name = topic_name.substr(0, topic_name.size() - suffix.size());
      }

      {
        std::lock_guard<std::mutex> cb_lock(transport_->callback_mutex_);
        if (transport_->receive_callback_) {
          HOLOSCAN_LOG_DEBUG(
              "FastDdsTransport::SidecarReaderListener: received native descriptor ({} bytes) "
              "from publisher {}",
              data.serialized_data.size(),
              publisher_gid.to_string());
          transport_->enqueue_sidecar_receive(
              publisher_gid, std::move(data.serialized_data), metadata);
        }
      }
    }
  }

  void on_subscription_matched([[maybe_unused]] DataReader* reader,
                               const SubscriptionMatchedStatus& status) override {
    HOLOSCAN_LOG_DEBUG("FastDdsTransport: Sidecar reader subscription_matched: count={}, change={}",
                       status.current_count,
                       status.current_count_change);
  }

 private:
  FastDdsTransport* transport_;
};

//==============================================================================
// FastDdsTransport Implementation
//==============================================================================

FastDdsTransport::FastDdsTransport(FastDdsPubSubContext* context) : context_(context) {
  if (!context_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport: null FastDdsPubSubContext");
  }
}

FastDdsTransport::~FastDdsTransport() {
  if (initialized_) {
    shutdown();
  }
}

Expected<void> FastDdsTransport::initialize() {
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::initialize() called");

  // Bridge FastDDS logging into Holoscan's spdlog-based logger.
  //
  if (initialized_) {
    HOLOSCAN_LOG_DEBUG("FastDdsTransport::initialize: already initialized");
    return Expected<void>();
  }

  // FastDDS has its own logging system that defaults to stdout/stderr.  We
  // replace the default consumers with a HoloscanLogConsumer that routes all
  // FastDDS messages through HOLOSCAN_LOG_* macros.  This gives us:
  //   - Consistent formatting and timestamps with the rest of Holoscan
  //   - Level filtering via HOLOSCAN_LOG_LEVEL
  //   - Category, filename, line number, and function name in each message
  //
  // Verbosity mapping (Holoscan -> FastDDS):
  //   TRACE       -> Info   (all FastDDS messages, routed to HOLOSCAN_LOG_TRACE)
  //   DEBUG       -> Info   (all FastDDS messages, routed to HOLOSCAN_LOG_DEBUG)
  //   INFO+       -> Warning (suppress chatty RTPS/transport traces)
  //
  // IMPORTANT: FastDDS Info-level logging (EPROSIMA_LOG_INFO) is compiled out
  // in release builds.  Only Warning and Error messages will appear unless
  // FastDDS was built with -DLOG_NO_INFO=OFF -DFASTDDS_ENFORCE_LOG_INFO=ON.
  // The Holoscan cmake option HOLOSCAN_FASTDDS_BUILD_FROM_SOURCE=ON now
  // passes these flags automatically.
  {
    using FastDDSLog = eprosima::fastdds::dds::Log;

    // Remove the default StdoutErrConsumer so messages don't double-print
    FastDDSLog::ClearConsumers();

    // Register our bridge consumer
    FastDDSLog::RegisterConsumer(std::unique_ptr<LogConsumer>(new HoloscanLogConsumer()));

    // Set FastDDS verbosity based on Holoscan's current log level.
    // At DEBUG or TRACE we enable FastDDS Info so all internal messages flow
    // through; at INFO or above we limit to Warning to keep logs clean.
    const auto hl_level = holoscan::log_level();
    if (hl_level <= holoscan::LogLevel::DEBUG) {
      FastDDSLog::SetVerbosity(FastDDSLog::Kind::Info);
      HOLOSCAN_LOG_DEBUG("FastDdsTransport: FastDDS log verbosity set to Info (Holoscan level={})",
                         static_cast<int>(hl_level));
    } else {
      FastDDSLog::SetVerbosity(FastDDSLog::Kind::Warning);
      HOLOSCAN_LOG_DEBUG(
          "FastDdsTransport: FastDDS log verbosity set to Warning (Holoscan level={})",
          static_cast<int>(hl_level));
    }

    HOLOSCAN_LOG_DEBUG("FastDdsTransport: registered HoloscanLogConsumer for FastDDS logging");
  }

  if (!context_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::initialize: null context");
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  auto* participant = context_->participant();
  if (!participant) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::initialize: null DomainParticipant");
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  HOLOSCAN_LOG_TRACE("FastDdsTransport::initialize: got DomainParticipant");

  // Create type support for unbounded HoloscanEntity type
  // Store as member to ensure proper lifetime throughout transport's use
  // Note: max_serialized_type_size is set to 0 (unbounded) per FastDDS best practices.
  // FastDDS uses calculate_serialized_size() for dynamic per-message allocation.
  type_support_.reset(new FastDdsHoloscanEntityTypeSupport());
  HOLOSCAN_LOG_DEBUG("FastDdsTransport: created type support for unbounded HoloscanEntity");

  if (participant->register_type(type_support_) != RETCODE_OK) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::initialize: failed to register type");
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }
  registered_type_name_ = type_support_.get_type_name();
  type_registered_ = true;
  HOLOSCAN_LOG_TRACE("FastDdsTransport::initialize: registered type '{}'", registered_type_name_);

  // Create Publisher
  publisher_ = participant->create_publisher(PUBLISHER_QOS_DEFAULT, nullptr);
  if (!publisher_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::initialize: failed to create Publisher");
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }
  HOLOSCAN_LOG_TRACE("FastDdsTransport::initialize: created Publisher");

  // Create Subscriber
  subscriber_ = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);
  if (!subscriber_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::initialize: failed to create Subscriber");
    participant->delete_publisher(publisher_);
    publisher_ = nullptr;
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }
  HOLOSCAN_LOG_TRACE("FastDdsTransport::initialize: created Subscriber");

  // Create listeners
  reader_listener_ = std::make_unique<ReaderListener>(this);
  writer_listener_ = std::make_unique<WriterListener>(this);

  initialized_ = true;

  // Create the sidecar dispatch queue (callback provider captures receive_callback_)
  sidecar_dispatch_queue_ = std::make_unique<SidecarDispatchQueue>([this]() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    return receive_callback_;
  });

  if (native_buffers_enabled_) {
    sidecar_dispatch_queue_->start();
  }
  HOLOSCAN_LOG_DEBUG("FastDdsTransport initialized successfully");
  return Expected<void>();
}

Expected<void> FastDdsTransport::shutdown() {
  if (!initialized_) {
    return Expected<void>();
  }

  std::unique_lock<std::mutex> lock(endpoints_mutex_);

  auto* participant = context_ ? context_->participant() : nullptr;

  // Wait for in-flight messages to be acknowledged before deleting writers
  // (important for RELIABLE QoS to avoid data loss)
  for (auto& [gid, writer_info] : writers_) {
    if (writer_info.writer) {
      Duration_t ack_wait{1, 0};  // 1 second timeout
      auto ack_ret = writer_info.writer->wait_for_acknowledgments(ack_wait);
      if (ack_ret != RETCODE_OK) {
        HOLOSCAN_LOG_DEBUG(
            "FastDdsTransport::shutdown: wait_for_acknowledgments timed out for topic '{}'",
            writer_info.topic_name);
      }
    }
  }

  // Shutdown follows the recommended bottom-up deletion order:
  //   1. DataWriters/DataReaders (from Publishers/Subscribers)
  //   2. Publishers/Subscribers (from DomainParticipant)
  //   3. Topics (from DomainParticipant)

  // Step 1a: Delete sidecar DataWriters
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting {} sidecar DataWriter(s)",
                     sidecar_writers_.size());
  for (auto& [name, sw_info] : sidecar_writers_) {
    if (publisher_ && sw_info.writer) {
      publisher_->delete_datawriter(sw_info.writer);
    }
  }
  sidecar_writers_.clear();

  // Step 1b: Delete sidecar DataReaders
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting {} sidecar DataReader(s)",
                     sidecar_readers_.size());
  for (auto& [name, sr_info] : sidecar_readers_) {
    if (subscriber_ && sr_info.reader) {
      subscriber_->delete_datareader(sr_info.reader);
    }
  }
  sidecar_readers_.clear();
  sidecar_reader_listener_.reset();
  {
    std::lock_guard<std::mutex> cb_lock(callback_mutex_);
    receive_callback_ = {};
  }
  auto* sidecar_queue = sidecar_dispatch_queue_.get();
  lock.unlock();
  if (sidecar_queue) {
    sidecar_queue->stop();
  }
  lock.lock();
  sidecar_dispatch_queue_.reset();

  // Step 1c: Delete all DataWriters
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting {} DataWriter(s)", writers_.size());
  for (auto& [gid, writer_info] : writers_) {
    if (publisher_ && writer_info.writer) {
      publisher_->delete_datawriter(writer_info.writer);
    }
  }
  writers_.clear();

  // Step 1d: Delete all DataReaders
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting {} DataReader(s)", readers_.size());
  for (auto& [gid, reader_info] : readers_) {
    if (subscriber_ && reader_info.reader) {
      subscriber_->delete_datareader(reader_info.reader);
    }
  }
  readers_.clear();

  // Clear listeners (safe now that all readers/writers are gone)
  reader_listener_.reset();
  writer_listener_.reset();

  // Step 2: Delete Publisher and Subscriber
  if (participant) {
    if (publisher_) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting Publisher");
      participant->delete_publisher(publisher_);
      publisher_ = nullptr;
    }
    if (subscriber_) {
      HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting Subscriber");
      participant->delete_subscriber(subscriber_);
      subscriber_ = nullptr;
    }
  }

  // Step 3: Delete Topics
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: deleting {} Topic(s)", topics_.size());
  for (auto& [name, topic] : topics_) {
    if (participant && topic) {
      participant->delete_topic(topic);
    }
  }
  topics_.clear();

  // Type support is automatically unregistered when the participant is deleted
  registered_type_name_.clear();
  type_registered_ = false;

  initialized_ = false;
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::shutdown: complete");
  return Expected<void>();
}

bool FastDdsTransport::is_initialized() const {
  return initialized_;
}

//==============================================================================
// Connection Management (mostly no-ops for DDS)
//==============================================================================

Expected<void> FastDdsTransport::connect_to(
    const nvidia::gxf::EndpointInfo& /* remote_endpoint */) {
  // No-op for DDS - discovery is automatic via SPDP/SEDP
  return Expected<void>();
}

Expected<void> FastDdsTransport::disconnect_from(const Gid& /* remote_gid */) {
  // No-op for DDS - endpoint lifecycle managed by DDS
  return Expected<void>();
}

bool FastDdsTransport::is_connected_to(const Gid& /* remote_gid */) const {
  // DDS handles connections automatically - always return true
  return initialized_;
}

//==============================================================================
// Topic-Based Endpoint Lifecycle
//==============================================================================

Expected<void> FastDdsTransport::create_publisher_endpoint(const std::string& topic_name,
                                                           const Gid& publisher_gid,
                                                           const nvidia::gxf::QoSProfile& qos) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::create_publisher_endpoint: not initialized");
    return Unexpected(GXF_UNINITIALIZED_VALUE);
  }

  std::lock_guard<std::mutex> lock(endpoints_mutex_);

  // Check if writer already exists
  if (writers_.find(publisher_gid) != writers_.end()) {
    HOLOSCAN_LOG_WARN(
        "FastDdsTransport::create_publisher_endpoint: writer already exists for gid {}",
        publisher_gid.to_string());
    return Expected<void>();
  }

  // Get or create topic
  auto* dds_topic = get_or_create_topic(topic_name);
  if (!dds_topic) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::create_publisher_endpoint: failed to create topic '{}'",
                       topic_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Map QoSProfile struct directly to FastDDS QoS (no named profile lookup)
  DataWriterQos writer_qos = DATAWRITER_QOS_DEFAULT;
  dds_qos::apply_writer_qos(writer_qos, qos);

  // Create DataWriter
  auto* writer = publisher_->create_datawriter(dds_topic, writer_qos, writer_listener_.get());
  if (!writer) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport::create_publisher_endpoint: failed to create DataWriter for topic '{}'",
        topic_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Store writer info by GID
  WriterInfo info;
  info.writer = writer;
  info.topic = dds_topic;
  info.topic_name = topic_name;
  writers_[publisher_gid] = info;

  HOLOSCAN_LOG_DEBUG("FastDdsTransport: created publisher endpoint for topic '{}' (gid={})",
                     topic_name,
                     publisher_gid.to_string());
  return Expected<void>();
}

Expected<void> FastDdsTransport::create_subscriber_endpoint(const std::string& topic_name,
                                                            const Gid& subscriber_gid,
                                                            const nvidia::gxf::QoSProfile& qos) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::create_subscriber_endpoint: not initialized");
    return Unexpected(GXF_UNINITIALIZED_VALUE);
  }

  std::lock_guard<std::mutex> lock(endpoints_mutex_);

  // Check if reader already exists
  if (readers_.find(subscriber_gid) != readers_.end()) {
    HOLOSCAN_LOG_WARN(
        "FastDdsTransport::create_subscriber_endpoint: reader already exists for gid {}",
        subscriber_gid.to_string());
    return Expected<void>();
  }

  // Get or create topic
  auto* dds_topic = get_or_create_topic(topic_name);
  if (!dds_topic) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::create_subscriber_endpoint: failed to create topic '{}'",
                       topic_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Map QoSProfile struct directly to FastDDS QoS
  DataReaderQos reader_qos = DATAREADER_QOS_DEFAULT;
  dds_qos::apply_reader_qos(reader_qos, qos);

  // Create DataReader with listener for async message reception
  auto* reader = subscriber_->create_datareader(dds_topic, reader_qos, reader_listener_.get());
  if (!reader) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport::create_subscriber_endpoint: failed to create DataReader for topic '{}'",
        topic_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Store reader info
  ReaderInfo info;
  info.reader = reader;
  info.topic = dds_topic;
  info.topic_name = topic_name;
  info.subscriber_gid = subscriber_gid;
  readers_[subscriber_gid] = info;

  // If native buffers are enabled, also create a sidecar reader for this topic
  if (native_buffers_enabled_) {
    auto sidecar_result = create_sidecar_reader(topic_name, subscriber_gid);
    if (!sidecar_result) {
      HOLOSCAN_LOG_WARN(
          "FastDdsTransport: failed to create sidecar reader for topic '{}' (non-fatal)",
          topic_name);
    }
  }

  HOLOSCAN_LOG_DEBUG("FastDdsTransport: created subscriber endpoint for topic '{}' (gid={})",
                     topic_name,
                     subscriber_gid.to_string());
  return Expected<void>();
}

Expected<void> FastDdsTransport::remove_publisher_endpoint(const Gid& publisher_gid) {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);

  auto it = writers_.find(publisher_gid);
  if (it == writers_.end()) {
    HOLOSCAN_LOG_WARN("FastDdsTransport::remove_publisher_endpoint: writer not found for gid {}",
                      publisher_gid.to_string());
    return Unexpected(GXF_ENTITY_NOT_FOUND);
  }

  // Wait for in-flight acknowledgments before deletion (RELIABLE QoS)
  if (it->second.writer) {
    Duration_t ack_wait{1, 0};  // 1 second timeout
    auto ack_ret = it->second.writer->wait_for_acknowledgments(ack_wait);
    if (ack_ret != RETCODE_OK) {
      HOLOSCAN_LOG_DEBUG(
          "FastDdsTransport::remove_publisher_endpoint: wait_for_acknowledgments timed out "
          "for topic '{}'",
          it->second.topic_name);
    }
  }

  if (publisher_ && it->second.writer) {
    publisher_->delete_datawriter(it->second.writer);
  }

  writers_.erase(it);
  HOLOSCAN_LOG_DEBUG("FastDdsTransport: removed publisher endpoint for gid {}",
                     publisher_gid.to_string());
  return Expected<void>();
}

Expected<void> FastDdsTransport::remove_subscriber_endpoint(const Gid& subscriber_gid) {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);

  auto it = readers_.find(subscriber_gid);
  if (it == readers_.end()) {
    HOLOSCAN_LOG_WARN("FastDdsTransport::remove_subscriber_endpoint: reader not found for gid {}",
                      subscriber_gid.to_string());
    return Unexpected(GXF_ENTITY_NOT_FOUND);
  }

  if (subscriber_ && it->second.reader) {
    subscriber_->delete_datareader(it->second.reader);
  }

  readers_.erase(it);
  HOLOSCAN_LOG_DEBUG("FastDdsTransport: removed subscriber endpoint for gid {}",
                     subscriber_gid.to_string());
  return Expected<void>();
}

//==============================================================================
// Data Plane
//==============================================================================

eprosima::fastdds::dds::Topic* FastDdsTransport::get_or_create_topic(
    const std::string& topic_name) {
  // Check if topic already exists
  auto it = topics_.find(topic_name);
  if (it != topics_.end()) {
    HOLOSCAN_LOG_TRACE("FastDdsTransport: reusing existing topic '{}'", topic_name);
    return it->second;
  }

  // Create new topic
  auto* participant = context_->participant();
  if (!participant) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::get_or_create_topic: null participant");
    return nullptr;
  }

  auto* topic = participant->create_topic(topic_name, registered_type_name_, TOPIC_QOS_DEFAULT);

  if (!topic) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::get_or_create_topic: failed to create topic '{}'",
                       topic_name);
    return nullptr;
  }

  topics_[topic_name] = topic;
  HOLOSCAN_LOG_DEBUG(
      "FastDdsTransport: created topic '{}' with type '{}'", topic_name, registered_type_name_);
  return topic;
}

Expected<void> FastDdsTransport::send_impl(const std::string& topic_name,
                                           const std::vector<uint8_t>& payload,
                                           const MessageMetadata& metadata) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::send: not initialized");
    return Unexpected(GXF_UNINITIALIZED_VALUE);
  }

  // Look up writer while holding the lock, but release before write()
  // to avoid deadlock with on_data_available callback (which also needs the lock)
  DataWriter* writer = nullptr;
  {
    std::lock_guard<std::mutex> lock(endpoints_mutex_);
    auto it = writers_.find(metadata.publisher_gid);
    if (it != writers_.end()) {
      writer = it->second.writer;
    }
  }
  // Lock released here - safe to call write() which may trigger callbacks

  if (!writer) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::send: no writer for topic '{}'", topic_name);
    return Unexpected(GXF_PUBSUB_NO_SUBSCRIBERS);
  }

  // Log express flag if set (DDS does not support per-message express)
  if (metadata.express) {
    HOLOSCAN_LOG_TRACE("FastDdsTransport::send: express flag set (ignored for DDS)");
  }

  // Build HoloscanEntityData
  HoloscanEntityData data;
  data.serialized_data = payload;
  data.source_operator = "";  // Could be populated from metadata if available
  data.timestamp_ns = static_cast<int64_t>(metadata.source_timestamp_ns);
  data.contains_gpu_tensors = false;  // Set by FastDdsSerializer if GPU data is present
  data.gpu_device_id = 0;

  // Query matched readers before writing for diagnostic purposes
  PublicationMatchedStatus pub_matched;
  writer->get_publication_matched_status(pub_matched);
  HOLOSCAN_LOG_DEBUG(
      "FastDdsTransport::send: writing {} bytes to topic '{}' "
      "(matched_readers={}, timestamp={})",
      payload.size(),
      topic_name,
      pub_matched.current_count,
      data.timestamp_ns);

  // Write to DDS (may trigger on_data_available synchronously for intra-process)
  auto ret = writer->write(&data);
  if (ret != RETCODE_OK) {
    switch (ret) {
      case RETCODE_TIMEOUT:
        // History buffer is full and write timed out waiting for space.
        // This can occur when subscribers are slow or not acknowledging.
        HOLOSCAN_LOG_WARN(
            "FastDdsTransport::send: write timed out on topic '{}' ({} bytes). "
            "History may be full - consider increasing history depth or using BEST_EFFORT QoS.",
            topic_name,
            payload.size());
        // Attempt to wait for acknowledgments and retry once.
        // We intentionally do not add backoff/sleep here — this runs on the
        // scheduler thread and blocking would stall the pipeline.  Persistent
        // history-full conditions should be addressed via QoS tuning or
        // congestion control policy at the application level.
        {
          Duration_t ack_wait{1, 0};  // 1 second
          if (writer->wait_for_acknowledgments(ack_wait) == RETCODE_OK) {
            ret = writer->write(&data);
            if (ret == RETCODE_OK) {
              HOLOSCAN_LOG_DEBUG("FastDdsTransport::send: write succeeded after waiting for acks");
              return Expected<void>();
            }
          }
          HOLOSCAN_LOG_ERROR(
              "FastDdsTransport::send: write retry failed on topic '{}' after waiting for acks",
              topic_name);
        }
        return Unexpected(GXF_PUBSUB_SEND_TIMEOUT);

      case RETCODE_OUT_OF_RESOURCES:
        // History is exhausted - no space available for new samples.
        HOLOSCAN_LOG_ERROR(
            "FastDdsTransport::send: out of resources on topic '{}' ({} bytes). "
            "History is full. Consider increasing resource_limits.max_samples "
            "or history depth.",
            topic_name,
            payload.size());
        return Unexpected(GXF_PUBSUB_QUEUE_FULL);

      default:
        HOLOSCAN_LOG_ERROR("FastDdsTransport::send: write failed on topic '{}' with code {}",
                           topic_name,
                           static_cast<int>(ret));
        return Unexpected(GXF_FAILURE);
    }
  }

  HOLOSCAN_LOG_DEBUG("FastDdsTransport::send: write completed successfully on topic '{}'",
                     topic_name);
  return Expected<void>();
}

Expected<void> FastDdsTransport::send(const Gid& /* destination_gid */,
                                      const std::vector<uint8_t>& payload,
                                      const MessageMetadata& metadata) {
  // GID-based fallback: determine topic from the publisher_gid
  std::string topic_name;
  {
    std::lock_guard<std::mutex> lock(endpoints_mutex_);
    auto it = writers_.find(metadata.publisher_gid);
    if (it != writers_.end()) {
      topic_name = it->second.topic_name;
    }
  }

  if (topic_name.empty()) {
    // Use metadata.topic_name if available
    topic_name = metadata.topic_name;
  }

  if (topic_name.empty()) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::send(GID): no topic found for publisher_gid {}",
                       metadata.publisher_gid.to_string());
    return Unexpected(GXF_ENTITY_NOT_FOUND);
  }

  // Sanity check: metadata.topic_name should match if populated
  if (!metadata.topic_name.empty() && metadata.topic_name != topic_name) {
    HOLOSCAN_LOG_WARN("FastDdsTransport::send(GID): metadata.topic_name '{}' != writer topic '{}'",
                      metadata.topic_name,
                      topic_name);
  }

  return send_impl(topic_name, payload, metadata);
}

Expected<void> FastDdsTransport::send(const std::string& topic_name,
                                      const std::vector<uint8_t>& payload,
                                      const MessageMetadata& metadata) {
  // Topic-based send: direct topic→DataWriter lookup (preferred path for DDS)
  return send_impl(topic_name, payload, metadata);
}

void FastDdsTransport::set_on_receive(ReceiveCallback callback) {
  HOLOSCAN_LOG_DEBUG("FastDdsTransport::set_on_receive: callback {}", callback ? "set" : "cleared");
  std::lock_guard<std::mutex> lock(callback_mutex_);
  receive_callback_ = std::move(callback);
}

void FastDdsTransport::set_on_connection_established(ConnectionEstablishedCallback callback) {
  connection_established_callback_ = std::move(callback);
}

void FastDdsTransport::set_on_connection_lost(ConnectionLostCallback callback) {
  connection_lost_callback_ = std::move(callback);
}

void FastDdsTransport::set_native_buffers_enabled(bool enabled) {
  native_buffers_enabled_ = enabled;
  if (initialized_ && sidecar_dispatch_queue_) {
    if (enabled) {
      sidecar_dispatch_queue_->start();
    } else {
      sidecar_dispatch_queue_->stop();
    }
  }
}

//==============================================================================
// Native Buffer (CUDA IPC) — Sidecar Topic Support
//==============================================================================

static std::string sidecar_topic_name(const std::string& base_topic) {
  return base_topic + "/_native_desc";
}

DataWriter* FastDdsTransport::get_or_create_sidecar_writer(const std::string& topic_name) {
  std::string sidecar_name = sidecar_topic_name(topic_name);

  auto it = sidecar_writers_.find(topic_name);
  if (it != sidecar_writers_.end()) {
    return it->second.writer;
  }

  // Create sidecar topic
  auto* dds_topic = get_or_create_topic(sidecar_name);
  if (!dds_topic) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport: failed to create sidecar topic '{}'", sidecar_name);
    return nullptr;
  }

  // Sidecar QoS: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(2)
  DataWriterQos writer_qos = DATAWRITER_QOS_DEFAULT;
  writer_qos.reliability().kind = eprosima::fastdds::dds::RELIABLE_RELIABILITY_QOS;
  writer_qos.durability().kind = eprosima::fastdds::dds::TRANSIENT_LOCAL_DURABILITY_QOS;
  writer_qos.history().kind = eprosima::fastdds::dds::KEEP_LAST_HISTORY_QOS;
  writer_qos.history().depth = 2;

  auto* writer = publisher_->create_datawriter(dds_topic, writer_qos, nullptr);
  if (!writer) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport: failed to create sidecar DataWriter for '{}'",
                       sidecar_name);
    return nullptr;
  }

  SidecarWriterInfo info;
  info.writer = writer;
  info.topic = dds_topic;
  sidecar_writers_[topic_name] = info;

  HOLOSCAN_LOG_DEBUG("FastDdsTransport: created sidecar DataWriter for '{}'", sidecar_name);
  return writer;
}

Expected<void> FastDdsTransport::create_sidecar_reader(const std::string& topic_name,
                                                       const Gid& subscriber_gid) {
  std::string sidecar_name = sidecar_topic_name(topic_name);

  if (sidecar_readers_.find(topic_name) != sidecar_readers_.end()) {
    return Expected<void>();
  }

  if (!sidecar_reader_listener_) {
    sidecar_reader_listener_ = std::make_unique<SidecarReaderListener>(this);
  }

  auto* dds_topic = get_or_create_topic(sidecar_name);
  if (!dds_topic) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport: failed to create sidecar topic '{}'", sidecar_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Sidecar QoS: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(2)
  DataReaderQos reader_qos = DATAREADER_QOS_DEFAULT;
  reader_qos.reliability().kind = eprosima::fastdds::dds::RELIABLE_RELIABILITY_QOS;
  reader_qos.durability().kind = eprosima::fastdds::dds::TRANSIENT_LOCAL_DURABILITY_QOS;
  reader_qos.history().kind = eprosima::fastdds::dds::KEEP_LAST_HISTORY_QOS;
  reader_qos.history().depth = 2;

  auto* reader =
      subscriber_->create_datareader(dds_topic, reader_qos, sidecar_reader_listener_.get());
  if (!reader) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport: failed to create sidecar DataReader for '{}'",
                       sidecar_name);
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  SidecarReaderInfo info;
  info.reader = reader;
  info.topic = dds_topic;
  sidecar_readers_[topic_name] = info;

  HOLOSCAN_LOG_DEBUG("FastDdsTransport: created sidecar DataReader for '{}' (subscriber={})",
                     sidecar_name,
                     subscriber_gid.to_string());
  return Expected<void>();
}

Expected<void> FastDdsTransport::send_native_descriptor(
    const std::string& topic_name, const nvidia::gxf::NativeDescriptorPayload& descriptor,
    const MessageMetadata& metadata) {
  if (!initialized_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::send_native_descriptor: not initialized");
    return Unexpected(GXF_UNINITIALIZED_VALUE);
  }

  if (!native_buffers_enabled_) {
    HOLOSCAN_LOG_ERROR("FastDdsTransport::send_native_descriptor: native buffers not enabled");
    return Unexpected(GXF_NOT_IMPLEMENTED);
  }

  DataWriter* writer = nullptr;
  Gid dds_publisher_gid;
  {
    std::lock_guard<std::mutex> lock(endpoints_mutex_);
    writer = get_or_create_sidecar_writer(topic_name);

    // Resolve the main-topic writer's DDS GUID so the subscriber can
    // associate this sidecar message with the publisher it discovered.
    // metadata.publisher_gid is the Holoscan-level GID which the remote
    // subscriber does not know; the DDS writer GUID is what discovery
    // registers on the remote side.
    auto it = writers_.find(metadata.publisher_gid);
    if (it != writers_.end() && it->second.writer != nullptr) {
      dds_publisher_gid = guid_to_gid(it->second.writer->guid());
    }
  }

  if (!writer) {
    return Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  if (dds_publisher_gid.is_null()) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport::send_native_descriptor: no main-topic writer found for '{}'",
        topic_name);
    return Unexpected(GXF_ENTITY_NOT_FOUND);
  }

  HoloscanEntityData data;
  data.serialized_data = descriptor.descriptor_bytes;
  data.publisher_gid = dds_publisher_gid.to_string();
  data.timestamp_ns = static_cast<int64_t>(metadata.source_timestamp_ns);
  data.contains_gpu_tensors = false;  // GPU data via IPC, not in payload
  data.gpu_device_id = 0;
  data.descriptor_format_version = metadata.descriptor_format_version != 0
                                       ? metadata.descriptor_format_version
                                       : descriptor.descriptor_format_version;
  data.protocol_name =
      !metadata.protocol_name.empty() ? metadata.protocol_name : descriptor.protocol_name;

  auto ret = writer->write(&data);
  if (ret != RETCODE_OK) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsTransport::send_native_descriptor: write failed on sidecar topic '{}' (code={})",
        sidecar_topic_name(topic_name),
        static_cast<int>(ret));
    return Unexpected(GXF_FAILURE);
  }

  HOLOSCAN_LOG_DEBUG("FastDdsTransport::send_native_descriptor: sent {} bytes on '{}'",
                     descriptor.descriptor_bytes.size(),
                     sidecar_topic_name(topic_name));
  return Expected<void>();
}

void FastDdsTransport::enqueue_sidecar_receive(nvidia::gxf::Gid publisher_gid,
                                               std::vector<uint8_t>&& payload,
                                               nvidia::gxf::MessageMetadata metadata) {
  sidecar_dispatch_queue_->enqueue(publisher_gid, std::move(payload), std::move(metadata));
}

//==============================================================================
// Metrics
//==============================================================================

size_t FastDdsTransport::get_send_queue_size() const {
  // DDS doesn't expose send queue size directly
  // Could potentially query DataWriter history cache, but for now return 0
  return 0;
}

size_t FastDdsTransport::get_receive_queue_size() const {
  // DDS doesn't expose receive queue size directly
  // Could potentially query DataReader history cache, but for now return 0
  return 0;
}

size_t FastDdsTransport::get_connection_count() const {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);
  // Return number of writers + readers as a proxy for "connections"
  return writers_.size() + readers_.size();
}

}  // namespace holoscan
