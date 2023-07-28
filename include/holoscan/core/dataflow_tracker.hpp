/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CORE_DATAFLOW_TRACKER_HPP
#define CORE_DATAFLOW_TRACKER_HPP

#include <limits.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "./forward_def.hpp"

namespace holoscan {

constexpr uint64_t kDefaultNumStartMessagesToSkip = 10;
constexpr uint64_t kDefaultNumLastMessagesToDiscard = 10;
constexpr int kDefaultLatencyThreshold = 0;
constexpr uint64_t kDefaultNumBufferedMessages = 100;
constexpr const char* kDefaultLogfileName = "logger.log";

enum class DataFlowMetric {
  kMaxMessageID,
  kMinMessageID,
  kMaxE2ELatency,
  kAvgE2ELatency,
  kMinE2ELatency,
  kNumSrcMessages,
  kNumDstMessages,
};

static const std::unordered_map<DataFlowMetric, std::string> metricToString = {
    {DataFlowMetric::kMaxE2ELatency, "Max end-to-end Latency (ms)"},
    {DataFlowMetric::kMaxMessageID, "Max Latency Message No"},
    {DataFlowMetric::kAvgE2ELatency, "Avg end-to-end Latency (ms)"},
    {DataFlowMetric::kMinE2ELatency, "Min end-to-end Latency (ms)"},
    {DataFlowMetric::kMinMessageID, "Min Latency Message No"},
    {DataFlowMetric::kNumDstMessages, "Number of messages"}};

class PathMetrics {
 public:
  PathMetrics() {
    path = "";
    metrics[DataFlowMetric::kMaxE2ELatency] = INT_MIN;
    metrics[DataFlowMetric::kMaxMessageID] = -1;
    metrics[DataFlowMetric::kAvgE2ELatency] = 0;
    metrics[DataFlowMetric::kMinE2ELatency] = INT_MAX;
    metrics[DataFlowMetric::kMinMessageID] = -1;
    metrics[DataFlowMetric::kNumDstMessages] = 0;
    num_skipped_messages = 0;
  }

  uint64_t get_buffer_size();

  std::string path;
  std::unordered_map<DataFlowMetric, double> metrics;
  std::queue<double> latency_buffer;
  uint64_t num_skipped_messages;
};

/**
 * @brief The DataFlowTracker class is used to track the data flow metrics for different paths
 * between the root operators and leaf operators. This class is used by the developers to get the
 * metrics data for flow during the execution of the application and at the end of it.
 *
 * This class uses mutex locks on metric properties so that multiple threads on multiple operators
 * can update the metrics without conflicts.
 *
 */
class DataFlowTracker {
 public:
  DataFlowTracker() {}

  ~DataFlowTracker();

  /**
   * @brief Set the number of messages to skip at the beginning of the execution.
   *
   * This does not affect the log file or the number of source messages metric.
   *
   * @param num The number of messages to skip.
   */
  void set_skip_starting_messages(uint64_t num) { num_start_messages_to_skip_ = num; }

  /**
   * @brief Set the threshold latency for which the end-to-end latency calculations will be done.
   * Any latency strictly less than the threshold latency will be ignored.
   *
   * This does not affect the log file or the number of source messages metric.
   *
   * @param threshold The threshold latency in milliseconds.
   */
  void set_skip_latencies(int threshold);

  /**
   * @brief Set the number of messages to discard at the end of the execution.
   *
   * This does not affect the log file or the number of source messages metric.
   *
   * @param num The number of messages to discard.
   */
  void set_discard_last_messages(uint64_t num) { num_last_messages_to_discard_ = num; }

  /**
   * @brief Enable message logging at the end of the every execution of a leaf
   * Operator.
   *
   * A path consisting of an array of tuples in the form of (an Operator name, message
   * receive timestamp, message publish timestamp) is logged in a file. The logging does not take
   * into account the number of message to skip or discard or the threshold latency.
   *
   * This function buffers a number of lines set by the @num_buffered_messages parameter before
   * flushing the buffer to the log file.
   *
   * @param filename The name of the log file.
   * @param num_buffered_messages The number of messages to be buffered before flushing the buffer
   * to the log file.
   */
  void enable_logging(std::string filename = kDefaultLogfileName,
                      uint64_t num_buffered_messages = kDefaultNumBufferedMessages);

  /**
   * @brief Print the result of the data flow tracking in pretty-printed format to the standard
   * output.
   *
   */
  void print() const;

  /**
   * @brief Return the number of tracked paths.
   *
   * @return The number of tracked paths.
   */
  int get_num_paths();

  /**
   * @brief Return an array of strings which are path names. Each path name is a
   * comma-separated list of Operator names in a path. The paths are agnostic to the edges between
   * two Operators.
   *
   * @return An array of the path names.
   */
  std::vector<std::string> get_path_strings();

  /**
   * @brief Return the value of a metric m for a given path.
   *
   * If m is DataFlowMetric::kNumSrcMessages, then the function returns -1.
   *
   * @param pathstring The path name string for which the metric is being queried.
   * @param metric The metric to be queried.
   * @return The value of the metric m for the given path.
   */
  double get_metric(std::string pathstring, holoscan::DataFlowMetric metric);

  /**
   * @brief Return the value of a metric.
   *
   * The metric must be DataFlowMetric::kNumSrcMessages.
   *
   * @param metric The metric to be queried.
   * @return The map of source names to the number of published
   * messages.
   */
  std::map<std::string, uint64_t> get_metric(
      holoscan::DataFlowMetric metric = DataFlowMetric::kNumSrcMessages);

  /**
   * @brief Write out the remaining messages from the log buffer and close the ofstream
   */
  void end_logging();

 protected:
  // Making DFFTCollector friend class to access update_latency,
  // update_source_messages_number, and write_to_logfile.
  friend class DFFTCollector;

  /**
   * @brief Update the tracker with the current latency for a given path.
   *
   * The function internally takes care of skipping beginning messages or discarding last messages.
   *
   * This function is not intended to be called by the developers. However, they may choose to
   * update the latencies manually to account for some external overheads.
   *
   * @param pathstring The path name string for which the latency is being updated.
   * @param current_latency The current latency value.
   */
  void update_latency(std::string pathstring, double current_latency);

  /**
   * @brief Update the tracker with the number of published messages for a given source
   * Operator.
   *
   * This function is not intended to be called by the developers. However, they may choose to
   * update the number of messages externally.
   *
   * @param source The name of the source in the form of [OperatorName->OutputName].
   * @param num The new number of published messages.
   */
  void update_source_messages_number(std::string source, uint64_t num);

  /**
   * @brief Writes to a log file only if file logging is enabled. Otherwise, the
   * function does nothing. It also takes care of buffering the messages and flushing them to the
   * log file periodically.
   *
   * @param text The new text to be written to the log file.
   */
  void write_to_logfile(std::string text);

 private:
  std::map<std::string, uint64_t>
      source_messages_;  ///< The map of source names to the number of published messages.
  std::mutex source_messages_mutex_;  ///< The mutex for the source_messages_.

  std::map<std::string, std::shared_ptr<holoscan::PathMetrics>>
      all_path_metrics_;               ///< The map of path names to the path metrics.
  std::mutex all_path_metrics_mutex_;  ///< The mutex for the all_path_metrics_.

  /// The number of messages to skip at the beginning of the execution of an application graph.
  /// This is also known as the warm-up period.
  uint64_t num_start_messages_to_skip_ = kDefaultNumStartMessagesToSkip;

  int latency_threshold_ = 0;  ///< The latency threshold in milliseconds below which we need
                               ///< to ignore latencies for end-to-end latency calculations.

  uint64_t num_last_messages_to_discard_ =
      kDefaultNumLastMessagesToDiscard;  ///< The number of messages to discard at the end of the
                                      ///< execution of an application graph.

  bool is_file_logging_enabled_ = false;  ///< The variable to indicate if file logging is enabled.
  std::string logger_filename_;           ///< The name of the log file.
  uint64_t num_buffered_messages_ =
      100;  ///< The number of messages to be buffered before flushing the buffer to the log file.
  std::ofstream logger_ofstream_;  ///< The output file stream for the log file.

  std::vector<std::string> buffered_messages_;  ///< The buffer for the log file.
  std::mutex buffered_messages_mutex_;          ///< The mutex for the buffered_messages_.

  uint64_t logfile_messages_ =
      0;  ///< The number of messages logged to the log file, used for writing to the log file.
};
}  // namespace holoscan

#endif /* CORE_DATAFLOW_TRACKER_HPP */
