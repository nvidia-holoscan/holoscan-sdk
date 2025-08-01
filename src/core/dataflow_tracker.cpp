/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/core/dataflow_tracker.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

uint64_t PathMetrics::get_buffer_size() {
  return latency_buffer.size();
}

DataFlowTracker::~DataFlowTracker() {
  end_logging();
}

void DataFlowTracker::end_logging() {
  if (!logger_ofstream_.is_open())
    return;

  // Write out the remaining messages from the log buffer and close ofstream
  for (const auto& it : buffered_messages_) {
    logger_ofstream_ << it << "\n";
  }
  logger_ofstream_.close();
}

void DataFlowTracker::print() const {
  std::cout << "Data Flow Tracking Results:\n";
  std::cout << "Total paths: " << all_path_metrics_.size() << "\n\n";
  int i = 0;
  for (const auto& it : all_path_metrics_) {
    std::cout << "Path " << ++i << ": " << it.first << "\n";
    for (const auto& it2 : it.second->metrics) {
      std::cout << metricToString.at(it2.first) << ": " << it2.second << "\n";
    }
    std::cout << "\n";
  }

  if (source_messages_.empty()) {
    std::cout << "No source messages found.\n";
  } else {
    std::cout << "Number of source messages [format: source operator->transmitter name: number of "
                 "messages]:\n";
    for (const auto& it : source_messages_) {
      std::cout << it.first << ": " << it.second << "\n";
    }
  }

  std::cout.flush();  // flush standard output; otherwise output may not be printed
}

void DataFlowTracker::update_latency(std::string pathstring, double current_latency) {
  std::scoped_lock lock(all_path_metrics_mutex_);

  if (all_path_metrics_.find(pathstring) == all_path_metrics_.end()) {
    all_path_metrics_[pathstring] = std::make_shared<PathMetrics>();
    all_path_metrics_[pathstring]->path = pathstring;
  }

  // If the current latency is less than the threshold, then skip this message from latency
  // calculations
  if (current_latency < latency_threshold_) {
    // Do not track this message
    return;
  }

  // For a path, if the number of skipped messages at the beginning is less than the
  // num_start_messages_to_skip_, then do not track this message
  if (all_path_metrics_[pathstring]->num_skipped_messages < num_start_messages_to_skip_) {
    all_path_metrics_[pathstring]->num_skipped_messages++;
    return;
  }

  // Push the current latency to the buffer
  all_path_metrics_[pathstring]->latency_buffer.push(current_latency);

  // If the size of the buffer in this path has exceeded the num_last_messages_to_discard_, then get
  // the oldest element from the buffer and treat it as current latency
  if (all_path_metrics_[pathstring]->get_buffer_size() > num_last_messages_to_discard_) {
    // Get the oldest latency from the buffer
    current_latency = all_path_metrics_[pathstring]->latency_buffer.front();
    // Remove the oldest latency from the buffer
    all_path_metrics_[pathstring]->latency_buffer.pop();
    // Sanity check to make sure that the size of the buffer is equal to
    // num_last_messages_to_discard_
    assert(num_last_messages_to_discard_ == all_path_metrics_[pathstring]->get_buffer_size());

    // Update Max E2E Latency
    double prev_max_latency =
        all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMaxE2ELatency];

    all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMaxE2ELatency] =
        std::max(current_latency, prev_max_latency);

    // Update Min E2E Latency
    double prev_min_latency =
        all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMinE2ELatency];

    all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMinE2ELatency] =
        std::min(current_latency, prev_min_latency);

    // Calculate the average latency from total messages and avg latency till now
    auto tmp_avg_lat = all_path_metrics_[pathstring]->metrics[DataFlowMetric::kAvgE2ELatency];

    auto tmp_tot_messages = all_path_metrics_[pathstring]->metrics[DataFlowMetric::kNumDstMessages];

    all_path_metrics_[pathstring]->metrics[DataFlowMetric::kAvgE2ELatency] =
        (tmp_avg_lat * tmp_tot_messages + current_latency) / (tmp_tot_messages + 1);

    // Update total number of messages
    all_path_metrics_[pathstring]->metrics[DataFlowMetric::kNumDstMessages] += 1;

    // Update kMaxMessageID
    if (all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMaxE2ELatency] == current_latency) {
      all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMaxMessageID] =
          all_path_metrics_[pathstring]->metrics[DataFlowMetric::kNumDstMessages];
    }

    // Update kMinMessageID
    if (all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMinE2ELatency] == current_latency) {
      all_path_metrics_[pathstring]->metrics[DataFlowMetric::kMinMessageID] =
          all_path_metrics_[pathstring]->metrics[DataFlowMetric::kNumDstMessages];
    }
  }
}

void DataFlowTracker::update_source_messages_number(std::string source, uint64_t num) {
  std::scoped_lock lock(source_messages_mutex_);
  source_messages_[source] = num;
}

int DataFlowTracker::get_num_paths() {
  return all_path_metrics_.size();
}

std::vector<std::string> DataFlowTracker::get_path_strings() {
  std::vector<std::string> all_pathstrings;
  all_pathstrings.reserve(all_path_metrics_.size());
  for (const auto& it : all_path_metrics_) {
    all_pathstrings.push_back(it.first);
  }
  return all_pathstrings;
}

double DataFlowTracker::get_metric(std::string pathstring, holoscan::DataFlowMetric metric) {
  if (metric == DataFlowMetric::kNumSrcMessages) {
    HOLOSCAN_LOG_ERROR("metric with pathstring must not be DataFlowMetric::kNumSrcMessages");
    return -1;
  } else if (all_path_metrics_.find(pathstring) == all_path_metrics_.end()) {
    HOLOSCAN_LOG_ERROR(
        "pathstring not found. make sure messages are not skipped at the beginning or end or with "
        "set_skip_latencies.");
    return -1;
  }
  return all_path_metrics_[pathstring]->metrics[metric];
}

std::map<std::string, uint64_t> DataFlowTracker::get_metric(holoscan::DataFlowMetric metric) {
  if (metric != DataFlowMetric::kNumSrcMessages) {
    HOLOSCAN_LOG_ERROR("metric without pathstring must be DataFlowMetric::kNumSrcMessages");
    return {};
  }
  return source_messages_;
}

void DataFlowTracker::set_skip_latencies(int threshold) {
  latency_threshold_ = threshold;
}

void DataFlowTracker::enable_logging(std::string filename, uint64_t num_buffered_messages) {
  is_file_logging_enabled_ = true;
  this->num_buffered_messages_ = num_buffered_messages;
  logger_filename_ = std::move(filename);
  std::scoped_lock lock(buffered_messages_mutex_);
  buffered_messages_.reserve(this->num_buffered_messages_);
  logfile_messages_ = 0;
}

void DataFlowTracker::write_to_logfile(std::string text) {
  if (!text.empty() && is_file_logging_enabled_) {
    if (!logger_ofstream_.is_open()) {
      logger_ofstream_.open(logger_filename_);
    }
    std::scoped_lock lock(buffered_messages_mutex_);
    buffered_messages_.push_back(std::to_string(++logfile_messages_) + ":\n" + text);
    // When the vector's size is equal to buffered number of messages,
    // flush out the buffer to file
    // and clear the vector to re-reserve the memory
    if (buffered_messages_.size() == num_buffered_messages_) {
      for (const auto& it : buffered_messages_) {
        logger_ofstream_ << it << "\n";
      }
      logger_ofstream_ << std::flush;
      buffered_messages_.clear();
      buffered_messages_.reserve(num_buffered_messages_);
    }
  }
}

}  // namespace holoscan
