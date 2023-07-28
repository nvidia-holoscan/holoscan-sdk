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

#include <limits.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

OperatorTimestampLabel& OperatorTimestampLabel::operator=(const OperatorTimestampLabel& o) {
  if (this != &o) {
    this->operator_ptr = o.operator_ptr;
    this->rec_timestamp = o.rec_timestamp;
    this->pub_timestamp = o.pub_timestamp;
  }
  return *this;
}

int64_t MessageLabel::get_e2e_latency(int index) {
  if (message_paths.empty()) {
    HOLOSCAN_LOG_ERROR("MessageLabel::get_e2e_latency - message_paths is empty");
    return -1;
  }

  auto cur_path = message_paths[index];

  return (cur_path.back().pub_timestamp - cur_path.front().rec_timestamp);
}

void MessageLabel::print_all() {
  if (!num_paths()) {
    std::cout << "No paths in MessageLabel.\n";
    return;
  }
  std::cout << "All paths in MessageLabel:\n";

  std::cout << this->to_string();
}

std::string MessageLabel::to_string() const {
  auto msg_buf = fmt::memory_buffer();
  for (auto it : message_paths) {
    for (auto it2 : it) {
      if (!it2.operator_ptr) {
        HOLOSCAN_LOG_ERROR("MessageLabel::to_string - Operator pointer is null");
      } else {
        fmt::format_to(std::back_inserter(msg_buf),
                       "({},{},{}) -> ",
                       it2.operator_ptr->name(),
                       std::to_string(it2.rec_timestamp),
                       std::to_string(it2.pub_timestamp));
      }
    }
    msg_buf.resize(msg_buf.size() - 3);
    fmt::format_to(std::back_inserter(msg_buf), "\n");
  }
  return fmt::to_string(msg_buf);
}

std::vector<std::string> MessageLabel::get_all_path_names() {
  std::vector<std::string> all_paths;
  all_paths.reserve(message_paths.size());
  for (auto path : message_paths) {
    auto pathstring = fmt::memory_buffer();
    for (auto oplabel : path) {
      if (!oplabel.operator_ptr) {
        HOLOSCAN_LOG_ERROR(
            "MessageLabel::get_all_path_names - Operator pointer is null. Path until now: {}.",
            fmt::to_string(pathstring));
      } else {
        fmt::format_to(std::back_inserter(pathstring), "{},", oplabel.operator_ptr->name());
      }
    }
    pathstring.resize(pathstring.size() - 1);
    all_paths.push_back(fmt::to_string(pathstring));
  }
  return all_paths;
}

void MessageLabel::add_new_op_timestamp(holoscan::OperatorTimestampLabel o_timestamp) {
  if (message_paths.empty()) {
    // By default, allocate space for DEFAULT_PATH_LENGTH Operators in a path
    TimestampedPath newPath;
    newPath.reserve(DEFAULT_PATH_LENGTH);

    message_paths.push_back(newPath);
    message_paths[0].push_back(o_timestamp);
  } else {
    for (auto& m : message_paths) {
      // By default, allocate space for DEFAULT_PATH_LENGTH Operators in a path
      if (m.capacity() < DEFAULT_PATH_LENGTH) m.reserve(DEFAULT_PATH_LENGTH);
      m.push_back(o_timestamp);
    }
  }
}

void MessageLabel::update_last_op_publish() {
  for (auto& path : message_paths) { path.back().pub_timestamp = get_current_time_us(); }
}

MessageLabel::TimestampedPath MessageLabel::get_path(int index) {
  return message_paths[index];
}

OperatorTimestampLabel& MessageLabel::get_operator(int path_index, int op_index) {
  return message_paths[path_index][op_index];
}

void MessageLabel::set_operator_pub_timestamp(int path_index, int op_index, int64_t pub_timestamp) {
  message_paths[path_index][op_index].pub_timestamp = pub_timestamp;
}

void MessageLabel::set_operator_rec_timestamp(int path_index, int op_index, int64_t rec_timestamp) {
  message_paths[path_index][op_index].rec_timestamp = rec_timestamp;
}

}  // namespace holoscan
