/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/logger/spdlog_logger.hpp"

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace spdlog {

namespace sinks {

template <typename ConsoleMutex>
class ansicolor_file_sink : public ansicolor_sink<ConsoleMutex> {
 public:
  explicit ansicolor_file_sink(FILE* file, color_mode mode = color_mode::automatic)
      : ansicolor_sink<ConsoleMutex>(file, mode) {}
};

}  // namespace sinks

static inline std::shared_ptr<logger> create_file_logger(const std::string name, FILE* file) {
  // Do not register to spdlog registry
  spdlog::details::registry::instance().set_automatic_registration(false);

  return spdlog::synchronous_factory::template create<
      spdlog::sinks::ansicolor_file_sink<spdlog::details::console_mutex>>(
      std::move(name), file, spdlog::color_mode::automatic);
}

}  // namespace spdlog

namespace nvidia {

/// Namespace for the NVIDIA logger functionality.
namespace logger {

static void ensure_log_level(int level) {
  if (level < 0 || level > spdlog::level::n_levels - 2) {
    std::fprintf(stderr, "SpdlogLogger: Invalid log level %d\n", level);
    std::abort();
  }
}

/// Default spdlog Logger implementation.
class DefaultSpdlogLogger : public ILogger {
 public:
  DefaultSpdlogLogger(std::string& name, std::string& pattern, int& level,
                      std::vector<void*>& sinks);
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg = nullptr) override;

  void pattern(const char* pattern) override;
  const char* pattern() const override;

  void level(int level) override;
  int level() const override;

  void redirect(int level, void* output) override;
  void* redirect(int level) const override;

 protected:
  std::string& name_;                 ///< logger name
  std::string& pattern_;              ///< log pattern
  int& level_;                        ///< log level
  std::vector<void*>& sinks_;         ///< log sinks
  std::shared_ptr<void> loggers_[6];  ///< spdlog loggers
};

std::string& SpdlogLogger::pattern_string() {
  return pattern_;
}

SpdlogLogger::SpdlogLogger(const char* name, const std::shared_ptr<ILogger>& logger,
                           const LogFunction& func)
    : Logger(logger, func), name_(name) {
  if (logger_ == nullptr && func_ == nullptr) {
    logger_ = std::make_shared<DefaultSpdlogLogger>(name_, pattern_, level_, sinks_);
  }

  // Set default sinks (stderr)
  for (int level = spdlog::level::n_levels - 2; level >= 0; --level) { redirect(level, stderr); }

  // Set default log level and pattern
  level(spdlog::level::info);
  pattern("[%^%l%$] [%s:%#] %v");
}

DefaultSpdlogLogger::DefaultSpdlogLogger(std::string& name, std::string& pattern, int& level,
                                         std::vector<void*>& sinks)
    : name_(name), pattern_(pattern), level_(level), sinks_(sinks) {}

void DefaultSpdlogLogger::log(const char* file, int line, const char* name, int level,
                              const char* log, void*) {
  auto logger = std::static_pointer_cast<spdlog::logger>(loggers_[level]);
  if (logger) {
    if (file != nullptr) {
      logger->log(
          spdlog::source_loc{file, line, name}, static_cast<spdlog::level::level_enum>(level), log);
    } else {
      logger->log(static_cast<spdlog::level::level_enum>(level), log);
    }
  }
}

void DefaultSpdlogLogger::pattern(const char* pattern) {
  std::shared_ptr<spdlog::logger> old_logger;
  for (int level = spdlog::level::n_levels - 2; level >= 0; --level) {
    auto logger = std::static_pointer_cast<spdlog::logger>(loggers_[level]);
    if (old_logger != logger) {
      old_logger = logger;
      logger->set_pattern(pattern);
    }
  }
}

const char* DefaultSpdlogLogger::pattern() const {
  return pattern_.c_str();
}

void DefaultSpdlogLogger::level(int level) {
  std::shared_ptr<spdlog::logger> old_logger;
  for (int lv = spdlog::level::n_levels - 2; lv >= 0; --lv) {
    auto logger = std::static_pointer_cast<spdlog::logger>(loggers_[lv]);
    if (old_logger != logger) {
      old_logger = logger;
      logger->set_level(static_cast<spdlog::level::level_enum>(level));
    }
  }
}

int DefaultSpdlogLogger::level() const {
  return level_;
}

void DefaultSpdlogLogger::redirect(int level, void* output) {
  ensure_log_level(level);

  bool logger_exists = false;
  std::shared_ptr<spdlog::logger> logger;
  for (int lv = spdlog::level::n_levels - 2; lv >= 0; --lv) {
    if (sinks_[lv] == output) {
      logger_exists = true;
      logger = std::static_pointer_cast<spdlog::logger>(loggers_[lv]);
      break;
    }
  }

  if (!logger_exists && output != nullptr) {
    logger = spdlog::create_file_logger(name_, reinterpret_cast<std::FILE*>(output));
    // Set pattern and level for the new logger
    logger->set_pattern(pattern_);
    logger->set_level(static_cast<spdlog::level::level_enum>(level_));
  }
  loggers_[level] = logger;
}

void* DefaultSpdlogLogger::redirect(int level) const {
  return sinks_[level];
}

}  // namespace logger

}  // namespace nvidia
