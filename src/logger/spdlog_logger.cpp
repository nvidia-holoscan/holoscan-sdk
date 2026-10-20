/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Array subscript access to loggers_ by log level is intentional for performance in hot paths.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)

#include <holoscan/logger/spdlog_logger.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdio>
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

static inline std::shared_ptr<logger> create_file_logger(const std::string& name, FILE* file) {
  // Do not register to spdlog registry
  spdlog::details::registry::instance().set_automatic_registration(false);

  return spdlog::synchronous_factory::template create<
      spdlog::sinks::ansicolor_file_sink<spdlog::details::console_mutex>>(
      name, file, spdlog::color_mode::automatic);
}

}  // namespace spdlog

namespace holoscan::logger {

constexpr int kMaxSeverity = 255;

// ----------------------------------------------------------------------------
// Logger (custom-logger / log-function dispatch base)
// ----------------------------------------------------------------------------

Logger::Logger(std::shared_ptr<ILogger> logger) : logger_(std::move(logger)) {}

Logger::Logger(LogFunction func) : func_(std::move(func)) {}

Logger::Logger(std::shared_ptr<ILogger> logger, LogFunction func)
    : logger_(std::move(logger)), func_(std::move(func)) {}

void Logger::log(const char* file, int line, const char* name, int level, const char* message) {
  // func_ (LogFunction) has higher priority than logger_ (ILogger)
  if (func_) {
    func_(file, line, name, level, message, func_arg_);
  } else if (logger_) {
    logger_->log(file, line, name, level, message, nullptr);
  }
}

void Logger::logger(std::shared_ptr<ILogger> logger) {
  logger_ = std::move(logger);
}

std::shared_ptr<ILogger> Logger::logger() const {
  return logger_;
}

void Logger::func(LogFunction log_func, void* arg) {
  func_ = std::move(log_func);
  func_arg_ = arg;
}

LogFunction Logger::func() const {
  return func_;
}

void* Logger::arg() const {
  return func_arg_;
}

void Logger::pattern(const char* pattern) {
  if (logger_) {
    logger_->pattern(pattern);
  }
  pattern_ = pattern;
}

const char* Logger::pattern() const {
  if (logger_) {
    return logger_->pattern();
  }
  return pattern_.c_str();
}

void Logger::level(int level) {
  if (logger_) {
    logger_->level(level);
  }
  level_ = level;
}

int Logger::level() const {
  if (logger_) {
    return logger_->level();
  }
  return level_;
}

void Logger::redirect(int level, void* output) {
  bool valid = false;
  if (level >= 0 && level <= kMaxSeverity) {
    // Resize the sinks_ vector if necessary
    if (static_cast<int>(sinks_.size()) <= level) {
      sinks_.resize(level + 1);
    }
    valid = true;
  }
  if (logger_) {
    logger_->redirect(level, output);
  }
  if (valid) {
    sinks_[level] = output;
  }
}

void* Logger::redirect(int level) const {
  if (logger_) {
    return logger_->redirect(level);
  }
  if (level >= 0 && level < static_cast<int>(sinks_.size())) {
    return sinks_[level];
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
// DefaultSpdlogLogger (default spdlog-backed ILogger implementation)
// ----------------------------------------------------------------------------

static void ensure_log_level(int level) {
  if (level < 0 || level > spdlog::level::n_levels - 2) {
    std::fprintf(stderr, "SpdlogLogger: Invalid log level %d\n", level);
    std::abort();
  }
}

class DefaultSpdlogLogger : public ILogger {
 public:
  DefaultSpdlogLogger(std::string& name, std::string& pattern, int& level,
                      std::vector<void*>& sinks);
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg) override;

  void pattern(const char* pattern) override;
  [[nodiscard]] const char* pattern() const override;

  void level(int level) override;
  [[nodiscard]] int level() const override;

  void redirect(int level, void* output) override;
  [[nodiscard]] void* redirect(int level) const override;

 protected:
  static constexpr std::size_t kLoggerCount = spdlog::level::n_levels - 1;

  std::string& name_;                                        ///< logger name
  std::string& pattern_;                                     ///< log pattern
  int& level_;                                               ///< log level
  std::vector<void*>& sinks_;                                ///< log sinks
  std::array<std::shared_ptr<void>, kLoggerCount> loggers_;  ///< spdlog loggers
};

std::string& SpdlogLogger::pattern_string() {
  return pattern_;
}

SpdlogLogger::SpdlogLogger(const char* name, std::shared_ptr<ILogger> logger, LogFunction func)
    : Logger(std::move(logger), std::move(func)), name_(name) {
  if (logger_ == nullptr && func_ == nullptr) {
    logger_ = std::make_shared<DefaultSpdlogLogger>(name_, pattern_, level_, sinks_);
  }

  // Set default sinks (stderr)
  for (int level = spdlog::level::n_levels - 2; level >= 0; --level) {
    redirect(level, stderr);
  }

  // Set default log level and pattern
  level(spdlog::level::info);
  pattern("[%^%l%$] [%s:%#] %v");
}

DefaultSpdlogLogger::DefaultSpdlogLogger(std::string& name, std::string& pattern, int& level,
                                         std::vector<void*>& sinks)
    : name_(name), pattern_(pattern), level_(level), sinks_(sinks) {}

void DefaultSpdlogLogger::log(const char* file, int line, const char* name, int level,
                              const char* message, [[maybe_unused]] void* arg) {
  auto logger = std::static_pointer_cast<spdlog::logger>(loggers_[level]);
  if (logger != nullptr) {
    if (file != nullptr) {
      logger->log(spdlog::source_loc{file, line, name},
                  static_cast<spdlog::level::level_enum>(level),
                  message);
    } else {
      logger->log(static_cast<spdlog::level::level_enum>(level), message);
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

}  // namespace holoscan::logger

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
