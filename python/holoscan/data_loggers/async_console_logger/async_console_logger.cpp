/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <variant>
#include <vector>

#include "./pydoc.hpp"

#include <holoscan/core/arg.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/async_data_logger.hpp>
#include <holoscan/core/resources/data_logger.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/data_loggers/async_console_logger/async_console_logger.hpp>
#include <holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp>
#include "../../core/component_util.hpp"
#include "../../core/gil_guarded_pyobject.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::data_loggers {

class PyAsyncConsoleLogger : public AsyncConsoleLogger {
 public:
  /* Inherit the constructors */
  using AsyncConsoleLogger::AsyncConsoleLogger;

  // Define a constructor that fully initializes the object.
  PyAsyncConsoleLogger(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      std::shared_ptr<SimpleTextSerializer> serializer = nullptr, bool log_inputs = true,
      bool log_outputs = true, bool log_metadata = true, bool log_tensor_data_content = true,
      bool use_scheduler_clock = false,
      std::optional<std::shared_ptr<Resource>> clock = std::nullopt,
      const std::vector<std::string>& allowlist_patterns = {},
      const std::vector<std::string>& denylist_patterns = {}, size_t max_queue_size = 50000,
      int64_t worker_sleep_time = 50000,
      // when loading the enum from the YAML config via **app.kwargs(key) it will become a string.
      // use of std::variant here is to support that case
      std::variant<AsyncQueuePolicy, std::string> queue_policy = AsyncQueuePolicy::kReject,
      size_t large_data_max_queue_size = 1000, int64_t large_data_worker_sleep_time = 200000,
      std::variant<AsyncQueuePolicy, std::string> large_data_queue_policy =
          AsyncQueuePolicy::kReject,
      bool enable_large_data_queue = true, int64_t shutdown_wait_period_ms = -1,
      std::variant<DataLoggerQueueType, std::string> queue_type = DataLoggerQueueType::LockFree,
      const std::string& name = resource_default_name_v<AsyncConsoleLogger>)
      : AsyncConsoleLogger(
            ArgList{Arg{"log_inputs", log_inputs},
                    Arg{"log_outputs", log_outputs},
                    Arg{"log_metadata", log_metadata},
                    Arg{"log_tensor_data_content", log_tensor_data_content},
                    Arg{"use_scheduler_clock", use_scheduler_clock},
                    Arg{"allowlist_patterns", allowlist_patterns},
                    Arg{"denylist_patterns", denylist_patterns},
                    Arg{"max_queue_size", max_queue_size},
                    Arg{"worker_sleep_time", worker_sleep_time},
                    Arg{"large_data_max_queue_size", large_data_max_queue_size},
                    Arg{"large_data_worker_sleep_time", large_data_worker_sleep_time},
                    Arg{"enable_large_data_queue", enable_large_data_queue},
                    Arg{"shutdown_wait_period_ms", shutdown_wait_period_ms}}) {
    if (serializer) {
      this->add_arg(Arg{"serializer", serializer});
    }
    if (clock.has_value()) {
      this->add_arg(Arg{"clock", clock.value()});
    }
    if (std::holds_alternative<std::string>(queue_policy)) {
      // C++ layer supports YAML::Node -> enum conversion via the registered argument setter
      this->add_arg(Arg("queue_policy", YAML::Node(std::get<std::string>(queue_policy))));
    } else {
      this->add_arg(Arg("queue_policy", std::get<holoscan::AsyncQueuePolicy>(queue_policy)));
    }
    if (std::holds_alternative<std::string>(large_data_queue_policy)) {
      // C++ layer supports YAML::Node -> enum conversion via the registered argument setter
      this->add_arg(Arg("large_data_queue_policy",
                        YAML::Node(std::get<std::string>(large_data_queue_policy))));
    } else {
      this->add_arg(Arg("large_data_queue_policy",
                        std::get<holoscan::AsyncQueuePolicy>(large_data_queue_policy)));
    }
    if (std::holds_alternative<std::string>(queue_type)) {
      // C++ layer supports YAML::Node -> enum conversion via the registered argument setter.
      this->add_arg(Arg("queue_type", YAML::Node(std::get<std::string>(queue_type))));
    } else {
      this->add_arg(Arg("queue_type", std::get<holoscan::DataLoggerQueueType>(queue_type)));
    }
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};
/* The python module */

PYBIND11_MODULE(_async_console_logger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK AsyncConsoleLogger Python Bindings
        -----------------------------------------------
        .. currentmodule:: _async_console_logger
    )pbdoc";

  py::class_<AsyncConsoleLogger,
             PyAsyncConsoleLogger,
             AsyncDataLoggerResource,
             DataLoggerResource,
             std::shared_ptr<AsyncConsoleLogger>>(
      m, "AsyncConsoleLogger", doc::AsyncConsoleLogger::doc_AsyncConsoleLogger)
      .def(py::init<const std::variant<Fragment*, Subgraph*>&,
                    std::shared_ptr<SimpleTextSerializer>,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    std::optional<std::shared_ptr<Resource>>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    size_t,
                    int64_t,
                    std::variant<AsyncQueuePolicy, std::string>,
                    size_t,
                    int64_t,
                    std::variant<AsyncQueuePolicy, std::string>,
                    bool,
                    int64_t,
                    std::variant<DataLoggerQueueType, std::string>,
                    const std::string&>(),
           "fragment"_a,
           "serializer"_a = py::none(),
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_metadata"_a = true,
           "log_tensor_data_content"_a = true,
           "use_scheduler_clock"_a = false,
           "clock"_a = py::none(),
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "max_queue_size"_a = 50000,
           "worker_sleep_time"_a = 50000,
           "queue_policy"_a = AsyncQueuePolicy::kReject,
           "large_data_max_queue_size"_a = 1000,
           "large_data_worker_sleep_time"_a = 50000,
           "large_data_queue_policy"_a = AsyncQueuePolicy::kReject,
           "enable_large_data_queue"_a = true,
           "shutdown_wait_period_ms"_a = -1,
           "queue_type"_a = DataLoggerQueueType::LockFree,
           "name"_a = std::string(resource_default_name_v<AsyncConsoleLogger>),
           doc::AsyncConsoleLogger::doc_AsyncConsoleLogger);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::data_loggers
