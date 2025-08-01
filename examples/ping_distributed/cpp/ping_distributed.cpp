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
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp>
#include <holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp>

class Fragment1 : public holoscan::Fragment {
 public:
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  Fragment1(bool gpu_tensor, int64_t count, int32_t batch_size, int32_t rows, int32_t columns,
            int32_t channels, std::string data_type)
      : gpu_tensor_(gpu_tensor),
        batch_size_(batch_size),
        count_(count),
        rows_(rows),
        columns_(columns),
        channels_(channels),
        data_type_(std::move(data_type)) {}

  void compose() override {
    using namespace holoscan;

    auto tx = make_operator<ops::PingTensorTxOp>(
        "tx",
        make_condition<CountCondition>(count_),
        Arg("storage_type", std::string{gpu_tensor_ ? "device" : "system"}),
        Arg("batch_size", batch_size_),
        Arg("rows", rows_),
        Arg("columns", columns_),
        Arg("channels", channels_),
        Arg("data_type", data_type_));
    add_operator(tx);
  }

 private:
  bool gpu_tensor_ = false;
  int64_t count_ = 10;
  int32_t batch_size_ = 0;
  int32_t rows_ = 32;
  int32_t columns_ = 64;
  int32_t channels_ = 0;
  std::string data_type_{"uint8_t"};
};

class Fragment2 : public holoscan::Fragment {
 public:
  Fragment2() = default;

  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<ops::PingTensorRxOp>("rx");
    add_operator(rx);
  }
};

class App : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void set_options(bool gpu_tensor = false, int64_t count = 10, int32_t batch_size = 0,
                   int32_t rows = 32, int32_t columns = 1024, int32_t channels = 0,
                   const std::string& data_type = "uint8_t") {
    HOLOSCAN_LOG_INFO("Configuring application to use {} tensors", gpu_tensor ? "GPU" : "host");
    gpu_tensor_ = gpu_tensor;
    count_ = count;
    batch_size_ = batch_size;
    rows_ = rows;
    columns_ = columns;
    channels_ = channels;
    data_type_ = data_type;
  }

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<Fragment1>(
        "fragment1", gpu_tensor_, count_, batch_size_, rows_, columns_, channels_, data_type_);
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    // Connect the two fragments (tx.out -> rx.in)
    // We can skip the "out" and "in" suffixes, as they are the default
    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }

 private:
  bool gpu_tensor_ = false;
  int64_t count_ = 10;
  int32_t batch_size_ = 0;
  int32_t rows_ = 32;
  int32_t columns_ = 64;
  int32_t channels_ = 0;
  std::string data_type_{"uint8_t"};
};

std::optional<bool> get_boolean_arg(std::vector<std::string> args, const std::string& name) {
  if (std::find(args.begin(), args.end(), name) != std::end(args)) {
    return true;
  }
  return {};
}

std::optional<int32_t> get_int32_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    try {
      return std::stoi(*loc);
    } catch (std::exception& e) {
      HOLOSCAN_LOG_ERROR("Unable to parse provided argument '{}'", name);
      return {};
    }
  }
  return {};
}
std::optional<int64_t> get_int64_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    try {
      return std::stoll(*loc);
    } catch (std::exception& e) {
      HOLOSCAN_LOG_ERROR("Unable to parse provided argument '{}'", name);
      return {};
    }
  }
  return {};
}

std::optional<std::string> get_str_arg(std::vector<std::string> args, const std::string& name) {
  auto loc = std::find(args.begin(), args.end(), name);
  if ((loc != std::end(args)) && (++loc != std::end(args))) {
    return *loc;
  }
  return {};
}

int main() {
  // Print info on the additional command line options supported by this app.
  // Note: -h and --help are intercepted by the Application class to handle the built-in options
  //       for distributed apps.
  std::cout << "Additional supported arguments: \n"
            << "  --gpu               If specified, tensors will be in GPU memory.\n"
            << "  --count COUNT       The number of times to send the tensor.\n"
            << "  --batch_size BATCH  The batch size of the tensor (dimension omitted if 0).\n"
            << "  --rows ROWS         The number of rows in the tensor.\n"
            << "  --columns COLUMNS   The number of columns (dimension omitted if 0).\n"
            << "  --channels CHANNELS The number of channels (dimension omitted if 0).\n"
            << "  --data_type TYPE    The C++ type of the data elements. Must be one of \n"
            << "                      {'int8_t', 'int16_t', 'int32_t', 'int64_t',  \n"
            << "                       'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',  \n"
            << "                       'float', 'double', 'complex<float>', complex<double>'}.\n"
            << "  --track             If specified, data flow tracking will be enabled.\n"
            << std::endl;

  auto app = holoscan::make_application<App>();

  // Parse args that are defined for all applications.
  std::vector<std::string>& remaining_args = app->argv();

  // Parse any additional supported arguments
  bool tensor_on_gpu = get_boolean_arg(remaining_args, "--gpu").value_or(false);
  int64_t count = get_int64_arg(remaining_args, "--count").value_or(10);
  int32_t batch_size = get_int32_arg(remaining_args, "--batch_size").value_or(0);
  int32_t rows = get_int32_arg(remaining_args, "--rows").value_or(32);
  int32_t columns = get_int32_arg(remaining_args, "--columns").value_or(64);
  int32_t channels = get_int32_arg(remaining_args, "--channels").value_or(0);
  std::string data_type = get_str_arg(remaining_args, "--data_type").value_or("uint8_t");
  bool data_flow_tracking_enabled = get_boolean_arg(remaining_args, "--track").value_or(false);

  HOLOSCAN_LOG_INFO("Running ping with tensors on {}.", tensor_on_gpu ? "GPU" : "host");

  // configure tensor on host vs. GPU and set the count and shape
  app->set_options(tensor_on_gpu, count, batch_size, rows, columns, channels, data_type);

  if (data_flow_tracking_enabled) {
    // enable data flow tracking for a distributed app
    auto trackers = app->track_distributed(0, 0, 0);

    // run the application
    app->run();

    // print data flow tracking results
    for (const auto& [name, tracker] : trackers) {
      std::cout << "Fragment: " << name << std::endl;
      tracker->print();
    }
  } else {
    app->run();
  }

  return 0;
}
