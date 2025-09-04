/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/test_ops/tx_dtype_test.hpp"

#include <any>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace holoscan::ops {

void DataTypeTxTestOp::setup(OperatorSpec& spec) {
  spec.output<std::any>("out");

  spec.param(
      data_type_,
      "data_type",
      "data type for the tensor elements",
      "Must be one of {'bool', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t',"
      "'uint32_t', 'uint64_t', 'float', 'double', 'complex<float>', 'complex<double>', "
      "'std::string'} or a std::vector<T> or std::vector<std::vector<T>> of any of the above",
      std::string{"double"});
}

void DataTypeTxTestOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                               [[maybe_unused]] ExecutionContext& context) {
  using namespace std;

  auto dtype = data_type_.get();
  if (dtype == "bool") {
    op_output.emit(true);
  } else if (dtype == "int8_t") {
    op_output.emit(int8_t(1));
  } else if (dtype == "int16_t") {
    op_output.emit(int16_t(1));
  } else if (dtype == "int32_t") {
    op_output.emit(int32_t(1));
  } else if (dtype == "int64_t") {
    op_output.emit(int64_t(1));
  } else if (dtype == "int") {
    op_output.emit(1);
  } else if (dtype == "long") {
    op_output.emit(1L);
  } else if (dtype == "long long") {
    op_output.emit(1LL);
  } else if (dtype == "uint8_t") {
    op_output.emit(uint8_t(1));
  } else if (dtype == "uint16_t") {
    op_output.emit(uint16_t(1));
  } else if (dtype == "uint32_t") {
    op_output.emit(uint32_t(1));
  } else if (dtype == "uint64_t") {
    op_output.emit(uint64_t(1));
  } else if (dtype == "float") {
    op_output.emit(float(1));
  } else if (dtype == "double") {
    op_output.emit(double(1));
  } else if (dtype == "std::string") {
    op_output.emit(string("test-string"));
  } else if (dtype == "std::complex<float>") {
    op_output.emit(complex<float>(1, 1));
  } else if (dtype == "std::complex<double>") {
    op_output.emit(complex<double>(1, 1));
  } else if (dtype == "std::nullptr_t") {
    op_output.emit(nullptr);
    // Vector types
  } else if (dtype == "std::vector<bool>") {
    op_output.emit(vector<bool>{true, false, true});
  } else if (dtype == "std::vector<int8_t>") {
    op_output.emit(vector<int8_t>{1, 2, 3});
  } else if (dtype == "std::vector<int16_t>") {
    op_output.emit(vector<int16_t>{1, 2, 3});
  } else if (dtype == "std::vector<int32_t>") {
    op_output.emit(vector<int32_t>{1, 2, 3});
  } else if (dtype == "std::vector<int64_t>") {
    op_output.emit(vector<int64_t>{1, 2, 3});
  } else if (dtype == "std::vector<uint8_t>") {
    op_output.emit(vector<uint8_t>{1, 2, 3});
  } else if (dtype == "std::vector<uint16_t>") {
    op_output.emit(vector<uint16_t>{1, 2, 3});
  } else if (dtype == "std::vector<uint32_t>") {
    op_output.emit(vector<uint32_t>{1, 2, 3});
  } else if (dtype == "std::vector<uint64_t>") {
    op_output.emit(vector<uint64_t>{1, 2, 3});
  } else if (dtype == "std::vector<float>") {
    op_output.emit(vector<float>{1.0f, 2.0f, 3.0f});
  } else if (dtype == "std::vector<double>") {
    op_output.emit(vector<double>{1.0, 2.0, 3.0});
  } else if (dtype == "std::vector<std::string>") {
    op_output.emit(vector<string>{"test1", "test2", "test3"});
  } else if (dtype == "std::vector<std::complex<float>>") {
    op_output.emit(vector<complex<float>>{{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}});
  } else if (dtype == "std::vector<std::complex<double>>") {
    op_output.emit(vector<complex<double>>{{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}});
    // Vector of vector types
  } else if (dtype == "std::vector<std::vector<bool>>") {
    op_output.emit(vector<vector<bool>>{{true, false}, {false, true}});
  } else if (dtype == "std::vector<std::vector<int8_t>>") {
    op_output.emit(vector<vector<int8_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<int16_t>>") {
    op_output.emit(vector<vector<int16_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<int32_t>>") {
    op_output.emit(vector<vector<int32_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<int64_t>>") {
    op_output.emit(vector<vector<int64_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<uint8_t>>") {
    op_output.emit(vector<vector<uint8_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<uint16_t>>") {
    op_output.emit(vector<vector<uint16_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<uint32_t>>") {
    op_output.emit(vector<vector<uint32_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<uint64_t>>") {
    op_output.emit(vector<vector<uint64_t>>{{1, 2}, {3, 4}});
  } else if (dtype == "std::vector<std::vector<float>>") {
    op_output.emit(vector<vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
  } else if (dtype == "std::vector<std::vector<double>>") {
    op_output.emit(vector<vector<double>>{{1.0, 2.0}, {3.0, 4.0}});
  } else if (dtype == "std::vector<std::vector<std::string>>") {
    op_output.emit(vector<vector<string>>{{"test1", "test2"}, {"test3", "test4"}});
  } else if (dtype == "std::vector<std::vector<std::complex<float>>>") {
    op_output.emit(
        vector<vector<complex<float>>>{{{1.0f, 1.0f}, {2.0f, 2.0f}}, {{3.0f, 3.0f}, {4.0f, 4.0f}}});
  } else if (dtype == "std::vector<std::vector<std::complex<double>>>") {
    op_output.emit(
        vector<vector<complex<double>>>{{{1.0, 1.0}, {2.0, 2.0}}, {{3.0, 3.0}, {4.0, 4.0}}});
    // Unordered map
  } else if (dtype == "std::unordered_map<std::string, std::string>") {
    op_output.emit(unordered_map<string, string>{{"key1", "value1"}, {"key2", "value2"}});
    // Shared pointer to basic types types
  } else if (dtype == "std::shared_ptr<std::string>") {
    op_output.emit(make_shared<string>("test-string"));
  } else if (dtype == "std::shared_ptr<float>") {
    op_output.emit(make_shared<float>(1.0f));
  } else if (dtype == "std::shared_ptr<double>") {
    op_output.emit(make_shared<double>(1.0));
  } else if (dtype == "std::shared_ptr<bool>") {
    op_output.emit(make_shared<bool>(true));
  } else if (dtype == "std::shared_ptr<int8_t>") {
    op_output.emit(make_shared<int8_t>(1));
  } else if (dtype == "std::shared_ptr<int16_t>") {
    op_output.emit(make_shared<int16_t>(1));
  } else if (dtype == "std::shared_ptr<int32_t>") {
    op_output.emit(make_shared<int32_t>(1));
  } else if (dtype == "std::shared_ptr<int64_t>") {
    op_output.emit(make_shared<int64_t>(1));
  } else if (dtype == "std::shared_ptr<uint8_t>") {
    op_output.emit(make_shared<uint8_t>(1));
  } else if (dtype == "std::shared_ptr<uint16_t>") {
    op_output.emit(make_shared<uint16_t>(1));
  } else if (dtype == "std::shared_ptr<uint32_t>") {
    op_output.emit(make_shared<uint32_t>(1));
  } else if (dtype == "std::shared_ptr<uint64_t>") {
    op_output.emit(make_shared<uint64_t>(1));
  } else if (dtype == "std::shared_ptr<std::complex<float>>") {
    op_output.emit(make_shared<complex<float>>(1.0f, 1.0f));
  } else if (dtype == "std::shared_ptr<std::complex<double>>") {
    op_output.emit(make_shared<complex<double>>(1.0, 1.0));
    // Vector of basic types
  } else if (dtype == "std::shared_ptr<std::vector<bool>>") {
    auto vec = vector<bool>{true, false, true};
    op_output.emit(make_shared<vector<bool>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<int8_t>>") {
    auto vec = vector<int8_t>{1, 2, 3};
    op_output.emit(make_shared<vector<int8_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<int16_t>>") {
    auto vec = vector<int16_t>{1, 2, 3};
    op_output.emit(make_shared<vector<int16_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<int32_t>>") {
    auto vec = vector<int32_t>{1, 2, 3};
    op_output.emit(make_shared<vector<int32_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<int64_t>>") {
    auto vec = vector<int64_t>{1, 2, 3};
    op_output.emit(make_shared<vector<int64_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<uint8_t>>") {
    auto vec = vector<uint8_t>{1, 2, 3};
    op_output.emit(make_shared<vector<uint8_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<uint16_t>>") {
    auto vec = vector<uint16_t>{1, 2, 3};
    op_output.emit(make_shared<vector<uint16_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<uint32_t>>") {
    auto vec = vector<uint32_t>{1, 2, 3};
    op_output.emit(make_shared<vector<uint32_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<uint64_t>>") {
    auto vec = vector<uint64_t>{1, 2, 3};
    op_output.emit(make_shared<vector<uint64_t>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<float>>") {
    auto vec = vector<float>{1.0f, 2.0f, 3.0f};
    op_output.emit(make_shared<vector<float>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<double>>") {
    auto vec = vector<double>{1.0, 2.0, 3.0};
    op_output.emit(make_shared<vector<double>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::string>>") {
    auto vec = vector<string>{"test1", "test2", "test3"};
    op_output.emit(make_shared<vector<string>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::complex<float>>>") {
    auto vec = vector<complex<float>>{{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
    op_output.emit(make_shared<vector<complex<float>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::complex<double>>>") {
    auto vec = vector<complex<double>>{{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    op_output.emit(make_shared<vector<complex<double>>>(move(vec)));
    // Vector of vector types
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<bool>>>") {
    auto vec = vector<vector<bool>>{{true, false}, {false, true}};
    op_output.emit(make_shared<vector<vector<bool>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<int8_t>>>") {
    auto vec = vector<vector<int8_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<int8_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<int16_t>>>") {
    auto vec = vector<vector<int16_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<int16_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<int32_t>>>") {
    auto vec = vector<vector<int32_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<int32_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<int64_t>>>") {
    auto vec = vector<vector<int64_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<int64_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<uint8_t>>>") {
    auto vec = vector<vector<uint8_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<uint8_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<uint16_t>>>") {
    auto vec = vector<vector<uint16_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<uint16_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<uint32_t>>>") {
    auto vec = vector<vector<uint32_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<uint32_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<uint64_t>>>") {
    auto vec = vector<vector<uint64_t>>{{1, 2}, {3, 4}};
    op_output.emit(make_shared<vector<vector<uint64_t>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<float>>>") {
    auto vec = vector<vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}};
    op_output.emit(make_shared<vector<vector<float>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<double>>>") {
    auto vec = vector<vector<double>>{{1.0, 2.0}, {3.0, 4.0}};
    op_output.emit(make_shared<vector<vector<double>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<std::string>>>") {
    auto vec = vector<vector<std::string>>{{"test1", "test2"}, {"test3", "test4"}};
    op_output.emit(make_shared<vector<vector<string>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<std::complex<float>>>>") {
    auto vec =
        vector<vector<complex<float>>>{{{1.0f, 1.0f}, {2.0f, 2.0f}}, {{3.0f, 3.0f}, {4.0f, 4.0f}}};
    op_output.emit(make_shared<vector<vector<complex<float>>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::vector<std::vector<std::complex<double>>>>") {
    auto vec = vector<vector<complex<double>>>{{{1.0, 1.0}, {2.0, 2.0}}, {{3.0, 3.0}, {4.0, 4.0}}};
    op_output.emit(make_shared<vector<vector<complex<double>>>>(move(vec)));
  } else if (dtype == "std::shared_ptr<std::unordered_map<std::string, std::string>>") {
    auto map = unordered_map<string, string>{{"key1", "value1"}, {"key2", "value2"}};
    op_output.emit(make_shared<unordered_map<string, string>>(move(map)));
  } else {
    throw std::invalid_argument("Invalid data type: " + dtype);
  }
}

}  // namespace holoscan::ops
