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

#include <gtest/gtest.h>

#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>
#include <holoscan/core/gxf/gxf_extension_registrar.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>
#include <holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp>

#include "receive_tensor_gxf.hpp"

#include "../env_wrapper.hpp"
#include "distributed_app_fixture.hpp"
#include "utility_apps.hpp"

namespace holoscan {

namespace {

// Define an operator that wraps the GXF Codelet that receives a tensor
// (`nvidia::gxf::test::ReceiveTensor` class in receive_tensor_gxf.hpp)
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(GXFReceiveTensorOp, "nvidia::gxf::test::ReceiveTensor")

class TensorTransmitFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    int32_t rows = 1;
    int32_t columns = 8;
    // GXFReceiveTensorOp expects a device tensor named "tensor"
    std::string tensor_name{"tensor"};
    std::string storage_type{"device"};
    auto tx = make_operator<ops::PingTensorTxOp>("tx",
                                                 Arg("rows", rows),
                                                 Arg("columns", columns),
                                                 Arg("tensor_name", tensor_name),
                                                 Arg("storage_type", storage_type),
                                                 make_condition<CountCondition>(5));
    add_operator(tx);
  }
};

class GXFTensorReceiveFragment : public holoscan::Fragment {
 public:
  void register_gxf_codelets() {
    HOLOSCAN_LOG_INFO("REGISTERING GXF CODELETS FOR APP");
    gxf_context_t context = executor().context();

    holoscan::gxf::GXFExtensionRegistrar extension_factory(
        context, "TensorReceiver", "Extension for sending and receiving tensors");

    extension_factory.add_component<::nvidia::gxf::test::ReceiveTensor, ::nvidia::gxf::Codelet>(
        "ReceiveTensor class");

    if (!extension_factory.register_extension()) {
      HOLOSCAN_LOG_ERROR("Failed to register GXF Codelet");
      return;
    } else {
      HOLOSCAN_LOG_INFO("SUCCESS IN GXF CODELET REGISTRATION");
    }
  }

  void compose() override {
    using namespace holoscan;

    register_gxf_codelets();

    auto rx = make_operator<GXFReceiveTensorOp>("rx", make_condition<CountCondition>(5));

    add_operator(rx);
  }
};

// Test application added to cover issue 4638505:
//    Verifies that a GXFOperator in the downstream fragment can receive a tensor sent from the
//    upstream fragment.
class GXFOperatorsDistributedApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto tx_fragment = make_fragment<TensorTransmitFragment>("tx_fragment");
    auto rx_fragment = make_fragment<GXFTensorReceiveFragment>("rx_fragment");

    add_flow(tx_fragment, rx_fragment, {{"tx.out", "rx.signal"}});
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

class DistributedGXFOperatorApps : public DistributedApp {};

TEST_F(DistributedGXFOperatorApps, TestDistributedAppGXFOperatorReceive) {
  auto app = make_application<GXFOperatorsDistributedApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) { HOLOSCAN_LOG_ERROR("Exception: {}", e.what()); }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Failed to access in tensor") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Sent message 5") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
