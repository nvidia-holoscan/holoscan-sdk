/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <imgui.h>

#include <holoviz/holoviz.hpp>
#include <util/unique_value.hpp>
#include "headless_fixture.hpp"

namespace viz = holoscan::viz;

using UniqueImGuiContext =
    viz::UniqueValue<ImGuiContext*, decltype(&ImGui::DestroyContext), &ImGui::DestroyContext>;

// Fixture that initializes Holoviz
class ImGuiLayer : public TestHeadless {
 protected:
  ImGuiLayer() : TestHeadless(256, 256) {}

  void SetUp() override {
    if (!ImGui::GetCurrentContext()) {
      im_gui_context_.reset(ImGui::CreateContext());
      ASSERT_TRUE(im_gui_context_);
    }
    ASSERT_NO_THROW(viz::ImGuiSetCurrentContext(ImGui::GetCurrentContext()));

    // call base class
    ::TestHeadless::SetUp();
  }

  void TearDown() override {
    // call base class
    ::TestHeadless::TearDown();

    ASSERT_NO_THROW(viz::ImGuiSetCurrentContext(nullptr));
  }

  UniqueImGuiContext im_gui_context_;
};

TEST_F(ImGuiLayer, Window) {
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::BeginImGuiLayer());

    ImGui::Begin("Window");
    ImGui::Text("Some Text");
    ImGui::End();

    EXPECT_NO_THROW(viz::EndLayer());
    EXPECT_NO_THROW(viz::End());
  }

  CompareResultCRC32({
    0xb374795b,  // RTX 6000, RTX A5000
    0x40899a2e   // RTX A6000
    });
}

TEST_F(ImGuiLayer, Errors) {
  std::vector<float> data{0.5f, 0.5f};

  EXPECT_NO_THROW(viz::Begin());

  // it's an error to call call BeginImGuiLayer no valid ImGui context is set
  EXPECT_NO_THROW(viz::ImGuiSetCurrentContext(nullptr));
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::ImGuiSetCurrentContext(im_gui_context_.get()));

  // it's an error to call BeginImGuiLayer again without calling EndLayer
  EXPECT_NO_THROW(viz::BeginImGuiLayer());
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);
  EXPECT_NO_THROW(viz::EndLayer());

  // multiple ImGui layers per frame are not supported
  EXPECT_THROW(viz::BeginImGuiLayer(), std::runtime_error);

  EXPECT_NO_THROW(viz::End());
}
