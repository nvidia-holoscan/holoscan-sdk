/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/std_ops.hpp>

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    if (is_aja_source_) {
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));
    }

    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");
    auto plax_cham_resized = make_operator<ops::FormatConverterOp>("plax_cham_resized",
                                                                   from_config("plax_cham_resized"),
                                                                   Arg("in_dtype") = in_dtype,
                                                                   Arg("pool") = pool_resource);

    auto plax_cham_pre = make_operator<ops::FormatConverterOp>("plax_cham_pre",
                                                               from_config("plax_cham_pre"),
                                                               Arg("in_dtype") = in_dtype,
                                                               Arg("pool") = pool_resource);

    auto aortic_ste_pre = make_operator<ops::FormatConverterOp>("aortic_ste_pre",
                                                                from_config("aortic_ste_pre"),
                                                                Arg("in_dtype") = in_dtype,
                                                                Arg("pool") = pool_resource);

    auto b_mode_pers_pre = make_operator<ops::FormatConverterOp>("b_mode_pers_pre",
                                                                 from_config("b_mode_pers_pre"),
                                                                 Arg("in_dtype") = in_dtype,
                                                                 Arg("pool") = pool_resource);

    auto multiai_inference = make_operator<ops::MultiAIInferenceOp>(
        "multiai_inference", from_config("multiai_inference"), Arg("allocator") = pool_resource);

    auto multiai_postprocessor =
        make_operator<ops::MultiAIPostprocessorOp>("multiai_postprocessor",
                                                   from_config("multiai_postprocessor"),
                                                   Arg("allocator") = pool_resource);

    auto visualizer_icardio = make_operator<ops::VisualizerICardioOp>(
        "visualizer_icardio", from_config("visualizer_icardio"), Arg("allocator") = pool_resource);

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz", from_config("holoviz"), Arg("allocator") = pool_resource);

    // Flow definition

    if (is_aja_source_) {
      const std::set<std::pair<std::string, std::string>> aja_ports = {{"video_buffer_output", ""}};
      add_flow(source, plax_cham_resized, aja_ports);
      add_flow(source, plax_cham_pre, aja_ports);
      add_flow(source, aortic_ste_pre, aja_ports);
      add_flow(source, b_mode_pers_pre, aja_ports);
    } else {
      add_flow(source, plax_cham_resized);
      add_flow(source, plax_cham_pre);
      add_flow(source, aortic_ste_pre);
      add_flow(source, b_mode_pers_pre);
    }

    add_flow(plax_cham_resized, holoviz, {{"", "receivers"}});

    add_flow(plax_cham_pre, multiai_inference, {{"", "receivers"}});
    add_flow(aortic_ste_pre, multiai_inference, {{"", "receivers"}});
    add_flow(b_mode_pers_pre, multiai_inference, {{"", "receivers"}});

    add_flow(multiai_inference, multiai_postprocessor, {{"transmitter", "receivers"}});
    add_flow(multiai_postprocessor, visualizer_icardio, {{"transmitter", "receivers"}});

    add_flow(visualizer_icardio, holoviz, {{"keypoints", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_1", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_2", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_3", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_4", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_5", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"lines", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"logo", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  app->run();

  return 0;
}
