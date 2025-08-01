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

#include <fmt/format.h>
#include <cmath>  // for M_PI
#include <memory>

#include <holoscan/holoscan.hpp>
#include <holoscan/pose_tree/math/pose3.hpp>         // Pose3d declaration
#include <holoscan/pose_tree/pose_tree_manager.hpp>  // for PoseTreeManager

constexpr double TWO_PI = 2.0 * M_PI;
constexpr double kDay = 24.0 * 3600.0;  // 1 simulated day in seconds

// fmt support for holoscan::Pose3d -------------------------------------------
namespace fmt {
template <>
// NOLINTNEXTLINE(altera-struct-pack-align)
struct formatter<holoscan::Pose3d> {
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const holoscan::Pose3d& p, FormatContext& ctx) const {
    const auto& t = p.translation;
    const auto& q = p.rotation.quaternion();
    return format_to(ctx.out(),
                     "T=({:.3f},{:.3f},{:.3f}) "
                     "Q=({:.3f},{:.3f},{:.3f},{:.3f})",
                     t.x(),
                     t.y(),
                     t.z(),  // translation
                     q.x(),
                     q.y(),
                     q.z(),
                     q.w());  // quaternion
  }
};
}  // namespace fmt
// ---------------------------------------------------------------------------

// ─────────────────── 1. OrbitSetterOp ─────────────────────────────────────────
class OrbitSetterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OrbitSetterOp)

  OrbitSetterOp() = default;

  void initialize() override {
    Operator::initialize();
    pose_tree_ = service<holoscan::PoseTreeManager>("pose_tree_manager")->tree();

    // Create frames & edges once.
    pose_tree_->create_frame("sun");
    pose_tree_->create_frame("earth");
    pose_tree_->create_frame("moon");
    pose_tree_->create_edges("sun", "earth");
    pose_tree_->create_edges("earth", "moon");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& ctx) override {
    constexpr double t_earth = 365.25 * 24 * 3600;  // orbital period (s)
    constexpr double t_moon = 27.32 * 24 * 3600;
    constexpr double r_earth = 1.0;              // 1 AU (scaled units)
    constexpr double r_moon = 384400.0 / 1.5e8;  // ~0.00256 AU

    // Advance simulation by one day
    sim_time_ += kDay;
    double now = sim_time_;  // store poses at simulated time

    double theta_e = TWO_PI * (sim_time_ / t_earth);
    double theta_m = TWO_PI * (sim_time_ / t_moon);

    // Sun → Earth pose (translation + rotation about Z)
    holoscan::Vector3d earth_pos(r_earth * std::cos(theta_e), r_earth * std::sin(theta_e), 0.0);
    constexpr double half_pi = M_PI / 2.0;
    holoscan::SO3d earth_rot = holoscan::SO3d::from_axis_angle({0.0, 0.0, 1.0}, theta_e + half_pi);
    pose_tree_->set("sun", "earth", now, {earth_rot, earth_pos});

    // Earth → Moon pose
    holoscan::Vector3d moon_pos(r_moon * std::cos(theta_m), r_moon * std::sin(theta_m), 0.0);
    holoscan::SO3d moon_rot = holoscan::SO3d::from_axis_angle({0.0, 0.0, 1.0}, theta_m);
    pose_tree_->set("earth", "moon", now, {moon_rot, moon_pos});
  }

 private:
  double sim_time_ = 0.0;  // advances by kDay every tick
  std::shared_ptr<holoscan::PoseTree> pose_tree_;
};

// ─────────────────── 2. TransformPrinterOp ────────────────────────────────────
class TransformPrinterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TransformPrinterOp)

  TransformPrinterOp() = default;

  void initialize() override {
    Operator::initialize();
    pose_tree_ = service<holoscan::PoseTreeManager>("pose_tree_manager")->tree();
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Advance our local simulation clock by one day
    sim_time_ += kDay;
    double now = sim_time_;

    auto sun_to_earth = pose_tree_->get("sun", "earth", now).value();
    auto earth_to_moon = pose_tree_->get("earth", "moon", now).value();
    auto sun_to_moon = pose_tree_->get("sun", "moon", now).value();

    double day = now / kDay;
    HOLOSCAN_LOG_INFO("[day {:.0f}]  Sun → Earth  : {}", day, sun_to_earth);
    HOLOSCAN_LOG_INFO("[day {:.0f}]  Earth → Moon : {}", day, earth_to_moon);
    HOLOSCAN_LOG_INFO("[day {:.0f}]  Sun → Moon   : {}", day, sun_to_moon);

    // Highlight interpolation: query pose at an intermediate time (12 hours ago)
    if (std::abs(day - 365.0) < 1e-6) {
      double interpolated_time = now - kDay / 2.0;
      auto sun_to_moon_interp = pose_tree_->get("sun", "moon", interpolated_time).value();
      HOLOSCAN_LOG_INFO(
          "[day {:.1f}]  Sun → Moon   : {} (interpolated)", day - 0.5, sun_to_moon_interp);
    }
  }

 private:
  std::shared_ptr<holoscan::PoseTree> pose_tree_;
  double sim_time_ = 0.0;  // matches OrbitSetterOp's clock
};

class PoseTreeOrbitApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // In a Holoscan application, a PoseTree is managed by the PoseTreeManager resource, which
    // acts as a FragmentService. This allows multiple operators within the same fragment to
    // share access to the same pose data, ensuring consistency.
    auto pose_tree_manager = make_resource<PoseTreeManager>(
        "pose_tree_manager",
        // Parameters to initialize the underlying PoseTree can be configured here.
        // These values are for demonstration; production apps should tune them.
        from_config("pose_tree_config"),
        // You can override some values
        // (see `pose_tree_basic.yaml` for the default values)
        Arg("number_frames", static_cast<int32_t>(64)));
    register_service(pose_tree_manager);

    // Operators
    auto orbit_setter_op =
        make_operator<OrbitSetterOp>("orbit_setter_op", make_condition<CountCondition>(365));
    auto transform_printer_op = make_operator<TransformPrinterOp>("transform_printer_op");

    // Connect – single edge is enough
    add_flow(orbit_setter_op, transform_printer_op);
  }
};

int main(int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("pose_tree_basic.yaml");
  if (argc >= 2) {
    config_path = argv[1];
  }

  auto app = holoscan::make_application<PoseTreeOrbitApp>();
  app->config(config_path);
  app->run();

  return 0;
}
