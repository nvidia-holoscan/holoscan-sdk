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

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "holoscan/pose_tree/math/pose2.hpp"
#include "holoscan/pose_tree/math/pose3.hpp"
#include "holoscan/pose_tree/math/so2.hpp"
#include "holoscan/pose_tree/math/so3.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_client.hpp"
#include "holoscan/pose_tree/pose_tree_ucx_server.hpp"

namespace py = pybind11;

namespace holoscan {

void init_pose_tree_geometry(py::module_& m) {
  py::class_<SO2d>(m, "SO2")
      .def(py::init<>())  // Default constructor
      .def_static("identity", &SO2d::identity)
      .def_static("from_angle", &SO2d::from_angle)
      .def_static("from_direction",
                  [](const Eigen::Vector2d& direction) { return SO2d::from_direction(direction); })
      .def_static("from_normalized",
                  [](const Eigen::Vector2d& direction) { return SO2d::from_normalized(direction); })
      .def("angle", &SO2d::angle)
      .def("cos", &SO2d::cos)
      .def("sin", &SO2d::sin)
      .def("inverse", &SO2d::inverse)
      .def("matrix", &SO2d::matrix)
      .def("__matmul__", [](const SO2d& lhs, const SO2d& rhs) { return lhs * rhs; })
      .def("__repr__", [](const SO2d& p) {
        std::stringstream ss;
        ss << "SO2(angle=" << p.angle() << ")";
        return ss.str();
      });

  py::class_<Pose2d>(m, "Pose2")
      .def(py::init<>())  // Default constructor
      .def(py::init<const SO2d&,
                    const Eigen::Vector2d&>(),  // Constructor with translation and rotation
           py::arg("rotation"),
           py::arg("translation"))
      .def_static("identity", &Pose2d::identity)
      .def_static(
          "from_translation",
          [](const Eigen::Vector2d& translation) { return Pose2d::from_translation(translation); })
      .def_static("from_rotation", &Pose2d::from_rotation)
      .def_static("from_xy_a", &Pose2d::from_xy_a)
      .def_static("from_matrix", &Pose2d::from_matrix)
      .def_readwrite("translation", &Pose2d::translation)
      .def_readwrite("rotation", &Pose2d::rotation)
      .def_property(
          "angle",
          [](Pose2d& p) { return p.rotation.angle(); },
          [](Pose2d& p, double a) { p.rotation = SO2d::from_angle(a); })
      .def("inverse", &Pose2d::inverse)
      .def("matrix", &Pose2d::matrix)
      .def("__matmul__", [](const Pose2d& lhs, const Pose2d& rhs) { return lhs * rhs; })
      .def("__mul__", [](const Pose2d& lhs, const Eigen::Vector2d& rhs) { return lhs * rhs; })
      .def("__repr__", [](const Pose2d& p) {
        std::stringstream ss;
        ss << "Pose2d(translation=" << p.translation.transpose() << ", ";
        ss << "rotation=" << p.rotation.angle() << ")";
        return ss.str();
      });

  // Bind Pose3d
  py::class_<SO3d>(m, "SO3")
      .def(py::init<>())  // Default constructor
      .def_static("identity", &SO3d::identity)
      .def_static("from_scaled_axis", &SO3d::from_scaled_axis)
      .def_static("from_axis_angle", &SO3d::from_axis_angle)
      .def_static("from_quaternion",
                  [](const Eigen::Vector4d& quaternion) {
                    return SO3d::from_quaternion(Quaterniond(quaternion.data()));
                  })
      .def_static("from_normalized_quaternion",
                  [](const Eigen::Vector4d& quaternion) {
                    return SO3d::from_normalized_quaternion(Quaterniond(quaternion.data()));
                  })
      .def_static("from_so2_xy", &SO3d::from_so2_xy)
      .def_static("from_matrix", &SO3d::from_matrix)
      .def("axis", &SO3d::axis)
      .def("angle", &SO3d::angle)
      .def_property(
          "quaternion",
          [](SO3d& q) { return q.quaternion().coeffs(); },
          [](SO3d& q, const Vector4d& v) { q = SO3d::from_quaternion(Quaterniond(v.data())); })
      .def("matrix", &SO3d::matrix)
      .def("euler_angles_rpy", &SO3d::euler_angles_rpy)
      .def("inverse", &SO3d::inverse)
      .def("__matmul__", [](const SO3d& lhs, const SO3d& rhs) { return lhs * rhs; })
      .def("__repr__", [](const SO3d& p) {
        std::stringstream ss;
        ss << "SO3(quaternion=" << p.quaternion().coeffs().transpose() << ")";
        return ss.str();
      });

  py::class_<Pose3d>(m, "Pose3")
      .def(py::init<>())  // Default constructor
      .def(py::init<const SO3d&,
                    const Eigen::Vector3d&>(),  // Constructor with translation and rotation
           py::arg("rotation"),
           py::arg("translation"))
      .def_static("identity", &Pose3d::identity)
      .def_static(
          "from_translation",
          [](const Eigen::Vector3d& translation) { return Pose3d::from_translation(translation); })
      .def_static("from_rotation", &Pose3d::from_rotation)
      .def_static("from_matrix", &Pose3d::from_matrix)
      .def_readwrite("translation", &Pose3d::translation)
      .def_readwrite("rotation", &Pose3d::rotation)
      .def("inverse", &Pose3d::inverse)
      .def("to_pose2_xy", &Pose3d::to_pose2_xy)
      .def_static("from_pose2_xy", &Pose3d::from_pose2_xy)
      .def("__matmul__", [](const Pose3d& lhs, const Pose3d& rhs) { return lhs * rhs; })
      .def("__mul__", [](const Pose3d& lhs, const Eigen::Vector3d& rhs) { return lhs * rhs; })
      .def("__repr__", [](const Pose3d& p) {
        std::stringstream ss;
        ss << "Pose3d(translation=" << p.translation.transpose() << ", ";
        ss << "rotation="
           << p.rotation.matrix().format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"))
           << ")";
        return ss.str();
      });
}
void init_pose_tree(py::module_& m) {
  // First bind the Error enum
  py::enum_<PoseTree::Error>(m, "PoseTreeError")
      .value("INVALID_ARGUMENT", PoseTree::Error::kInvalidArgument)
      .value("OUT_OF_MEMORY", PoseTree::Error::kOutOfMemory)
      .value("FRAME_NOT_FOUND", PoseTree::Error::kFrameNotFound)
      .value("ALREADY_EXISTS", PoseTree::Error::kAlreadyExists)
      .value("CYCLING_DEPENDENCY", PoseTree::Error::kCyclingDependency)
      .value("FRAMES_NOT_LINKED", PoseTree::Error::kFramesNotLinked)
      .value("POSE_OUT_OF_ORDER", PoseTree::Error::kPoseOutOfOrder)
      .value("LOGIC_ERROR", PoseTree::Error::kLogicError);

  // Bind the AccessMethod enum
  py::enum_<PoseTreeEdgeHistory::AccessMethod>(m, "PoseTreeAccessMethod")
      .value("NEAREST", PoseTreeEdgeHistory::AccessMethod::kNearest)
      .value("INTERPOLATE_LINEARLY", PoseTreeEdgeHistory::AccessMethod::kInterpolateLinearly)
      .value("EXTRAPOLATE_LINEARLY", PoseTreeEdgeHistory::AccessMethod::kExtrapolateLinearly)
      .value("INTERPOLATE_SLERP", PoseTreeEdgeHistory::AccessMethod::kInterpolateSlerp)
      .value("EXTRAPOLATE_SLERP", PoseTreeEdgeHistory::AccessMethod::kExtrapolateSlerp)
      .value("PREVIOUS", PoseTreeEdgeHistory::AccessMethod::kPrevious)
      .value("DEFAULT", PoseTreeEdgeHistory::AccessMethod::kDefault);

  // Bind the PoseTree class
  py::class_<PoseTree, std::shared_ptr<PoseTree>>(m, "PoseTree")
      .def(py::init<>())
      // Initialize method
      .def(
          "init",
          [](PoseTree& self,
             int32_t maximum_number_frames,
             int32_t maximum_number_edges,
             int32_t history_length,
             int32_t default_number_edges,
             int32_t default_history_length,
             int32_t edges_chunk_size,
             int32_t history_chunk_size) {
            auto ret = self.init(maximum_number_frames,
                                 maximum_number_edges,
                                 history_length,
                                 default_number_edges,
                                 default_history_length,
                                 edges_chunk_size,
                                 history_chunk_size);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
          },
          py::arg("number_frames") = 1024,
          py::arg("number_edges") = 16384,
          py::arg("history_length") = 1048576,
          py::arg("default_number_edges") = 16,
          py::arg("default_history_length") = 1024,
          py::arg("edges_chunk_size") = 4,
          py::arg("history_chunk_size") = 64)
      .def("deinit", &PoseTree::deinit)
      .def(
          "set_multithreading_info",
          [](PoseTree& self, PoseTree::frame_t start_id, PoseTree::frame_t increment) {
            auto ret = self.set_multithreading_info(start_id, increment);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
          },
          py::arg("start_id") = 1,
          py::arg("increment") = 1)

      // Version methods
      .def("get_pose_tree_version", &PoseTree::get_pose_tree_version)

      // Frame creation and management
      .def(
          "create_frame",
          [](PoseTree& self, std::string_view name, int32_t number_edges) {
            auto ret = self.create_frame(name, number_edges);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name") = "",
          py::arg("number_edges") = 16)

      .def(
          "create_frame_with_id",
          [](PoseTree& self, PoseTree::frame_t id, std::string_view name) {
            auto ret = self.create_frame_with_id(id, name);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("id"),
          py::arg("name") = "")

      .def(
          "find_frame",
          [](PoseTree& self, std::string_view name) {
            auto ret = self.find_frame(name);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name"))

      .def(
          "find_or_create_frame",
          [](PoseTree& self, std::string_view name, int32_t number_edges) {
            auto ret = self.find_or_create_frame(name, number_edges);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name"),
          py::arg("number_edges") = 16)

      .def(
          "get_frame_name",
          [](PoseTree& self, PoseTree::frame_t uid) {
            auto ret = self.get_frame_name(uid);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("uid"))

      // Edge management
      .def(
          "create_edges",
          [](PoseTree& self,
             PoseTree::frame_t lhs,
             PoseTree::frame_t rhs,
             int32_t number_edges,
             PoseTreeEdgeHistory::AccessMethod method) {
            auto ret = self.create_edges(lhs, rhs, number_edges, method);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("number_edges") = 16,
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault)

      .def(
          "create_edges",
          [](PoseTree& self,
             std::string_view lhs,
             std::string_view rhs,
             int32_t number_edges,
             PoseTreeEdgeHistory::AccessMethod method) {
            auto ret = self.create_edges(lhs, rhs, number_edges, method);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("number_edges") = 16,
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault)

      .def(
          "delete_edge",
          [](PoseTree& self, PoseTree::frame_t lhs, PoseTree::frame_t rhs) {
            auto ret = self.delete_edge(lhs, rhs);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"))

      .def(
          "delete_edge",
          [](PoseTree& self, std::string_view lhs, std::string_view rhs) {
            auto ret = self.delete_edge(lhs, rhs);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"))

      .def(
          "disconnect_edge",
          [](PoseTree& self, PoseTree::frame_t lhs, PoseTree::frame_t rhs, double time) {
            auto ret = self.disconnect_edge(lhs, rhs, time);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"))

      .def(
          "disconnect_edge",
          [](PoseTree& self, std::string_view lhs, std::string_view rhs, double time) {
            auto ret = self.disconnect_edge(lhs, rhs, time);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"))

      .def(
          "delete_frame",
          [](PoseTree& self, PoseTree::frame_t uid) {
            auto ret = self.delete_frame(uid);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("uid"))

      .def(
          "delete_frame",
          [](PoseTree& self, std::string_view name) {
            auto ret = self.delete_frame(name);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name"))

      .def(
          "disconnect_frame",
          [](PoseTree& self, PoseTree::frame_t uid, double time) {
            auto ret = self.disconnect_frame(uid, time);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name"),
          py::arg("time"))
      .def(
          "disconnect_frame",
          [](PoseTree& self, std::string_view name, double time) {
            auto ret = self.disconnect_frame(name, time);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("name"),
          py::arg("time"))

      .def(
          "get_latest",
          [](PoseTree& self, PoseTree::frame_t lhs, PoseTree::frame_t rhs) {
            auto ret = self.get_latest(lhs, rhs);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"))

      .def(
          "get_latest",
          [](PoseTree& self, std::string_view lhs, std::string_view rhs) {
            auto ret = self.get_latest(lhs, rhs);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"))

      // Get pose methods
      .def(
          "get",
          [](PoseTree& self,
             PoseTree::frame_t lhs,
             PoseTree::frame_t rhs,
             double time,
             PoseTreeEdgeHistory::AccessMethod method,
             PoseTree::version_t version) {
            if (version == 0) {
              version = self.get_pose_tree_version();
            }
            auto ret = self.get(lhs, rhs, time, method, version);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time") = std::numeric_limits<double>::max(),
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault,
          py::arg("version") = 0)
      .def(
          "get",
          [](PoseTree& self,
             std::string_view lhs,
             std::string_view rhs,
             double time,
             PoseTreeEdgeHistory::AccessMethod method,
             PoseTree::version_t version) {
            if (version == 0) {
              version = self.get_pose_tree_version();
            }
            auto ret = self.get(lhs, rhs, time, method, version);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time") = std::numeric_limits<double>::max(),
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault,
          py::arg("version") = 0)

      .def(
          "get_pose2_xy",
          [](PoseTree& self,
             PoseTree::frame_t lhs,
             PoseTree::frame_t rhs,
             double time,
             PoseTreeEdgeHistory::AccessMethod method,
             PoseTree::version_t version) {
            if (version == 0) {
              version = self.get_pose_tree_version();
            }
            auto ret = self.get_pose2_xy(lhs, rhs, time, method, version);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time") = std::numeric_limits<double>::max(),
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault,
          py::arg("version") = 0)
      .def(
          "get_pose2_xy",
          [](PoseTree& self,
             std::string_view lhs,
             std::string_view rhs,
             double time,
             PoseTreeEdgeHistory::AccessMethod method,
             PoseTree::version_t version) {
            if (version == 0) {
              version = self.get_pose_tree_version();
            }
            auto ret = self.get_pose2_xy(lhs, rhs, time, method, version);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time") = std::numeric_limits<double>::max(),
          py::arg("method") = PoseTreeEdgeHistory::AccessMethod::kDefault,
          py::arg("version") = 0)

      // Pose operations
      .def(
          "set",
          [](PoseTree& self,
             PoseTree::frame_t lhs,
             PoseTree::frame_t rhs,
             double time,
             const Pose3d& pose) {
            auto ret = self.set(lhs, rhs, time, pose);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"),
          py::arg("pose"))
      .def(
          "set",
          [](PoseTree& self,
             std::string_view lhs,
             std::string_view rhs,
             double time,
             const Pose3d& pose) {
            auto ret = self.set(lhs, rhs, time, pose);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"),
          py::arg("pose"))
      .def(
          "set",
          [](PoseTree& self,
             PoseTree::frame_t lhs,
             PoseTree::frame_t rhs,
             double time,
             const Pose2d& pose) {
            auto ret = self.set(lhs, rhs, time, pose);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"),
          py::arg("pose"))
      .def(
          "set",
          [](PoseTree& self,
             std::string_view lhs,
             std::string_view rhs,
             double time,
             const Pose2d& pose) {
            auto ret = self.set(lhs, rhs, time, pose);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return ret.value();
          },
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("time"),
          py::arg("pose"))

      // Frame and edge listing
      .def(
          "get_frame_uids",
          [](PoseTree& self, int32_t maximum_number_frames) {
            std::vector<PoseTree::frame_t> container;
            container.reserve(maximum_number_frames);  // reasonable initial capacity
            auto ret = self.get_frame_uids(container);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return container;
          },
          py::arg("maximum_number_frames") = 1024)
      .def(
          "get_frame_names",
          [](PoseTree& self, int32_t maximum_number_frames) {
            std::vector<std::string_view> container;
            container.reserve(maximum_number_frames);  // reasonable initial capacity
            auto ret = self.get_frame_names(container);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return container;
          },
          py::arg("maximum_number_frames") = 1024)
      .def(
          "get_edge_uids",
          [](PoseTree& self, int32_t maximum_number_edges) {
            std::vector<std::pair<PoseTree::frame_t, PoseTree::frame_t>> container;
            container.reserve(maximum_number_edges);  // reasonable initial capacity
            auto ret = self.get_edge_uids(container);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return container;
          },
          py::arg("maximum_number_edges") = 1024)
      .def(
          "get_edge_names",
          [](PoseTree& self, int32_t maximum_number_edges) {
            std::vector<std::pair<std::string_view, std::string_view>> container;
            container.reserve(maximum_number_edges);  // reasonable initial capacity
            auto ret = self.get_edge_names(container);
            if (!ret) {
              throw std::runtime_error(PoseTree::error_to_str(ret.error()));
            }
            return container;
          },
          py::arg("maximum_number_edges") = 1024);
}

void init_pose_tree_ucx(py::module_& m) {
  // Bind PoseTreeUCXServerConfig
  py::class_<PoseTreeUCXServerConfig>(m, "PoseTreeUCXServerConfig")
      .def(py::init<>())
      .def_readwrite("worker_progress_sleep_us", &PoseTreeUCXServerConfig::worker_progress_sleep_us)
      .def_readwrite("shutdown_timeout_ms", &PoseTreeUCXServerConfig::shutdown_timeout_ms)
      .def_readwrite("shutdown_poll_sleep_ms", &PoseTreeUCXServerConfig::shutdown_poll_sleep_ms)
      .def_readwrite("maximum_clients", &PoseTreeUCXServerConfig::maximum_clients);

  // Bind PoseTreeUCXServer Error enum
  py::enum_<PoseTreeUCXServer::Error>(m, "PoseTreeUCXServerError")
      .value("ALREADY_RUNNING", PoseTreeUCXServer::Error::kAlreadyRunning)
      .value("INVALID_ARGUMENT", PoseTreeUCXServer::Error::kInvalidArgument)
      .value("STARTUP_FAILED", PoseTreeUCXServer::Error::kStartupFailed)
      .value("NOT_RUNNING", PoseTreeUCXServer::Error::kNotRunning)
      .value("SHUTDOWN_TIMEOUT", PoseTreeUCXServer::Error::kShutdownTimeout)
      .value("INTERNAL_ERROR", PoseTreeUCXServer::Error::kInternalError);

  // Bind PoseTreeUCXServer.
  // The shared_ptr holder type is needed as it is passed as a shared_ptr to the client.
  py::class_<PoseTreeUCXServer, std::shared_ptr<PoseTreeUCXServer>>(m, "PoseTreeUCXServer")
      .def(py::init<std::shared_ptr<PoseTree>, PoseTreeUCXServerConfig>(),
           py::arg("pose_tree"),
           py::arg("config") = PoseTreeUCXServerConfig{})
      .def(
          "start",
          [](PoseTreeUCXServer& self, uint16_t port) {
            auto result = self.start(port);
            if (!result) {
              throw std::runtime_error(PoseTreeUCXServer::error_to_str(result.error()));
            }
          },
          py::arg("port"))
      .def("stop",
           [](PoseTreeUCXServer& self) {
             auto result = self.stop();
             if (!result) {
               throw std::runtime_error(PoseTreeUCXServer::error_to_str(result.error()));
             }
           })
      .def_property_readonly("is_running", &PoseTreeUCXServer::is_running)
      .def_static("error_to_str", &PoseTreeUCXServer::error_to_str);

  // Bind PoseTreeUCXClientConfig
  py::class_<PoseTreeUCXClientConfig>(m, "PoseTreeUCXClientConfig")
      .def(py::init<>())
      .def_readwrite("request_timeout_ms", &PoseTreeUCXClientConfig::request_timeout_ms)
      .def_readwrite("request_poll_sleep_us", &PoseTreeUCXClientConfig::request_poll_sleep_us)
      .def_readwrite("worker_progress_sleep_us",
                     &PoseTreeUCXClientConfig::worker_progress_sleep_us);

  // Bind PoseTreeUCXClient Error enum
  py::enum_<PoseTreeUCXClient::Error>(m, "PoseTreeUCXClientError")
      .value("ALREADY_CONNECTED", PoseTreeUCXClient::Error::kAlreadyConnected)
      .value("INVALID_ARGUMENT", PoseTreeUCXClient::Error::kInvalidArgument)
      .value("CONNECTION_FAILED", PoseTreeUCXClient::Error::kConnectionFailed)
      .value("NOT_CONNECTED", PoseTreeUCXClient::Error::kNotConnected)
      .value("THREAD_ERROR", PoseTreeUCXClient::Error::kThreadError)
      .value("SHUTDOWN_ERROR", PoseTreeUCXClient::Error::kShutdownError)
      .value("INTERNAL_ERROR", PoseTreeUCXClient::Error::kInternalError);

  // unique_ptr holder for a non-copyable class
  using PoseTreeUCXClientUPtr = std::unique_ptr<PoseTreeUCXClient>;
  py::class_<PoseTreeUCXClient, PoseTreeUCXClientUPtr>(m, "PoseTreeUCXClient")
      .def(py::init<std::shared_ptr<PoseTree>, PoseTreeUCXClientConfig>(),
           py::arg("pose_tree"),
           py::arg("config") = PoseTreeUCXClientConfig{})
      .def(
          "connect",
          [](PoseTreeUCXClient& self, std::string_view host, uint16_t port, bool request_snapshot) {
            auto result = self.connect(host, port, request_snapshot);
            if (!result) {
              throw std::runtime_error(PoseTreeUCXClient::error_to_str(result.error()));
            }
          },
          py::arg("host"),
          py::arg("port"),
          py::arg("request_snapshot"))
      .def("disconnect",
           [](PoseTreeUCXClient& self) {
             auto result = self.disconnect();
             if (!result) {
               throw std::runtime_error(PoseTreeUCXClient::error_to_str(result.error()));
             }
           })
      .def_property_readonly("is_running", &PoseTreeUCXClient::is_running)
      .def_static("error_to_str", &PoseTreeUCXClient::error_to_str);
}

PYBIND11_MODULE(_pose_tree, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK PoseTree Python Bindings
        ---------------------------------
        .. currentmodule:: _pose_tree
    )pbdoc";

  init_pose_tree_geometry(m);
  init_pose_tree(m);
  init_pose_tree_ucx(m);
}

}  // namespace holoscan
