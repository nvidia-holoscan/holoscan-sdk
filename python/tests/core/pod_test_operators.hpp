/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::python::test {

// Raw-POD payload exchanged between Python and C++ Operators in test_pod_payload.py. The fields
// cover an unsigned and a signed integer, a float, and integer and floating-point arrays.
struct TestPose {
  std::uint64_t sequence;
  double position[3];
  float confidence;
  std::int16_t quadrant;
  std::uint8_t flags[2];
};

// Raw-POD payload whose one-element arrays have the same element count as its scalar field;
// test_pod_payload.py exchanges the arrays as one-element tuples.
struct TestSingleton {
  std::uint32_t count;
  std::uint32_t ids[1];
  std::array<float, 1> gains;
};

// Opaque bytes and unsigned integers have distinct Python array representations.
struct TestByteRecord {
  std::byte tag;
  std::array<std::byte, 1> singleton;
  std::byte cbytes[3];
  std::array<std::byte, 3> bytes;
  std::uint8_t numeric[2];
  std::uint8_t value;
};

}  // namespace holoscan::python::test

namespace holoscan {

template <>
struct ChannelTypeIdentity<python::test::TestPose> {
  static constexpr std::string_view name = "holoscan.test.Pose";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<python::test::TestPose> {
  using Pose = python::test::TestPose;
  static constexpr auto fields =
      pod_fields(pod_field("sequence", &Pose::sequence, offsetof(Pose, sequence)),
                 pod_field("position", &Pose::position, offsetof(Pose, position)),
                 pod_field("confidence", &Pose::confidence, offsetof(Pose, confidence)),
                 pod_field("quadrant", &Pose::quadrant, offsetof(Pose, quadrant)),
                 pod_field("flags", &Pose::flags, offsetof(Pose, flags)));
};

template <>
struct ChannelTypeIdentity<python::test::TestSingleton> {
  static constexpr std::string_view name = "holoscan.test.Singleton";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<python::test::TestSingleton> {
  using Singleton = python::test::TestSingleton;
  static constexpr auto fields =
      pod_fields(pod_field("count", &Singleton::count, offsetof(Singleton, count)),
                 pod_field("ids", &Singleton::ids, offsetof(Singleton, ids)),
                 pod_field("gains", &Singleton::gains, offsetof(Singleton, gains)));
};

template <>
struct ChannelTypeIdentity<python::test::TestByteRecord> {
  static constexpr std::string_view name = "holoscan.test.ByteRecord";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelPodLayout<python::test::TestByteRecord> {
  using Record = python::test::TestByteRecord;
  static constexpr auto fields =
      pod_fields(pod_field("tag", &Record::tag, offsetof(Record, tag)),
                 pod_field("singleton", &Record::singleton, offsetof(Record, singleton)),
                 pod_field("cbytes", &Record::cbytes, offsetof(Record, cbytes)),
                 pod_field("bytes", &Record::bytes, offsetof(Record, bytes)),
                 pod_field("numeric", &Record::numeric, offsetof(Record, numeric)),
                 pod_field("value", &Record::value, offsetof(Record, value)));
};

}  // namespace holoscan

namespace holoscan::python::test {

// The value TestPoseSource emits, which test_pod_payload.py expects field by field.
inline constexpr TestPose kTestPose{
    .sequence = 7U,
    .position = {1.5, -2.25, 3.0},
    .confidence = 0.5F,
    .quadrant = -3,
    .flags = {1U, 254U},
};

// Records the last pose TestPoseSink received.
class TestPoseObservation final {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    last_.reset();
  }

  static void record(const TestPose& pose) {
    const std::lock_guard lock(mutex_);
    last_ = pose;
  }

  [[nodiscard]] static std::optional<TestPose> last() {
    const std::lock_guard lock(mutex_);
    return last_;
  }

 private:
  inline static std::mutex mutex_;
  inline static std::optional<TestPose> last_;
};

// Emits kTestPose once.
class TestPoseSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return out.emit(kTestPose);
  }

  Output<TestPose> out;
};

// Records each pose it receives in TestPoseObservation.
class TestPoseSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(const TestPose pose, in.receive_data());
    TestPoseObservation::record(pose);
    return {};
  }

  Input<TestPose> in;
};

}  // namespace holoscan::python::test
