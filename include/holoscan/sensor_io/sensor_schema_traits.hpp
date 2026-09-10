/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "holoscan/core/schema_identity.hpp"
#include "holoscan/schema/flatbuffer_schema_traits.hpp"

/// @file
/// @brief Identities for the sensor payload schemas this library owns.
///
/// Image and Imu are domain payloads rather than transport primitives, so they live here rather
/// than beside tensor.fbs and time.fbs in Core. The practical consequence is that adding a camera
/// field or a new sensor type no longer amends Core's public contract, and a break like the Imu
/// quaternion reorder is scoped to a component a deployment can choose not to install.
///
/// The namespace is still \c holoscan::schema and deliberately so. A FlatBuffers namespace is part
/// of the wire identity -- it is the \c canonical_name below -- so renaming it to \c
/// holoscan::sensor_io would reissue both schemas under new identities and invalidate every
/// recorded payload. Ownership moved; the format did not.

namespace holoscan::schema {

struct Image;
struct Imu;

/// @brief Describe the SDK Image root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<Image> {
  using root_type = Image;  ///< Public generated semantic root type.
  // Version 4 replaces the plane_pitches row-stride array with plane_layouts, a vector of
  // ImagePlaneLayout structs carrying an offset, a row stride, and an extent per plane. A stride
  // alone could not locate a plane: an Image carries one Tensor whose single shape and stride set
  // gives no independent base per plane, so a consumer had to reconstruct offsets from assumed
  // subsampling and could not bound its reads. Field id 9 is reused because no consumer has
  // recorded a version 4 payload.
  //
  // Version 4 also renames analog_gain to gain, and follows Header renaming its sample counter to
  // device_sequence. The sensor interfaces this schema is meant to describe report one combined
  // gain figure -- SIPL exposes a single sensorGain -- so the old name promised a digital
  // counterpart that was never going to arrive, and a field named for a distinction the hardware
  // does not make invites producers to invent one. Neither rename touches the wire layout: same
  // field ids, types, and offsets. But flatc records field names in the binary schema, so a rename
  // alone moves the digest, which is why both ride a version.
  //
  // Version 3 had moved the counter into Header, where every sensor payload shares one definition
  // of it, and renumbered the fields it left behind. Version 2 had added color space, significant
  // bits, ROI origin, plane pitches, exposure, and gain, plus the YUV, float, 10-bit, and Bayer
  // encodings.
  //
  // Versions 2 through 4 exist only on this branch: the last released Image was version 1. The
  // renames are therefore free of any migration story, which is exactly why they are worth making
  // now rather than after a consumer has recorded a payload under the old names.
  //
  // The digest covers the embedded binary schema -- the bfbsData array in the generated
  // <stem>_bfbs_generated.h -- because that is the artifact Runtime validates a package against.
  // Those bytes record the logical filename of every declaration file, so they depend on the
  // directory layout codegen sees and not only on the schema text: src/sensor_io/schemas/
  // CMakeLists.txt stages the Core schemas this one includes flat beside it so each is recorded as
  // "//<name>.fbs", and tests/sensor_io/test_schema_identity.cpp fails if that staging stops
  // working.
  //
  // These constants are hand-written only because the generated identity path cannot yet accept a
  // staged input. Because these schemas include Core-owned ones from another directory, they are
  // compiled from the staging directory above and so are named by absolute path, while the path's
  // reflection step spells the same file relatively. The filenames are part of the bytes, so the
  // two copies disagree and trip its own byte-for-byte guard. A follow-up against
  // HoloscanSchemaTools.cmake pins the spelling, after which these constants and this comment are
  // generated.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.Image",
      .version = 4U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "7781dbf9de20cb07c376c6a4f4410aaa7af996132f0dd01619e860dfbbb4e3ea"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("14a7331cc03e92662ebd4f58435b350f")},
  };
  static constexpr std::uint32_t schema_version = 4U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  // Metadata-only Image FlatBuffers stay small because FlatBuffers omits fields that hold their
  // default, so the added fields cost nothing until a producer sets them; a 4K HDR frame that set
  // every one of them landed at 296 bytes when plane_layouts was still a row-stride array. The
  // change is bounded rather than open-ended: a vector of structs is stored inline, so its cost is
  // exactly the plane count times 24 bytes with no offset or vtable per plane, and
  // kMaximumImagePlanes caps the count at four. The widest layout therefore adds at most 84 bytes
  // over the four-entry array it replaces, alignment padding included, which leaves the same frame
  // inside the 512-byte descriptor default. Callers can spell a larger explicit
  // DescriptorFlatBufferPayload<Image, N> when metadata strings need it.
  static constexpr std::size_t default_inline_capacity = 2048U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      512U;  ///< Descriptor metadata authoring default.
};

/// @brief Describe the SDK Imu root's complete identity and authoring defaults.
template <>
struct FlatBufferSchemaTraits<Imu> {
  using root_type = Imu;  ///< Public generated semantic root type.
  // Version 3 moves the geometry structs to geometry.fbs so a rotation has one definition, changes
  // the quaternion wire order to (x, y, z, w) to match ROS, MCAP, Newton, Isaac Lab, and Warp,
  // moves the sample counter into Header as device_sequence, and drops die temperature as a
  // device-specific quantity that belongs in a vendor table. The field set is now the ROS
  // sensor_msgs/Imu set.
  // Version 2 had added the per-sample flags and the three covariance matrices and stated the
  // units.
  //
  // The quaternion reorder is a silent break for any recorded version 2 payload and for positional
  // construction, which is why it rides a version bump rather than a patch. See the digest note on
  // FlatBufferSchemaTraits<Image>.
  //
  // Nothing in imu.fbs itself changed for the device_sequence rename; the digest moved because the
  // field names of an included schema are recorded in the binary schema too. Versions 2 and 3 exist
  // only on this branch, so no released payload carries the old name.
  /// @brief Carry the exact name, version, artifact digest, and derived compact identity.
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.schema.Imu",
      .version = 3U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "d4d53e356fdbc1a60008b9bc45b5d60854b34691e7a6547773cc9f56379f6228"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("5097e72b6c4cb26ca49cc29ebcc48f86")},
  };
  static constexpr std::uint32_t schema_version = 3U;  ///< Exact semantic schema version.
  static constexpr std::string_view root_name =
      identity.canonical_name;  ///< Canonical globally scoped UTF-8 name.
  static constexpr std::size_t default_inline_capacity = 512U;  ///< Inline authoring default.
  static constexpr std::size_t default_descriptor_metadata_capacity =
      512U;  ///< Descriptor metadata authoring default.
};

}  // namespace holoscan::schema

namespace holoscan {

/// @brief Return the generated SDK Image schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::Image>() {
  return schema::FlatBufferSchemaTraits<schema::Image>::identity;
}

/// @brief Return the generated SDK Imu schema identity.
template <>
consteval SchemaIdentity schema_identity<schema::Imu>() {
  return schema::FlatBufferSchemaTraits<schema::Imu>::identity;
}

}  // namespace holoscan
