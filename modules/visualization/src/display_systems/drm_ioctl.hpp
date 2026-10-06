/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal kernel DRM UAPI (KMS) client: the handful of libdrm entry points the
// display-ownership path needs, issued as raw ioctls against an already-open
// /dev/dri/cardN fd. HoloViz deliberately does not link libdrm: on NVIDIA IGX /
// Tegra systems the loader resolves libdrm.so.2 to NVIDIA's Tegra shim (placed
// before the multiarch directory by ld.so.conf), which cannot drive a discrete
// GPU's KMS node, and glibc's soname de-duplication means no RUNPATH or dlopen
// trick reliably reaches the Mesa library once the shim is in the process.
// Speaking to the kernel directly removes the dependency altogether.
//
// Results are owned values (no drmModeFree* twins). Every fallible call returns
// expected<T> whose error carries the ioctl name and the errno text. The ioctl
// entry point is a function pointer so the two-pass array protocol can be
// unit-tested against a fake without a device (see tests/unit/test_drm_ioctl.cpp).
// Linux-only.

#pragma once

#include <drm/drm.h>
#include <drm/drm_mode.h>

#include <array>
#include <cstdint>
#include <vector>

#include <holoviz/expected.hpp>

namespace holoviz::display_systems::drm {

/// ioctl(2)-compatible entry point: returns 0 on success, -1 with errno set on failure.
using IoctlFn = int (*)(int fd, unsigned long request, void* arg);

/// The real ::ioctl.
int default_ioctl(int fd, unsigned long request, void* arg);

/// An open DRM primary node plus the ioctl entry point used to talk to it.
struct Node {
  int fd = -1;
  IoctlFn ioctl_fn = default_ioctl;
};

/// Issue \p request on \p node, retrying on EINTR/EAGAIN like libdrm's drmIoctl.
/// @return 0 on success, -errno on failure.
[[nodiscard]] int drm_ioctl(const Node& node, unsigned long request, void* arg);

/// Return whether \p node currently holds DRM master. Mirrors libdrm's drmIsMaster:
/// DRM_IOCTL_AUTH_MAGIC with magic 0 fails with EACCES only for non-masters (a master
/// gets EINVAL for the unknown magic, which counts as "is master").
[[nodiscard]] bool is_master(const Node& node);

/// DRM_IOCTL_SET_MASTER.
[[nodiscard]] expected<void> set_master(const Node& node);

/// DRM_IOCTL_DROP_MASTER.
[[nodiscard]] expected<void> drop_master(const Node& node);

/// The connector ids of a device (DRM_IOCTL_MODE_GETRESOURCES). Only the connector
/// array is fetched; the fb/crtc/encoder counts are reported but their ids are not needed.
struct Resources {
  std::vector<uint32_t> connectors;
  uint32_t count_fbs = 0;
  uint32_t count_crtcs = 0;
  uint32_t count_encoders = 0;
};

/// Enumerate the device's KMS resources.
[[nodiscard]] expected<Resources> get_resources(const Node& node);

/// Values of drm_mode_get_connector::connection (the UAPI header leaves them unnamed;
/// libdrm spells them DRM_MODE_CONNECTED / DISCONNECTED / UNKNOWNCONNECTION).
inline constexpr uint32_t kConnectionConnected = 1;
inline constexpr uint32_t kConnectionDisconnected = 2;
inline constexpr uint32_t kConnectionUnknown = 3;

/// A connector (DRM_IOCTL_MODE_GETCONNECTOR). The arrays are fetched to keep the
/// kernel's two-pass protocol honest; the ownership path only reads the scalar fields.
struct Connector {
  uint32_t connector_id = 0;
  uint32_t connector_type = 0;
  uint32_t connector_type_id = 0;
  uint32_t encoder_id = 0;                   ///< current encoder, 0 when none
  uint32_t connection = kConnectionUnknown;  ///< one of the kConnection* values
  std::vector<uint32_t> encoders;
  std::vector<drm_mode_modeinfo> modes;
  std::vector<uint32_t> props;
  std::vector<uint64_t> prop_values;
};

/// Fetch connector \p connector_id. The first (counting) pass sends every count as 0,
/// which, like libdrm's drmModeGetConnector, makes the kernel force-probe the connector
/// when the caller is DRM master.
[[nodiscard]] expected<Connector> get_connector(const Node& node, uint32_t connector_id);

/// An encoder (DRM_IOCTL_MODE_GETENCODER).
struct Encoder {
  uint32_t encoder_id = 0;
  uint32_t encoder_type = 0;
  uint32_t crtc_id = 0;  ///< current CRTC, 0 when none
};

/// Fetch encoder \p encoder_id.
[[nodiscard]] expected<Encoder> get_encoder(const Node& node, uint32_t encoder_id);

/// A CRTC (DRM_IOCTL_MODE_GETCRTC).
struct Crtc {
  uint32_t crtc_id = 0;
  bool mode_valid = false;
  drm_mode_modeinfo mode{};
};

/// Fetch CRTC \p crtc_id.
[[nodiscard]] expected<Crtc> get_crtc(const Node& node, uint32_t crtc_id);

/// The properties attached to a KMS object (DRM_IOCTL_MODE_OBJ_GETPROPERTIES).
struct ObjectProperties {
  std::vector<uint32_t> props;
  std::vector<uint64_t> prop_values;
};

/// Fetch the properties of \p object_id of type \p object_type (DRM_MODE_OBJECT_*).
[[nodiscard]] expected<ObjectProperties> get_object_properties(const Node& node, uint32_t object_id,
                                                               uint32_t object_type);

/// A NUL-terminated property name (DRM_PROP_NAME_LEN bytes plus terminator).
using PropertyName = std::array<char, DRM_PROP_NAME_LEN + 1>;

/// Fetch the name of property \p property_id (DRM_IOCTL_MODE_GETPROPERTY; values and
/// enum blobs are not requested).
[[nodiscard]] expected<PropertyName> get_property_name(const Node& node, uint32_t property_id);

/// Fetch the bytes of blob \p blob_id (DRM_IOCTL_MODE_GETPROPBLOB), e.g. a connector's EDID.
[[nodiscard]] expected<std::vector<uint8_t>> get_property_blob(const Node& node, uint32_t blob_id);

}  // namespace holoviz::display_systems::drm
