/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "drm_ioctl.hpp"

#include <sys/ioctl.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace holoviz::display_systems::drm {

namespace {

/// Bound on the count-then-fill retries when a hotplug changes an array size between passes.
constexpr int kMaxTwoPassRetries = 4;

template <typename T>
uint64_t user_ptr(std::vector<T>& v) {
  return v.empty() ? 0 : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v.data()));
}

Error ioctl_error(const char* name, int rc) {
  return Error{
      .code = ErrorCode::kFailure,
      .message = std::string("[holoviz] ") + name + " failed (" + std::strerror(-rc) + ")"};
}

Error retry_exhausted_error(const char* name) {
  return Error{.code = ErrorCode::kFailure,
               .message = std::string("[holoviz] ") + name +
                          " failed (resource counts kept changing between passes, hotplug?)"};
}

}  // namespace

int default_ioctl(int fd, unsigned long request, void* arg) {
  return ::ioctl(fd, request, arg);
}

int drm_ioctl(const Node& node, unsigned long request, void* arg) {
  int rc = 0;
  do {
    rc = node.ioctl_fn(node.fd, request, arg);
  } while (rc == -1 && (errno == EINTR || errno == EAGAIN));
  return rc == 0 ? 0 : -errno;
}

bool is_master(const Node& node) {
  drm_auth auth{};
  auth.magic = 0;
  return drm_ioctl(node, DRM_IOCTL_AUTH_MAGIC, &auth) != -EACCES;
}

expected<void> set_master(const Node& node) {
  if (const int rc = drm_ioctl(node, DRM_IOCTL_SET_MASTER, nullptr); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_SET_MASTER", rc));
  }
  return {};
}

expected<void> drop_master(const Node& node) {
  if (const int rc = drm_ioctl(node, DRM_IOCTL_DROP_MASTER, nullptr); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_DROP_MASTER", rc));
  }
  return {};
}

expected<Resources> get_resources(const Node& node) {
  drm_mode_card_res res{};
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETRESOURCES, &res); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETRESOURCES", rc));
  }

  for (int attempt = 0; attempt < kMaxTwoPassRetries; ++attempt) {
    Resources out;
    out.connectors.resize(res.count_connectors);
    out.count_fbs = res.count_fbs;
    out.count_crtcs = res.count_crtcs;
    out.count_encoders = res.count_encoders;

    drm_mode_card_res fill{};
    fill.count_connectors = static_cast<uint32_t>(out.connectors.size());
    fill.connector_id_ptr = user_ptr(out.connectors);
    // fb/crtc/encoder ids are not needed: their pointers stay 0 and the kernel only
    // reports the counts.
    if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETRESOURCES, &fill); rc != 0) {
      return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETRESOURCES", rc));
    }
    if (fill.count_connectors <= out.connectors.size()) {
      out.connectors.resize(fill.count_connectors);
      out.count_fbs = fill.count_fbs;
      out.count_crtcs = fill.count_crtcs;
      out.count_encoders = fill.count_encoders;
      return out;
    }
    res = fill;  // the array grew (hotplug); size for the new count and retry
  }
  return make_unexpected(retry_exhausted_error("DRM_IOCTL_MODE_GETRESOURCES"));
}

expected<Connector> get_connector(const Node& node, uint32_t connector_id) {
  // Every count 0 on the first pass: the kernel force-probes the connector when we are
  // master (libdrm's drmModeGetConnector semantics), which is what the ownership
  // classification wants — a fresh connection status.
  drm_mode_get_connector counts{};
  counts.connector_id = connector_id;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETCONNECTOR, &counts); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETCONNECTOR", rc));
  }

  for (int attempt = 0; attempt < kMaxTwoPassRetries; ++attempt) {
    Connector out;
    out.encoders.resize(counts.count_encoders);
    out.modes.resize(counts.count_modes);
    out.props.resize(counts.count_props);
    out.prop_values.resize(counts.count_props);

    drm_mode_get_connector fill{};
    fill.connector_id = connector_id;
    fill.count_encoders = static_cast<uint32_t>(out.encoders.size());
    fill.encoders_ptr = user_ptr(out.encoders);
    fill.count_modes = static_cast<uint32_t>(out.modes.size());
    fill.modes_ptr = user_ptr(out.modes);
    fill.count_props = static_cast<uint32_t>(out.props.size());
    fill.props_ptr = user_ptr(out.props);
    fill.prop_values_ptr = user_ptr(out.prop_values);
    if (out.modes.empty()) {
      // count_modes == 0 would re-probe; a one-element scratch keeps the fill pass cheap.
      out.modes.resize(1);
      fill.count_modes = 1;
      fill.modes_ptr = user_ptr(out.modes);
    }
    if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETCONNECTOR, &fill); rc != 0) {
      return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETCONNECTOR", rc));
    }
    if (fill.count_encoders <= out.encoders.size() && fill.count_modes <= out.modes.size() &&
        fill.count_props <= out.props.size()) {
      out.encoders.resize(fill.count_encoders);
      out.modes.resize(fill.count_modes);
      out.props.resize(fill.count_props);
      out.prop_values.resize(fill.count_props);
      out.connector_id = fill.connector_id;
      out.connector_type = fill.connector_type;
      out.connector_type_id = fill.connector_type_id;
      out.encoder_id = fill.encoder_id;
      out.connection = fill.connection;
      return out;
    }
    counts = fill;  // an array grew; size for the new counts and retry
  }
  return make_unexpected(retry_exhausted_error("DRM_IOCTL_MODE_GETCONNECTOR"));
}

expected<Encoder> get_encoder(const Node& node, uint32_t encoder_id) {
  drm_mode_get_encoder enc{};
  enc.encoder_id = encoder_id;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETENCODER, &enc); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETENCODER", rc));
  }
  return Encoder{
      .encoder_id = enc.encoder_id, .encoder_type = enc.encoder_type, .crtc_id = enc.crtc_id};
}

expected<Crtc> get_crtc(const Node& node, uint32_t crtc_id) {
  drm_mode_crtc crtc{};
  crtc.crtc_id = crtc_id;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETCRTC, &crtc); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETCRTC", rc));
  }
  return Crtc{.crtc_id = crtc.crtc_id, .mode_valid = crtc.mode_valid != 0, .mode = crtc.mode};
}

expected<ObjectProperties> get_object_properties(const Node& node, uint32_t object_id,
                                                 uint32_t object_type) {
  drm_mode_obj_get_properties counts{};
  counts.obj_id = object_id;
  counts.obj_type = object_type;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &counts); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_OBJ_GETPROPERTIES", rc));
  }

  for (int attempt = 0; attempt < kMaxTwoPassRetries; ++attempt) {
    ObjectProperties out;
    out.props.resize(counts.count_props);
    out.prop_values.resize(counts.count_props);

    drm_mode_obj_get_properties fill{};
    fill.obj_id = object_id;
    fill.obj_type = object_type;
    fill.count_props = static_cast<uint32_t>(out.props.size());
    fill.props_ptr = user_ptr(out.props);
    fill.prop_values_ptr = user_ptr(out.prop_values);
    if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &fill); rc != 0) {
      return make_unexpected(ioctl_error("DRM_IOCTL_MODE_OBJ_GETPROPERTIES", rc));
    }
    if (fill.count_props <= out.props.size()) {
      out.props.resize(fill.count_props);
      out.prop_values.resize(fill.count_props);
      return out;
    }
    counts = fill;
  }
  return make_unexpected(retry_exhausted_error("DRM_IOCTL_MODE_OBJ_GETPROPERTIES"));
}

expected<PropertyName> get_property_name(const Node& node, uint32_t property_id) {
  // Values and enum blobs are not requested (counts 0); the kernel fills the name anyway.
  drm_mode_get_property prop{};
  prop.prop_id = property_id;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETPROPERTY, &prop); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETPROPERTY", rc));
  }
  PropertyName name{};
  std::memcpy(name.data(), prop.name, DRM_PROP_NAME_LEN);
  name[DRM_PROP_NAME_LEN] = '\0';
  return name;
}

expected<std::vector<uint8_t>> get_property_blob(const Node& node, uint32_t blob_id) {
  drm_mode_get_blob counts{};
  counts.blob_id = blob_id;
  if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETPROPBLOB, &counts); rc != 0) {
    return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETPROPBLOB", rc));
  }

  for (int attempt = 0; attempt < kMaxTwoPassRetries; ++attempt) {
    std::vector<uint8_t> out(counts.length);
    drm_mode_get_blob fill{};
    fill.blob_id = blob_id;
    fill.length = static_cast<uint32_t>(out.size());
    fill.data = user_ptr(out);
    if (const int rc = drm_ioctl(node, DRM_IOCTL_MODE_GETPROPBLOB, &fill); rc != 0) {
      return make_unexpected(ioctl_error("DRM_IOCTL_MODE_GETPROPBLOB", rc));
    }
    if (fill.length <= out.size()) {
      out.resize(fill.length);
      return out;
    }
    counts = fill;
  }
  return make_unexpected(retry_exhausted_error("DRM_IOCTL_MODE_GETPROPBLOB"));
}

}  // namespace holoviz::display_systems::drm
