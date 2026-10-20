/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the kernel DRM ioctl client (display_systems/drm_ioctl.hpp) against a fake device:
// the EINTR/EAGAIN retry, drmIsMaster's EACCES contract, and the count-then-fill array
// protocol (including the retry when a count grows between passes, and the kernel rule
// that an undersized array is left unfilled). Headless — no /dev/dri, no GPU.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "display_systems/drm_ioctl.hpp"
#include "vulkan/vk_display_util.hpp"

namespace drm = holoviz::display_systems::drm;

namespace {

/// In-memory KMS device the fake ioctl serves. One instance per test (file-static because
/// the seam is a plain function pointer).
struct FakeDevice {
  // Result scripting: each ioctl pops the front entry; a nonzero errno fails the call, 0
  // succeeds without touching the argument. An empty queue serves the device model.
  std::deque<int> scripted_errnos;
  // Resources.
  std::vector<uint32_t> connectors;
  // When set, the connector list grows by this id after the first counting pass
  // (simulates a hotplug between passes).
  uint32_t hotplug_connector_after_count = 0;
  // Connector 7.
  uint32_t connector_type = DRM_MODE_CONNECTOR_DisplayPort;
  uint32_t connector_type_id = 4;
  uint32_t connection = drm::kConnectionConnected;
  uint32_t encoder_id = 30;
  std::vector<uint32_t> encoders{30};
  std::vector<drm_mode_modeinfo> modes;
  std::vector<uint32_t> connector_props{100, 101};
  std::vector<uint64_t> connector_prop_values{1, 500};
  // Property names by id.
  std::string prop_name_100 = "vrr_capable";
  std::string prop_name_101 = "EDID";
  std::vector<uint8_t> blob_500;
  // Encoder 30 / CRTC 40.
  uint32_t crtc_id = 40;
  drm_mode_modeinfo crtc_mode{};
  // Observations.
  int ioctl_calls = 0;
  uint32_t first_connector_count_modes = UINT32_MAX;
  bool master = true;
};

FakeDevice* g_device = nullptr;

template <typename T>
T* as_user(uint64_t ptr) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(ptr));
}

template <typename T>
void fill_array(uint64_t ptr, uint32_t& count_io, const std::vector<T>& values) {
  // Kernel rule: only fill when the caller's capacity covers the whole array.
  if (count_io >= values.size() && !values.empty() && ptr != 0) {
    std::memcpy(as_user<T>(ptr), values.data(), values.size() * sizeof(T));
  }
  count_io = static_cast<uint32_t>(values.size());
}

// Keep the supported ioctl cases together so the fake device protocol remains easy to compare.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int fake_ioctl(int /*fd*/, unsigned long request, void* arg) {
  FakeDevice& d = *g_device;
  ++d.ioctl_calls;
  if (!d.scripted_errnos.empty()) {
    const int e = d.scripted_errnos.front();
    d.scripted_errnos.pop_front();
    if (e == 0) {
      return 0;
    }
    errno = e;
    return -1;
  }
  switch (request) {
    case DRM_IOCTL_AUTH_MAGIC:
      errno = d.master ? EINVAL : EACCES;  // a master gets "no such magic"
      return -1;
    case DRM_IOCTL_SET_MASTER:
    case DRM_IOCTL_DROP_MASTER:
      return 0;
    case DRM_IOCTL_MODE_GETRESOURCES: {
      auto* res = static_cast<drm_mode_card_res*>(arg);
      fill_array(res->connector_id_ptr, res->count_connectors, d.connectors);
      res->count_crtcs = 1;
      res->count_encoders = 1;
      res->count_fbs = 0;
      if (d.hotplug_connector_after_count != 0) {
        d.connectors.push_back(d.hotplug_connector_after_count);
        d.hotplug_connector_after_count = 0;
      }
      return 0;
    }
    case DRM_IOCTL_MODE_GETCONNECTOR: {
      auto* c = static_cast<drm_mode_get_connector*>(arg);
      if (c->connector_id != 7) {
        errno = ENOENT;
        return -1;
      }
      if (d.first_connector_count_modes == UINT32_MAX) {
        d.first_connector_count_modes = c->count_modes;
      }
      fill_array(c->encoders_ptr, c->count_encoders, d.encoders);
      fill_array(c->modes_ptr, c->count_modes, d.modes);
      const uint32_t props_capacity = c->count_props;
      fill_array(c->props_ptr, c->count_props, d.connector_props);
      uint32_t values_capacity = props_capacity;
      fill_array(c->prop_values_ptr, values_capacity, d.connector_prop_values);
      c->encoder_id = d.encoder_id;
      c->connector_type = d.connector_type;
      c->connector_type_id = d.connector_type_id;
      c->connection = d.connection;
      return 0;
    }
    case DRM_IOCTL_MODE_GETENCODER: {
      auto* e = static_cast<drm_mode_get_encoder*>(arg);
      if (e->encoder_id != 30) {
        errno = ENOENT;
        return -1;
      }
      e->encoder_type = DRM_MODE_ENCODER_TMDS;
      e->crtc_id = d.crtc_id;
      return 0;
    }
    case DRM_IOCTL_MODE_GETCRTC: {
      auto* c = static_cast<drm_mode_crtc*>(arg);
      if (c->crtc_id != 40) {
        errno = ENOENT;
        return -1;
      }
      c->mode_valid = 1;
      c->mode = d.crtc_mode;
      return 0;
    }
    case DRM_IOCTL_MODE_OBJ_GETPROPERTIES: {
      auto* p = static_cast<drm_mode_obj_get_properties*>(arg);
      if (p->obj_id != 7 || p->obj_type != DRM_MODE_OBJECT_CONNECTOR) {
        errno = ENOENT;
        return -1;
      }
      const uint32_t capacity = p->count_props;
      fill_array(p->props_ptr, p->count_props, d.connector_props);
      uint32_t values_capacity = capacity;
      fill_array(p->prop_values_ptr, values_capacity, d.connector_prop_values);
      return 0;
    }
    case DRM_IOCTL_MODE_GETPROPERTY: {
      auto* p = static_cast<drm_mode_get_property*>(arg);
      const std::string* name = p->prop_id == 100   ? &d.prop_name_100
                                : p->prop_id == 101 ? &d.prop_name_101
                                                    : nullptr;
      if (name == nullptr) {
        errno = ENOENT;
        return -1;
      }
      std::memset(p->name, 0, sizeof(p->name));
      std::memcpy(p->name, name->data(), name->size());
      p->count_values = 1;
      p->count_enum_blobs = 0;
      return 0;
    }
    case DRM_IOCTL_MODE_GETPROPBLOB: {
      auto* b = static_cast<drm_mode_get_blob*>(arg);
      if (b->blob_id != 500) {
        errno = ENOENT;
        return -1;
      }
      fill_array(b->data, b->length, d.blob_500);
      return 0;
    }
    default:
      errno = ENOTTY;
      return -1;
  }
}

class DrmIoctl : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = FakeDevice{};
    device_.connectors = {7};
    device_.modes.resize(2);
    device_.modes[0].hdisplay = 1920;
    device_.modes[0].vdisplay = 1080;
    device_.modes[0].vrefresh = 60;
    device_.crtc_mode.hdisplay = 2560;
    device_.crtc_mode.vdisplay = 1440;
    device_.crtc_mode.vrefresh = 144;
    // A base EDID block whose descriptor slot 1 is a range-limits descriptor (48-165 Hz).
    device_.blob_500.assign(128, 0);
    device_.blob_500[54 + 3] = 0xFD;
    device_.blob_500[54 + 5] = 48;
    device_.blob_500[54 + 6] = 165;
    g_device = &device_;
  }
  void TearDown() override { g_device = nullptr; }

  FakeDevice device_;
  drm::Node node_{.fd = 42, .ioctl_fn = fake_ioctl};
};

TEST_F(DrmIoctl, RetriesOnEintrAndEagain) {
  device_.scripted_errnos = {EINTR, EAGAIN, 0};
  EXPECT_EQ(drm::drm_ioctl(node_, DRM_IOCTL_SET_MASTER, nullptr), 0);
  EXPECT_EQ(device_.ioctl_calls, 3);
}

TEST_F(DrmIoctl, ReturnsNegativeErrnoOnOtherFailures) {
  device_.scripted_errnos = {EPERM};
  EXPECT_EQ(drm::drm_ioctl(node_, DRM_IOCTL_SET_MASTER, nullptr), -EPERM);
  EXPECT_EQ(device_.ioctl_calls, 1);
}

TEST_F(DrmIoctl, IsMasterFollowsAuthMagicContract) {
  device_.master = true;  // AUTH_MAGIC(0) -> EINVAL: still master
  EXPECT_TRUE(drm::is_master(node_));
  device_.master = false;  // -> EACCES
  EXPECT_FALSE(drm::is_master(node_));
  device_.scripted_errnos = {0};  // an outright success also means master
  EXPECT_TRUE(drm::is_master(node_));
}

TEST_F(DrmIoctl, SetAndDropMasterSurfaceErrnoText) {
  EXPECT_TRUE(drm::set_master(node_).has_value());
  EXPECT_TRUE(drm::drop_master(node_).has_value());
  device_.scripted_errnos = {EACCES};
  const auto failed = drm::set_master(node_);
  ASSERT_FALSE(failed.has_value());
  EXPECT_NE(failed.error().message.find("DRM_IOCTL_SET_MASTER"), std::string::npos);
  EXPECT_NE(failed.error().message.find(std::strerror(EACCES)), std::string::npos);
}

TEST_F(DrmIoctl, GetResourcesTwoPass) {
  device_.connectors = {7, 8, 9};
  const auto res = drm::get_resources(node_);
  ASSERT_TRUE(res.has_value()) << res.error().message;
  EXPECT_EQ(res->connectors, (std::vector<uint32_t>{7, 8, 9}));
  EXPECT_EQ(res->count_crtcs, 1u);
  EXPECT_EQ(device_.ioctl_calls, 2);
}

TEST_F(DrmIoctl, GetResourcesRetriesWhenCountGrowsBetweenPasses) {
  device_.connectors = {7};
  device_.hotplug_connector_after_count = 8;
  const auto res = drm::get_resources(node_);
  ASSERT_TRUE(res.has_value()) << res.error().message;
  EXPECT_EQ(res->connectors, (std::vector<uint32_t>{7, 8}));
  EXPECT_EQ(device_.ioctl_calls, 3);  // count, undersized fill, refill
}

TEST_F(DrmIoctl, GetResourcesFailureMessage) {
  device_.scripted_errnos = {EOPNOTSUPP};
  const auto res = drm::get_resources(node_);
  ASSERT_FALSE(res.has_value());
  EXPECT_NE(res.error().message.find("DRM_IOCTL_MODE_GETRESOURCES"), std::string::npos);
  EXPECT_NE(res.error().message.find(std::strerror(EOPNOTSUPP)), std::string::npos);
}

TEST_F(DrmIoctl, GetConnectorProbesThenFills) {
  const auto connector = drm::get_connector(node_, 7);
  ASSERT_TRUE(connector.has_value()) << connector.error().message;
  // The counting pass sends count_modes == 0 (force-probe when master, like libdrm).
  EXPECT_EQ(device_.first_connector_count_modes, 0u);
  EXPECT_EQ(connector->connector_id, 7u);
  EXPECT_EQ(connector->connector_type, DRM_MODE_CONNECTOR_DisplayPort);
  EXPECT_EQ(connector->connector_type_id, 4u);
  EXPECT_EQ(connector->connection, drm::kConnectionConnected);
  EXPECT_EQ(connector->encoder_id, 30u);
  EXPECT_EQ(connector->encoders, (std::vector<uint32_t>{30}));
  ASSERT_EQ(connector->modes.size(), 2u);
  EXPECT_EQ(connector->modes[0].hdisplay, 1920);
  EXPECT_EQ(connector->props, (std::vector<uint32_t>{100, 101}));
  EXPECT_EQ(connector->prop_values, (std::vector<uint64_t>{1, 500}));
  EXPECT_EQ(device_.ioctl_calls, 2);
}

TEST_F(DrmIoctl, GetConnectorWithoutModes) {
  device_.modes.clear();
  device_.connection = drm::kConnectionDisconnected;
  const auto connector = drm::get_connector(node_, 7);
  ASSERT_TRUE(connector.has_value()) << connector.error().message;
  EXPECT_TRUE(connector->modes.empty());
  EXPECT_EQ(connector->connection, drm::kConnectionDisconnected);
}

TEST_F(DrmIoctl, GetConnectorUnknownIdFails) {
  const auto connector = drm::get_connector(node_, 99);
  ASSERT_FALSE(connector.has_value());
  EXPECT_NE(connector.error().message.find(std::strerror(ENOENT)), std::string::npos);
}

TEST_F(DrmIoctl, EncoderAndCrtcScalars) {
  const auto encoder = drm::get_encoder(node_, 30);
  ASSERT_TRUE(encoder.has_value()) << encoder.error().message;
  EXPECT_EQ(encoder->crtc_id, 40u);
  const auto crtc = drm::get_crtc(node_, 40);
  ASSERT_TRUE(crtc.has_value()) << crtc.error().message;
  EXPECT_TRUE(crtc->mode_valid);
  EXPECT_EQ(crtc->mode.hdisplay, 2560);
  EXPECT_EQ(crtc->mode.vdisplay, 1440);
  EXPECT_EQ(crtc->mode.vrefresh, 144u);
  EXPECT_FALSE(drm::get_crtc(node_, 41).has_value());
}

TEST_F(DrmIoctl, ConnectorPropertiesResolveVrrAndEdidFloor) {
  // The exact lookup display_ownership_drm.cpp performs.
  const auto props = drm::get_object_properties(node_, 7, DRM_MODE_OBJECT_CONNECTOR);
  ASSERT_TRUE(props.has_value()) << props.error().message;
  ASSERT_EQ(props->props.size(), 2u);
  bool vrr_capable = false;
  uint32_t edid_min_hz = 0;
  for (size_t i = 0; i < props->props.size(); ++i) {
    const auto name = drm::get_property_name(node_, props->props[i]);
    ASSERT_TRUE(name.has_value()) << name.error().message;
    if (std::strcmp(name->data(), "vrr_capable") == 0) {
      vrr_capable = props->prop_values[i] != 0;
    } else if (std::strcmp(name->data(), "EDID") == 0) {
      const auto blob = drm::get_property_blob(node_, static_cast<uint32_t>(props->prop_values[i]));
      ASSERT_TRUE(blob.has_value()) << blob.error().message;
      EXPECT_EQ(blob->size(), 128u);
      edid_min_hz = holoviz::vulkan::edid_min_vertical_rate_hz(*blob).value_or(0);
    }
  }
  EXPECT_TRUE(vrr_capable);
  EXPECT_EQ(edid_min_hz, 48u);
}

TEST_F(DrmIoctl, PropertyNameIsNulTerminatedAtFullLength) {
  device_.prop_name_100.assign(DRM_PROP_NAME_LEN, 'x');  // fills the kernel buffer exactly
  const auto name = drm::get_property_name(node_, 100);
  ASSERT_TRUE(name.has_value());
  EXPECT_EQ(std::strlen(name->data()), static_cast<size_t>(DRM_PROP_NAME_LEN));
}

TEST_F(DrmIoctl, EmptyBlobAndUnknownBlob) {
  device_.blob_500.clear();
  const auto empty = drm::get_property_blob(node_, 500);
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
  EXPECT_FALSE(drm::get_property_blob(node_, 501).has_value());
}

}  // namespace
