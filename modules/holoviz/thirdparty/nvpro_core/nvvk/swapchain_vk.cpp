/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "swapchain_vk.hpp"
#include "error_vk.hpp"

#include <assert.h>

#include <nvh/nvprint.hpp>
#include <nvvk/debug_util_vk.hpp>
namespace nvvk {
void SwapChain::init(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue,
                     uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkFormat format,
                     VkColorSpaceKHR color_space, VkImageUsageFlags imageUsage) {
  assert(!m_device);
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_swapchain = VK_NULL_HANDLE;
  m_queue = queue;
  m_queueFamilyIndex = queueFamilyIndex;
  m_changeID = 0;
  m_currentSemaphore = 0;
  m_surface = surface;
  m_imageUsage = imageUsage;
  m_surfaceFormat = format;
  m_surfaceColor = color_space;

  uint32_t count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(
      physicalDevice, nullptr, &count, extensionProperties.data()));

  for (const auto& extension : extensionProperties) {
    if (strcmp(extension.extensionName, VK_KHR_PRESENT_ID_EXTENSION_NAME) == 0) {
      m_has_present_id_extension = true;
      break;
    }
  }
}

bool SwapChain::update(int width, int height, VkPresentModeKHR presentMode,
                       VkExtent2D* dimensions) {
  m_changeID++;
  m_presentMode = presentMode;

  VkSwapchainKHR oldSwapchain = m_swapchain;

  if (NVVK_CHECK(waitIdle()))
    return false;

  // Check the surface capabilities and formats
  VkSurfaceCapabilitiesKHR surfCapabilities;
  if (NVVK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
          m_physicalDevice, m_surface, &surfCapabilities)))
    return false;

  VkExtent2D swapchainExtent;
  // width and height are either both -1, or both not -1.
  if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
    // If the surface size is undefined, the size is set to
    // the size of the images requested.
    swapchainExtent.width = width;
    swapchainExtent.height = height;
  } else {
    // If the surface size is defined, the swap chain size must match
    swapchainExtent = surfCapabilities.currentExtent;
  }

  // test against valid size, typically hit when windows are minimized, the app must
  // prevent triggering this code accordingly
  assert(swapchainExtent.width && swapchainExtent.height);

  // Determine the number of VkImage's to use in the swap chain (we desire to
  // own only 1 image at a time, besides the images being displayed and
  // queued for display):
  uint32_t desiredNumberOfSwapchainImages = surfCapabilities.minImageCount;
  if ((surfCapabilities.maxImageCount > 0) &&
      (desiredNumberOfSwapchainImages > surfCapabilities.maxImageCount)) {
    // Application must settle for fewer images than desired:
    desiredNumberOfSwapchainImages = surfCapabilities.maxImageCount;
  }

  VkSurfaceTransformFlagBitsKHR preTransform;
  if (surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  } else {
    preTransform = surfCapabilities.currentTransform;
  }

  VkSwapchainCreateInfoKHR swapchain = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain.surface = m_surface;
  swapchain.minImageCount = desiredNumberOfSwapchainImages;
  swapchain.imageFormat = m_surfaceFormat;
  swapchain.imageColorSpace = m_surfaceColor;
  swapchain.imageExtent = swapchainExtent;
  swapchain.imageUsage = m_imageUsage;
  swapchain.preTransform = preTransform;
  swapchain.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain.imageArrayLayers = 1;
  swapchain.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchain.queueFamilyIndexCount = 1;
  swapchain.pQueueFamilyIndices = &m_queueFamilyIndex;
  swapchain.presentMode = m_presentMode;
  swapchain.oldSwapchain = oldSwapchain;
  swapchain.clipped = true;

  if (NVVK_CHECK(vkCreateSwapchainKHR(m_device, &swapchain, nullptr, &m_swapchain)))
    return false;

  nvvk::DebugUtil debugUtil(m_device);

  debugUtil.setObjectName(m_swapchain, "SwapChain::m_swapchain");

  // If we just re-created an existing swapchain, we should destroy the old
  // swapchain at this point.
  // Note: destroying the swapchain also cleans up all its associated
  // presentable images once the platform is done with them.
  if (oldSwapchain != VK_NULL_HANDLE) {
    for (auto&& it : m_entries) {
      vkDestroyImageView(m_device, it.imageView, nullptr);
      vkDestroySemaphore(m_device, it.readSemaphore, nullptr);
      vkDestroySemaphore(m_device, it.writtenSemaphore, nullptr);
    }
    vkDestroySwapchainKHR(m_device, oldSwapchain, nullptr);
  }

  if (NVVK_CHECK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &m_imageCount, nullptr)))
    return false;

  m_entries.resize(m_imageCount);
  m_barriers.resize(m_imageCount);

  std::vector<VkImage> images(m_imageCount);

  if (NVVK_CHECK(vkGetSwapchainImagesKHR(m_device, m_swapchain, &m_imageCount, images.data())))
    return false;
  //
  // Image views
  //
  for (uint32_t i = 0; i < m_imageCount; i++) {
    Entry& entry = m_entries[i];

    // image
    entry.image = images[i];

    // imageview
    VkImageViewCreateInfo viewCreateInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                            nullptr,
                                            0,
                                            entry.image,
                                            VK_IMAGE_VIEW_TYPE_2D,
                                            m_surfaceFormat,
                                            {VK_COMPONENT_SWIZZLE_R,
                                             VK_COMPONENT_SWIZZLE_G,
                                             VK_COMPONENT_SWIZZLE_B,
                                             VK_COMPONENT_SWIZZLE_A},
                                            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

    if (NVVK_CHECK(vkCreateImageView(m_device, &viewCreateInfo, nullptr, &entry.imageView)))
      return false;

    // semaphore
    VkSemaphoreCreateInfo semCreateInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    if (NVVK_CHECK(vkCreateSemaphore(m_device, &semCreateInfo, nullptr, &entry.readSemaphore)))
      return false;
    if (NVVK_CHECK(vkCreateSemaphore(m_device, &semCreateInfo, nullptr, &entry.writtenSemaphore)))
      return false;

    // initial barriers
    VkImageSubresourceRange range = {0};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = VK_REMAINING_MIP_LEVELS;
    range.baseArrayLayer = 0;
    range.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memBarrier.dstAccessMask = 0;
    memBarrier.srcAccessMask = 0;
    memBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    memBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    memBarrier.image = entry.image;
    memBarrier.subresourceRange = range;

    m_barriers[i] = memBarrier;

    debugUtil.setObjectName(entry.image, "swapchainImage:" + std::to_string(i));
    debugUtil.setObjectName(entry.imageView, "swapchainImageView:" + std::to_string(i));
    debugUtil.setObjectName(entry.readSemaphore, "swapchainReadSemaphore:" + std::to_string(i));
    debugUtil.setObjectName(entry.writtenSemaphore,
                            "swapchainWrittenSemaphore:" + std::to_string(i));
  }

  m_updateWidth = width;
  m_updateHeight = height;
  m_extent = swapchainExtent;

  m_currentSemaphore = 0;
  m_currentImage = 0;

  if (dimensions)
    *dimensions = swapchainExtent;
  return true;
}

void SwapChain::deinitResources() {
  if (!m_device)
    return;

  VkResult result = waitIdle();
  if (nvvk::checkResult(result, __FILE__, __LINE__)) {
    exit(-1);
  }

  for (auto&& it : m_entries) {
    vkDestroyImageView(m_device, it.imageView, nullptr);
    vkDestroySemaphore(m_device, it.readSemaphore, nullptr);
    vkDestroySemaphore(m_device, it.writtenSemaphore, nullptr);
  }

  if (m_swapchain) {
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
  }

  m_entries.clear();
  m_barriers.clear();
}

void SwapChain::deinit() {
  deinitResources();

  m_physicalDevice = VK_NULL_HANDLE;
  m_device = VK_NULL_HANDLE;
  m_surface = VK_NULL_HANDLE;
  m_changeID = 0;
}

bool SwapChain::acquire(bool* pRecreated, SwapChainAcquireState* pOut) {
  return acquireCustom(VK_NULL_HANDLE, m_updateWidth, m_updateHeight, pRecreated, pOut);
}

bool SwapChain::acquireAutoResize(int width, int height, bool* pRecreated,
                                  SwapChainAcquireState* pOut) {
  return acquireCustom(VK_NULL_HANDLE, width, height, pRecreated, pOut);
}

bool SwapChain::acquireCustom(VkSemaphore argSemaphore, bool* pRecreated,
                              SwapChainAcquireState* pOut) {
  return acquireCustom(argSemaphore, m_updateWidth, m_updateHeight, pRecreated, pOut);
}

bool SwapChain::acquireCustom(VkSemaphore argSemaphore, int width, int height, bool* pRecreated,
                              SwapChainAcquireState* pOut) {
  bool didRecreate = false;

  if (width != m_updateWidth || height != m_updateHeight) {
    deinitResources();
    update(width, height, m_presentMode);
    m_updateWidth = width;
    m_updateHeight = height;
    didRecreate = true;
  }
  if (pRecreated != nullptr) {
    *pRecreated = didRecreate;
  }

  // try recreation a few times
  for (int i = 0; i < 2; i++) {
    VkSemaphore semaphore = argSemaphore ? argSemaphore : getActiveReadSemaphore();
    VkResult result;
    result = vkAcquireNextImageKHR(
        m_device, m_swapchain, UINT64_MAX, semaphore, (VkFence)VK_NULL_HANDLE, &m_currentImage);

    if (result == VK_SUCCESS) {
      if (pOut != nullptr) {
        pOut->image = getActiveImage();
        pOut->view = getActiveImageView();
        pOut->index = getActiveImageIndex();
        pOut->waitSem = getActiveReadSemaphore();
        pOut->signalSem = getActiveWrittenSemaphore();
      }
      return true;
    } else if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      deinitResources();
      update(width, height, m_presentMode);
    } else {
      return false;
    }
  }

  return false;
}

VkSemaphore SwapChain::getActiveWrittenSemaphore() const {
  return m_entries[(m_currentSemaphore % m_imageCount)].writtenSemaphore;
}

VkSemaphore SwapChain::getActiveReadSemaphore() const {
  return m_entries[(m_currentSemaphore % m_imageCount)].readSemaphore;
}

VkImage SwapChain::getActiveImage() const {
  return m_entries[m_currentImage].image;
}

VkImageView SwapChain::getActiveImageView() const {
  return m_entries[m_currentImage].imageView;
}

VkImage SwapChain::getImage(uint32_t i) const {
  if (i >= m_imageCount)
    return nullptr;
  return m_entries[i].image;
}

void SwapChain::present(VkQueue queue) {
  VkResult result;
  VkPresentInfoKHR presentInfo;

  presentCustom(presentInfo);

  VkPresentIdKHR present_id = {VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
  if (m_has_present_id_extension) {
    present_id.swapchainCount = 1;
    present_id.pPresentIds = &m_current_present_id;
    presentInfo.pNext = &present_id;
  }

  result = vkQueuePresentKHR(queue, &presentInfo);
  if ((result == VK_SUCCESS) && m_has_present_id_extension) {
    m_current_present_id++;
  }
  // assert(result == VK_SUCCESS); // can fail on application exit
}

void SwapChain::presentCustom(VkPresentInfoKHR& presentInfo) {
  VkSemaphore& written = m_entries[(m_currentSemaphore % m_imageCount)].writtenSemaphore;

  presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  presentInfo.swapchainCount = 1;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &written;
  presentInfo.pSwapchains = &m_swapchain;
  presentInfo.pImageIndices = &m_currentImage;

  m_currentSemaphore++;
}

void SwapChain::cmdUpdateBarriers(VkCommandBuffer cmd) const {
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       0,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       m_imageCount,
                       m_barriers.data());
}

uint32_t SwapChain::getChangeID() const {
  return m_changeID;
}

VkImageView SwapChain::getImageView(uint32_t i) const {
  if (i >= m_imageCount)
    return nullptr;
  return m_entries[i].imageView;
}

}  // namespace nvvk
