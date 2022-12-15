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

#include "vulkan.hpp"

#include <vulkan/spv/geometry_shader.glsl.frag.h>
#include <vulkan/spv/geometry_shader.glsl.vert.h>
#include <vulkan/spv/geometry_text_shader.glsl.frag.h>
#include <vulkan/spv/geometry_text_shader.glsl.vert.h>
#include <vulkan/spv/image_lut_float_shader.glsl.frag.h>
#include <vulkan/spv/image_lut_uint_shader.glsl.frag.h>
#include <vulkan/spv/image_shader.glsl.frag.h>
#include <vulkan/spv/image_shader.glsl.vert.h>
#include <unistd.h>
#include <nvmath/nvmath.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../cuda/convert.hpp"
#include "../cuda/cuda_service.hpp"

#include <nvh/fileoperations.hpp>
#include <nvvk/appbase_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>

#include "../layers/layer.hpp"
#include "framebuffer_sequence.hpp"

namespace holoscan::viz {

struct PushConstantFragment {
    float opacity;
};

struct PushConstantVertex {
    nvmath::mat4f matrix;
    float point_size;
    std::array<float, 4> color;
};

struct PushConstantTextVertex {
    nvmath::mat4f matrix;
};

struct Vulkan::Texture {
    Texture(uint32_t width, uint32_t height, ImageFormat format, nvvk::ResourceAllocator *alloc)
        : width_(width)
        , height_(height)
        , format_(format)
        , alloc_(alloc) {
    }

    const uint32_t width_;
    const uint32_t height_;
    const ImageFormat format_;
    nvvk::ResourceAllocator *const alloc_;

    enum class State {
        UNKNOWN,
        UPLOADED,
        RENDERED
    };
    State state_ = State::UNKNOWN;

    nvvk::Texture texture_{};
    UniqueCUexternalMemory external_mem_;
    UniqueCUmipmappedArray mipmap_;

    /// this semaphore is used to synchronize uploading and rendering, it's signaled by Cuda
    /// on upload and waited on by vulkan on rendering
    VkSemaphore upload_semaphore_ = nullptr;
    UniqueCUexternalSemaphore cuda_upload_semaphore_;

    /// this semaphore is used to synchronize rendering and uploading, it's signaled by Vulkan
    /// on rendering and waited on by Cuda on uploading
    VkSemaphore render_semaphore_ = nullptr;
    UniqueCUexternalSemaphore cuda_render_semaphore_;

    VkFence fence_ = nullptr;  ///< last usage of the texture, need to sync before destroying memory
};

struct Vulkan::Buffer {
    Buffer(size_t size, nvvk::ResourceAllocator *alloc)
        : size_(size)
        , alloc_(alloc) {
    }

    const size_t size_;
    nvvk::ResourceAllocator *const alloc_;

    nvvk::Buffer buffer_{};

    VkFence fence_ = nullptr;  ///< last usage of the buffer, need to sync before destroying memory
};

class Vulkan::Impl {
 public:
    Impl() = default;
    virtual ~Impl();

    void setup(Window *window);

    void begin_transfer_pass();
    void end_transfer_pass();
    void begin_render_pass();
    void end_render_pass();
    void cleanup_transfer_jobs();

    void prepare_frame();
    void submit_frame();
    uint32_t get_active_image_index() const {
        return fb_sequence_.get_active_image_index();
    }
    const std::vector<VkCommandBuffer> &get_command_buffers() {
        return command_buffers_;
    }

    Texture *create_texture_for_cuda_upload(uint32_t width, uint32_t height,
                                            ImageFormat format, VkFilter filter,
                                            bool normalized);
    Texture *create_texture(uint32_t width, uint32_t height, ImageFormat format,
                            size_t data_size, const void *data,
                            VkFilter filter, bool normalized);
    void destroy_texture(Texture *texture);

    void upload_to_texture(CUdeviceptr device_ptr, Texture *texture, CUstream stream);
    void upload_to_texture(const void *host_ptr, Texture *texture);

    void draw_texture(Texture *texture, Texture *lut, float opacity);

    Buffer *create_buffer(size_t data_size, VkBufferUsageFlags usage, const void *data = nullptr);

    void destroy_buffer(Buffer *buffer);

    void draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, Buffer *buffer,
            float opacity, const std::array<float, 4> &color, float point_size, float line_width);

    void draw_indexed(VkDescriptorSet desc_set, Buffer *vertex_buffer, Buffer *index_buffer,
                      VkIndexType index_type, uint32_t index_count, uint32_t first_index,
                      uint32_t vertex_offset, float opacity);

    void read_framebuffer(ImageFormat fmt, size_t buffer_size, CUdeviceptr device_ptr,
                                                                     CUstream stream);

 private:
    void init_im_gui();
    bool create_framebuffer_sequence();
    void create_depth_buffer();
    void create_render_pass();

    /**
     * Create all the framebuffers in which the image will be rendered
     * - Swapchain need to be created before calling this
     */
    void create_frame_buffers();

    /**
     * Callback when the window is resized
     * - Destroy allocated frames, then rebuild them with the new size
     */
    void on_framebuffer_size(int w, int h);

    VkCommandBuffer create_temp_cmd_buffer();
    void submit_temp_cmd_buffer(VkCommandBuffer cmdBuffer);

    uint32_t get_memory_type(uint32_t typeBits, const VkMemoryPropertyFlags &properties) const;

    VkPipeline create_pipeline(VkPipelineLayout pipeline_layout, const uint32_t *vertex_shader,
                               size_t vertex_shader_size, const uint32_t *fragment_shader,
                               size_t fragment_shader_size, VkPrimitiveTopology topology,
                               const std::vector<VkDynamicState> dynamic_state = {},
                               bool imgui_attrib_desc = false);
    UniqueCUexternalSemaphore import_semaphore_to_cuda(VkSemaphore semaphore);

    Window *window_ = nullptr;

    // Vulkan low level
    VkSurfaceKHR surface_              = VK_NULL_HANDLE;
    VkQueue queue_                     = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_            = VK_NULL_HANDLE;
    VkDescriptorPool im_gui_desc_pool_ = VK_NULL_HANDLE;

    // Drawing/Surface
    FramebufferSequence fb_sequence_;
    std::vector<VkFramebuffer> framebuffers_;          // All framebuffers,
                                                       //  correspond to the Swapchain
    std::vector<VkCommandBuffer> command_buffers_;     // Command buffer per nb element in Swapchain
    std::vector<VkFence> wait_fences_;                 // Fences per nb element in Swapchain
    VkImage depth_image_         = VK_NULL_HANDLE;     // Depth/Stencil
    VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;     // Depth/Stencil
    VkImageView depth_view_      = VK_NULL_HANDLE;     // Depth/Stencil
    VkRenderPass render_pass_    = VK_NULL_HANDLE;     // Base render pass
    VkExtent2D size_{0, 0};                            // Size of the window
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;  // Cache for pipeline/shaders

    // Depth buffer format
    VkFormat depth_format_{VK_FORMAT_UNDEFINED};

    // allocators
    nvvk::ResourceAllocatorDma alloc_;            ///< Allocator for buffer, images,
                                                  ///        acceleration structures
    nvvk::ExportResourceAllocator export_alloc_;  ///< Allocator for allocations
                                                  ///        which can be exported

    nvvk::Context vk_ctx_{};

    nvvk::BatchSubmission batch_submission_;

    nvvk::CommandPool transfer_cmd_pool_;
    struct TransferJob {
        VkCommandBuffer cmd_buffer_ = nullptr;
        VkSemaphore semaphore_      = nullptr;
        VkFence fence_              = nullptr;
        VkFence frame_fence_        = nullptr;
    };
    std::list<TransferJob> transfer_jobs_;

    nvvk::Buffer vertex_buffer_{};
    nvvk::Buffer index_buffer_{};

    VkPipelineLayout image_pipeline_layout_         = nullptr;
    VkPipelineLayout image_lut_pipeline_layout_     = nullptr;
    VkPipelineLayout geometry_pipeline_layout_      = nullptr;
    VkPipelineLayout geometry_text_pipeline_layout_ = nullptr;

    const uint32_t bindings_offset_texture_     = 0;
    const uint32_t bindings_offset_texture_lut_ = 1;

    nvvk::DescriptorSetBindings desc_set_layout_bind_;
    VkDescriptorSetLayout desc_set_layout_ = nullptr;

    nvvk::DescriptorSetBindings desc_set_layout_bind_lut_;
    VkDescriptorSetLayout desc_set_layout_lut_ = nullptr;

    nvvk::DescriptorSetBindings desc_set_layout_bind_text_;
    VkDescriptorSetLayout desc_set_layout_text_ = nullptr;
    VkDescriptorPool desc_pool_text_            = nullptr;
    VkDescriptorSet desc_set_text_              = nullptr;
    VkSampler sampler_text_                     = nullptr;

    VkPipeline image_pipeline_           = nullptr;
    VkPipeline image_lut_uint_pipeline_  = nullptr;
    VkPipeline image_lut_float_pipeline_ = nullptr;

    VkPipeline geometry_point_pipeline_      = nullptr;
    VkPipeline geometry_line_pipeline_       = nullptr;
    VkPipeline geometry_line_strip_pipeline_ = nullptr;
    VkPipeline geometry_triangle_pipeline_   = nullptr;
    VkPipeline geometry_text_pipeline_       = nullptr;

    ImGuiContext *im_gui_context_ = nullptr;
};

Vulkan::Impl::~Impl() {
    if (vk_ctx_.m_device) {
        NVVK_CHECK(vkDeviceWaitIdle(vk_ctx_.m_device));

        cleanup_transfer_jobs();

        vkDestroyDescriptorSetLayout(vk_ctx_.m_device, desc_set_layout_lut_, nullptr);
        vkDestroyDescriptorSetLayout(vk_ctx_.m_device, desc_set_layout_, nullptr);

        vkDestroyDescriptorSetLayout(vk_ctx_.m_device, desc_set_layout_text_, nullptr);
        vkDestroyDescriptorPool(vk_ctx_.m_device, desc_pool_text_, nullptr);
        alloc_.releaseSampler(sampler_text_);

        vkDestroyPipeline(vk_ctx_.m_device, geometry_text_pipeline_, nullptr);
        geometry_text_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, geometry_triangle_pipeline_, nullptr);
        geometry_triangle_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, geometry_line_strip_pipeline_, nullptr);
        geometry_line_strip_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, geometry_line_pipeline_, nullptr);
        geometry_line_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, geometry_point_pipeline_, nullptr);
        geometry_point_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, image_lut_float_pipeline_, nullptr);
        image_lut_float_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, image_lut_uint_pipeline_, nullptr);
        image_lut_uint_pipeline_ = nullptr;
        vkDestroyPipeline(vk_ctx_.m_device, image_pipeline_, nullptr);
        image_pipeline_ = nullptr;

        vkDestroyPipelineLayout(vk_ctx_.m_device, geometry_text_pipeline_layout_, nullptr);
        geometry_text_pipeline_layout_ = nullptr;
        vkDestroyPipelineLayout(vk_ctx_.m_device, geometry_pipeline_layout_, nullptr);
        geometry_pipeline_layout_ = nullptr;
        vkDestroyPipelineLayout(vk_ctx_.m_device, image_lut_pipeline_layout_, nullptr);
        image_lut_pipeline_layout_ = nullptr;
        vkDestroyPipelineLayout(vk_ctx_.m_device, image_pipeline_layout_, nullptr);
        image_pipeline_layout_ = nullptr;

        alloc_.destroy(index_buffer_);
        alloc_.destroy(vertex_buffer_);

        transfer_cmd_pool_.deinit();

        if (ImGui::GetCurrentContext() != nullptr) {
            ImGui_ImplVulkan_Shutdown();
            if (im_gui_context_) {
                ImGui::DestroyContext(im_gui_context_);
            }
        }

        vkDestroyRenderPass(vk_ctx_.m_device, render_pass_, nullptr);

        vkDestroyImageView(vk_ctx_.m_device, depth_view_, nullptr);
        vkDestroyImage(vk_ctx_.m_device, depth_image_, nullptr);
        vkFreeMemory(vk_ctx_.m_device, depth_memory_, nullptr);
        vkDestroyPipelineCache(vk_ctx_.m_device, pipeline_cache_, nullptr);

        for (uint32_t i = 0; i < fb_sequence_.get_image_count(); i++) {
            vkDestroyFence(vk_ctx_.m_device, wait_fences_[i], nullptr);

            vkDestroyFramebuffer(vk_ctx_.m_device, framebuffers_[i], nullptr);

            vkFreeCommandBuffers(vk_ctx_.m_device, cmd_pool_, 1, &command_buffers_[i]);
        }
        fb_sequence_.deinit();
        vkDestroyDescriptorPool(vk_ctx_.m_device, im_gui_desc_pool_, nullptr);
        vkDestroyCommandPool(vk_ctx_.m_device, cmd_pool_, nullptr);

        if (surface_) {
            vkDestroySurfaceKHR(vk_ctx_.m_instance, surface_, nullptr);
        }

        export_alloc_.deinit();
        alloc_.deinit();
    }

    vk_ctx_.deinit();
}

void Vulkan::Impl::setup(Window *window) {
    window_ = window;

#ifdef NDEBUG
    nvvk::ContextCreateInfo context_info;
#else
    nvvk::ContextCreateInfo context_info(true /*bUseValidation*/);
#endif

    context_info.setVersion(1, 2);  // Using Vulkan 1.2

    // Requesting Vulkan extensions and layers
    uint32_t count{0};
    const char **req_extensions = window_->get_required_instance_extensions(&count);
    for (uint32_t ext_id = 0; ext_id < count; ext_id++)
        context_info.addInstanceExtension(req_extensions[ext_id]);

    // Allow debug names
    context_info.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);
    context_info.addInstanceExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);

    req_extensions = window_->get_required_device_extensions(&count);
    for (uint32_t ext_id = 0; ext_id < count; ext_id++)
        context_info.addDeviceExtension(req_extensions[ext_id]);

    context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    // Creating Vulkan base application
    if (!vk_ctx_.initInstance(context_info)) {
        throw std::runtime_error("Failed to create the Vulkan instance.");
    }

    // Find all compatible devices
    const std::vector<uint32_t> compatible_devices = vk_ctx_.getCompatibleDevices(context_info);
    if (compatible_devices.empty()) {
        throw std::runtime_error("No Vulkan capable GPU present.");
    }
    // Use a compatible device
    vk_ctx_.initDevice(compatible_devices[0], context_info);

    // Find the most suitable depth format
    {
        const VkFormatFeatureFlagBits feature = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
        for (const auto &f : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                                         VK_FORMAT_D16_UNORM_S8_UINT}) {
            VkFormatProperties formatProp{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
            vkGetPhysicalDeviceFormatProperties(vk_ctx_.m_physicalDevice, f, &formatProp);
            if ((formatProp.optimalTilingFeatures & feature) == feature) {
                depth_format_ = f;
                break;
            }
        }
        if (depth_format_ == VK_FORMAT_UNDEFINED) {
            throw std::runtime_error("Could not find a suitable depth format.");
        }
    }

    // create a surface, headless windows don't have a surface
    surface_ = window_->create_surface(vk_ctx_.m_physicalDevice, vk_ctx_.m_instance);
    if (surface_) {
        if (!vk_ctx_.setGCTQueueWithPresent(surface_)) {
            throw std::runtime_error("Surface not supported by queue");
        }
    }

    vkGetDeviceQueue(vk_ctx_.m_device, vk_ctx_.m_queueGCT.familyIndex, 0, &queue_);

    VkCommandPoolCreateInfo pool_reate_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_reate_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    NVVK_CHECK(vkCreateCommandPool(vk_ctx_.m_device, &pool_reate_info, nullptr, &cmd_pool_));

    VkPipelineCacheCreateInfo pipeline_cache_info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    NVVK_CHECK(vkCreatePipelineCache(vk_ctx_.m_device, &pipeline_cache_info,
                                                nullptr, &pipeline_cache_));

    alloc_.init(vk_ctx_.m_instance, vk_ctx_.m_device, vk_ctx_.m_physicalDevice);
    export_alloc_.init(vk_ctx_.m_device, vk_ctx_.m_physicalDevice, alloc_.getMemoryAllocator());

    create_framebuffer_sequence();
    create_depth_buffer();
    create_render_pass();
    create_frame_buffers();

    // init batch submission
    batch_submission_.init(vk_ctx_.m_queueGCT);

    // init command pool
    transfer_cmd_pool_.init(vk_ctx_.m_device, vk_ctx_.m_queueT.familyIndex);

    // allocate the vertex and index buffer for the image draw pass
    {
        nvvk::CommandPool cmd_buf_get(vk_ctx_.m_device, vk_ctx_.m_queueGCT.familyIndex);
        VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

        const std::vector<float> vertices{-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
        vertex_buffer_ = alloc_.createBuffer(cmd_buf, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        const std::vector<uint16_t> indices{0, 2, 1, 2, 0, 3};
        index_buffer_ = alloc_.createBuffer(cmd_buf, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        cmd_buf_get.submitAndWait(cmd_buf);
        alloc_.finalizeAndReleaseStaging();
    }

    // create the descriptor sets
    desc_set_layout_bind_.addBinding(bindings_offset_texture_,
                                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_ =
        desc_set_layout_bind_.createLayout(vk_ctx_.m_device,
                                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    desc_set_layout_bind_lut_.addBinding(bindings_offset_texture_,
                                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_bind_lut_.addBinding(bindings_offset_texture_lut_,
                                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_lut_ = desc_set_layout_bind_lut_.createLayout(
        vk_ctx_.m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter     = VK_FILTER_LINEAR;
        info.minFilter     = VK_FILTER_LINEAR;
        info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.minLod        = -1000;
        info.maxLod        = 1000;
        info.maxAnisotropy = 1.0f;
        sampler_text_      = alloc_.acquireSampler(info);
    }
    desc_set_layout_bind_text_.addBinding(bindings_offset_texture_,
                                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                          VK_SHADER_STAGE_FRAGMENT_BIT, &sampler_text_);
    desc_set_layout_text_ = desc_set_layout_bind_text_.createLayout(vk_ctx_.m_device);
    desc_pool_text_       = desc_set_layout_bind_text_.createPool(vk_ctx_.m_device);
    desc_set_text_        = nvvk::allocateDescriptorSet(vk_ctx_.m_device,
                                                        desc_pool_text_,
                                                        desc_set_layout_text_);

    // create the pipeline layout for images
    {
        // Push constants
        VkPushConstantRange push_constant_ranges{};
        push_constant_ranges.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant_ranges.offset     = 0;
        push_constant_ranges.size       = sizeof(PushConstantFragment);

        // Creating the Pipeline Layout
        VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        create_info.setLayoutCount         = 1;
        create_info.pSetLayouts            = &desc_set_layout_;
        create_info.pushConstantRangeCount = 1;
        create_info.pPushConstantRanges    = &push_constant_ranges;
        NVVK_CHECK(vkCreatePipelineLayout(vk_ctx_.m_device, &create_info,
                                          nullptr, &image_pipeline_layout_));
    }

    // Create the Pipeline
    image_pipeline_ = create_pipeline(
        image_pipeline_layout_, image_shader_glsl_vert,
        sizeof(image_shader_glsl_vert) / sizeof(image_shader_glsl_vert[0]), image_shader_glsl_frag,
        sizeof(image_shader_glsl_frag) / sizeof(image_shader_glsl_frag[0]),
                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // create the pipeline layout for images with lut
    {
        // Push constants
        VkPushConstantRange push_constant_ranges{};
        push_constant_ranges.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant_ranges.offset     = 0;
        push_constant_ranges.size       = sizeof(PushConstantFragment);

        // Creating the Pipeline Layout
        VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        create_info.setLayoutCount         = 1;
        create_info.pSetLayouts            = &desc_set_layout_lut_;
        create_info.pushConstantRangeCount = 1;
        create_info.pPushConstantRanges    = &push_constant_ranges;
        NVVK_CHECK(vkCreatePipelineLayout(vk_ctx_.m_device, &create_info,
                                          nullptr, &image_lut_pipeline_layout_));
    }

    image_lut_uint_pipeline_ = create_pipeline(
        image_lut_pipeline_layout_, image_shader_glsl_vert,
        sizeof(image_shader_glsl_vert) / sizeof(image_shader_glsl_vert[0]),
        image_lut_uint_shader_glsl_frag,
        sizeof(image_lut_uint_shader_glsl_frag) / sizeof(image_lut_uint_shader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    image_lut_float_pipeline_ = create_pipeline(
        image_lut_pipeline_layout_, image_shader_glsl_vert,
        sizeof(image_shader_glsl_vert) / sizeof(image_shader_glsl_vert[0]),
        image_lut_float_shader_glsl_frag,
        sizeof(image_lut_float_shader_glsl_frag) / sizeof(image_lut_float_shader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // create the pipeline layout for geometry
    {
        // Push constants
        VkPushConstantRange push_constant_ranges[2]{};
        push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push_constant_ranges[0].offset     = 0;
        push_constant_ranges[0].size       = sizeof(PushConstantVertex);
        push_constant_ranges[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant_ranges[1].offset     = sizeof(PushConstantVertex);
        push_constant_ranges[1].size       = sizeof(PushConstantFragment);

        VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        create_info.pushConstantRangeCount = 2;
        create_info.pPushConstantRanges    = push_constant_ranges;
        NVVK_CHECK(vkCreatePipelineLayout(vk_ctx_.m_device, &create_info,
                                          nullptr, &geometry_pipeline_layout_));
    }

    geometry_point_pipeline_ = create_pipeline(
        geometry_pipeline_layout_, geometry_shader_glsl_vert,
        sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
        geometry_shader_glsl_frag,
        sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    geometry_line_pipeline_       = create_pipeline(geometry_pipeline_layout_,
                                                    geometry_shader_glsl_vert,
                                                    sizeof(geometry_shader_glsl_vert) /
                                                     sizeof(geometry_shader_glsl_vert[0]),
                                                    geometry_shader_glsl_frag,
                                                    sizeof(geometry_shader_glsl_frag) /
                                                     sizeof(geometry_shader_glsl_frag[0]),
                                                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                    {VK_DYNAMIC_STATE_LINE_WIDTH});
    geometry_line_strip_pipeline_ = create_pipeline(
        geometry_pipeline_layout_, geometry_shader_glsl_vert,
        sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
        geometry_shader_glsl_frag,
        sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        {VK_DYNAMIC_STATE_LINE_WIDTH});
    geometry_triangle_pipeline_ = create_pipeline(
        geometry_pipeline_layout_, geometry_shader_glsl_vert,
        sizeof(geometry_shader_glsl_vert) / sizeof(geometry_shader_glsl_vert[0]),
        geometry_shader_glsl_frag,
        sizeof(geometry_shader_glsl_frag) / sizeof(geometry_shader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // create the pipeline layout for text geometry
    {
        // Push constants
        VkPushConstantRange push_constant_ranges[2]{};
        push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push_constant_ranges[0].offset     = 0;
        push_constant_ranges[0].size       = sizeof(PushConstantTextVertex);
        push_constant_ranges[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant_ranges[1].offset     = sizeof(PushConstantTextVertex);
        push_constant_ranges[1].size       = sizeof(PushConstantFragment);

        VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        create_info.setLayoutCount         = 1;
        create_info.pSetLayouts            = &desc_set_layout_text_;
        create_info.pushConstantRangeCount = 2;
        create_info.pPushConstantRanges    = push_constant_ranges;
        NVVK_CHECK(vkCreatePipelineLayout(vk_ctx_.m_device, &create_info,
                                          nullptr, &geometry_text_pipeline_layout_));
    }

    geometry_text_pipeline_ =
        create_pipeline(geometry_text_pipeline_layout_, geometry_text_shader_glsl_vert,
                        sizeof(geometry_text_shader_glsl_vert) /
                         sizeof(geometry_text_shader_glsl_vert[0]),
                        geometry_text_shader_glsl_frag,
                        sizeof(geometry_text_shader_glsl_frag) /
                         sizeof(geometry_text_shader_glsl_frag[0]),
                        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, {}, true);

    // ImGui initialization
    init_im_gui();
    window_->setup_callbacks([this](int width, int height)
                             { this->on_framebuffer_size(width, height); });
    window_->init_im_gui();
}

void Vulkan::Impl::init_im_gui() {
    // if the app did not specify a context, create our own
    if (!ImGui::GetCurrentContext()) {
        im_gui_context_ = ImGui::CreateContext();
    }

    ImGuiIO &io    = ImGui::GetIO();
    io.IniFilename = nullptr;  // Avoiding the INI file
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking

    std::vector<VkDescriptorPoolSize> pool_size{{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
                                                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets       = 2;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes    = pool_size.data();
    NVVK_CHECK(vkCreateDescriptorPool(vk_ctx_.m_device, &pool_info, nullptr, &im_gui_desc_pool_));

    // Setup Platform/Renderer back ends
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance        = vk_ctx_.m_instance;
    init_info.PhysicalDevice  = vk_ctx_.m_physicalDevice;
    init_info.Device          = vk_ctx_.m_device;
    init_info.QueueFamily     = vk_ctx_.m_queueGCT.familyIndex;
    init_info.Queue           = queue_;
    init_info.PipelineCache   = VK_NULL_HANDLE;
    init_info.DescriptorPool  = im_gui_desc_pool_;
    init_info.Subpass         = 0;
    init_info.MinImageCount   = 2;
    init_info.ImageCount      = static_cast<int>(fb_sequence_.get_image_count());
    init_info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = nullptr;
    init_info.Allocator       = nullptr;

    ImGui_ImplVulkan_Init(&init_info, render_pass_);

    // Upload Fonts
    VkCommandBuffer cmd_buf = create_temp_cmd_buffer();
    ImGui_ImplVulkan_CreateFontsTexture(cmd_buf);
    submit_temp_cmd_buffer(cmd_buf);

    // set the default font
    ImGui::SetCurrentFont(ImGui::GetDefaultFont());
}

void Vulkan::Impl::begin_transfer_pass() {
    // create a new transfer job and a command buffer
    transfer_jobs_.emplace_back();
    TransferJob &transfer_job = transfer_jobs_.back();

    transfer_job.cmd_buffer_ = transfer_cmd_pool_.createCommandBuffer();
}

void Vulkan::Impl::end_transfer_pass() {
    if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_ != nullptr)) {
        throw std::runtime_error("Not in transfer pass.");
    }

    TransferJob &transfer_job = transfer_jobs_.back();

    // end the command buffer for this job
    NVVK_CHECK(vkEndCommandBuffer(transfer_job.cmd_buffer_));

    // create the fence and semaphore needed for submission
    VkSemaphoreCreateInfo semaphore_create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    NVVK_CHECK(vkCreateSemaphore(vk_ctx_.m_device, &semaphore_create_info,
                                 nullptr, &transfer_job.semaphore_));
    VkFenceCreateInfo fence_create_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    NVVK_CHECK(vkCreateFence(vk_ctx_.m_device, &fence_create_info, nullptr, &transfer_job.fence_));

    // finalize the staging job for later cleanup of resources
    // associates all current staging resources with the transfer fence
    alloc_.finalizeStaging(transfer_job.fence_);

    // submit staged transfers
    VkSubmitInfo submit_info = nvvk::makeSubmitInfo(1, &transfer_job.cmd_buffer_, 1,
                                                          &transfer_job.semaphore_);
    NVVK_CHECK(vkQueueSubmit(vk_ctx_.m_queueT.queue, 1, &submit_info, transfer_job.fence_));

    // next graphics submission must wait for transfer completion
    batch_submission_.enqueueWait(transfer_job.semaphore_, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

void Vulkan::Impl::begin_render_pass() {
    // Acquire the next image
    prepare_frame();

    // Get the command buffer for the frame.
    //  There are n command buffers equal to the number of in-flight frames.
    const uint32_t cur_frame      = get_active_image_index();
    const VkCommandBuffer cmd_buf = command_buffers_[cur_frame];

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    NVVK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_info));

    // Clearing values
    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color        = {{0.f, 0.f, 0.f, 1.f}};
    clear_values[1].depthStencil = {1.0f, 0};

    // Begin rendering
    VkRenderPassBeginInfo render_pass_begin_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_pass_begin_info.clearValueCount = 2;
    render_pass_begin_info.pClearValues    = clear_values.data();
    render_pass_begin_info.renderPass      = render_pass_;
    render_pass_begin_info.framebuffer     = framebuffers_[cur_frame];
    render_pass_begin_info.renderArea      = {{0, 0}, size_};
    vkCmdBeginRenderPass(cmd_buf, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    // set the dynamic viewport
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(size_.width),
                        static_cast<float>(size_.height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd_buf, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, {size_.width, size_.height}};
    vkCmdSetScissor(cmd_buf, 0, 1, &scissor);
}

void Vulkan::Impl::end_render_pass() {
    const VkCommandBuffer cmd_buf = command_buffers_[get_active_image_index()];

    // End rendering
    vkCmdEndRenderPass(cmd_buf);

    // Submit for display
    NVVK_CHECK(vkEndCommandBuffer(cmd_buf));
    submit_frame();
}

void Vulkan::Impl::cleanup_transfer_jobs() {
    for (auto it = transfer_jobs_.begin(); it != transfer_jobs_.end();) {
        if (it->fence_) {
            // check if the upload fence was triggered, that means the copy has completed
            //  and cmd buffer can be destroyed
            const VkResult result = vkGetFenceStatus(vk_ctx_.m_device, it->fence_);
            if (result == VK_SUCCESS) {
                transfer_cmd_pool_.destroy(it->cmd_buffer_);
                it->cmd_buffer_ = nullptr;

                // before destroying the fence release all staging buffers using that fence
                alloc_.releaseStaging();

                vkDestroyFence(vk_ctx_.m_device, it->fence_, nullptr);
                it->fence_ = nullptr;
            } else if (result != VK_NOT_READY) {
                NVVK_CHECK(result);
            }
        }

        if (!it->fence_) {
            if (it->frame_fence_) {
                // check if the frame fence was triggered, that means the job can be destroyed
                const VkResult result = vkGetFenceStatus(vk_ctx_.m_device, it->frame_fence_);
                if (result == VK_SUCCESS) {
                    vkDestroySemaphore(vk_ctx_.m_device, it->semaphore_, nullptr);
                    it->semaphore_ = nullptr;
                    /// @todo instead of allocating and destroying semaphore and fences, move to
                    ///  unused list and reuse
                    ///  (call 'NVVK_CHECK(vkResetFences(vk_ctx_.m_device, 1,
                    ///   &it->fence_));' to reuse)
                    it = transfer_jobs_.erase(it);
                    continue;
                } else if (result != VK_NOT_READY) {
                    NVVK_CHECK(result);
                }
            } else {
                // this is a stale transfer buffer (no end_transfer_pass()?), remove it
                it = transfer_jobs_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

void Vulkan::Impl::prepare_frame() {
    if (!transfer_jobs_.empty() && (transfer_jobs_.back().fence_ == nullptr)) {
        throw std::runtime_error("Transfer pass is active!");
    }

    // Acquire the next image from the framebuffer sequence
    if (!fb_sequence_.acquire()) {
        throw std::runtime_error("Failed to acquire next framebuffer sequence image.");
    }

    // Use a fence to wait until the command buffer has finished execution before using it again
    const uint32_t image_index = get_active_image_index();

    VkResult result{VK_SUCCESS};
    do {
        result = vkWaitForFences(vk_ctx_.m_device, 1, &wait_fences_[image_index],
                                                             VK_TRUE, 1'000'000);
    } while (result == VK_TIMEOUT);

    if (result != VK_SUCCESS) {
        // This allows Aftermath to do things and exit below
        usleep(1000);
        NVVK_CHECK(result);
        exit(-1);
    }

    // reset the fence to be re-used
    NVVK_CHECK(vkResetFences(vk_ctx_.m_device, 1, &wait_fences_[image_index]));

    // if there is a pending transfer job assign the frame fence of the frame
    //  which is about to be rendered.
    if (!transfer_jobs_.empty() && !transfer_jobs_.back().frame_fence_) {
        transfer_jobs_.back().frame_fence_ = wait_fences_[image_index];
    }

    // try to free previous transfer jobs
    cleanup_transfer_jobs();
}

void Vulkan::Impl::submit_frame() {
    const uint32_t image_index = get_active_image_index();

    batch_submission_.enqueue(command_buffers_[image_index]);

    // wait for the previous frame's semaphore
    if (fb_sequence_.get_active_read_semaphore() != VK_NULL_HANDLE) {
        batch_submission_.enqueueWait(fb_sequence_.get_active_read_semaphore(),
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    // and signal this frames semaphore on completion
    batch_submission_.enqueueSignal(fb_sequence_.get_active_written_semaphore());

    NVVK_CHECK(batch_submission_.execute(wait_fences_[image_index], 0b0000'0001));

    // Presenting frame
    fb_sequence_.present(queue_);
}

bool Vulkan::Impl::create_framebuffer_sequence() {
    window_->get_framebuffer_size(&size_.width, &size_.height);

    if (!fb_sequence_.init(&alloc_, vk_ctx_, queue_, surface_)) {
        return false;
    }

    if (!fb_sequence_.update(size_.width, size_.height, &size_)) {
        return false;
    }

    // Create Synchronization Primitives
    VkFenceCreateInfo fence_create_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    wait_fences_.resize(fb_sequence_.get_image_count());
    for (auto &fence : wait_fences_) {
        NVVK_CHECK(vkCreateFence(vk_ctx_.m_device, &fence_create_info, nullptr, &fence));
    }

    // Command buffers store a reference to the frame buffer inside their render pass info
    // so for static usage without having to rebuild them each frame, we use one per frame buffer
    VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate_info.commandPool        = cmd_pool_;
    allocate_info.commandBufferCount = fb_sequence_.get_image_count();
    allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffers_.resize(fb_sequence_.get_image_count());
    NVVK_CHECK(vkAllocateCommandBuffers(vk_ctx_.m_device, &allocate_info, command_buffers_.data()));

    const VkCommandBuffer cmd_buffer = create_temp_cmd_buffer();
    fb_sequence_.cmd_update_barriers(cmd_buffer);
    submit_temp_cmd_buffer(cmd_buffer);

#ifdef _DEBUG
    for (size_t i = 0; i < command_buffers_.size(); i++) {
        std::string name = std::string("AppBase") + std::to_string(i);

        VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectHandle = (uint64_t)command_buffers_[i];
        name_info.objectType   = VK_OBJECT_TYPE_COMMAND_BUFFER;
        name_info.pObjectName  = name.c_str();
        NVVK_CHECK(vkSetDebugUtilsObjectNameEXT(vk_ctx_.m_device, &name_info));
    }
#endif  // _DEBUG

    return true;
}

void Vulkan::Impl::create_depth_buffer() {
    if (depth_view_) {
        vkDestroyImageView(vk_ctx_.m_device, depth_view_, nullptr);
    }

    if (depth_image_) {
        vkDestroyImage(vk_ctx_.m_device, depth_image_, nullptr);
    }

    if (depth_memory_) {
        vkFreeMemory(vk_ctx_.m_device, depth_memory_, nullptr);
    }

    // Depth information
    const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    VkImageCreateInfo depth_stencil_create_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    depth_stencil_create_info.imageType   = VK_IMAGE_TYPE_2D;
    depth_stencil_create_info.extent      = VkExtent3D{size_.width, size_.height, 1};
    depth_stencil_create_info.format      = depth_format_;
    depth_stencil_create_info.mipLevels   = 1;
    depth_stencil_create_info.arrayLayers = 1;
    depth_stencil_create_info.samples     = VK_SAMPLE_COUNT_1_BIT;
    depth_stencil_create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // Create the depth image
    NVVK_CHECK(vkCreateImage(vk_ctx_.m_device, &depth_stencil_create_info, nullptr, &depth_image_));

#ifdef _DEBUG
    std::string name = std::string("AppBaseDepth");
    VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectHandle = (uint64_t)depth_image_;
    name_info.objectType   = VK_OBJECT_TYPE_IMAGE;
    name_info.pObjectName  = R"(AppBase)";
    NVVK_CHECK(vkSetDebugUtilsObjectNameEXT(vk_ctx_.m_device, &name_info));
#endif  // _DEBUG

    // Allocate the memory
    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk_ctx_.m_device, depth_image_, &mem_reqs);
    VkMemoryAllocateInfo memAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memAllocInfo.allocationSize  = mem_reqs.size;
    memAllocInfo.memoryTypeIndex = get_memory_type(mem_reqs.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    NVVK_CHECK(vkAllocateMemory(vk_ctx_.m_device, &memAllocInfo, nullptr, &depth_memory_));

    // Bind image and memory
    NVVK_CHECK(vkBindImageMemory(vk_ctx_.m_device, depth_image_, depth_memory_, 0));

    const VkCommandBuffer cmd_buffer = create_temp_cmd_buffer();

    // Put barrier on top, Put barrier inside setup command buffer
    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask = aspect;
    subresource_range.levelCount = 1;
    subresource_range.layerCount = 1;
    VkImageMemoryBarrier image_memory_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image_memory_barrier.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    image_memory_barrier.newLayout            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    image_memory_barrier.image                = depth_image_;
    image_memory_barrier.subresourceRange     = subresource_range;
    image_memory_barrier.srcAccessMask        = VkAccessFlags();
    image_memory_barrier.dstAccessMask        = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    const VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    const VkPipelineStageFlags destStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

    vkCmdPipelineBarrier(cmd_buffer, src_stage_mask, destStageMask, VK_FALSE, 0, nullptr,
                         0, nullptr, 1,
                         &image_memory_barrier);
    submit_temp_cmd_buffer(cmd_buffer);

    // Setting up the view
    VkImageViewCreateInfo depth_stencil_view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depth_stencil_view.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depth_stencil_view.format           = depth_format_;
    depth_stencil_view.subresourceRange = subresource_range;
    depth_stencil_view.image            = depth_image_;
    NVVK_CHECK(vkCreateImageView(vk_ctx_.m_device, &depth_stencil_view, nullptr, &depth_view_));
}

void Vulkan::Impl::create_render_pass() {
    if (render_pass_) {
        vkDestroyRenderPass(vk_ctx_.m_device, render_pass_, nullptr);
    }

    std::array<VkAttachmentDescription, 2> attachments{};
    // Color attachment
    attachments[0].format      = fb_sequence_.get_format();
    attachments[0].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].finalLayout = surface_ ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].samples     = VK_SAMPLE_COUNT_1_BIT;

    // Depth attachment
    attachments[1].format        = depth_format_;
    attachments[1].loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].samples       = VK_SAMPLE_COUNT_1_BIT;

    // One color, one depth
    const VkAttachmentReference color_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth_reference{1,
                                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    std::array<VkSubpassDependency, 1> subpass_dependencies{};
    // Transition from final to initial
    //  (VK_SUBPASS_EXTERNAL refers to all commands executed outside of the actual renderpass)
    subpass_dependencies[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    subpass_dependencies[0].dstSubpass    = 0;
    subpass_dependencies[0].srcStageMask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    subpass_dependencies[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    subpass_dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpass_dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkSubpassDescription subpass_description{};
    subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_description.colorAttachmentCount    = 1;
    subpass_description.pColorAttachments       = &color_reference;
    subpass_description.pDepthStencilAttachment = &depth_reference;

    VkRenderPassCreateInfo render_pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments    = attachments.data();
    render_pass_info.subpassCount    = 1;
    render_pass_info.pSubpasses      = &subpass_description;
    render_pass_info.dependencyCount = static_cast<uint32_t>(subpass_dependencies.size());
    render_pass_info.pDependencies   = subpass_dependencies.data();

    NVVK_CHECK(vkCreateRenderPass(vk_ctx_.m_device, &render_pass_info, nullptr, &render_pass_));

#ifdef _DEBUG
    VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectHandle = (uint64_t)render_pass_;
    name_info.objectType   = VK_OBJECT_TYPE_RENDER_PASS;
    name_info.pObjectName  = R"(AppBaseVk)";
    NVVK_CHECK(vkSetDebugUtilsObjectNameEXT(vk_ctx_.m_device, &name_info));
#endif  // _DEBUG
}

void Vulkan::Impl::create_frame_buffers() {
    // Recreate the frame buffers
    for (auto framebuffer : framebuffers_) {
        vkDestroyFramebuffer(vk_ctx_.m_device, framebuffer, nullptr);
    }

    // Array of attachment (color, depth)
    std::array<VkImageView, 2> attachments{};

    // Create frame buffers for every swap chain image
    VkFramebufferCreateInfo framebuffer_create_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_create_info.renderPass      = render_pass_;
    framebuffer_create_info.attachmentCount = 2;
    framebuffer_create_info.width           = size_.width;
    framebuffer_create_info.height          = size_.height;
    framebuffer_create_info.layers          = 1;
    framebuffer_create_info.pAttachments    = attachments.data();

    // Create frame buffers for every swap chain image
    framebuffers_.resize(fb_sequence_.get_image_count());
    for (uint32_t i = 0; i < fb_sequence_.get_image_count(); i++) {
        attachments[0] = fb_sequence_.get_image_view(i);
        attachments[1] = depth_view_;
        NVVK_CHECK(vkCreateFramebuffer(vk_ctx_.m_device, &framebuffer_create_info, nullptr,
                                       &framebuffers_[i]));
    }

#ifdef _DEBUG
    for (size_t i = 0; i < framebuffers_.size(); i++) {
        std::string name = std::string("AppBase") + std::to_string(i);
        VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        name_info.objectHandle = (uint64_t)framebuffers_[i];
        name_info.objectType   = VK_OBJECT_TYPE_FRAMEBUFFER;
        name_info.pObjectName  = name.c_str();
        NVVK_CHECK(vkSetDebugUtilsObjectNameEXT(vk_ctx_.m_device, &name_info));
    }
#endif  // _DEBUG
}

void Vulkan::Impl::on_framebuffer_size(int w, int h) {
    if ((w == 0) || (h == 0)) {
        return;
    }

    // Update imgui
    if (ImGui::GetCurrentContext() != nullptr) {
        auto &imgui_io       = ImGui::GetIO();
        imgui_io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));
    }

    // Wait to finish what is currently drawing
    NVVK_CHECK(vkDeviceWaitIdle(vk_ctx_.m_device));
    NVVK_CHECK(vkQueueWaitIdle(queue_));

    // Request new swapchain image size
    fb_sequence_.update(w, h, &size_); {
        const VkCommandBuffer cmd_buffer = create_temp_cmd_buffer();
        fb_sequence_.cmd_update_barriers(cmd_buffer);  // Make them presentable
        submit_temp_cmd_buffer(cmd_buffer);
    }

    if ((size_.width != w) || (size_.height != h)) {
        LOGW("Requested size (%d, %d) is different from created size (%u, %u) ", w, h,
              size_.width, size_.height);
    }

    // Recreating other resources
    create_depth_buffer();
    create_frame_buffers();
}

VkCommandBuffer Vulkan::Impl::create_temp_cmd_buffer() {
    // Create an image barrier to change the layout from undefined to DepthStencilAttachmentOptimal
    VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate_info.commandBufferCount = 1;
    allocate_info.commandPool        = cmd_pool_;
    allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VkCommandBuffer cmd_buffer;
    NVVK_CHECK(vkAllocateCommandBuffers(vk_ctx_.m_device, &allocate_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    NVVK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));
    return cmd_buffer;
}

void Vulkan::Impl::submit_temp_cmd_buffer(VkCommandBuffer cmdBuffer) {
    NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &cmdBuffer;
    NVVK_CHECK(vkQueueSubmit(queue_, 1, &submit_info, {}));
    NVVK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(vk_ctx_.m_device, cmd_pool_, 1, &cmdBuffer);
}

uint32_t Vulkan::Impl::get_memory_type(uint32_t typeBits,
                                       const VkMemoryPropertyFlags &properties) const {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(vk_ctx_.m_physicalDevice, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if (((typeBits & (1 << i)) > 0) &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    std::string err = "Unable to find memory type " + std::to_string(properties);
    LOGE(err.c_str());
    return ~0u;
}

VkPipeline Vulkan::Impl::create_pipeline(VkPipelineLayout pipeline_layout,
                                         const uint32_t *vertex_shader,
                                         size_t vertex_shader_size, const uint32_t *fragment_shader,
                                         size_t fragment_shader_size, VkPrimitiveTopology topology,
                                         const std::vector<VkDynamicState> dynamic_state,
                                         bool imgui_attrib_desc) {
    nvvk::GraphicsPipelineGeneratorCombined gpb(vk_ctx_.m_device, pipeline_layout, render_pass_);
    gpb.depthStencilState.depthTestEnable = true;

    std::vector<uint32_t> code;
    code.assign(vertex_shader, vertex_shader + vertex_shader_size);
    gpb.addShader(code, VK_SHADER_STAGE_VERTEX_BIT);
    code.assign(fragment_shader, fragment_shader + fragment_shader_size);
    gpb.addShader(code, VK_SHADER_STAGE_FRAGMENT_BIT);

    gpb.inputAssemblyState.topology = topology;

    if (imgui_attrib_desc) {
        gpb.addBindingDescription({0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX});
        gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32_SFLOAT,
                                       IM_OFFSETOF(ImDrawVert, pos)}});
        gpb.addAttributeDescriptions({{1, 0, VK_FORMAT_R32G32_SFLOAT,
                                       IM_OFFSETOF(ImDrawVert, uv)}});
        gpb.addAttributeDescriptions({{2, 0, VK_FORMAT_R8G8B8A8_UNORM,
                                       IM_OFFSETOF(ImDrawVert, col)}});
    } else {
        gpb.addBindingDescription({0, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX});
        gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}});
    }

    // disable culling
    gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // enable blending
    VkPipelineColorBlendAttachmentState color_blend_attachment_state{};
    color_blend_attachment_state.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                                                            | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment_state.blendEnable         = VK_TRUE;
    color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_state.colorBlendOp        = VK_BLEND_OP_ADD;
    color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment_state.alphaBlendOp        = VK_BLEND_OP_ADD;
    // remove the default blend attachment state
    gpb.clearBlendAttachmentStates();
    gpb.addBlendAttachmentState(color_blend_attachment_state);

    for (auto &&state : dynamic_state) {
        gpb.addDynamicStateEnable(state);
    }

    return gpb.createPipeline();
}

static void format_info(ImageFormat format, uint32_t *src_channels, uint32_t *dst_channels,
                        uint32_t *component_size) {
    switch (format) {
    case ImageFormat::R8_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint8_t);
        break;
    case ImageFormat::R16_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint16_t);
        break;
    case ImageFormat::R16_SFLOAT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint16_t);
        break;
    case ImageFormat::R32_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint32_t);
        break;
    case ImageFormat::R32_SFLOAT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(float);
        break;
    case ImageFormat::R8G8B8_UNORM:
        *src_channels   = 3u;
        *dst_channels   = 4u;
        *component_size = sizeof(uint8_t);
        break;
    case ImageFormat::B8G8R8_UNORM:
        *src_channels   = 3u;
        *dst_channels   = 4u;
        *component_size = sizeof(uint8_t);
        break;
    case ImageFormat::R8G8B8A8_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint8_t);
        break;
    case ImageFormat::B8G8R8A8_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint8_t);
        break;
    case ImageFormat::R16G16B16A16_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint16_t);
        break;
    case ImageFormat::R16G16B16A16_SFLOAT:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint16_t);
        break;
    case ImageFormat::R32G32B32A32_SFLOAT:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint32_t);
        break;
    default:
        throw std::runtime_error("Unhandled image format.");
    }
}

static VkFormat to_vulkan_format(ImageFormat format) {
    VkFormat vk_format;

    switch (format) {
    case ImageFormat::R8_UINT:
        vk_format = VK_FORMAT_R8_UINT;
        break;
    case ImageFormat::R16_UINT:
        vk_format = VK_FORMAT_R16_UINT;
        break;
    case ImageFormat::R16_SFLOAT:
        vk_format = VK_FORMAT_R16_SFLOAT;
        break;
    case ImageFormat::R32_UINT:
        vk_format = VK_FORMAT_R32_UINT;
        break;
    case ImageFormat::R32_SFLOAT:
        vk_format = VK_FORMAT_R32_SFLOAT;
        break;
    case ImageFormat::R8G8B8_UNORM:
        vk_format = VK_FORMAT_R8G8B8A8_UNORM;
        break;
    case ImageFormat::B8G8R8_UNORM:
        vk_format = VK_FORMAT_B8G8R8A8_UNORM;
        break;
    case ImageFormat::R8G8B8A8_UNORM:
        vk_format = VK_FORMAT_R8G8B8A8_UNORM;
        break;
    case ImageFormat::B8G8R8A8_UNORM:
        vk_format = VK_FORMAT_B8G8R8A8_UNORM;
        break;
    case ImageFormat::R16G16B16A16_UNORM:
        vk_format = VK_FORMAT_R16G16B16A16_UNORM;
        break;
    case ImageFormat::R16G16B16A16_SFLOAT:
        vk_format = VK_FORMAT_R16G16B16A16_SFLOAT;
        break;
    case ImageFormat::R32G32B32A32_SFLOAT:
        vk_format = VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
    default:
        throw std::runtime_error("Unhandled image format.");
    }

    return vk_format;
}

UniqueCUexternalSemaphore Vulkan::Impl::import_semaphore_to_cuda(VkSemaphore semaphore) {
    VkSemaphoreGetFdInfoKHR semaphore_get_fd_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    semaphore_get_fd_info.semaphore  = semaphore;
    semaphore_get_fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    UniqueValue<int, decltype(&close), &close> file_handle;
    file_handle.reset([this, &semaphore_get_fd_info] {
        int handle;
        NVVK_CHECK(vkGetSemaphoreFdKHR(vk_ctx_.m_device, &semaphore_get_fd_info, &handle));
        return handle;
    }());

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semaphore_handle_desc{};
    semaphore_handle_desc.type      = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
    semaphore_handle_desc.handle.fd = file_handle.get();

    UniqueCUexternalSemaphore cuda_semaphore;
    cuda_semaphore.reset([&semaphore_handle_desc] {
        CUexternalSemaphore ext_semaphore;
        CudaCheck(cuImportExternalSemaphore(&ext_semaphore, &semaphore_handle_desc));
        return ext_semaphore;
    }());

    // don't need to close the file handle if it had been successfully imported
    file_handle.release();

    return cuda_semaphore;
}

Vulkan::Texture *Vulkan::Impl::create_texture_for_cuda_upload(uint32_t width, uint32_t height,
                                                              ImageFormat format,
                                                              VkFilter filter, bool normalized) {
    if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_ != nullptr)) {
        throw std::runtime_error(
            "Transfer command buffer not set. Calls to create_texture_for_cuda_upload() "
            "need to be enclosed by "
            "begin_transfer_pass() and "
            "end_transfer_pass()");
    }

    const VkFormat vk_format = to_vulkan_format(format);
    uint32_t src_channels, dst_channels, component_size;
    format_info(format, &src_channels, &dst_channels, &component_size);

    // create the image
    const VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(
                                                            VkExtent2D{width, height}, vk_format);
    // the VkExternalMemoryImageCreateInfoKHR struct is appended by nvvk::ExportResourceAllocator
    const nvvk::Image image = export_alloc_.createImage(image_create_info,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // create the texture
    std::unique_ptr<Texture> texture = std::make_unique<Texture>(width, height, format,
                                                                                &export_alloc_);

    // create the vulkan texture
    VkSamplerCreateInfo sampler_create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_create_info.minFilter               = filter;
    sampler_create_info.magFilter               = filter;
    sampler_create_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_create_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.maxLod                  = normalized ? FLT_MAX : 0;
    sampler_create_info.unnormalizedCoordinates = normalized ? VK_FALSE : VK_TRUE;

    const VkImageViewCreateInfo image_view_info = nvvk::makeImageViewCreateInfo(image.image,
                                                                              image_create_info);
    texture->texture_ = export_alloc_.createTexture(image, image_view_info, sampler_create_info);

    // create the semaphores, one for upload and one for rendering
    VkSemaphoreCreateInfo semaphore_create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkExportSemaphoreCreateInfoKHR export_semaphore_create_info
                                    {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR};
    export_semaphore_create_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    semaphore_create_info.pNext              = &export_semaphore_create_info;
    NVVK_CHECK(vkCreateSemaphore(vk_ctx_.m_device, &semaphore_create_info, nullptr,
                                                     &texture->upload_semaphore_));
    NVVK_CHECK(vkCreateSemaphore(vk_ctx_.m_device, &semaphore_create_info, nullptr,
                                                     &texture->render_semaphore_));

    {
        const CudaService::ScopedPush cuda_context = CudaService::get().PushContext();

        // import memory to CUDA
        const nvvk::MemAllocator::MemInfo memInfo = export_alloc_
                                              .getMemoryAllocator()->getMemoryInfo(image.memHandle);

        VkMemoryGetFdInfoKHR memory_get_fd_info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        memory_get_fd_info.memory     = memInfo.memory;
        memory_get_fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        UniqueValue<int, decltype(&close), &close> file_handle;
        file_handle.reset([this, &memory_get_fd_info] {
            int handle;
            NVVK_CHECK(vkGetMemoryFdKHR(vk_ctx_.m_device, &memory_get_fd_info, &handle));
            return handle;
        }());

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memmory_handle_desc{};
        memmory_handle_desc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memmory_handle_desc.handle.fd = file_handle.get();
        memmory_handle_desc.size      = memInfo.offset + memInfo.size;

        texture->external_mem_.reset([&memmory_handle_desc] {
            CUexternalMemory external_mem;
            CudaCheck(cuImportExternalMemory(&external_mem, &memmory_handle_desc));
            return external_mem;
        }());
        // don't need to close the file handle if it had been successfully imported
        file_handle.release();

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapped_array_desc{};
        mipmapped_array_desc.arrayDesc.Width  = width;
        mipmapped_array_desc.arrayDesc.Height = height;
        mipmapped_array_desc.arrayDesc.Depth  = 0;
        switch (component_size) {
        case 1:
            mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            break;
        case 2:
            mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            break;
        case 4:
            mipmapped_array_desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            break;
        default:
            throw std::runtime_error("Unhandled component size");
        }
        mipmapped_array_desc.arrayDesc.NumChannels = dst_channels;
        mipmapped_array_desc.arrayDesc.Flags       = CUDA_ARRAY3D_SURFACE_LDST;

        mipmapped_array_desc.numLevels = 1;
        mipmapped_array_desc.offset    = memInfo.offset;

        texture->mipmap_.reset([external_mem = texture->external_mem_.get(),
                                                                      &mipmapped_array_desc] {
            CUmipmappedArray mipmaped_array;
            CudaCheck(cuExternalMemoryGetMappedMipmappedArray(&mipmaped_array, external_mem,
                                                                    &mipmapped_array_desc));
            return mipmaped_array;
        }());

        // import the semaphores to Cuda
        texture->cuda_upload_semaphore_ = import_semaphore_to_cuda(texture->upload_semaphore_);
        texture->cuda_render_semaphore_ = import_semaphore_to_cuda(texture->render_semaphore_);
    }

    // transition to shader layout
    nvvk::cmdBarrierImageLayout(transfer_jobs_.back().cmd_buffer_, texture->texture_.image,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return texture.release();
}

Vulkan::Texture *Vulkan::Impl::create_texture(uint32_t width, uint32_t height, ImageFormat format,
                                              size_t data_size, const void *data,
                                              VkFilter filter, bool normalized) {
    if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_ != nullptr)) {
        throw std::runtime_error(
            "Transfer command buffer not set. Calls to create_texture() need to be enclosed by "
            "begin_transfer_pass() and end_transfer_pass()");
    }

    const VkFormat vk_format = to_vulkan_format(format);
    uint32_t src_channels, dst_channels, component_size;
    format_info(format, &src_channels, &dst_channels, &component_size);

    if (data && (data_size != width * height * src_channels * component_size)) {
        throw std::runtime_error("The size of the data array is wrong");
    }

    const VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(
                                                              VkExtent2D{width, height}, vk_format);
    const nvvk::Image image = alloc_.createImage(transfer_jobs_.back().cmd_buffer_, data_size, data,
                                                                                 image_create_info);

    // create the texture
    std::unique_ptr<Texture> texture = std::make_unique<Texture>(width, height, format, &alloc_);

    // create the Vulkan texture
    VkSamplerCreateInfo sampler_create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_create_info.minFilter               = filter;
    sampler_create_info.magFilter               = filter;
    sampler_create_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_create_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.maxLod                  = normalized ? FLT_MAX : 0;
    sampler_create_info.unnormalizedCoordinates = normalized ? VK_FALSE : VK_TRUE;

    const VkImageViewCreateInfo image_view_info = nvvk::makeImageViewCreateInfo(image.image,
                                                                                image_create_info);
    texture->texture_                           = alloc_.createTexture(image, image_view_info,
                                                                              sampler_create_info);

    // transition to shader layout
    /// @todo I don't know if this is defined. Should the old layout be
    ///        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, like it would be if we uploaded using Vulkan?
    nvvk::cmdBarrierImageLayout(transfer_jobs_.back().cmd_buffer_, image.image,
                                image_create_info.initialLayout,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return texture.release();
}

void Vulkan::Impl::destroy_texture(Texture *texture) {
    if (texture->fence_) {
        // if the texture had been tagged with a fence, wait for it before freeing the memory
        NVVK_CHECK(vkWaitForFences(vk_ctx_.m_device, 1, &texture->fence_, VK_TRUE, 1000000));
    }

    // check if this texture had been imported to CUDA
    if (texture->mipmap_ || texture->external_mem_) {
        const CudaService::ScopedPush cuda_context = CudaService::get().PushContext();

        texture->mipmap_.reset();
        texture->external_mem_.reset();
        texture->cuda_upload_semaphore_.reset();
        texture->cuda_render_semaphore_.reset();
    }

    if (texture->upload_semaphore_) {
        vkDestroySemaphore(vk_ctx_.m_device, texture->upload_semaphore_, nullptr);
    }
    if (texture->render_semaphore_) {
        vkDestroySemaphore(vk_ctx_.m_device, texture->render_semaphore_, nullptr);
    }

    texture->alloc_->destroy(texture->texture_);

    delete texture;
}

void Vulkan::Impl::upload_to_texture(CUdeviceptr device_ptr, Texture *texture, CUstream stream) {
    if (!texture->mipmap_) {
        throw std::runtime_error("Texture had not been imported to CUDA, can't upload data.");
    }

    const CudaService::ScopedPush cuda_context = CudaService::get().PushContext();

    if (texture->state_ == Texture::State::RENDERED) {
        // if the texture had been used in rendering, wait for Vulkan to finish it's work
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS ext_wait_params{};
        const CUexternalSemaphore external_semaphore = texture->cuda_render_semaphore_.get();
        CudaCheck(cuWaitExternalSemaphoresAsync(&external_semaphore, &ext_wait_params, 1, stream));
        texture->state_ = Texture::State::UNKNOWN;
    }

    CUarray array;
    CudaCheck(cuMipmappedArrayGetLevel(&array, texture->mipmap_.get(), 0));

    uint32_t src_channels, dst_channels, component_size;
    format_info(texture->format_, &src_channels, &dst_channels, &component_size);
    const uint32_t src_pitch = texture->width_ * src_channels * component_size;

    if (src_channels != dst_channels) {
        // three channel texture data is not hardware natively supported, convert to four channel
        if ((src_channels != 3) || (dst_channels != 4) || (component_size != 1)) {
            throw std::runtime_error("Unhandled conversion.");
        }
        ConvertR8G8B8ToR8G8B8A8(texture->width_, texture->height_, device_ptr, src_pitch,
                                                                          array, stream);
    } else {
        // else just copy
        CUDA_MEMCPY2D memcpy_2d{};
        memcpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy_2d.srcDevice     = device_ptr;
        memcpy_2d.srcPitch      = src_pitch;
        memcpy_2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        memcpy_2d.dstArray      = array;
        memcpy_2d.WidthInBytes  = texture->width_ * dst_channels * component_size;
        memcpy_2d.Height        = texture->height_;
        CudaCheck(cuMemcpy2DAsync(&memcpy_2d, stream));
    }

    // signal the semaphore
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_signal_params{};
    const CUexternalSemaphore external_semaphore = texture->cuda_upload_semaphore_.get();
    CudaCheck(cuSignalExternalSemaphoresAsync(&external_semaphore, &ext_signal_params, 1, stream));

    // the texture is now in uploaded state, Vulkan needs to wait for it
    texture->state_ = Texture::State::UPLOADED;
}

void Vulkan::Impl::upload_to_texture(const void *host_ptr, Texture *texture) {
    if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_ != nullptr)) {
        throw std::runtime_error(
            "Transfer command buffer not set. Calls to upload_to_texture() need to be enclosed by "
            "begin_transfer_pass() and end_transfer_pass()");
    }

    if ((texture->state_ != Texture::State::RENDERED) &&
        (texture->state_ != Texture::State::UNKNOWN)) {
        throw std::runtime_error("When uploading to texture, the texture should be in rendered "
                                                                            "or unknown state");
    }

    const VkCommandBuffer cmd_buf = transfer_jobs_.back().cmd_buffer_;

    // Copy buffer to image
    VkImageSubresourceRange subresource_range{};
    subresource_range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseArrayLayer = 0;
    subresource_range.baseMipLevel   = 0;
    subresource_range.layerCount     = 1;
    subresource_range.levelCount     = 1;

    nvvk::cmdBarrierImageLayout(cmd_buf, texture->texture_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresource_range);

    VkOffset3D offset                    = {0};
    VkImageSubresourceLayers subresource = {0};
    subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.layerCount               = 1;

    uint32_t src_channels, dst_channels, component_size;
    format_info(texture->format_, &src_channels, &dst_channels, &component_size);

    const VkDeviceSize data_size = texture->width_ * texture->height_ * dst_channels *
                                                                                    component_size;
    void *mapping                = alloc_.getStaging()->cmdToImage(cmd_buf, texture->texture_.image,
                                                                   offset,
                                                                   VkExtent3D{texture->width_,
                                                                   texture->height_, 1},
                                                                   subresource,
                                                                   data_size, nullptr);

    if (src_channels != dst_channels) {
        // three channel texture data is not hardware natively supported, convert to four channel
        if ((src_channels != 3) || (dst_channels != 4) || (component_size != 1)) {
            throw std::runtime_error("Unhandled conversion.");
        }
        const uint8_t *src = reinterpret_cast<const uint8_t *>(host_ptr);
        uint32_t *dst      = reinterpret_cast<uint32_t *>(mapping);
        for (uint32_t y = 0; y < texture->height_; ++y) {
            for (uint32_t x = 0; x < texture->width_; ++x) {
                const uint8_t data[4]{src[0], src[1], src[2], 0xFF};
                *dst = *reinterpret_cast<const uint32_t *>(&data);
                src += 3;
                ++dst;
            }
        }
    } else {
        memcpy(mapping, host_ptr, data_size);
    }

    // Setting final image layout
    nvvk::cmdBarrierImageLayout(cmd_buf, texture->texture_.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // no need to set the texture state here, the transfer command buffer submission is
    // always synchronized to the render command buffer submission.
}

Vulkan::Buffer *Vulkan::Impl::create_buffer(size_t data_size, VkBufferUsageFlags usage,
                                            const void *data) {
    std::unique_ptr<Buffer> buffer(new Buffer(data_size, &alloc_));
    if (data) {
        if (transfer_jobs_.empty() || (transfer_jobs_.back().fence_ != nullptr)) {
            throw std::runtime_error(
                "Transfer command buffer not set. Calls to create_buffer() with data need to be "
                "enclosed by begin_transfer_pass() and end_transfer_pass()");
        }

        buffer->buffer_ =
            alloc_.createBuffer(transfer_jobs_.back().cmd_buffer_,
                                static_cast<VkDeviceSize>(data_size), data, usage);
    } else {
        buffer->buffer_ = alloc_.createBuffer(static_cast<VkDeviceSize>(data_size), usage);
    }

    return buffer.release();
}

void Vulkan::Impl::destroy_buffer(Buffer *buffer) {
    if (buffer->fence_) {
        // if the buffer had been tagged with a fence, wait for it before freeing the memory
        NVVK_CHECK(vkWaitForFences(vk_ctx_.m_device, 1, &buffer->fence_, VK_TRUE, 100'000'000));
    }

    buffer->alloc_->destroy(buffer->buffer_);
    delete buffer;
}

void Vulkan::Impl::draw_texture(Texture *texture, Texture *lut, float opacity) {
    const VkCommandBuffer cmd_buf = command_buffers_[get_active_image_index()];

    if (texture->state_ == Texture::State::UPLOADED) {
        // enqueue the semaphore signalled by Cuda to be waited on by rendering
        batch_submission_.enqueueWait(texture->upload_semaphore_,
                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        // also signal the render semapore which will be waited on by Cuda
        batch_submission_.enqueueSignal(texture->render_semaphore_);
        texture->state_ = Texture::State::RENDERED;
    }

    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;

    // update descriptor sets
    std::vector<VkWriteDescriptorSet> writes;
    if (lut) {
        if (lut->state_ == Texture::State::UPLOADED) {
            // enqueue the semaphore signalled by Cuda to be waited on by rendering
            batch_submission_.enqueueWait(lut->upload_semaphore_,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            // also signal the render semapore which will be waited on by Cuda
            batch_submission_.enqueueSignal(lut->render_semaphore_);
            lut->state_ = Texture::State::RENDERED;
        }

        if ((texture->format_ == ImageFormat::R8_UINT) ||
            (texture->format_ == ImageFormat::R16_UINT) ||
            (texture->format_ == ImageFormat::R32_UINT)) {
            pipeline = image_lut_uint_pipeline_;
        } else {
            pipeline = image_lut_float_pipeline_;
        }

        pipeline_layout = image_lut_pipeline_layout_;

        writes.emplace_back(
            desc_set_layout_bind_lut_.makeWrite(nullptr, bindings_offset_texture_,
                                                  &texture->texture_.descriptor));
        writes.emplace_back(
            desc_set_layout_bind_lut_.makeWrite(nullptr, bindings_offset_texture_lut_,
                                                          &lut->texture_.descriptor));
    } else {
        pipeline        = image_pipeline_;
        pipeline_layout = image_pipeline_layout_;

        writes.emplace_back(
            desc_set_layout_bind_.makeWrite(nullptr, bindings_offset_texture_,
            &texture->texture_.descriptor));
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    vkCmdPushDescriptorSetKHR(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // push the constants
    PushConstantFragment push_constants;
    push_constants.opacity = opacity;
    vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantFragment), &push_constants);

    // bind the buffers
    VkDeviceSize offset{0};
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vertex_buffer_.buffer, &offset);
    vkCmdBindIndexBuffer(cmd_buf, index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT16);

    // draw
    vkCmdDrawIndexed(cmd_buf, 6, 1, 0, 0, 0);

    // tag the texture and lut with the current fence
    texture->fence_ = wait_fences_[get_active_image_index()];
    if (lut) {
        lut->fence_ = wait_fences_[get_active_image_index()];
    }
}

void Vulkan::Impl::draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first,
                        Buffer *buffer, float opacity, const std::array<float, 4> &color,
                        float point_size, float line_width) {
    const VkCommandBuffer cmd_buf = command_buffers_[get_active_image_index()];

    switch (topology) {
    case VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_point_pipeline_);
        break;
    case VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_line_pipeline_);
        vkCmdSetLineWidth(cmd_buf, line_width);
        break;
    case VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_line_strip_pipeline_);
        vkCmdSetLineWidth(cmd_buf, line_width);
        break;
    case VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
        vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_triangle_pipeline_);
        break;
    default:
        throw std::runtime_error("Unhandled primitive type");
    }

    // push the constants
    PushConstantFragment push_constants_fragment;
    push_constants_fragment.opacity = opacity;
    vkCmdPushConstants(cmd_buf, geometry_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       sizeof(PushConstantVertex), sizeof(PushConstantFragment),
                       &push_constants_fragment);

    PushConstantVertex push_constant_vertex;
    push_constant_vertex.matrix.identity();
    push_constant_vertex.matrix.scale({2.f, 2.f, 1.f});
    push_constant_vertex.matrix.translate({-.5f, -.5f, 0.f});
    push_constant_vertex.point_size = point_size;
    push_constant_vertex.color      = color;
    vkCmdPushConstants(cmd_buf, geometry_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstantVertex), &push_constant_vertex);

    VkDeviceSize offset{0};
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &buffer->buffer_.buffer, &offset);

    // draw
    vkCmdDraw(cmd_buf, count, 1, first, 0);

    // tag the buffer with the current fence
    buffer->fence_ = wait_fences_[get_active_image_index()];
}

void Vulkan::Impl::draw_indexed(VkDescriptorSet desc_set, Buffer *vertex_buffer,
                                Buffer *index_buffer, VkIndexType index_type,
                                uint32_t index_count, uint32_t first_index,
                                uint32_t vertex_offset, float opacity) {
    const VkCommandBuffer cmd_buf = command_buffers_[get_active_image_index()];

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_text_pipeline_);

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            geometry_text_pipeline_layout_, 0, 1, &desc_set,
                            0, nullptr);

    // push the constants
    PushConstantFragment push_constants;
    push_constants.opacity = opacity;
    vkCmdPushConstants(cmd_buf, geometry_text_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       sizeof(PushConstantTextVertex), sizeof(PushConstantFragment),
                                                                   &push_constants);

    PushConstantTextVertex push_constant_vertex;
    push_constant_vertex.matrix.identity();
    push_constant_vertex.matrix.scale({2.f, 2.f, 1.f});
    push_constant_vertex.matrix.translate({-.5f, -.5f, 0.f});
    vkCmdPushConstants(cmd_buf, geometry_text_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(PushConstantTextVertex), &push_constant_vertex);

    // bind the buffers
    VkDeviceSize offset{0};
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vertex_buffer->buffer_.buffer, &offset);
    vkCmdBindIndexBuffer(cmd_buf, index_buffer->buffer_.buffer, 0, index_type);

    // draw
    vkCmdDrawIndexed(cmd_buf, index_count, 1, first_index, vertex_offset, 0);

    // tag the buffers with the current fence
    vertex_buffer->fence_ = index_buffer->fence_ = wait_fences_[get_active_image_index()];
}

void Vulkan::Impl::read_framebuffer(ImageFormat fmt, size_t buffer_size,
                                    CUdeviceptr device_ptr, CUstream stream) {
    if (fmt != ImageFormat::R8G8B8A8_UNORM) {
        throw std::runtime_error("Unsupported image format, supported formats: R8G8B8A8_UNORM.");
    }

    const VkFormat vk_format = to_vulkan_format(fmt);
    uint32_t src_channels, dst_channels, component_size;
    format_info(fmt, &src_channels, &dst_channels, &component_size);

    const size_t data_size = size_.width * size_.height * dst_channels * component_size;
    if (buffer_size < data_size) {
        throw std::runtime_error("The size of the buffer is too small");
    }

    /// @todo need to wait for frame, use semaphores to sync
    batch_submission_.waitIdle();

    // create a command buffer
    nvvk::CommandPool cmd_buf_get(vk_ctx_.m_device, vk_ctx_.m_queueGCT.familyIndex);
    VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

    // Make the image layout VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL to copy to buffer
    VkImageSubresourceRange subresource_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    nvvk::cmdBarrierImageLayout(cmd_buf, fb_sequence_.get_active_image(),
                                surface_ ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource_range);

    // allocate the buffer
    /// @todo keep the buffer and the mapping to Cuda to avoid allocations
    Vulkan::Buffer *const transfer_buffer = create_buffer(data_size,
                                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Copy the image to the buffer
    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent.width           = size_.width;
    copy_region.imageExtent.height          = size_.height;
    copy_region.imageExtent.depth           = 1;
    vkCmdCopyImageToBuffer(cmd_buf, fb_sequence_.get_active_image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           transfer_buffer->buffer_.buffer, 1, &copy_region);

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmd_buf, fb_sequence_.get_active_image(),
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                surface_ ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                subresource_range);
    /// @todo avoid wait, use semaphore to sync
    cmd_buf_get.submitAndWait(cmd_buf);

    {
        const CudaService::ScopedPush cuda_context = CudaService::get().PushContext();

        // import memory to CUDA
        const nvvk::MemAllocator::MemInfo memInfo =
            export_alloc_.getMemoryAllocator()->getMemoryInfo(transfer_buffer->buffer_.memHandle);

        VkMemoryGetFdInfoKHR memory_get_fd_info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        memory_get_fd_info.memory     = memInfo.memory;
        memory_get_fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        UniqueValue<int, decltype(&close), &close> file_handle;
        file_handle.reset([this, &memory_get_fd_info] {
            int handle;
            NVVK_CHECK(vkGetMemoryFdKHR(vk_ctx_.m_device, &memory_get_fd_info, &handle));
            return handle;
        }());

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memmory_handle_desc{};
        memmory_handle_desc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memmory_handle_desc.handle.fd = file_handle.get();
        memmory_handle_desc.size      = memInfo.offset + memInfo.size;

        UniqueCUexternalMemory external_mem;
        external_mem.reset([&memmory_handle_desc] {
            CUexternalMemory external_mem;
            CudaCheck(cuImportExternalMemory(&external_mem, &memmory_handle_desc));
            return external_mem;
        }());
        // don't need to close the file handle if it had been successfully imported
        file_handle.release();

        CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc{};
        buffer_desc.size   = data_size;
        buffer_desc.offset = memInfo.offset;

        UniqueCUdeviceptr transfer_mem;
        transfer_mem.reset([&external_mem, &buffer_desc] {
            CUdeviceptr device_ptr;
            CudaCheck(cuExternalMemoryGetMappedBuffer(&device_ptr, external_mem.get(),
                                                                       &buffer_desc));
            return device_ptr;
        }());

        if (fb_sequence_.get_format() == VkFormat::VK_FORMAT_B8G8R8A8_UNORM) {
            // convert from B8G8R8A8 to R8G8B8A8
            ConvertB8G8R8A8ToR8G8B8A8(size_.width, size_.height, transfer_mem.get(),
                                      size_.width * dst_channels * component_size, device_ptr,
                                      size_.width * dst_channels * component_size, stream);
        } else if (fb_sequence_.get_format() == VkFormat::VK_FORMAT_R8G8B8A8_UNORM) {
            CUDA_MEMCPY2D memcpy2d{};
            memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            memcpy2d.srcDevice     = transfer_mem.get();
            memcpy2d.srcPitch      = size_.width * dst_channels * component_size;
            memcpy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            memcpy2d.dstDevice     = device_ptr;
            memcpy2d.dstPitch      = size_.width * dst_channels * component_size;
            memcpy2d.WidthInBytes  = size_.width * dst_channels * component_size;
            memcpy2d.Height        = size_.height;
            CudaCheck(cuMemcpy2DAsync(&memcpy2d, stream));
        } else {
            throw std::runtime_error("Unhandled framebuffer format.");
        }
    }

    destroy_buffer(transfer_buffer);
}

Vulkan::Vulkan()
    : impl_(new Vulkan::Impl) {
}

Vulkan::~Vulkan() {}

void Vulkan::setup(Window *window) {
    impl_->setup(window);
}

void Vulkan::begin_transfer_pass() {
    impl_->begin_transfer_pass();
}

void Vulkan::end_transfer_pass() {
    impl_->end_transfer_pass();
}

void Vulkan::begin_render_pass() {
    impl_->begin_render_pass();
}

void Vulkan::end_render_pass() {
    impl_->end_render_pass();
}

VkCommandBuffer Vulkan::get_command_buffer() {
    return impl_->get_command_buffers()[impl_->get_active_image_index()];
}

Vulkan::Texture *Vulkan::create_texture_for_cuda_upload(uint32_t width, uint32_t height,
                                                        ImageFormat format,
                                                        VkFilter filter, bool normalized) {
    return impl_->create_texture_for_cuda_upload(width, height, format, filter, normalized);
}

Vulkan::Texture *Vulkan::create_texture(uint32_t width, uint32_t height, ImageFormat format,
                                        size_t data_size, const void *data,
                                        VkFilter filter, bool normalized) {
    return impl_->create_texture(width, height, format, data_size, data, filter, normalized);
}

void Vulkan::destroy_texture(Texture *texture) {
    impl_->destroy_texture(texture);
}

void Vulkan::upload_to_texture(CUdeviceptr device_ptr, Texture *texture, CUstream stream) {
    impl_->upload_to_texture(device_ptr, texture, stream);
}

void Vulkan::upload_to_texture(const void *host_ptr, Texture *texture) {
    impl_->upload_to_texture(host_ptr, texture);
}

void Vulkan::draw_texture(Texture *texture, Texture *lut, float opacity) {
    impl_->draw_texture(texture, lut, opacity);
}

Vulkan::Buffer *Vulkan::create_buffer(size_t data_size, const void *data,
                                      VkBufferUsageFlags usage) {
    return impl_->create_buffer(data_size, usage, data);
}

void Vulkan::destroy_buffer(Buffer *buffer) {
    impl_->destroy_buffer(buffer);
}

void Vulkan::draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, Buffer *buffer,
                  float opacity, const std::array<float, 4> &color,
                  float point_size, float line_width) {
    impl_->draw(topology, count, first, buffer, opacity, color, point_size, line_width);
}

void Vulkan::draw_indexed(VkDescriptorSet desc_set, Buffer *vertex_buffer, Buffer *index_buffer,
                          VkIndexType index_type, uint32_t index_count, uint32_t first_index,
                          uint32_t vertex_offset, float opacity) {
    impl_->draw_indexed(desc_set, vertex_buffer, index_buffer, index_type, index_count,
                        first_index, vertex_offset, opacity);
}

void Vulkan::read_framebuffer(ImageFormat fmt, size_t buffer_size,
                              CUdeviceptr buffer, CUstream stream) {
    impl_->read_framebuffer(fmt, buffer_size, buffer, stream);
}

}  // namespace holoscan::viz
