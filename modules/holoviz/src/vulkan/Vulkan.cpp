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

#include "Vulkan.h"

#include "../layers/Layer.h"

#include <vulkan/spv/geometryShader.glsl.frag.h>
#include <vulkan/spv/geometryShader.glsl.vert.h>
#include <vulkan/spv/geometryTextShader.glsl.frag.h>
#include <vulkan/spv/geometryTextShader.glsl.vert.h>
#include <vulkan/spv/imageLUTFloatShader.glsl.frag.h>
#include <vulkan/spv/imageLUTUIntShader.glsl.frag.h>
#include <vulkan/spv/imageShader.glsl.frag.h>
#include <vulkan/spv/imageShader.glsl.vert.h>

#include "../cuda/CudaService.h"
#include "../cuda/Convert.h"

#include <nvvk/appbase_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvh/fileoperations.hpp>
#include <nvmath/nvmath.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <unistd.h>

namespace clara::holoviz
{

struct PushConstantFragment
{
    float opacity;
};

struct PushConstantVertex
{
    nvmath::mat4f matrix;
    float point_size;
    std::array<float, 4> color;
};

struct PushConstantTextVertex
{
    nvmath::mat4f matrix;
};

struct Vulkan::Texture
{
    Texture(uint32_t width, uint32_t height, ImageFormat format, nvvk::ResourceAllocator *alloc)
        : width_(width)
        , height_(height)
        , format_(format)
        , alloc_(alloc)
    {
    }

    const uint32_t width_;
    const uint32_t height_;
    const ImageFormat format_;
    nvvk::ResourceAllocator *const alloc_;

    nvvk::Texture texture_{};
    CUexternalMemory external_mem_ = nullptr;
    CUmipmappedArray mipmap_       = nullptr;
};

struct Vulkan::Buffer
{
    Buffer(size_t size, nvvk::ResourceAllocator *alloc)
        : size_(size)
        , alloc_(alloc)
    {
    }

    const size_t size_;
    nvvk::ResourceAllocator *const alloc_;

    nvvk::Buffer buffer_{};
};

class Vulkan::Impl : public nvvk::AppBaseVk
{
public:
    Impl() = default;
    virtual ~Impl();

    void Setup(Window *window);

    // begin AppBaseVk overrides
    void prepareFrame() override;
    // end AppBaseVk overrides

    Texture *CreateTextureForCudaUpload(uint32_t width, uint32_t height, ImageFormat format, VkFilter filter,
                                        bool normalized);
    Texture *CreateTexture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size, const void *data,
                           VkFilter filter, bool normalized);
    void DestroyTexture(Texture *texture);

    void UploadToTexture(CUdeviceptr device_ptr, const Texture *texture);

    void DrawTexture(const Texture *texture, const Texture *lut, float opacity);

    Buffer *CreateBuffer(size_t data_size, const void *data, VkBufferUsageFlags usage);

    void DestroyBuffer(Buffer *buffer);

    void Draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, const Buffer *buffer, float opacity,
              const std::array<float, 4> &color, float point_size, float line_width);

    void DrawIndexed(VkDescriptorSet desc_set, const Buffer *vertex_buffer, const Buffer *index_buffer,
                     VkIndexType index_type, uint32_t index_count, uint32_t first_index, uint32_t vertex_offset,
                     float opacity);

private:
    void initImGui();
    VkPipeline CreatePipeline(VkPipelineLayout pipeline_layout, const uint32_t *vertex_shader,
                              size_t vertex_shader_size, const uint32_t *fragment_shader, size_t fragment_shader_size,
                              VkPrimitiveTopology topology, const std::vector<VkDynamicState> dynamic_state = {},
                              bool imgui_attrib_desc = false);
    VkFormat ToVulkanFormat(ImageFormat format, uint32_t *src_channels, uint32_t *dst_channels,
                            uint32_t *component_size);

    Window *window_ = nullptr;

    nvvk::ResourceAllocatorDma alloc_;           // Allocator for buffer, images, acceleration structures
    nvvk::ExportResourceAllocator export_alloc_; // Allocator for allocations which can be exported
    nvvk::Context vkctx_{};

    nvvk::Buffer vertex_buffer_{};
    nvvk::Buffer index_buffer_{};

    VkPipelineLayout image_pipeline_layout_         = nullptr;
    VkPipelineLayout image_lut_pipeline_layout_     = nullptr;
    VkPipelineLayout geometry_pipeline_layout_      = nullptr;
    VkPipelineLayout geometry_text_pipeline_layout_ = nullptr;

    const uint32_t bindings_offset_texture_     = 0;
    const uint32_t bindings_offset_texture_lut_ = 1;

    nvvk::DescriptorSetBindings desc_set_layout_bind_;
    VkDescriptorPool desc_pool_            = nullptr;
    VkDescriptorSetLayout desc_set_layout_ = nullptr;
    VkDescriptorSet desc_set_              = nullptr;

    nvvk::DescriptorSetBindings desc_set_layout_bind_lut_;
    VkDescriptorPool desc_pool_lut_            = nullptr;
    VkDescriptorSetLayout desc_set_layout_lut_ = nullptr;
    VkDescriptorSet desc_set_lut_              = nullptr;

    VkPipeline image_pipeline_           = nullptr;
    VkPipeline image_lut_uint_pipeline_  = nullptr;
    VkPipeline image_lut_float_pipeline_ = nullptr;

    VkPipeline geometry_point_pipeline_      = nullptr;
    VkPipeline geometry_line_pipeline_       = nullptr;
    VkPipeline geometry_line_strip_pipeline_ = nullptr;
    VkPipeline geometry_triangle_pipeline_   = nullptr;
    VkPipeline geometry_text_pipeline_       = nullptr;

    CUcontext cuda_context_ = nullptr;
};

Vulkan::Impl::~Impl()
{
    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS)
    {
        LOGE("cuDevicePrimaryCtxRelease failed.");
    }

    NVVK_CHECK(vkDeviceWaitIdle(getDevice()));

    vkDestroyDescriptorSetLayout(m_device, desc_set_layout_lut_, nullptr);
    vkDestroyDescriptorPool(m_device, desc_pool_lut_, nullptr);
    vkDestroyDescriptorSetLayout(m_device, desc_set_layout_, nullptr);
    vkDestroyDescriptorPool(m_device, desc_pool_, nullptr);

    vkDestroyPipeline(m_device, geometry_text_pipeline_, nullptr);
    geometry_text_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, geometry_triangle_pipeline_, nullptr);
    geometry_triangle_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, geometry_line_strip_pipeline_, nullptr);
    geometry_line_strip_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, geometry_line_pipeline_, nullptr);
    geometry_line_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, geometry_point_pipeline_, nullptr);
    geometry_point_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, image_lut_float_pipeline_, nullptr);
    image_lut_float_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, image_lut_uint_pipeline_, nullptr);
    image_lut_uint_pipeline_ = nullptr;
    vkDestroyPipeline(m_device, image_pipeline_, nullptr);
    image_pipeline_ = nullptr;

    vkDestroyPipelineLayout(m_device, geometry_text_pipeline_layout_, nullptr);
    geometry_text_pipeline_layout_ = nullptr;
    vkDestroyPipelineLayout(m_device, geometry_pipeline_layout_, nullptr);
    geometry_pipeline_layout_ = nullptr;
    vkDestroyPipelineLayout(m_device, image_lut_pipeline_layout_, nullptr);
    image_lut_pipeline_layout_ = nullptr;
    vkDestroyPipelineLayout(m_device, image_pipeline_layout_, nullptr);
    image_pipeline_layout_ = nullptr;

    alloc_.destroy(index_buffer_);
    alloc_.destroy(vertex_buffer_);

    export_alloc_.deinit();
    alloc_.deinit();
    AppBaseVk::destroy();

    vkctx_.deinit();
}

void Vulkan::Impl::Setup(Window *window)
{
    window_ = window;

    uint32_t count{0};
    const char **req_extensions = window_->GetRequiredInstanceExtensions(&count);

#ifdef NDEBUG
    nvvk::ContextCreateInfo context_info;
#else
    nvvk::ContextCreateInfo context_info(true /*bUseValidation*/);
#endif

    context_info.setVersion(1, 2); // Using Vulkan 1.2

    // Requesting Vulkan extensions and layers
    for (uint32_t ext_id = 0; ext_id < count; ext_id++) // Adding required extensions (surface, win32, linux, ..)
        context_info.addInstanceExtension(req_extensions[ext_id]);
    context_info.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true); // Allow debug names
    context_info.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);           // Enabling ability to present rendering

    context_info.addInstanceExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    context_info.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);

    // Creating Vulkan base application
    vkctx_.initInstance(context_info);

    // Find all compatible devices
    const std::vector<uint32_t> compatible_devices = vkctx_.getCompatibleDevices(context_info);
    if (compatible_devices.empty())
    {
        throw std::runtime_error("No Vulkan capable GPU present.");
    }
    // Use a compatible device
    vkctx_.initDevice(compatible_devices[0], context_info);

    m_surface = window_->CreateSurface(vkctx_.m_physicalDevice, vkctx_.m_instance);
    if (!vkctx_.setGCTQueueWithPresent(m_surface))
    {
        std::runtime_error("Surface not supported by queue");
    }

    {
        uint32_t surface_format_count;
        NVVK_CHECK(
            vkGetPhysicalDeviceSurfaceFormatsKHR(vkctx_.m_physicalDevice, m_surface, &surface_format_count, nullptr));
        std::vector<VkSurfaceFormatKHR> surfaceFormats(surface_format_count);
        NVVK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkctx_.m_physicalDevice, m_surface, &surface_format_count,
                                                        surfaceFormats.data()));

        // pick a preferred format or use the first available one
        const VkFormat format = VkFormat::VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        m_colorFormat         = surfaceFormats[0].format;
        for (auto &f : surfaceFormats)
        {
            if (format == f.format)
            {
                m_colorFormat = f.format;
                break;
            }
        }
    }

    AppBaseVk::setup(vkctx_.m_instance, vkctx_.m_device, vkctx_.m_physicalDevice, vkctx_.m_queueGCT.familyIndex);

    alloc_.init(vkctx_.m_instance, vkctx_.m_device, vkctx_.m_physicalDevice);
    export_alloc_.init(vkctx_.m_device, vkctx_.m_physicalDevice, alloc_.getMemoryAllocator());

    uint32_t framebuffer_width, framebuffer_height;
    window_->GetFramebufferSize(&framebuffer_width, &framebuffer_height);
    createSwapchain(m_surface, framebuffer_width, framebuffer_height, m_colorFormat);
    createDepthBuffer();
    createRenderPass();
    createFrameBuffers();

    // allocate the vertex and index buffer for the image draw pass
    {
        nvvk::CommandPool cmd_buf_get(vkctx_.m_device, m_graphicsQueueIndex);
        VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

        const std::vector<float> vertices{-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
        vertex_buffer_ = alloc_.createBuffer(cmd_buf, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        const std::vector<uint16_t> indices{0, 2, 1, 2, 0, 3};
        index_buffer_ = alloc_.createBuffer(cmd_buf, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        cmd_buf_get.submitAndWait(cmd_buf);
        alloc_.finalizeAndReleaseStaging();
    }

    // create the descriptor sets
    desc_set_layout_bind_.addBinding(bindings_offset_texture_, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                     VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_ = desc_set_layout_bind_.createLayout(m_device);
    desc_pool_       = desc_set_layout_bind_.createPool(m_device);
    desc_set_        = nvvk::allocateDescriptorSet(m_device, desc_pool_, desc_set_layout_);

    desc_set_layout_bind_lut_.addBinding(bindings_offset_texture_, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_bind_lut_.addBinding(bindings_offset_texture_lut_, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT);
    desc_set_layout_lut_ = desc_set_layout_bind_lut_.createLayout(m_device);
    desc_pool_lut_       = desc_set_layout_bind_lut_.createPool(m_device);
    desc_set_lut_        = nvvk::allocateDescriptorSet(m_device, desc_pool_lut_, desc_set_layout_lut_);

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
        NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &image_pipeline_layout_));
    }

    // Create the Pipeline
    image_pipeline_ = CreatePipeline(
        image_pipeline_layout_, imageShader_glsl_vert, sizeof(imageShader_glsl_vert) / sizeof(imageShader_glsl_vert[0]),
        imageShader_glsl_frag, sizeof(imageShader_glsl_frag) / sizeof(imageShader_glsl_frag[0]),
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
        NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &image_lut_pipeline_layout_));
    }

    image_lut_uint_pipeline_ =
        CreatePipeline(image_lut_pipeline_layout_, imageShader_glsl_vert,
                       sizeof(imageShader_glsl_vert) / sizeof(imageShader_glsl_vert[0]), imageLUTUIntShader_glsl_frag,
                       sizeof(imageLUTUIntShader_glsl_frag) / sizeof(imageLUTUIntShader_glsl_frag[0]),
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    image_lut_float_pipeline_ =
        CreatePipeline(image_lut_pipeline_layout_, imageShader_glsl_vert,
                       sizeof(imageShader_glsl_vert) / sizeof(imageShader_glsl_vert[0]), imageLUTFloatShader_glsl_frag,
                       sizeof(imageLUTFloatShader_glsl_frag) / sizeof(imageLUTFloatShader_glsl_frag[0]),
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
        NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &geometry_pipeline_layout_));
    }

    geometry_point_pipeline_ = CreatePipeline(
        geometry_pipeline_layout_, geometryShader_glsl_vert,
        sizeof(geometryShader_glsl_vert) / sizeof(geometryShader_glsl_vert[0]), geometryShader_glsl_frag,
        sizeof(geometryShader_glsl_frag) / sizeof(geometryShader_glsl_frag[0]), VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
    geometry_line_pipeline_ =
        CreatePipeline(geometry_pipeline_layout_, geometryShader_glsl_vert,
                       sizeof(geometryShader_glsl_vert) / sizeof(geometryShader_glsl_vert[0]), geometryShader_glsl_frag,
                       sizeof(geometryShader_glsl_frag) / sizeof(geometryShader_glsl_frag[0]),
                       VK_PRIMITIVE_TOPOLOGY_LINE_LIST, {VK_DYNAMIC_STATE_LINE_WIDTH});
    geometry_line_strip_pipeline_ =
        CreatePipeline(geometry_pipeline_layout_, geometryShader_glsl_vert,
                       sizeof(geometryShader_glsl_vert) / sizeof(geometryShader_glsl_vert[0]), geometryShader_glsl_frag,
                       sizeof(geometryShader_glsl_frag) / sizeof(geometryShader_glsl_frag[0]),
                       VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, {VK_DYNAMIC_STATE_LINE_WIDTH});
    geometry_triangle_pipeline_ = CreatePipeline(
        geometry_pipeline_layout_, geometryShader_glsl_vert,
        sizeof(geometryShader_glsl_vert) / sizeof(geometryShader_glsl_vert[0]), geometryShader_glsl_frag,
        sizeof(geometryShader_glsl_frag) / sizeof(geometryShader_glsl_frag[0]), VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

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
        create_info.pSetLayouts            = &desc_set_layout_;
        create_info.pushConstantRangeCount = 2;
        create_info.pPushConstantRanges    = push_constant_ranges;
        NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &geometry_text_pipeline_layout_));
    }

    geometry_text_pipeline_ = CreatePipeline(
        geometry_text_pipeline_layout_, geometryTextShader_glsl_vert,
        sizeof(geometryTextShader_glsl_vert) / sizeof(geometryTextShader_glsl_vert[0]), geometryTextShader_glsl_frag,
        sizeof(geometryTextShader_glsl_frag) / sizeof(geometryTextShader_glsl_frag[0]),
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, {}, true);

    // ImGui initialization
    initImGui();
    window_->SetupCallbacks([this](int width, int height) { this->onFramebufferSize(width, height); });
    window_->InitImGui();

    // Initialize Cuda
    CudaCheck(cuInit(0));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, 0));
}

void Vulkan::Impl::initImGui()
{
    // if the app did not specify a context, create our own
    if (!ImGui::GetCurrentContext())
    {
        ImGui::CreateContext();
    }

    ImGuiIO &io    = ImGui::GetIO();
    io.IniFilename = nullptr; // Avoiding the INI file
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking

    std::vector<VkDescriptorPoolSize> pool_size{{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
                                                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets       = 2;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes    = pool_size.data();
    vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_imguiDescPool);

    // Setup Platform/Renderer back ends
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance        = m_instance;
    init_info.PhysicalDevice  = m_physicalDevice;
    init_info.Device          = m_device;
    init_info.QueueFamily     = m_graphicsQueueIndex;
    init_info.Queue           = m_queue;
    init_info.PipelineCache   = VK_NULL_HANDLE;
    init_info.DescriptorPool  = m_imguiDescPool;
    init_info.Subpass         = 0;
    init_info.MinImageCount   = 2;
    init_info.ImageCount      = static_cast<int>(m_swapChain.getImageCount());
    init_info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    init_info.CheckVkResultFn = nullptr;
    init_info.Allocator       = nullptr;

    ImGui_ImplVulkan_Init(&init_info, m_renderPass);

    // Upload Fonts
    VkCommandBuffer cmd_buf = createTempCmdBuffer();
    ImGui_ImplVulkan_CreateFontsTexture(cmd_buf);
    submitTempCmdBuffer(cmd_buf);
}

void Vulkan::Impl::prepareFrame()
{
    // Acquire the next image from the swap chain
    if (!m_swapChain.acquire())
    {
        throw std::runtime_error("Failed to acquire next swap chain image.");
    }

    // Use a fence to wait until the command buffer has finished execution before using it again
    const uint32_t image_index = m_swapChain.getActiveImageIndex();

    VkResult result{VK_SUCCESS};
    do
    {
        result = vkWaitForFences(m_device, 1, &m_waitFences[image_index], VK_TRUE, 1'000'000);
    } while (result == VK_TIMEOUT);

    if (result != VK_SUCCESS)
    {
        // This allows Aftermath to do things and later assert below
        usleep(1000);
    }
    assert(result == VK_SUCCESS);
}

VkPipeline Vulkan::Impl::CreatePipeline(VkPipelineLayout pipeline_layout, const uint32_t *vertex_shader,
                                        size_t vertex_shader_size, const uint32_t *fragment_shader,
                                        size_t fragment_shader_size, VkPrimitiveTopology topology,
                                        const std::vector<VkDynamicState> dynamic_state, bool imgui_attrib_desc)
{
    nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, pipeline_layout, getRenderPass());
    gpb.depthStencilState.depthTestEnable = true;

    std::vector<uint32_t> code;
    code.assign(vertex_shader, vertex_shader + vertex_shader_size);
    gpb.addShader(code, VK_SHADER_STAGE_VERTEX_BIT);
    code.assign(fragment_shader, fragment_shader + fragment_shader_size);
    gpb.addShader(code, VK_SHADER_STAGE_FRAGMENT_BIT);

    gpb.inputAssemblyState.topology = topology;

    if (imgui_attrib_desc)
    {
        gpb.addBindingDescription({0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX});
        gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, pos)}});
        gpb.addAttributeDescriptions({{1, 0, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, uv)}});
        gpb.addAttributeDescriptions({{2, 0, VK_FORMAT_R8G8B8A8_UNORM, IM_OFFSETOF(ImDrawVert, col)}});
    }
    else
    {
        gpb.addBindingDescription({0, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX});
        gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}});
    }

    // disable culling
    gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // enable blending
    VkPipelineColorBlendAttachmentState color_blend_attachment_state{};
    color_blend_attachment_state.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment_state.blendEnable         = VK_TRUE;
    color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_state.colorBlendOp        = VK_BLEND_OP_ADD;
    color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_state.alphaBlendOp        = VK_BLEND_OP_ADD;
    // remove the default blend attachment state
    gpb.clearBlendAttachmentStates();
    gpb.addBlendAttachmentState(color_blend_attachment_state);

    for (auto &&state : dynamic_state)
    {
        gpb.addDynamicStateEnable(state);
    }

    return gpb.createPipeline();
}

VkFormat Vulkan::Impl::ToVulkanFormat(ImageFormat format, uint32_t *src_channels, uint32_t *dst_channels,
                                      uint32_t *component_size)
{
    VkFormat vk_format;

    switch (format)
    {
    case ImageFormat::R8_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint8_t);
        vk_format                     = VK_FORMAT_R8_UINT;
        break;
    case ImageFormat::R16_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint16_t);
        vk_format                     = VK_FORMAT_R16_UINT;
        break;
    case ImageFormat::R16_SFLOAT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint16_t);
        vk_format                     = VK_FORMAT_R16_SFLOAT;
        break;
    case ImageFormat::R32_UINT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(uint32_t);
        vk_format                     = VK_FORMAT_R32_UINT;
        break;
    case ImageFormat::R32_SFLOAT:
        *src_channels = *dst_channels = 1u;
        *component_size               = sizeof(float);
        vk_format                     = VK_FORMAT_R32_SFLOAT;
        break;
    case ImageFormat::R8G8B8_UNORM:
        *src_channels   = 3u;
        *dst_channels   = 4u;
        *component_size = sizeof(uint8_t);
        vk_format       = VK_FORMAT_R8G8B8A8_UNORM;
        break;
    case ImageFormat::B8G8R8_UNORM:
        *src_channels   = 3u;
        *dst_channels   = 4u;
        *component_size = sizeof(uint8_t);
        vk_format       = VK_FORMAT_B8G8R8A8_UNORM;
        break;
    case ImageFormat::R8G8B8A8_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint8_t);
        vk_format                     = VK_FORMAT_R8G8B8A8_UNORM;
        break;
    case ImageFormat::B8G8R8A8_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint8_t);
        vk_format                     = VK_FORMAT_B8G8R8A8_UNORM;
        break;
    case ImageFormat::R16G16B16A16_UNORM:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint16_t);
        vk_format                     = VK_FORMAT_R16G16B16A16_UNORM;
        break;
    case ImageFormat::R16G16B16A16_SFLOAT:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint16_t);
        vk_format                     = VK_FORMAT_R16G16B16A16_SFLOAT;
        break;
    case ImageFormat::R32G32B32A32_SFLOAT:
        *src_channels = *dst_channels = 4u;
        *component_size               = sizeof(uint32_t);
        vk_format                     = VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
    default:
        throw std::runtime_error("Unhandled image format.");
    }

    return vk_format;
}

Vulkan::Texture *Vulkan::Impl::CreateTextureForCudaUpload(uint32_t width, uint32_t height, ImageFormat format,
                                                          VkFilter filter, bool normalized)
{
    uint32_t src_channels, dst_channels, component_size;
    const VkFormat vk_format = ToVulkanFormat(format, &src_channels, &dst_channels, &component_size);

    // create the image
    const VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{width, height}, vk_format);
    // the VkExternalMemoryImageCreateInfoKHR struct is appended by nvvk::ExportResourceAllocator
    const nvvk::Image image = export_alloc_.createImage(imageCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // create the texture
    Texture *texture = new Texture(width, height, format, &export_alloc_);

    // create the vulkan texture
    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCreateInfo.minFilter               = filter;
    samplerCreateInfo.magFilter               = filter;
    samplerCreateInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.maxLod                  = normalized ? FLT_MAX : 0;
    samplerCreateInfo.unnormalizedCoordinates = normalized ? VK_FALSE : VK_TRUE;

    const VkImageViewCreateInfo imageViewInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture->texture_                         = export_alloc_.createTexture(image, imageViewInfo, samplerCreateInfo);

    // import memory to CUDA
    const nvvk::MemAllocator::MemInfo memInfo = export_alloc_.getMemoryAllocator()->getMemoryInfo(image.memHandle);

    VkMemoryGetFdInfoKHR getFdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    getFdInfo.memory     = memInfo.memory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int handle           = -1;
    NVVK_CHECK(vkGetMemoryFdKHR(m_device, &getFdInfo, &handle));

    CudaCheck(cuCtxPushCurrent(cuda_context_));

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc{};
    memHandleDesc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memHandleDesc.handle.fd = handle;
    memHandleDesc.size      = memInfo.offset + memInfo.size;

    CudaCheck(cuImportExternalMemory(&texture->external_mem_, &memHandleDesc));

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapArrayDesc{};
    mipmapArrayDesc.arrayDesc.Width  = width;
    mipmapArrayDesc.arrayDesc.Height = height;
    mipmapArrayDesc.arrayDesc.Depth  = 0;
    switch (component_size)
    {
    case 1:
        mipmapArrayDesc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        break;
    case 2:
        mipmapArrayDesc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
        break;
    case 4:
        mipmapArrayDesc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
        break;
    default:
        throw std::runtime_error("Unhandled component size");
    }
    mipmapArrayDesc.arrayDesc.NumChannels = dst_channels;
    mipmapArrayDesc.arrayDesc.Flags       = CUDA_ARRAY3D_SURFACE_LDST;

    mipmapArrayDesc.numLevels = 1;
    mipmapArrayDesc.offset    = memInfo.offset;

    CudaCheck(cuExternalMemoryGetMappedMipmappedArray(&texture->mipmap_, texture->external_mem_, &mipmapArrayDesc));

    CUcontext popped_context;
    CudaCheck(cuCtxPopCurrent(&popped_context));
    if (popped_context != cuda_context_)
    {
        LOGE("Cuda: Unexpected context popped");
    }

    // don't need to close if successfully imported
    /*if (close(handle) != 0)
        {
            throw std::runtime_error("close failed.");
        }
        handle = -1;*/

    return texture;
}

Vulkan::Texture *Vulkan::Impl::CreateTexture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size,
                                             const void *data, VkFilter filter, bool normalized)
{
    uint32_t src_channels, dst_channels, component_size;
    const VkFormat vk_format = ToVulkanFormat(format, &src_channels, &dst_channels, &component_size);

    if (data_size != width * height * src_channels * component_size)
    {
        throw std::runtime_error("The size of the data array is wrong");
    }

    nvvk::CommandPool cmd_buf_get(m_device, m_graphicsQueueIndex);
    VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

    const VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{width, height}, vk_format);
    const nvvk::Image image                 = alloc_.createImage(cmd_buf, data_size, data, imageCreateInfo);

    // create the texture
    Texture *texture = new Texture(width, height, format, &alloc_);

    // create the Vulkan texture
    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    /// @todo this (nearest, unnormalizedCoordinates) is just for the LUT-texture
    samplerCreateInfo.minFilter               = filter;
    samplerCreateInfo.magFilter               = filter;
    samplerCreateInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.maxLod                  = normalized ? FLT_MAX : 0;
    samplerCreateInfo.unnormalizedCoordinates = normalized ? VK_FALSE : VK_TRUE;

    const VkImageViewCreateInfo imageViewInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture->texture_                         = alloc_.createTexture(image, imageViewInfo, samplerCreateInfo);

    // transition to shader layout
    /// @todo I don't know if this is defined. Should the old layout be VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, like it
    ///       would be if we uploaded using Vulkan?
    nvvk::cmdBarrierImageLayout(cmd_buf, image.image, imageCreateInfo.initialLayout,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cmd_buf_get.submitAndWait(cmd_buf);

    return texture;
}

void Vulkan::Impl::DestroyTexture(Texture *texture)
{
    // check if this texture had been imported to CUDA
    if (texture->mipmap_ || texture->external_mem_)
    {
        CudaCheck(cuCtxPushCurrent(cuda_context_));

        CudaCheck(cuMipmappedArrayDestroy(texture->mipmap_));
        texture->mipmap_ = nullptr;

        CudaCheck(cuDestroyExternalMemory(texture->external_mem_));
        texture->external_mem_ = nullptr;

        CUcontext popped_context;
        CudaCheck(cuCtxPopCurrent(&popped_context));
        if (popped_context != cuda_context_)
        {
            LOGE("Cuda: Unexpected context popped");
        }
    }

    texture->alloc_->destroy(texture->texture_);

    delete texture;
}

void Vulkan::Impl::UploadToTexture(CUdeviceptr device_ptr, const Texture *texture)
{
    if (!texture->mipmap_)
    {
        throw std::runtime_error("Texture had not been imported to CUDA, can't upload data.");
    }

    CUarray array;
    CudaCheck(cuMipmappedArrayGetLevel(&array, texture->mipmap_, 0));

    uint32_t src_channels, dst_channels, component_size;
    const VkFormat vk_format = ToVulkanFormat(texture->format_, &src_channels, &dst_channels, &component_size);
    const uint32_t src_pitch = texture->width_ * src_channels * component_size;

    if ((texture->format_ == ImageFormat::R8G8B8_UNORM) || (texture->format_ == ImageFormat::B8G8R8_UNORM))
    {
        // if R8G8B8 is not supported then convert to R8G8B8A8
        ConvertR8G8B8ToR8G8B8A8(texture->width_, texture->height_, device_ptr, src_pitch, array);
        /// @todo use streams
        CudaCheck(cuCtxSynchronize());
    }
    else
    {
        // else just copy
        CUDA_MEMCPY2D memcpy2d{};
        memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2d.srcDevice     = device_ptr;
        memcpy2d.srcPitch      = src_pitch;
        memcpy2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        memcpy2d.dstArray      = array;
        memcpy2d.WidthInBytes  = texture->width_ * dst_channels * component_size;
        memcpy2d.Height        = texture->height_;
        /// @todo use streams
        CudaCheck(cuMemcpy2D(&memcpy2d));
    }

    // transition to shader layout
    {
        nvvk::CommandPool cmd_buf_get(m_device, m_graphicsQueueIndex);
        VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();
        nvvk::cmdBarrierImageLayout(cmd_buf, texture->texture_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        cmd_buf_get.submitAndWait(cmd_buf);
    }
}

void Vulkan::Impl::DrawTexture(const Texture *texture, const Texture *lut, float opacity)
{
    const VkCommandBuffer cmd_buf = getCommandBuffers()[getCurFrame()];

    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSet desc_set;

    // update descriptor sets
    if (lut)
    {
        std::vector<VkWriteDescriptorSet> writes;
        writes.emplace_back(desc_set_layout_bind_lut_.makeWrite(desc_set_lut_, bindings_offset_texture_,
                                                                &texture->texture_.descriptor));
        writes.emplace_back(desc_set_layout_bind_lut_.makeWrite(desc_set_lut_, bindings_offset_texture_lut_,
                                                                &lut->texture_.descriptor));
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        if ((texture->format_ == ImageFormat::R8_UINT) || (texture->format_ == ImageFormat::R16_UINT) ||
            (texture->format_ == ImageFormat::R32_UINT))
        {
            pipeline = image_lut_uint_pipeline_;
        }
        else
        {
            pipeline = image_lut_float_pipeline_;
        }

        pipeline_layout = image_lut_pipeline_layout_;
        desc_set        = desc_set_lut_;
    }
    else
    {
        const VkWriteDescriptorSet write_descriptor_set =
            desc_set_layout_bind_.makeWrite(desc_set_, bindings_offset_texture_, &texture->texture_.descriptor);
        vkUpdateDescriptorSets(m_device, 1, &write_descriptor_set, 0, nullptr);

        pipeline        = image_pipeline_;
        pipeline_layout = image_pipeline_layout_;
        desc_set        = desc_set_;
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &desc_set, 0, nullptr);

    // push the constants
    PushConstantFragment push_constants;
    push_constants.opacity = opacity;
    vkCmdPushConstants(cmd_buf, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantFragment),
                       &push_constants);

    // bind the buffers
    VkDeviceSize offset{0};
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &vertex_buffer_.buffer, &offset);
    vkCmdBindIndexBuffer(cmd_buf, index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT16);

    // draw
    vkCmdDrawIndexed(cmd_buf, 6, 1, 0, 0, 0);
}

Vulkan::Buffer *Vulkan::Impl::CreateBuffer(size_t data_size, const void *data, VkBufferUsageFlags usage)
{
    std::unique_ptr<Buffer> buffer(new Buffer(data_size, &alloc_));

    nvvk::CommandPool cmd_buf_get(vkctx_.m_device, m_graphicsQueueIndex);
    VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

    buffer->buffer_ = alloc_.createBuffer(cmd_buf, static_cast<VkDeviceSize>(data_size), data, usage);

    cmd_buf_get.submitAndWait(cmd_buf);
    alloc_.finalizeAndReleaseStaging();

    return buffer.release();
}

void Vulkan::Impl::DestroyBuffer(Buffer *buffer)
{
    buffer->alloc_->destroy(buffer->buffer_);
    delete buffer;
}

void Vulkan::Impl::Draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, const Buffer *buffer,
                        float opacity, const std::array<float, 4> &color, float point_size, float line_width)
{
    const VkCommandBuffer cmd_buf = getCommandBuffers()[getCurFrame()];

    switch (topology)
    {
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
    vkCmdPushConstants(cmd_buf, geometry_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(PushConstantVertex),
                       sizeof(PushConstantFragment), &push_constants_fragment);

    PushConstantVertex push_constant_vertex;
    push_constant_vertex.matrix.identity();
    push_constant_vertex.matrix.scale({2.f, 2.f, 1.f});
    push_constant_vertex.matrix.translate({-.5f, -.5f, 0.f});
    push_constant_vertex.point_size = point_size;
    push_constant_vertex.color      = color;
    vkCmdPushConstants(cmd_buf, geometry_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantVertex),
                       &push_constant_vertex);

    VkDeviceSize offset{0};
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &buffer->buffer_.buffer, &offset);

    // draw
    vkCmdDraw(cmd_buf, count, 1, first, 0);
}

void Vulkan::Impl::DrawIndexed(VkDescriptorSet desc_set, const Buffer *vertex_buffer, const Buffer *index_buffer,
                               VkIndexType index_type, uint32_t index_count, uint32_t first_index,
                               uint32_t vertex_offset, float opacity)
{
    const VkCommandBuffer cmd_buf = getCommandBuffers()[getCurFrame()];

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_text_pipeline_layout_, 0, 1, &desc_set,
                            0, nullptr);

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, geometry_text_pipeline_);

    // push the constants
    PushConstantFragment push_constants;
    push_constants.opacity = opacity;
    vkCmdPushConstants(cmd_buf, geometry_text_pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       sizeof(PushConstantTextVertex), sizeof(PushConstantFragment), &push_constants);

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
}

Vulkan::Vulkan()
    : impl_(new Vulkan::Impl)
{
}

Vulkan::~Vulkan() {}

void Vulkan::Setup(Window *window)
{
    impl_->Setup(window);
}

void Vulkan::SubmitFrame(const std::list<std::unique_ptr<Layer>> &layers)
{
    // Acquire the next image
    impl_->prepareFrame();

    // Get the command buffer for the frame. There are n command buffers equal to the number of in-flight frames.
    const uint32_t cur_frame      = impl_->getCurFrame();
    const VkCommandBuffer cmd_buf = impl_->getCommandBuffers()[cur_frame];

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    NVVK_CHECK(vkBeginCommandBuffer(cmd_buf, &begin_info));

    // Clearing values
    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color        = {{0.f, 0.f, 0.f, 1.f}};
    clear_values[1].depthStencil = {1.0f, 0};

    // Begin rendering
    VkRenderPassBeginInfo renderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues    = clear_values.data();
    renderPassBeginInfo.renderPass      = impl_->getRenderPass();
    renderPassBeginInfo.framebuffer     = impl_->getFramebuffers()[cur_frame];
    renderPassBeginInfo.renderArea      = {{0, 0}, impl_->getSize()};
    vkCmdBeginRenderPass(cmd_buf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // set the dynamic viewport
    impl_->setViewport(cmd_buf);

    // sort layers (inverse because highest priority is drawn last)
    std::list<Layer *> sorted_layers;
    for (auto &&item : layers)
    {
        sorted_layers.emplace_back(item.get());
    }
    sorted_layers.sort([](Layer *a, Layer *b) { return a->GetPriority() < b->GetPriority(); });

    // render
    for (auto &&layer : sorted_layers)
    {
        layer->Render(this);
    }

    // End rendering
    vkCmdEndRenderPass(cmd_buf);

    // Submit for display
    NVVK_CHECK(vkEndCommandBuffer(cmd_buf));
    impl_->submitFrame();

    /// @todo need to wait for idle here because after this point the layers are destroyed and they destroy their Vulkan
    /// resources which is only allowed when the command buffer had completed execution. Replace this by using
    /// synchronization primitives.
    NVVK_CHECK(vkQueueWaitIdle(impl_->getQueue()));
}

VkCommandBuffer Vulkan::GetCommandBuffer()
{
    return impl_->getCommandBuffers()[impl_->getCurFrame()];
}

Vulkan::Texture *Vulkan::CreateTextureForCudaUpload(uint32_t width, uint32_t height, ImageFormat format,
                                                    VkFilter filter, bool normalized)
{
    return impl_->CreateTextureForCudaUpload(width, height, format, filter, normalized);
}

Vulkan::Texture *Vulkan::CreateTexture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size,
                                       const void *data, VkFilter filter, bool normalized)
{
    return impl_->CreateTexture(width, height, format, data_size, data, filter, normalized);
}

void Vulkan::DestroyTexture(Texture *texture)
{
    impl_->DestroyTexture(texture);
}

void Vulkan::UploadToTexture(CUdeviceptr device_ptr, const Texture *texture)
{
    impl_->UploadToTexture(device_ptr, texture);
}

void Vulkan::DrawTexture(const Texture *texture, const Texture *lut, float opacity)
{
    impl_->DrawTexture(texture, lut, opacity);
}

Vulkan::Buffer *Vulkan::CreateBuffer(size_t data_size, const void *data, VkBufferUsageFlags usage)
{
    return impl_->CreateBuffer(data_size, data, usage);
}

void Vulkan::DestroyBuffer(Buffer *buffer)
{
    impl_->DestroyBuffer(buffer);
}

void Vulkan::Draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, const Buffer *buffer, float opacity,
                  const std::array<float, 4> &color, float point_size, float line_width)
{
    impl_->Draw(topology, count, first, buffer, opacity, color, point_size, line_width);
}

void Vulkan::DrawIndexed(VkDescriptorSet desc_set, const Buffer *vertex_buffer, const Buffer *index_buffer,
                         VkIndexType index_type, uint32_t index_count, uint32_t first_index, uint32_t vertex_offset,
                         float opacity)
{
    impl_->DrawIndexed(desc_set, vertex_buffer, index_buffer, index_type, index_count, first_index, vertex_offset,
                       opacity);
}

} // namespace clara::holoviz
