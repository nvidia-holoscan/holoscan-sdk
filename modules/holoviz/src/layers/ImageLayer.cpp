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

#include "ImageLayer.h"

#include "../vulkan/Vulkan.h"

#include <nvvk/resourceallocator_vk.hpp>

#include <vector>

namespace clara::holoviz
{

struct ImageLayer::Impl
{
    bool CanBeReused(Impl &other) const
    {
        // we can reuse if the format/size and LUT match
        if ((format_ == other.format_) && (width_ == other.width_) && (height_ == other.height_) &&
            (lut_size_ == other.lut_size_) && (lut_format_ == other.lut_format_) && (lut_data_ == other.lut_data_))
        {
            // update the device pointer, data will be uploaded when drawing
            other.device_ptr_ = device_ptr_;
            return true;
        }

        return false;
    }

    // user provided state
    ImageFormat format_     = ImageFormat(-1);
    uint32_t width_         = 0;
    uint32_t height_        = 0;
    CUdeviceptr device_ptr_ = 0;

    uint32_t lut_size_      = 0;
    ImageFormat lut_format_ = ImageFormat(-1);
    std::vector<uint8_t> lut_data_;

    // internal state
    Vulkan *vulkan_               = nullptr;
    Vulkan::Texture *texture_     = nullptr;
    Vulkan::Texture *lut_texture_ = nullptr;
};

ImageLayer::ImageLayer()
    : Layer(Type::Image)
    , impl_(new ImageLayer::Impl)
{
}

ImageLayer::~ImageLayer()
{
    if (impl_->vulkan_)
    {
        if (impl_->texture_)
        {
            impl_->vulkan_->DestroyTexture(impl_->texture_);
        }
        if (impl_->lut_texture_)
        {
            impl_->vulkan_->DestroyTexture(impl_->lut_texture_);
        }
    }
}

void ImageLayer::ImageCudaDevice(uint32_t width, uint32_t height, ImageFormat fmt, CUdeviceptr device_ptr)
{
    impl_->width_      = width;
    impl_->height_     = height;
    impl_->format_     = fmt;
    impl_->device_ptr_ = device_ptr;
}

void ImageLayer::ImageCudaArray(ImageFormat fmt, CUarray array)
{
    throw std::runtime_error("Not implemented");
}

void ImageLayer::ImageHost(uint32_t width, uint32_t height, ImageFormat fmt, const void *data)
{
    throw std::runtime_error("Not implemented");
}

void ImageLayer::LUT(uint32_t size, ImageFormat fmt, size_t data_size, const void *data)
{
    impl_->lut_size_   = size;
    impl_->lut_format_ = fmt;
    impl_->lut_data_.assign(reinterpret_cast<const uint8_t *>(data),
                            reinterpret_cast<const uint8_t *>(data) + data_size);
}

bool ImageLayer::CanBeReused(Layer &other) const
{
    return Layer::CanBeReused(other) && impl_->CanBeReused(*static_cast<const ImageLayer &>(other).impl_.get());
}

void ImageLayer::Render(Vulkan *vulkan)
{
    if (impl_->device_ptr_)
    {
        // check if this is a reused layer, in this case we just have to upload the data to the texture
        if (!impl_->texture_)
        {
            /// @todo need to remember Vulkan instance for destroying texture, destroy should propably be handled by Vulkan class
            impl_->vulkan_ = vulkan;

            // check if we have a lut, if yes, the texture needs to be nearest samopled since it has index values
            const bool has_lut = !impl_->lut_data_.empty();

            // create a texture to which we can upload from CUDA
            impl_->texture_ = vulkan->CreateTextureForCudaUpload(impl_->width_, impl_->height_, impl_->format_,
                                                                 has_lut ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

            if (has_lut)
            {
                // create LUT texture
                impl_->lut_texture_ =
                    vulkan->CreateTexture(impl_->lut_size_, 1, impl_->lut_format_, impl_->lut_data_.size(),
                                          impl_->lut_data_.data(), VK_FILTER_NEAREST, false/*normalized*/);
            }
        }

        vulkan->UploadToTexture(impl_->device_ptr_, impl_->texture_);

        // draw
        vulkan->DrawTexture(impl_->texture_, impl_->lut_texture_, GetOpacity());
    }
}

} // namespace clara::holoviz
