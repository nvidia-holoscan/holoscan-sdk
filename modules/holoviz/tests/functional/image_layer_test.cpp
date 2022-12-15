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

#include <gtest/gtest.h>

#include "headless_fixture.hpp"
#include <holoviz/holoviz.hpp>
#include <cuda/cuda_service.hpp>

namespace viz = holoscan::viz;

enum class Source {
    HOST,
    CUDA_DEVICE
};

enum class Reuse {
    DISABLE,
    ENABLE
};

enum class UseLut {
    DISABLE,
    ENABLE
};

// define the '<<' operators to get a nice parameter string

#define CASE(VALUE)                \
    case VALUE:                    \
        os << std::string(#VALUE); \
        break;

// define the '<<' operator to get a nice parameter string
std::ostream &operator<<(std::ostream &os, const Source &source) {
    switch (source) {
        CASE(Source::HOST)
        CASE(Source::CUDA_DEVICE)
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Reuse &reuse) {
    switch (reuse) {
        CASE(Reuse::DISABLE)
        CASE(Reuse::ENABLE)
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const UseLut &use_lut) {
    switch (use_lut) {
        CASE(UseLut::DISABLE)
        CASE(UseLut::ENABLE)
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;
}

#undef CASE

class ImageLayer
    : public TestHeadless
    , public testing::WithParamInterface<std::tuple<Source, Reuse, UseLut>> {
};

TEST_P(ImageLayer, Image) {
    const Source source = std::get<0>(GetParam());
    const bool reuse    = std::get<1>(GetParam()) == Reuse::ENABLE;
    const bool use_lut  = std::get<2>(GetParam()) == UseLut::ENABLE;

    const viz::ImageFormat kFormat    = use_lut ? viz::ImageFormat::R8_UINT :
                                                  viz::ImageFormat::R8G8B8A8_UNORM;
    const viz::ImageFormat kLutFormat = viz::ImageFormat::R8G8B8A8_UNORM;

    SetupData(kFormat);

    std::vector<uint32_t> lut;
    std::vector<uint8_t> data_with_lut;

    viz::CudaService::ScopedPush cuda_context;
    viz::UniqueCUdeviceptr device_ptr;

    if (source == Source::CUDA_DEVICE) {
        cuda_context = viz::CudaService::get().PushContext();
        device_ptr.reset([this] {
            CUdeviceptr device_ptr;
            EXPECT_EQ(cuMemAlloc(&device_ptr, data_.size()), CUDA_SUCCESS);
            return device_ptr;
        }());

        EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), data_.data(), data_.size()), CUDA_SUCCESS);
    }

    if (use_lut) {
        std::srand(1);
        lut.resize(lut_size_);
        for (uint32_t index = 0; index < lut_size_; ++index) {
            lut[index] = static_cast<uint32_t>(std::rand()) | 0xFF000000;
        }

        // lookup color to produce result
        data_with_lut.resize(width_ * height_ * sizeof(uint32_t));
        for (size_t index = 0; index < width_ * height_; ++index) {
            const uint8_t value = data_[index];

            reinterpret_cast<uint32_t *>(data_with_lut.data())[index] = lut[value];
        }
    }

    for (uint32_t i = 0; i < (reuse ? 2 : 1); ++i) {
        EXPECT_NO_THROW(viz::Begin());

        EXPECT_NO_THROW(viz::BeginImageLayer());

        if (use_lut) {
            EXPECT_NO_THROW(viz::LUT(lut_size_, kLutFormat, lut.size() * sizeof(uint32_t),
                                                                             lut.data()));
        }

        switch (source) {
        case Source::HOST:
            EXPECT_NO_THROW(viz::ImageHost(width_, height_, kFormat,
                                           reinterpret_cast<void *>(data_.data())));
            break;
        case Source::CUDA_DEVICE:
            EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()));
            break;
        default:
            EXPECT_TRUE(false) << "Unhandled source type";
        }
        EXPECT_NO_THROW(viz::EndLayer());

        EXPECT_NO_THROW(viz::End());
    }

    if (use_lut) {
        std::swap(data_, data_with_lut);
        CompareResult();
        std::swap(data_with_lut, data_);
    } else {
        CompareResult();
    }
}

INSTANTIATE_TEST_SUITE_P(ImageLayer, ImageLayer,
                         testing::Combine(testing::Values(Source::HOST, Source::CUDA_DEVICE),
                                          testing::Values(Reuse::DISABLE, Reuse::ENABLE),
                                          testing::Values(UseLut::DISABLE, UseLut::ENABLE)));

TEST_F(ImageLayer, ImageCudaArray) {
    constexpr viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

    CUarray array;

    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    // Cuda array support is not yet implemented, the test below should throw. Add the cuda array
    // test to the test case above when cuda array support is implemented.
    EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
}

/**
 * Switch between host data and device data each frame to make sure that layers are not reused.
 *
 * Background: Holoviz is maintaining a layer cache to avoid re-creating resources each frame
 * to improve performance. The function `Layer::can_be_reused()` is used to check if a layer can
 * be reused. For the image layer there it's special case when switching from host to device memory,
 * That special case is tested here.
 */
TEST_F(ImageLayer, SwitchHostDevice) {
    const viz::ImageFormat kFormat = viz::ImageFormat::R8G8B8A8_UNORM;

    const viz::CudaService::ScopedPush cuda_context = viz::CudaService::get().PushContext();

    for (uint32_t i = 0; i < 3; ++i) {
        SetupData(kFormat, i);

        viz::UniqueCUdeviceptr device_ptr;
        device_ptr.reset([this] {
            CUdeviceptr device_ptr;
            EXPECT_EQ(cuMemAlloc(&device_ptr, data_.size()), CUDA_SUCCESS);
            return device_ptr;
        }());

        EXPECT_EQ(cuMemcpyHtoD(device_ptr.get(), data_.data(), data_.size()), CUDA_SUCCESS);

        EXPECT_NO_THROW(viz::Begin());

        EXPECT_NO_THROW(viz::BeginImageLayer());

        if (i & 1) {
            EXPECT_NO_THROW(viz::ImageHost(width_, height_, kFormat,
                                           reinterpret_cast<void *>(data_.data())));
        } else {
            EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()));
        }

        EXPECT_NO_THROW(viz::EndLayer());

        EXPECT_NO_THROW(viz::End());

        if (!CompareResult()) {
            break;
        }
    }
}

TEST_F(ImageLayer, Errors) {
    constexpr viz::ImageFormat kFormat    = viz::ImageFormat::R8G8B8A8_UNORM;
    constexpr viz::ImageFormat kLutFormat = viz::ImageFormat::R8G8B8A8_UNORM;

    std::vector<uint8_t> host_data(width_ * height_ * 4);
    std::vector<uint32_t> lut(lut_size_);

    const viz::CudaService::ScopedPush cuda_context = viz::CudaService::get().PushContext();

    viz::UniqueCUdeviceptr device_ptr;
    device_ptr.reset([this] {
        CUdeviceptr device_ptr;
        EXPECT_EQ(cuMemAlloc(&device_ptr, width_ * height_ * 4), CUDA_SUCCESS);
        return device_ptr;
    }());

    CUarray array;

    EXPECT_NO_THROW(viz::Begin());

    // it's an error to specify both a host and a device image for a layer.
    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::ImageHost(width_, height_, kFormat,
                                   reinterpret_cast<void *>(host_data.data())));
    EXPECT_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()),
                                                              std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_NO_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()));
    EXPECT_THROW(viz::ImageHost(width_, height_, kFormat,
                                reinterpret_cast<void *>(host_data.data())),
                 std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    // it's an error to call image functions without calling BeginImageLayer first
    EXPECT_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()),
                                                              std::runtime_error);
    EXPECT_THROW(viz::ImageHost(width_, height_, kFormat,
                 reinterpret_cast<void *>(host_data.data())),
                 std::runtime_error);
    EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
    EXPECT_THROW(viz::LUT(lut_size_, kLutFormat, lut.size() * sizeof(uint32_t), lut.data()),
                                                                        std::runtime_error);

    // it's an error to call BeginImageLayer again without calling EndLayer
    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_THROW(viz::BeginImageLayer(), std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    // it's an error to call image functions when a different layer is active
    EXPECT_NO_THROW(viz::BeginGeometryLayer());
    EXPECT_THROW(viz::ImageCudaDevice(width_, height_, kFormat, device_ptr.get()),
                                                              std::runtime_error);
    EXPECT_THROW(viz::ImageHost(width_, height_, kFormat,
                 reinterpret_cast<void *>(host_data.data())),
                 std::runtime_error);
    EXPECT_THROW(viz::ImageCudaArray(kFormat, array), std::runtime_error);
    EXPECT_THROW(viz::LUT(lut_size_, kLutFormat, lut.size() * sizeof(uint32_t), lut.data()),
                                                                        std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
}
