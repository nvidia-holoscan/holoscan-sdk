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

namespace viz = holoscan::viz;

// define the '<<' operator to get a nice parameter string
std::ostream &operator<<(std::ostream &os, const viz::PrimitiveTopology &topology) {
#define CASE(VALUE)                \
    case VALUE:                    \
        os << std::string(#VALUE); \
        break;

    switch (topology) {
        CASE(viz::PrimitiveTopology::POINT_LIST);
        CASE(viz::PrimitiveTopology::LINE_LIST);
        CASE(viz::PrimitiveTopology::LINE_STRIP);
        CASE(viz::PrimitiveTopology::TRIANGLE_LIST);
        CASE(viz::PrimitiveTopology::CROSS_LIST);
        CASE(viz::PrimitiveTopology::RECTANGLE_LIST);
        CASE(viz::PrimitiveTopology::OVAL_LIST);
    default:
        os.setstate(std::ios_base::failbit);
    }
    return os;

#undef CASE
}

// Fixture that creates initializes Holoviz
class GeometryLayer
    : public TestHeadless
    , public testing::WithParamInterface<viz::PrimitiveTopology> {
};

TEST_P(GeometryLayer, Primitive) {
    const viz::PrimitiveTopology topology = GetParam();

    uint32_t crc;
    uint32_t primitive_count;
    std::vector<float> data;
    switch (topology) {
    case viz::PrimitiveTopology::POINT_LIST:
        primitive_count = 1;
        data.push_back(0.5f);
        data.push_back(0.5f);
        crc = 0xE81FD1BB;
        break;
    case viz::PrimitiveTopology::LINE_LIST:
        primitive_count = 2;
        data.push_back(0.1f);
        data.push_back(0.1f);
        data.push_back(0.9f);
        data.push_back(0.9f);

        data.push_back(0.7f);
        data.push_back(0.3f);
        data.push_back(0.2f);
        data.push_back(0.4f);
        crc = 0xF7E63B21;
        break;
    case viz::PrimitiveTopology::LINE_STRIP:
        primitive_count = 2;
        data.push_back(0.1f);
        data.push_back(0.1f);
        data.push_back(0.7f);
        data.push_back(0.9f);

        data.push_back(0.3f);
        data.push_back(0.2f);
        crc = 0x392E35D8;
        break;
    case viz::PrimitiveTopology::TRIANGLE_LIST:
        primitive_count = 2;
        data.push_back(0.1f);
        data.push_back(0.1f);
        data.push_back(0.5f);
        data.push_back(0.9f);
        data.push_back(0.9f);
        data.push_back(0.1f);

        data.push_back(0.05f);
        data.push_back(0.7f);
        data.push_back(0.15f);
        data.push_back(0.8f);
        data.push_back(0.25f);
        data.push_back(0.6f);
        crc = 0xB29BAA37;
        break;
    case viz::PrimitiveTopology::CROSS_LIST:
        primitive_count = 2;
        data.push_back(0.5f);
        data.push_back(0.5f);
        data.push_back(0.1f);

        data.push_back(0.1f);
        data.push_back(0.3f);
        data.push_back(0.01f);
        crc = 0x16056A95;
        break;
    case viz::PrimitiveTopology::RECTANGLE_LIST:
        primitive_count = 2;
        data.push_back(0.1f);
        data.push_back(0.1f);
        data.push_back(0.9f);
        data.push_back(0.9f);

        data.push_back(0.3f);
        data.push_back(0.2f);
        data.push_back(0.5f);
        data.push_back(0.3f);
        crc = 0x355A2C00;
        break;
    case viz::PrimitiveTopology::OVAL_LIST:
        primitive_count = 2;
        data.push_back(0.5f);
        data.push_back(0.5f);
        data.push_back(0.2f);
        data.push_back(0.1f);

        data.push_back(0.6f);
        data.push_back(0.4f);
        data.push_back(0.05f);
        data.push_back(0.07f);
        crc = 0xA907614F;
        break;
    default:
        EXPECT_TRUE(false) << "Unhandled primitive topoplogy";
    }

    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginGeometryLayer());

    for (uint32_t i = 0; i < 3; ++i) {
        if (i == 1) {
            EXPECT_NO_THROW(viz::Color(1.f, 0.5f, 0.25f, 0.75f));
        } else if (i == 2) {
            EXPECT_NO_THROW(viz::PointSize(4.f));
            EXPECT_NO_THROW(viz::LineWidth(3.f));
        }

        EXPECT_NO_THROW(viz::Primitive(topology, primitive_count, data.size(), data.data()));

        for (auto &&item : data) {
            item += 0.1f;
        }
    }
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());

    CompareResultCRC32({crc});
}

INSTANTIATE_TEST_SUITE_P(GeometryLayer, GeometryLayer,
                         testing::Values(viz::PrimitiveTopology::POINT_LIST,
                                         viz::PrimitiveTopology::LINE_LIST,
                                         viz::PrimitiveTopology::LINE_STRIP,
                                         viz::PrimitiveTopology::TRIANGLE_LIST,
                                         viz::PrimitiveTopology::CROSS_LIST,
                                         viz::PrimitiveTopology::RECTANGLE_LIST,
                                         viz::PrimitiveTopology::OVAL_LIST));

TEST_F(GeometryLayer, Text) {
    EXPECT_NO_THROW(viz::Begin());

    EXPECT_NO_THROW(viz::BeginGeometryLayer());
    EXPECT_NO_THROW(viz::Text(0.4f, 0.4f, 0.4f, "Text"));
    EXPECT_NO_THROW(viz::Color(0.5f, 0.9f, 0.7f, 0.9f));
    EXPECT_NO_THROW(viz::Text(0.1f, 0.1f, 0.2f, "Colored"));
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());

    CompareResultCRC32({0xCA706DD0});
}

TEST_F(GeometryLayer, Reuse) {
    std::vector<float> data{0.5f, 0.5f};

    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_NO_THROW(viz::Begin());

        EXPECT_NO_THROW(viz::BeginGeometryLayer());
        EXPECT_NO_THROW(viz::Color(0.1f, 0.2f, 0.3f, 0.4f));
        EXPECT_NO_THROW(viz::LineWidth(2.f));
        EXPECT_NO_THROW(viz::PointSize(3.f));
        EXPECT_NO_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(),
                                                                            data.data()));
        EXPECT_NO_THROW(viz::Text(0.4f, 0.4f, 0.1f, "Text"));
        EXPECT_NO_THROW(viz::EndLayer());

        EXPECT_NO_THROW(viz::End());
    }
}

TEST_F(GeometryLayer, Errors) {
    std::vector<float> data{0.5f, 0.5f};

    EXPECT_NO_THROW(viz::Begin());

    // it's an error to call geometry functions without calling BeginGeometryLayer first
    EXPECT_THROW(viz::Color(0.f, 0.f, 0.f, 1.f), std::runtime_error);
    EXPECT_THROW(viz::LineWidth(1.0f), std::runtime_error);
    EXPECT_THROW(viz::PointSize(1.0f), std::runtime_error);
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(),
                                                 data.data()), std::runtime_error);
    EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"), std::runtime_error);

    // it's an error to call BeginGeometryLayer again without calling EndLayer
    EXPECT_NO_THROW(viz::BeginGeometryLayer());
    EXPECT_THROW(viz::BeginGeometryLayer(), std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    // it's an error to call geometry functions when a different layer is active
    EXPECT_NO_THROW(viz::BeginImageLayer());
    EXPECT_THROW(viz::Color(0.f, 0.f, 0.f, 1.f), std::runtime_error);
    EXPECT_THROW(viz::LineWidth(1.0f), std::runtime_error);
    EXPECT_THROW(viz::PointSize(1.0f), std::runtime_error);
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(),
                                                 data.data()), std::runtime_error);
    EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"), std::runtime_error);
    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::BeginGeometryLayer());

    // Primitive function errors, first call the passing function
    EXPECT_NO_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1,
                                              data.size(), data.data()));
    // it's an error to call Primitive with a primitive count of zero
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 0, data.size(), data.data()),
                 std::invalid_argument);
    // it's an error to call Primitive with a data size of zero
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, 0, data.data()),
                                                                std::invalid_argument);
    // it's an error to call Primitive with a null data pointer
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size(), nullptr),
                                                                      std::invalid_argument);
    // it's an error to call Primitive with a data size which is too small for the primitive count
    EXPECT_THROW(viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, data.size() - 1,
                                                     data.data()), std::runtime_error);

    // Text function errors, first call the passing function
    EXPECT_NO_THROW(viz::Text(0.5f, 0.5f, 0.1f, "Text"));
    // it's an error to call Text with a size of zero
    EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.0f, "Text"), std::invalid_argument);
    // it's an error to call Text with null text pointer
    EXPECT_THROW(viz::Text(0.5f, 0.5f, 0.1f, nullptr), std::invalid_argument);

    EXPECT_NO_THROW(viz::EndLayer());

    EXPECT_NO_THROW(viz::End());
}
