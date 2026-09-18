// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/common.hpp>
#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

/// \file
/// Re-exports the HoloViz core vocabulary (error type, expected) into
/// holoscan::visualization so the layer-renderer backends and the HolovizOp implementation
/// can name them unqualified. The backends are built on HoloViz and share its error
/// model rather than Holoscan Core's; these using-declarations shadow holoscan::expected,
/// holoscan::Error, and holoscan::ErrorCode for unqualified lookup inside this namespace.
///
/// Code in holoscan::visualization that mixes both vocabularies must qualify the Core side
/// explicitly (holoscan::expected, holoscan::Error, holoscan::make_unexpected): an unqualified
/// make_unexpected(holoscan::Error{...}) is ambiguous between this using-declaration and the
/// ADL candidate holoscan::make_unexpected.
///
/// Private to the operator implementation (not installed). The public headers include no HoloViz
/// header; their PresentMode is a distinct enum translated in "common/holoviz_conversions.hpp".

// A namespace alias needs its target declared; the text API itself is only pulled in by the
// text layer (<holoviz/text/text_renderer.hpp>).
namespace holoviz::text {}

namespace holoscan::visualization {

/// HoloViz text-rendering namespace alias (the text layer wraps a text::TextRenderer).
namespace text = holoviz::text;

using holoviz::Error;
using holoviz::ErrorCode;
using holoviz::expected;
using holoviz::forward_error;
using holoviz::make_unexpected;
using holoviz::unexpect;
using holoviz::unexpected;

}  // namespace holoscan::visualization
