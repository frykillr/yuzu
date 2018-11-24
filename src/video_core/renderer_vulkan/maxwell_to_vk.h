// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"

namespace Vulkan::MaxwellToVK {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using PixelFormat = VideoCore::Surface::PixelFormat;
using ComponentType = VideoCore::Surface::ComponentType;

vk::Format SurfaceFormat(PixelFormat pixel_format, ComponentType component_type);

vk::ShaderStageFlagBits ShaderStage(Maxwell::ShaderStage stage);

vk::PrimitiveTopology PrimitiveTopology(Maxwell::PrimitiveTopology topology);

vk::Format VertexFormat(Maxwell::VertexAttribute::Type type, Maxwell::VertexAttribute::Size size);

vk::CompareOp ComparisonOp(Maxwell::ComparisonOp comparison);

vk::IndexType IndexFormat(Maxwell::IndexFormat index_format);

} // namespace Vulkan::MaxwellToVK