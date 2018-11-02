// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/engines/maxwell_3d.h"

namespace Vulkan::MaxwellToVK {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

inline vk::ShaderStageFlagBits ShaderStage(Maxwell::ShaderProgram stage) {
    switch (stage) {
    case Maxwell::ShaderProgram::VertexA:
    case Maxwell::ShaderProgram::VertexB:
        return vk::ShaderStageFlagBits::eVertex;
    case Maxwell::ShaderProgram::Fragment:
        return vk::ShaderStageFlagBits::eFragment;
    }
    LOG_CRITICAL(Render_Vulkan, "Unimplemented shader stage={}", static_cast<u32>(stage));
    UNREACHABLE();
    return {};
}

inline vk::PrimitiveTopology PrimitiveTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return vk::PrimitiveTopology::ePointList;
    case Maxwell::PrimitiveTopology::Lines:
        return vk::PrimitiveTopology::eLineList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return vk::PrimitiveTopology::eLineStrip;
    case Maxwell::PrimitiveTopology::Triangles:
        return vk::PrimitiveTopology::eTriangleList;
    case Maxwell::PrimitiveTopology::TriangleStrip:
        return vk::PrimitiveTopology::eTriangleStrip;
    }
    LOG_CRITICAL(Render_Vulkan, "Unimplemented topology={}", static_cast<u32>(topology));
    UNREACHABLE();
    return {};
}

} // namespace Vulkan::MaxwellToGL