// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/surface.h"

namespace Vulkan::MaxwellToVK {

struct FormatTuple {
    vk::Format format;
    ComponentType component_type;
};

static constexpr std::array<FormatTuple, VideoCore::Surface::MaxPixelFormat> tex_format_tuples = {{
    {vk::Format::eA8B8G8R8UnormPack32, ComponentType::UNorm}, // ABGR8U
    {vk::Format::eUndefined, ComponentType::Invalid},         // ABGR8S
    {vk::Format::eUndefined, ComponentType::Invalid},         // ABGR8UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // B5G6R5U
    {vk::Format::eUndefined, ComponentType::Invalid},         // A2B10G10R10U
    {vk::Format::eUndefined, ComponentType::Invalid},         // A1B5G5R5U
    {vk::Format::eUndefined, ComponentType::Invalid},         // R8U
    {vk::Format::eUndefined, ComponentType::Invalid},         // R8UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA16F
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA16U
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA16UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // R11FG11FB10F
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA32UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXT1
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXT23
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXT45
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXN1
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXN2UNORM
    {vk::Format::eUndefined, ComponentType::Invalid},         // DXN2SNORM
    {vk::Format::eUndefined, ComponentType::Invalid},         // BC7U
    {vk::Format::eUndefined, ComponentType::Invalid},         // BC6H_UF16
    {vk::Format::eUndefined, ComponentType::Invalid},         // BC6H_SF16
    {vk::Format::eUndefined, ComponentType::Invalid},         // ASTC_2D_4X4
    {vk::Format::eUndefined, ComponentType::Invalid},         // G8R8U
    {vk::Format::eUndefined, ComponentType::Invalid},         // G8R8S
    {vk::Format::eUndefined, ComponentType::Invalid},         // BGRA8
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA32F
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG32F
    {vk::Format::eUndefined, ComponentType::Invalid},         // R32F
    {vk::Format::eUndefined, ComponentType::Invalid},         // R16F
    {vk::Format::eUndefined, ComponentType::Invalid},         // R16U
    {vk::Format::eUndefined, ComponentType::Invalid},         // R16S
    {vk::Format::eUndefined, ComponentType::Invalid},         // R16UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // R16I
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG16
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG16F
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG16UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG16I
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG16S
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGB32F
    {vk::Format::eUndefined, ComponentType::Invalid},         // RGBA8_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG8U
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG8S
    {vk::Format::eUndefined, ComponentType::Invalid},         // RG32UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // R32UI
    {vk::Format::eUndefined, ComponentType::Invalid},         // ASTC_2D_8X8
    {vk::Format::eUndefined, ComponentType::Invalid},         // ASTC_2D_8X5
    {vk::Format::eUndefined, ComponentType::Invalid},         // ASTC_2D_5X4

    // Compressed sRGB formats
    {vk::Format::eUndefined, ComponentType::Invalid}, // BGRA8_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // DXT1_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // DXT23_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // DXT45_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // BC7U_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_4X4_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_8X8_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_8X5_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_5X4_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_5X5
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_5X5_SRGB
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_10X8
    {vk::Format::eUndefined, ComponentType::Invalid}, // ASTC_2D_10X8_SRGB

    // Depth formats
    {vk::Format::eUndefined, ComponentType::Invalid}, // Z32F
    {vk::Format::eUndefined, ComponentType::Invalid}, // Z16

    // DepthStencil formats
    {vk::Format::eD24UnormS8Uint, ComponentType::UNorm}, // Z24S8
    {vk::Format::eUndefined, ComponentType::Invalid},    // S8Z24
    {vk::Format::eUndefined, ComponentType::Invalid},    // Z32FS8
}};

vk::Format SurfaceFormat(PixelFormat pixel_format, ComponentType component_type) {
    ASSERT(static_cast<std::size_t>(pixel_format) < tex_format_tuples.size());

    const auto& format = tex_format_tuples[static_cast<u32>(pixel_format)];
    UNIMPLEMENTED_IF_MSG(format.format == vk::Format::eUndefined,
                         "Unimplemented texture format with pixel format={} and component type={}",
                         static_cast<u32>(pixel_format), static_cast<u32>(component_type));
    ASSERT_MSG(component_type == format.component_type, "Component type mismatch");

    return format.format;
}

vk::ShaderStageFlagBits ShaderStage(Maxwell::ShaderStage stage) {
    switch (stage) {
    case Maxwell::ShaderStage::Vertex:
        return vk::ShaderStageFlagBits::eVertex;
    case Maxwell::ShaderStage::Fragment:
        return vk::ShaderStageFlagBits::eFragment;
    }
    UNIMPLEMENTED_MSG("Unimplemented shader stage={}", static_cast<u32>(stage));
    return {};
}

vk::PrimitiveTopology PrimitiveTopology(Maxwell::PrimitiveTopology topology) {
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
    UNIMPLEMENTED_MSG("Unimplemented topology={}", static_cast<u32>(topology));
    return {};
}

vk::Format VertexFormat(Maxwell::VertexAttribute::Type type, Maxwell::VertexAttribute::Size size) {
    switch (type) {
    case Maxwell::VertexAttribute::Type::SignedNorm:
    case Maxwell::VertexAttribute::Type::UnsignedNorm:
    case Maxwell::VertexAttribute::Type::SignedInt:
    case Maxwell::VertexAttribute::Type::UnsignedInt:
    case Maxwell::VertexAttribute::Type::UnsignedScaled:
    case Maxwell::VertexAttribute::Type::SignedScaled:
        break;
    case Maxwell::VertexAttribute::Type::Float:
        switch (size) {
        case Maxwell::VertexAttribute::Size::Size_32_32_32_32:
            return vk::Format::eR32G32B32A32Sfloat;
        case Maxwell::VertexAttribute::Size::Size_32_32_32:
            return vk::Format::eR32G32B32Sfloat;
        case Maxwell::VertexAttribute::Size::Size_32_32:
            return vk::Format::eR32G32Sfloat;
        }
        break;
    }
    UNIMPLEMENTED_MSG("Unimplemented vertex format of type={} and size={}", static_cast<u32>(type),
                      static_cast<u32>(size));
    return vk::Format::eR8Unorm;
}

vk::CompareOp ComparisonOp(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never:
    case Maxwell::ComparisonOp::NeverOld:
        return vk::CompareOp::eNever;
    case Maxwell::ComparisonOp::Less:
    case Maxwell::ComparisonOp::LessOld:
        return vk::CompareOp::eLess;
    case Maxwell::ComparisonOp::Equal:
    case Maxwell::ComparisonOp::EqualOld:
        return vk::CompareOp::eEqual;
    case Maxwell::ComparisonOp::LessEqual:
    case Maxwell::ComparisonOp::LessEqualOld:
        return vk::CompareOp::eLessOrEqual;
    case Maxwell::ComparisonOp::Greater:
    case Maxwell::ComparisonOp::GreaterOld:
        return vk::CompareOp::eGreater;
    case Maxwell::ComparisonOp::NotEqual:
    case Maxwell::ComparisonOp::NotEqualOld:
        return vk::CompareOp::eNotEqual;
    case Maxwell::ComparisonOp::GreaterEqual:
    case Maxwell::ComparisonOp::GreaterEqualOld:
        return vk::CompareOp::eGreaterOrEqual;
    case Maxwell::ComparisonOp::Always:
    case Maxwell::ComparisonOp::AlwaysOld:
        return vk::CompareOp::eAlways;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", static_cast<u32>(comparison));
    return vk::CompareOp::eAlways;
}

} // namespace Vulkan::MaxwellToVK