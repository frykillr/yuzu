// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "common/math_util.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_sync.h"
#include "video_core/utils.h"

namespace Vulkan {

static const u8 blit_vertex_code[] = {
    0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x07, 0x00, 0x08, 0x00, 0x27, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
    0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
    0x25, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x04, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x19, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x24, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x09, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x06, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x00, 0x04, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
    0x13, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x19, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x20, 0x00, 0x04, 0x00, 0x21, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00,
    0x24, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x25, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
    0x13, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x16, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
    0x1a, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x1d, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x50, 0x00, 0x07, 0x00, 0x07, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
    0x1e, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x91, 0x00, 0x05, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
    0x41, 0x00, 0x05, 0x00, 0x21, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x22, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x3d, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00,
    0x3e, 0x00, 0x03, 0x00, 0x24, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00,
    0x38, 0x00, 0x01, 0x00};

static const u8 blit_fragment_code[] = {
    0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x07, 0x00, 0x08, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
    0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x07, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e,
    0x00, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x10, 0x00, 0x03, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x09, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x03, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x0d, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x57, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x13, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00};

struct ScreenRectVertex {
    ScreenRectVertex(f32 x, f32 y, f32 u, f32 v) {
        position = {x, y};
        tex_coord = {u, v};
    }

    std::array<f32, 2> position;
    std::array<f32, 2> tex_coord;

    static vk::VertexInputBindingDescription GetDescription() {
        return vk::VertexInputBindingDescription(0, sizeof(ScreenRectVertex),
                                                 vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 2> GetAttributes() {
        return {vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat,
                                                    offsetof(ScreenRectVertex, position)),
                vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat,
                                                    offsetof(ScreenRectVertex, tex_coord))};
    }
};

struct ScreenUniformData {
    std::array<f32, 4 * 4> matrix;
};

static std::array<f32, 4 * 4> MakeOrthographicMatrix(const f32 width, const f32 height) {
    // clang-format off
    return { 2.f / width, 0.f,          0.f, 0.f,
             0.f,        -2.f / height, 0.f, 0.f,
             0.f,         0.f,          1.f, 0.f,
            -1.f,         1.f,          0.f, 1.f};
    // clang-format on
}

VulkanBlitScreen::VulkanBlitScreen(VideoCore::RasterizerInterface& rasterizer,
                                   Core::Frontend::EmuWindow& render_window,
                                   VulkanDevice& device_handler,
                                   VulkanResourceManager& resource_manager,
                                   VulkanMemoryManager& memory_manager, VulkanSwapchain& swapchain)
    : rasterizer(rasterizer), render_window(render_window), device(device_handler.GetLogical()),
      resource_manager(resource_manager), memory_manager(memory_manager), swapchain(swapchain),
      image_count(swapchain.GetImageCount()) {

    watches.resize(image_count);
    std::generate(watches.begin(), watches.end(),
                  []() { return std::make_unique<VulkanFenceWatch>(); });

    CreateShaders();
    CreateDescriptorPool();
    CreateRenderPass();
    CreateDescriptorSetLayout();
    CreateDescriptorSets();
    CreatePipelineLayout();
    CreateGraphicsPipeline();

    Recreate();
}

VulkanBlitScreen::~VulkanBlitScreen() = default;

void VulkanBlitScreen::Recreate() {
    CreateFramebuffers();
}

VulkanFence& VulkanBlitScreen::Draw(VulkanSync& sync, const Tegra::FramebufferConfig& framebuffer) {
    const VAddr framebuffer_addr{framebuffer.address + framebuffer.offset};
    if (rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr, framebuffer.stride)) {
        UNREACHABLE();
    }
    const u32 bytes_per_pixel{Tegra::FramebufferConfig::BytesPerPixel(framebuffer.pixel_format)};
    const u64 size_in_bytes{framebuffer.stride * framebuffer.height * bytes_per_pixel};
    const u32 image_index = swapchain.GetImageIndex();
    const vk::Extent2D& framebuffer_size{swapchain.GetSize()};

    RefreshRawImages(framebuffer);
    UpdateDescriptorSet(image_index);

    const auto& layout = render_window.GetFramebufferLayout();
    const auto& screen = layout.screen;

    u8* data = buffer_commit->GetData();

    auto* uniform_data = reinterpret_cast<ScreenUniformData*>(data + GetUniformDataOffset());
    uniform_data->matrix =
        MakeOrthographicMatrix(static_cast<f32>(layout.width), static_cast<f32>(layout.height));

    SetVertexData(framebuffer);

    Memory::RasterizerFlushVirtualRegion(framebuffer_addr, size_in_bytes, Memory::FlushMode::Flush);

    const u64 image_offset = GetRawImageOffset(framebuffer, image_index);
    VideoCore::MortonCopyPixels128(framebuffer.width, framebuffer.height, bytes_per_pixel, 4,
                                   Memory::GetPointer(framebuffer_addr), data + image_offset, true);

    VulkanFence& fence = sync.PrepareExecute(false);
    watches[image_index]->Watch(fence);

    // Record blitting.
    vk::CommandBuffer cmdbuf{sync.BeginRecord()};

    const vk::Image raw_image = *raw_images[image_index];
    SetImageLayout(cmdbuf, raw_image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eFragmentShader,
                   vk::PipelineStageFlagBits::eTransfer);

    const vk::BufferImageCopy copy(image_offset, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                   {0, 0, 0}, {framebuffer.width, framebuffer.height, 1});
    cmdbuf.copyBufferToImage(*buffer, raw_image, vk::ImageLayout::eTransferDstOptimal, {copy});

    SetImageLayout(cmdbuf, raw_image, vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                   vk::PipelineStageFlagBits::eTransfer,
                   vk::PipelineStageFlagBits::eFragmentShader);

    const vk::Extent2D size = swapchain.GetSize();
    const vk::ClearValue clear_color{std::array<f32, 4>{1.0f, 0.0f, 0.0f, 1.0f}};
    const vk::RenderPassBeginInfo renderpass_bi(*renderpass, *framebuffers[image_index],
                                                {{0, 0}, size}, 1, &clear_color);

    cmdbuf.setViewport(
        0, {{0.0f, 0.0f, static_cast<f32>(size.width), static_cast<f32>(size.height), 0.0f, 1.0f}});
    cmdbuf.setScissor(0, {{{0, 0}, size}});

    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline);
    {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cmdbuf.bindVertexBuffers(0, {*buffer}, {GetVertexDataOffset()});
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0,
                                  {descriptor_sets[image_index]}, {});
        cmdbuf.draw(4, 1, 0, 0);
    }
    cmdbuf.endRenderPass();

    sync.EndRecord(cmdbuf);
    sync.Execute();

    return fence;
}

void VulkanBlitScreen::CreateShaders() {
    vertex_shader = BuildShader(device, sizeof(blit_vertex_code), blit_vertex_code);
    fragment_shader = BuildShader(device, sizeof(blit_fragment_code), blit_fragment_code);
}

void VulkanBlitScreen::CreateDescriptorPool() {
    const std::array<vk::DescriptorPoolSize, 2> pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, image_count},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, image_count}};
    const vk::DescriptorPoolCreateInfo pool_ci({}, image_count, static_cast<u32>(pool_sizes.size()),
                                               pool_sizes.data());
    descriptor_pool = device.createDescriptorPoolUnique(pool_ci);
}

void VulkanBlitScreen::CreateRenderPass() {
    const vk::AttachmentDescription color_attachment(
        {}, swapchain.GetImageFormat(), vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
        vk::ImageLayout::ePresentSrcKHR);

    const vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);

    const vk::SubpassDescription subpass_description({}, vk::PipelineBindPoint::eGraphics, 0,
                                                     nullptr, 1, &color_attachment_ref, nullptr,
                                                     nullptr, 0, nullptr);

    const vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, {},
        vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, {});

    const vk::RenderPassCreateInfo renderpass_ci({}, 1, &color_attachment, 1, &subpass_description,
                                                 1, &dependency);
    renderpass = device.createRenderPassUnique(renderpass_ci);
}

void VulkanBlitScreen::CreateDescriptorSetLayout() {
    const std::array<vk::DescriptorSetLayoutBinding, 2> layout_bindings{
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eVertex, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment, nullptr)};

    const vk::DescriptorSetLayoutCreateInfo descriptor_layout_ci(
        {}, static_cast<u32>(layout_bindings.size()), layout_bindings.data());
    descriptor_set_layout = device.createDescriptorSetLayoutUnique(descriptor_layout_ci);
}

void VulkanBlitScreen::CreateDescriptorSets() {
    descriptor_sets.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        const vk::DescriptorSetLayout layout = *descriptor_set_layout;
        const vk::DescriptorSetAllocateInfo descriptor_set_ai(*descriptor_pool, 1, &layout);
        const auto sets = device.allocateDescriptorSets(descriptor_set_ai);
        descriptor_sets[i] = sets[0];
    }
}

void VulkanBlitScreen::CreatePipelineLayout() {
    const vk::PipelineLayoutCreateInfo pipeline_layout_ci({}, 1, &descriptor_set_layout.get(), 0,
                                                          nullptr);
    pipeline_layout = device.createPipelineLayoutUnique(pipeline_layout_ci);
}

void VulkanBlitScreen::CreateGraphicsPipeline() {
    const std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertex_shader,
                                          "main", nullptr),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragment_shader,
                                          "main", nullptr)};

    const auto vertex_binding_description = ScreenRectVertex::GetDescription();
    const auto vertex_attrs_description = ScreenRectVertex::GetAttributes();
    const vk::PipelineVertexInputStateCreateInfo vertex_input(
        {}, 1, &vertex_binding_description, static_cast<u32>(vertex_attrs_description.size()),
        vertex_attrs_description.data());

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly(
        {}, vk::PrimitiveTopology::eTriangleStrip, false);

    // Set a dummy viewport, it's going to be replaced by dynamic states.
    const vk::Viewport viewport(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f);
    const vk::Rect2D scissor({0, 0}, {1, 1});

    const vk::PipelineViewportStateCreateInfo viewport_state({}, 1, &viewport, 1, &scissor);

    const vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
        vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

    const vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1,
                                                               false, 0.0f, nullptr, false, false);

    const vk::PipelineColorBlendAttachmentState color_blend_attachment(
        false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
        vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    const vk::PipelineColorBlendStateCreateInfo color_blending(
        {}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment, {0.0f, 0.0f, 0.0f, 0.0f});

    const std::array<vk::DynamicState, 2> dynamic_states = {vk::DynamicState::eViewport,
                                                            vk::DynamicState::eScissor};

    const vk::PipelineDynamicStateCreateInfo dynamic_state(
        {}, static_cast<u32>(dynamic_states.size()), dynamic_states.data());

    const vk::GraphicsPipelineCreateInfo pipeline_ci(
        {}, static_cast<u32>(shader_stages.size()), shader_stages.data(), &vertex_input,
        &input_assembly, nullptr, &viewport_state, &rasterizer, &multisampling, nullptr,
        &color_blending, &dynamic_state, *pipeline_layout, *renderpass, 0, nullptr, 0);

    pipeline = device.createGraphicsPipelineUnique({}, pipeline_ci);
}

void VulkanBlitScreen::CreateFramebuffers() {
    const vk::Extent2D size{swapchain.GetSize()};
    framebuffers.clear();
    framebuffers.resize(image_count);

    for (u32 i = 0; i < image_count; ++i) {
        const vk::FramebufferCreateInfo framebuffer_ci(
            {}, *renderpass, 1, &swapchain.GetImageViewIndex(i), size.width, size.height, 1);
        framebuffers[i] = device.createFramebufferUnique(framebuffer_ci);
    }
}

void VulkanBlitScreen::UpdateDescriptorSet(u32 image_index) {
    const vk::DescriptorSet descriptor_set = descriptor_sets[image_index];

    const vk::DescriptorBufferInfo buffer_info(*buffer, GetUniformDataOffset(),
                                               sizeof(ScreenUniformData));
    const vk::WriteDescriptorSet ubo_write(descriptor_set, 0, 0, 1,
                                           vk::DescriptorType::eUniformBuffer, nullptr,
                                           &buffer_info, nullptr);

    const vk::DescriptorImageInfo image_info(*raw_samplers[image_index],
                                             *raw_image_views[image_index],
                                             vk::ImageLayout::eShaderReadOnlyOptimal);
    const vk::WriteDescriptorSet sampler_write(descriptor_set, 1, 0, 1,
                                               vk::DescriptorType::eCombinedImageSampler,
                                               &image_info, nullptr, nullptr);

    device.updateDescriptorSets({ubo_write, sampler_write}, {});
}

void VulkanBlitScreen::RefreshRawImages(const Tegra::FramebufferConfig& framebuffer) {
    if (framebuffer.width == raw_width && framebuffer.height == raw_height && !raw_images.empty()) {
        return;
    }
    raw_width = framebuffer.width;
    raw_height = framebuffer.height;

    for (u32 i = 0; i < static_cast<u32>(raw_images.size()); ++i) {
        watches[i]->Wait();
        raw_images[i].reset();
        memory_manager.Free(raw_buffer_commits[i]);
    }
    buffer.reset();
    memory_manager.Free(buffer_commit);

    const u32 bytes_per_pixel{Tegra::FramebufferConfig::BytesPerPixel(framebuffer.pixel_format)};
    const u64 size_in_bytes{framebuffer.stride * framebuffer.height * bytes_per_pixel};
    const u64 buffer_size = CalculateBufferSize(framebuffer);

    const vk::BufferCreateInfo buffer_ci({}, buffer_size,
                                         vk::BufferUsageFlagBits::eTransferSrc |
                                             vk::BufferUsageFlagBits::eVertexBuffer |
                                             vk::BufferUsageFlagBits::eUniformBuffer,
                                         vk::SharingMode::eExclusive, 0, nullptr);
    buffer = device.createBufferUnique(buffer_ci);
    buffer_commit = memory_manager.Commit(device.getBufferMemoryRequirements(*buffer), true);
    device.bindBufferMemory(*buffer, buffer_commit->GetMemory(), buffer_commit->GetOffset());

    raw_images.resize(image_count);
    raw_buffer_commits.resize(image_count);
    raw_image_views.resize(image_count);
    raw_samplers.resize(image_count);

    for (u32 i = 0; i < image_count; ++i) {
        const vk::ImageCreateInfo image_ci(
            {}, vk::ImageType::e2D, vk::Format::eA8B8G8R8UnormPack32,
            {framebuffer.width, framebuffer.height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eLinear,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::eUndefined);
        raw_images[i] = device.createImageUnique(image_ci);
        const vk::Image image = *raw_images[i];

        raw_buffer_commits[i] =
            memory_manager.Commit(device.getImageMemoryRequirements(image), false);

        device.bindImageMemory(image, raw_buffer_commits[i]->GetMemory(),
                               raw_buffer_commits[i]->GetOffset());

        const vk::ImageViewCreateInfo image_view_ci({}, image, vk::ImageViewType::e2D,
                                                    vk::Format::eA8B8G8R8UnormPack32, {},
                                                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        raw_image_views[i] = device.createImageViewUnique(image_view_ci);

        const vk::SamplerCreateInfo sampler_ci(
            {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eClampToBorder, vk::SamplerAddressMode::eClampToBorder,
            vk::SamplerAddressMode::eClampToBorder, 0.0f, false, 0.0f, false, vk::CompareOp::eNever,
            0.0f, 0.0f, vk::BorderColor::eFloatOpaqueBlack, false);
        raw_samplers[i] = device.createSamplerUnique(sampler_ci);
    }
}

void VulkanBlitScreen::SetVertexData(const Tegra::FramebufferConfig& framebuffer) {
    const auto& framebuffer_transform_flags = framebuffer.transform_flags;
    const auto& framebuffer_crop_rect = framebuffer.crop_rect;

    const MathUtil::Rectangle<f32> texcoords{0.f, 0.f, 1.f, 1.f};
    const auto left = texcoords.left;
    const auto right = texcoords.right;

    ASSERT_MSG(framebuffer_crop_rect.top == 0, "Unimplemented");
    ASSERT_MSG(framebuffer_crop_rect.left == 0, "Unimplemented");

    // FIXME(Rodrigo): Change raw_width and raw_height with screen_info.texture.width

    // Scale the output by the crop width/height. This is commonly used with 1280x720 rendering
    // (e.g. handheld mode) on a 1920x1080 framebuffer.
    f32 scale_u = 1.f, scale_v = 1.f;
    if (framebuffer_crop_rect.GetWidth() > 0) {
        scale_u = static_cast<f32>(framebuffer_crop_rect.GetWidth()) / raw_width;
    }
    if (framebuffer_crop_rect.GetHeight() > 0) {
        scale_v = static_cast<f32>(framebuffer_crop_rect.GetHeight()) / raw_height;
    }

    const auto& screen = render_window.GetFramebufferLayout().screen;
    const auto x = static_cast<f32>(screen.left);
    const auto y = static_cast<f32>(screen.top);
    const auto w = static_cast<f32>(screen.GetWidth());
    const auto h = static_cast<f32>(screen.GetHeight());

    u8* data = buffer_commit->GetData();
    auto* vertex_data = reinterpret_cast<ScreenRectVertex*>(data + GetVertexDataOffset());

    vertex_data[0] = ScreenRectVertex(x, y, texcoords.top * scale_u, left * scale_v);
    vertex_data[1] = ScreenRectVertex(x + w, y, texcoords.bottom * scale_u, left * scale_v);
    vertex_data[2] = ScreenRectVertex(x, y + h, texcoords.top * scale_u, right * scale_v);
    vertex_data[3] = ScreenRectVertex(x + w, y + h, texcoords.bottom * scale_u, right * scale_v);
}

u64 VulkanBlitScreen::CalculateBufferSize(const Tegra::FramebufferConfig& framebuffer) const {
    const u32 bytes_per_pixel{Tegra::FramebufferConfig::BytesPerPixel(framebuffer.pixel_format)};
    const u64 size_in_bytes{framebuffer.stride * framebuffer.height * bytes_per_pixel};

    return sizeof(ScreenUniformData) + sizeof(ScreenRectVertex) * 4 + size_in_bytes * image_count;
}

u64 VulkanBlitScreen::GetUniformDataOffset() const {
    return 0;
}

u64 VulkanBlitScreen::GetVertexDataOffset() const {
    return GetUniformDataOffset() + sizeof(ScreenUniformData);
}

u64 VulkanBlitScreen::GetRawImageOffset(const Tegra::FramebufferConfig& framebuffer,
                                        u32 image_index) const {
    const u32 bytes_per_pixel{Tegra::FramebufferConfig::BytesPerPixel(framebuffer.pixel_format)};
    const u64 size_in_bytes{framebuffer.stride * framebuffer.height * bytes_per_pixel};

    return GetVertexDataOffset() + sizeof(ScreenRectVertex) * 4 + size_in_bytes * image_index;
}

} // namespace Vulkan