// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <vector>
#include <vulkan/vulkan.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

vk::UniqueShaderModule BuildShader(vk::Device device, std::size_t code_size, const u8* code_data) {
    const vk::ShaderModuleCreateInfo shader_ci({}, code_size,
                                               reinterpret_cast<const u32*>(code_data));
    vk::ShaderModule shader_module;
    if (device.createShaderModule(&shader_ci, nullptr, &shader_module) != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Shader module failed to build!");
        UNREACHABLE();
    }
    return vk::UniqueShaderModule(shader_module, device);
}

} // namespace Vulkan