// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vulkan/vulkan.hpp>
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

#pragma optimize("", off)

namespace Vulkan {

/// Gets the address for the specified shader stage program
static VAddr GetShaderAddress(Maxwell::ShaderProgram program) {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();
    const auto& shader_config = gpu.regs.shader_config[static_cast<std::size_t>(program)];
    return *gpu.memory_manager.GpuToCpuAddress(gpu.regs.code_address.CodeAddress() +
                                               shader_config.offset);
}

/// Gets the shader program code from memory for the specified address
static VKShader::ProgramCode GetShaderCode(VAddr addr) {
    VKShader::ProgramCode program_code(VKShader::MAX_PROGRAM_CODE_LENGTH);
    Memory::ReadBlock(addr, program_code.data(), program_code.size() * sizeof(u64));
    return program_code;
}

CachedShader::CachedShader(VulkanDevice& device_handler, VAddr addr,
                           Maxwell::ShaderProgram program_type)
    : addr{addr}, program_type{program_type}, setup{GetShaderCode(addr)} {

    VKShader::ProgramResult program_result = [&]() {
        switch (program_type) {
        case Maxwell::ShaderProgram::VertexA:
            // VertexB is always enabled, so when VertexA is enabled, we have two vertex shaders.
            // Conventional HW does not support this, so we combine VertexA and VertexB into one
            // stage here.
            setup.SetProgramB(GetShaderCode(GetShaderAddress(Maxwell::ShaderProgram::VertexB)));
        case Maxwell::ShaderProgram::VertexB:
            return VKShader::GenerateVertexShader(setup);
        case Maxwell::ShaderProgram::Fragment:
            return VKShader::GenerateFragmentShader(setup);
        default:
            LOG_CRITICAL(HW_GPU, "Unimplemented program_type={}", static_cast<u32>(program_type));
            UNREACHABLE();
        }
    }();

    const vk::ShaderModuleCreateInfo shader_module_ci(
        {}, program_result.code.size(), reinterpret_cast<const u32*>(program_result.code.data()));
    shader_module = device_handler.GetLogical().createShaderModuleUnique(shader_module_ci);
}

VulkanShaderCache::VulkanShaderCache(VulkanDevice& device_handler)
    : device_handler(device_handler) {}

Shader VulkanShaderCache::GetStageProgram(Maxwell::ShaderProgram program) {
    const VAddr program_addr{GetShaderAddress(program)};

    // Look up shader in the cache based on address
    Shader shader{TryGet(program_addr)};

    if (!shader) {
        // No shader found - create a new one
        shader = std::make_shared<CachedShader>(device_handler, program_addr, program);
        Register(shader);
    }

    return shader;
}

} // namespace Vulkan