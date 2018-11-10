// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;

class CachedShader;
using Shader = std::shared_ptr<CachedShader>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

class CachedShader final : public RasterizerCacheObject {
public:
    CachedShader(VulkanDevice& device_handler, VAddr addr, Maxwell::ShaderProgram program_type);

    /// Gets a descriptor set from the internal pool.
    vk::DescriptorSet CommitDescriptorSet(VulkanFence& fence);

    VAddr GetAddr() const override {
        return addr;
    }

    std::size_t GetSizeInBytes() const override {
        return VKShader::MAX_PROGRAM_CODE_LENGTH * sizeof(u64);
    }

    // We do not have to flush this cache as things in it are never modified by us.
    void Flush() override {}

    /// Gets the module handle for the shader.
    vk::ShaderModule GetHandle(vk::PrimitiveTopology primitive_mode) {
        return *shader_module;
    }

    /// Gets the descriptor set layout of the shader.
    vk::DescriptorSetLayout GetSetLayout() const {
        return *descriptor_set_layout;
    }

    /// Gets the module entries for the shader.
    const VKShader::ShaderEntries& GetEntries() const {
        return entries;
    }

private:
    void CreateDescriptorSetLayout();
    void CreateDescriptorPool();

    const VAddr addr;
    const Maxwell::ShaderProgram program_type;
    const vk::Device device;

    VKShader::ShaderSetup setup;
    VKShader::ShaderEntries entries;

    vk::UniqueShaderModule shader_module;
    vk::UniqueDescriptorSetLayout descriptor_set_layout;
    vk::UniqueDescriptorPool descriptor_pool;
    vk::UniqueDescriptorSet descriptor_set;
};

class VulkanShaderCache final : public RasterizerCache<Shader> {
public:
    explicit VulkanShaderCache(VulkanDevice& device_handler);

    /// Gets the current specified shader stage program
    Shader GetStageProgram(Maxwell::ShaderProgram program);

private:
    VulkanDevice& device_handler;
};

} // namespace Vulkan