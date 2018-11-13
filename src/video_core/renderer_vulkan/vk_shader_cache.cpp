// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstddef>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

#pragma optimize("", off)

namespace Vulkan {

// How many sets are created per descriptor pool.
static constexpr std::size_t SETS_PER_POOL = 0x400;

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

class CachedShader::DescriptorPool final : public VulkanFencedPool {
public:
    explicit DescriptorPool(vk::Device device,
                            const std::vector<vk::DescriptorPoolSize>& pool_sizes,
                            const vk::DescriptorSetLayout layout)
        : pool_ci(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, SETS_PER_POOL,
                  static_cast<u32>(stored_pool_sizes.size()), stored_pool_sizes.data()),
          stored_pool_sizes(pool_sizes), layout(layout), device(device) {

        InitResizable(SETS_PER_POOL, SETS_PER_POOL);
    }

    ~DescriptorPool() = default;

    vk::DescriptorSet Commit(VulkanFence& fence) {
        const std::size_t index = ResourceCommit(fence);
        const auto pool_index = index / SETS_PER_POOL;
        const auto set_index = index % SETS_PER_POOL;
        return allocations[pool_index][set_index].get();
    }

protected:
    void Allocate(std::size_t begin, std::size_t end) override {
        ASSERT_MSG(begin % SETS_PER_POOL == 0 && end % SETS_PER_POOL == 0, "Not aligned.");

        auto pool = device.createDescriptorPoolUnique(pool_ci);
        std::vector<vk::DescriptorSetLayout> layout_clones(SETS_PER_POOL, layout);

        const vk::DescriptorSetAllocateInfo descriptor_set_ai(*pool, SETS_PER_POOL,
                                                              layout_clones.data());

        pools.push_back(std::move(pool));
        allocations.push_back(device.allocateDescriptorSetsUnique(descriptor_set_ai));
    }

private:
    const vk::Device device;
    const std::vector<vk::DescriptorPoolSize> stored_pool_sizes;
    const vk::DescriptorPoolCreateInfo pool_ci;
    const vk::DescriptorSetLayout layout;

    std::vector<vk::UniqueDescriptorPool> pools;
    std::vector<std::vector<vk::UniqueDescriptorSet>> allocations;
};

CachedShader::CachedShader(VulkanDevice& device_handler, VAddr addr,
                           Maxwell::ShaderProgram program_type)
    : addr(addr),
      program_type{program_type}, setup{GetShaderCode(addr)}, device{device_handler.GetLogical()} {

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

    entries = program_result.entries;

    const vk::ShaderModuleCreateInfo shader_module_ci(
        {}, program_result.code.size(), reinterpret_cast<const u32*>(program_result.code.data()));
    shader_module = device.createShaderModuleUnique(shader_module_ci);

    CreateDescriptorSetLayout();
    CreateDescriptorPool();
}

vk::DescriptorSet CachedShader::CommitDescriptorSet(VulkanFence& fence) {
    if (descriptor_pool == nullptr) {
        // If the descriptor pool has not been initialized, it means that the shader doesn't used
        // descriptors. Return a null descriptor set.
        return nullptr;
    }
    return descriptor_pool->Commit(fence);
}

void CachedShader::CreateDescriptorSetLayout() {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (const auto& cbuf_entry : entries.const_buffers) {
        bindings.push_back({cbuf_entry.GetBinding(), vk::DescriptorType::eUniformBuffer, 1,
                            MaxwellToVK::ShaderStage(program_type), nullptr});
    }

    descriptor_set_layout = device.createDescriptorSetLayoutUnique(
        {{}, static_cast<u32>(bindings.size()), bindings.data()});
}

void CachedShader::CreateDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> pool_sizes;

    if (u32 used_ubos = static_cast<u32>(entries.const_buffers.size()); used_ubos > 0) {
        pool_sizes.push_back(
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, used_ubos * SETS_PER_POOL));
    }
    if (u32 used_attrs = static_cast<u32>(entries.attributes.size()); used_attrs > 0) {
        pool_sizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment,
                                                    used_attrs * SETS_PER_POOL));
    }

    if (pool_sizes.size() == 0) {
        // If the shader doesn't use descriptor sets, skip the pool creation.
        return;
    }

    descriptor_pool = std::make_unique<DescriptorPool>(device, pool_sizes, *descriptor_set_layout);
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