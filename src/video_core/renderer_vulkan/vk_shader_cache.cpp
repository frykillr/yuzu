// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstddef>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/static_vector.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

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

static std::size_t GetStageFromProgram(std::size_t program) {
    return program == 0 ? 0 : program - 1;
}

static Maxwell::ShaderStage GetStageFromProgram(Maxwell::ShaderProgram program) {
    return static_cast<Maxwell::ShaderStage>(
        GetStageFromProgram(static_cast<std::size_t>(program)));
}

/// Gets the shader program code from memory for the specified address
static VKShader::ProgramCode GetShaderCode(VAddr addr) {
    VKShader::ProgramCode program_code(VKShader::MAX_PROGRAM_CODE_LENGTH);
    Memory::ReadBlock(addr, program_code.data(), program_code.size() * sizeof(u64));
    return program_code;
}

class CachedShader::DescriptorPool final : public VKFencedPool {
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

    vk::DescriptorSet Commit(VKFence& fence) {
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

CachedShader::CachedShader(VKDevice& device_handler, VAddr addr,
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

vk::DescriptorSet CachedShader::CommitDescriptorSet(VKFence& fence) {
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
                            MaxwellToVK::ShaderStage(GetStageFromProgram(program_type)), nullptr});
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

VKShaderCache::VKShaderCache(RasterizerVulkan& rasterizer, VKDevice& device_handler)
    : RasterizerCache{rasterizer},
      device_handler{device_handler}, device{device_handler.GetLogical()} {}

Pipeline VKShaderCache::GetPipeline(const PipelineParams& params) {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();

    Pipeline pipeline;
    ShaderPipeline shaders{};

    for (std::size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const auto& shader_config = gpu.regs.shader_config[index];
        const auto program{static_cast<Maxwell::ShaderProgram>(index)};

        // Skip stages that are not enabled
        if (!gpu.regs.IsShaderConfigEnabled(index)) {
            continue;
        }

        const VAddr program_addr{GetShaderAddress(program)};
        shaders[index] = program_addr;

        // Look up shader in the cache based on address
        Shader shader{TryGet(program_addr)};

        if (!shader) {
            // No shader found - create a new one
            shader = std::make_shared<CachedShader>(device_handler, program_addr, program);
            Register(shader);
        }

        const std::size_t stage = index == 0 ? 0 : index - 1;
        pipeline.shaders[stage] = std::move(shader);

        // When VertexA is enabled, we have dual vertex shaders
        if (program == Maxwell::ShaderProgram::VertexA) {
            // VertexB was combined with VertexA, so we skip the VertexB iteration
            index++;
        }
    }

    const auto [pair, is_cache_miss] = cache.try_emplace({shaders, params});
    auto& entry = pair->second;

    if (is_cache_miss) {
        entry = std::make_unique<CacheEntry>();
        entry->renderpass = CreateRenderPass(params);
        pipeline.renderpass = *entry->renderpass;

        entry->layout = CreatePipelineLayout(params, pipeline);
        pipeline.layout = *entry->layout;

        entry->pipeline = CreatePipeline(params, pipeline);
    }

    pipeline.handle = *entry->pipeline;
    pipeline.layout = *entry->layout;
    pipeline.renderpass = *entry->renderpass;
    return pipeline;
}

void VKShaderCache::ObjectInvalidated(const Shader& shader) {
    const VAddr invalidated_addr = shader->GetAddr();
    for (auto it = cache.begin(); it != cache.end();) {
        auto& entry = it->first;
        const bool has_addr = [&]() {
            const auto [shaders, params] = entry;
            for (auto& shader_addr : shaders) {
                if (shader_addr == invalidated_addr) {
                    return true;
                }
            }
            return false;
        }();
        if (has_addr) {
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

vk::UniquePipelineLayout VKShaderCache::CreatePipelineLayout(const PipelineParams& params,
                                                             const Pipeline& pipeline) const {
    StaticVector<Maxwell::MaxShaderStage, vk::DescriptorSetLayout> set_layouts;
    for (auto& shader : pipeline.shaders) {
        if (shader != nullptr) {
            set_layouts.Push(shader->GetDescriptorSetLayout());
        }
    }

    return device.createPipelineLayoutUnique(
        {{}, static_cast<u32>(set_layouts.Size()), set_layouts.data(), 0, nullptr});
}

vk::UniquePipeline VKShaderCache::CreatePipeline(const PipelineParams& params,
                                                 const Pipeline& pipeline) const {
    const auto& vertex_input = params.vertex_input;
    const auto& input_assembly = params.input_assembly;
    const auto& depth_stencil = params.depth_stencil;

    StaticVector<Maxwell::NumVertexArrays, vk::VertexInputBindingDescription> vertex_bindings;
    for (const auto& binding : params.vertex_input.bindings) {
        ASSERT(binding.divisor == 0);
        vertex_bindings.Push(vk::VertexInputBindingDescription(binding.index, binding.stride));
    }

    StaticVector<Maxwell::NumVertexArrays, vk::VertexInputAttributeDescription> vertex_attributes;
    for (const auto& attribute : params.vertex_input.attributes) {
        vertex_attributes.Push(vk::VertexInputAttributeDescription(
            attribute.index, attribute.buffer,
            MaxwellToVK::VertexFormat(attribute.type, attribute.size), attribute.offset));
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_ci(
        {}, static_cast<u32>(vertex_bindings.Size()), vertex_bindings.data(),
        static_cast<u32>(vertex_attributes.Size()), vertex_attributes.data());

    const vk::PrimitiveTopology primitive_topology =
        MaxwellToVK::PrimitiveTopology(input_assembly.topology);
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci(
        {}, primitive_topology, input_assembly.primitive_restart_enable);

    const vk::Viewport viewport(0.f, 0.f, 1280.f, 720.f, 0.f, 1.f);
    const vk::Rect2D scissor({0, 0}, {1280, 720});
    const vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, &viewport, 1, &scissor);

    const vk::PipelineRasterizationStateCreateInfo rasterizer_ci(
        {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
        vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

    const vk::PipelineMultisampleStateCreateInfo multisampling_ci(
        {}, vk::SampleCountFlagBits::e1, false, 0.0f, nullptr, false, false);

    const vk::CompareOp depth_test_compare =
        depth_stencil.depth_test_enable
            ? MaxwellToVK::ComparisonOp(depth_stencil.depth_test_function)
            : vk::CompareOp::eAlways;
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil_ci(
        {}, depth_stencil.depth_test_enable, depth_stencil.depth_write_enable, depth_test_compare,
        0, false, {}, {}, 0.f, 0.f);

    const vk::PipelineColorBlendAttachmentState color_blend_attachment(
        false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
        vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    const vk::PipelineColorBlendStateCreateInfo color_blending_ci(
        {}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment, {0.f, 0.f, 0.f, 0.f});

    StaticVector<Maxwell::MaxShaderStage, vk::PipelineShaderStageCreateInfo> shader_stages;
    for (std::size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        const auto& shader = pipeline.shaders[stage];
        if (shader == nullptr)
            continue;

        shader_stages.Push(vk::PipelineShaderStageCreateInfo(
            {}, MaxwellToVK::ShaderStage(static_cast<Maxwell::ShaderStage>(stage)),
            shader->GetHandle(primitive_topology), "main", nullptr));
    }

    const vk::GraphicsPipelineCreateInfo create_info(
        {}, static_cast<u32>(shader_stages.Size()), shader_stages.data(), &vertex_input_ci,
        &input_assembly_ci, nullptr, &viewport_state_ci, &rasterizer_ci, &multisampling_ci,
        &depth_stencil_ci, &color_blending_ci, nullptr, pipeline.layout, pipeline.renderpass, 0);
    return device.createGraphicsPipelineUnique(nullptr, create_info);
}

vk::UniqueRenderPass VKShaderCache::CreateRenderPass(const PipelineParams& params) const {
    const auto& p = params.renderpass;
    const bool preserve_contents = p.preserve_contents;
    ASSERT(p.color_map.Size() == 1);

    const vk::AttachmentLoadOp load_op =
        preserve_contents ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;

    StaticVector<Maxwell::NumRenderTargets + 1, vk::AttachmentDescription> descrs;
    const auto& first_map = p.color_map.data()[0];

    descrs.Push(vk::AttachmentDescription(
        {}, MaxwellToVK::SurfaceFormat(first_map.pixel_format, first_map.component_type),
        vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal));

    if (p.has_zeta) {
        descrs.Push(vk::AttachmentDescription(
            {}, MaxwellToVK::SurfaceFormat(p.zeta_pixel_format, p.zeta_component_type),
            vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore, load_op,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal));
    }

    // TODO(Rodrigo): Support multiple attachments
    const vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::AttachmentReference zeta_attachment_ref(
        1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    const vk::SubpassDescription subpass_description(
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr,
        p.has_zeta ? &zeta_attachment_ref : nullptr, 0, nullptr);

    vk::AccessFlags access{};
    vk::PipelineStageFlags stage{};
    if (preserve_contents)
        access |= vk::AccessFlagBits::eColorAttachmentRead;
    access |= vk::AccessFlagBits::eColorAttachmentWrite;
    stage |= vk::PipelineStageFlagBits::eColorAttachmentOutput;

    if (p.has_zeta) {
        if (preserve_contents)
            access |= vk::AccessFlagBits::eDepthStencilAttachmentRead;
        access |= vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        stage |= vk::PipelineStageFlagBits::eLateFragmentTests;
    }

    const vk::SubpassDependency subpass_dependency(VK_SUBPASS_EXTERNAL, 0, stage, stage, {}, access,
                                                   {});

    const vk::RenderPassCreateInfo create_info({}, static_cast<u32>(descrs.Size()), descrs.data(),
                                               1, &subpass_description, 1, &subpass_dependency);

    return device.createRenderPassUnique(create_info);
}

} // namespace Vulkan