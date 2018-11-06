// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vulkan/vulkan.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"
#include "video_core/renderer_vulkan/vk_sync.h"

#pragma optimize("", off)

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& renderer,
                                   VulkanScreenInfo& screen_info, VulkanDevice& device_handler,
                                   VulkanResourceManager& resource_manager,
                                   VulkanMemoryManager& memory_manager, VulkanSync& sync)
    : VideoCore::RasterizerInterface(), render_window(renderer), screen_info(screen_info),
      device_handler(device_handler), device(device_handler.GetLogical()),
      graphics_queue(device_handler.GetGraphicsQueue()), resource_manager(resource_manager),
      memory_manager(memory_manager), sync(sync),
      uniform_buffer_alignment(device_handler.GetUniformBufferAlignment()) {

    res_cache =
        std::make_unique<VulkanRasterizerCache>(device_handler, resource_manager, memory_manager);
    shader_cache = std::make_unique<VulkanShaderCache>(device_handler);
    buffer_cache = std::make_unique<VulkanBufferCache>(resource_manager, device_handler,
                                                       memory_manager, STREAM_BUFFER_SIZE);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();
    const auto& regs = gpu.regs;

    VulkanFence& fence = sync.PrepareExecute(true);

    const vk::PrimitiveTopology primitive_topology =
        MaxwellToVK::PrimitiveTopology(regs.draw.topology);

    std::size_t buffer_size = 0;
    buffer_size += Maxwell::MaxConstBuffers * (MaxConstbufferSize + uniform_buffer_alignment);

    buffer_cache->Reserve(buffer_size);

    SetupShaders(primitive_topology);

    buffer_cache->Send(sync, fence);

    vk::CommandBuffer cmdbuf = sync.BeginRecord();
    sync.EndRecord(cmdbuf);

    sync.Execute();
}

void RasterizerVulkan::Clear() {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    bool use_color{};
    bool use_depth{};
    bool use_stencil{};

    if (regs.clear_buffers.R || regs.clear_buffers.G || regs.clear_buffers.B ||
        regs.clear_buffers.A) {
        use_color = true;
    }
    if (regs.clear_buffers.Z) {
        UNIMPLEMENTED_MSG("Depth clear");
    }
    if (regs.clear_buffers.S) {
        UNIMPLEMENTED_MSG("Stencil clear");
    }

    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    ASSERT_MSG(use_color, "Unimplemented");

    VulkanFence& fence = sync.PrepareExecute(true);

    Surface color_surface = res_cache->GetColorBufferSurface(regs.clear_buffers.RT.Value(), false);
    ASSERT(color_surface);
    const auto color_params = color_surface->GetSurfaceParams();

    const vk::AttachmentDescription color_attachment(
        {}, color_surface->GetFormat(), vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal);

    const vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);

    const vk::SubpassDescription subpass_description({}, vk::PipelineBindPoint::eGraphics, 0,
                                                     nullptr, 1, &color_attachment_ref, nullptr,
                                                     nullptr, 0, nullptr);

    const vk::SubpassDependency subpass_dependency(
        VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, {},
        vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, {});

    const vk::RenderPassCreateInfo renderpass_ci({}, 1, &color_attachment, 1, &subpass_description,
                                                 1, &subpass_dependency);

    vk::RenderPass renderpass = resource_manager.CreateRenderPass(fence, renderpass_ci);

    // TODO(Rodrigo): Apply color mask here.
    const vk::ImageViewCreateInfo image_view_ci = color_surface->GetImageViewCreateInfo(
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    const vk::ImageView image_view = resource_manager.CreateImageView(fence, image_view_ci);

    const vk::FramebufferCreateInfo framebuffer_ci({}, renderpass, 1, &image_view,
                                                   color_params.width, color_params.height, 1);
    const vk::Framebuffer framebuffer = resource_manager.CreateFramebuffer(fence, framebuffer_ci);

    const vk::CommandBuffer cmdbuf = sync.BeginRecord();

    color_surface->Transition(cmdbuf, vk::ImageAspectFlagBits::eColor,
                              vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eColorAttachmentWrite);

    const vk::ClearValue clear_color(std::array<float, 4>{
        regs.clear_color[0], regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]});

    const vk::RenderPassBeginInfo renderpass_bi(renderpass, framebuffer,
                                                {{0, 0}, {color_params.width, color_params.height}},
                                                1, &clear_color);

    cmdbuf.beginRenderPass(&renderpass_bi, vk::SubpassContents::eInline);
    cmdbuf.endRenderPass();

    sync.EndRecord(cmdbuf);
    sync.Execute();
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::InvalidateRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) {}

bool RasterizerVulkan::AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                         VAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }

    const auto& surface{res_cache->TryFindFramebufferSurface(framebuffer_addr)};
    if (!surface) {
        return {};
    }

    // Verify that the cached surface is the same size and format as the requested framebuffer
    const auto& params{surface->GetSurfaceParams()};
    const auto& pixel_format{
        VideoCore::Surface::PixelFormatFromGPUPixelFormat(config.pixel_format)};
    ASSERT_MSG(params.width == config.width, "Framebuffer width is different");
    ASSERT_MSG(params.height == config.height, "Framebuffer height is different");
    ASSERT_MSG(params.pixel_format == pixel_format, "Framebuffer pixel_format is different");

    screen_info.image = surface.get();

    return true;
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    // TODO
    DrawArrays();
    return true;
}

void RasterizerVulkan::SetupShaders(vk::PrimitiveTopology primitive_topology) {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();

    constexpr u32 MAX_STAGES_COUNT = 6;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_STAGES_COUNT> stages;
    u32 stages_count = 0;

    for (std::size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const auto& shader_config = gpu.regs.shader_config[index];
        const auto program{static_cast<Maxwell::ShaderProgram>(index)};

        // Skip stages that are not enabled
        if (!gpu.regs.IsShaderConfigEnabled(index)) {
            continue;
        }

        Shader shader = shader_cache->GetStageProgram(program);

        const std::size_t stage{index == 0 ? 0 : index - 1}; // Stage indices are 0 - 5
        const vk::ShaderStageFlagBits stage_bits = MaxwellToVK::ShaderStage(program);

        stages[stages_count++] = vk::PipelineShaderStageCreateInfo(
            {}, stage_bits, shader->GetHandle(primitive_topology), "main", nullptr);

        SetupConstBuffers(shader, static_cast<Maxwell::ShaderStage>(stage), stage_bits);

        // When VertexA is enabled, we have dual vertex shaders
        if (program == Maxwell::ShaderProgram::VertexA) {
            // VertexB was combined with VertexA, so we skip the VertexB iteration
            index++;
        }
    }
}

void RasterizerVulkan::SetupConstBuffers(Shader shader, Maxwell::ShaderStage stage,
                                         vk::ShaderStageFlagBits stage_bits) {
    const auto& gpu = Core::System::GetInstance().GPU();
    const auto& maxwell3d = gpu.Maxwell3D();
    const auto& shader_stage = maxwell3d.state.shader_stages[static_cast<std::size_t>(stage)];
    const auto& entries = shader->GetEntries().const_buffer_entries;

    for (const auto& used_buffer : entries) {
        const auto& buffer = shader_stage.const_buffers[used_buffer.GetIndex()];

        std::size_t size = 0;

        if (used_buffer.IsIndirect()) {
            // Buffer is accessed indirectly, so upload the entire thing
            size = buffer.size;

            if (size > MaxConstbufferSize) {
                LOG_CRITICAL(HW_GPU, "indirect constbuffer size {} exceeds maximum {}", size,
                             MaxConstbufferSize);
                size = MaxConstbufferSize;
            }
        } else {
            // Buffer is accessed directly, upload just what we use
            size = used_buffer.GetSize() * sizeof(float);
        }

        // Align the actual size so it ends up being a multiple of vec4 to meet the OpenGL std140
        // UBO alignment requirements.
        size = Common::AlignUp(size, 4 * sizeof(float));
        ASSERT_MSG(size <= MaxConstbufferSize, "Constbuffer too big");

        buffer_cache->UploadMemory(buffer.address, size, uniform_buffer_alignment);
    }
}

} // namespace Vulkan