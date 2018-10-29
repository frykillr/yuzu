// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vulkan/vulkan.hpp>
#include "core/core.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_sync.h"

#pragma optimize("", off)

namespace Vulkan {

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& renderer,
                                   VulkanScreenInfo& screen_info, VulkanDevice& device_handler,
                                   VulkanResourceManager& resource_manager,
                                   VulkanMemoryManager& memory_manager, VulkanSync& sync)
    : VideoCore::RasterizerInterface(), render_window(renderer), screen_info(screen_info),
      device_handler(device_handler), device(device_handler.GetLogical()),
      resource_manager(resource_manager), memory_manager(memory_manager), sync(sync) {

    res_cache =
        std::make_unique<VulkanRasterizerCache>(device_handler, resource_manager, memory_manager);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {}

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

    VulkanFence& fence = sync.PrepareExecute(true);

    ASSERT_MSG(use_color, "Unimplemented");

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

    // TODO(Rodrigo): Resource manage this.
    vk::RenderPass renderpass = device.createRenderPass(renderpass_ci);

    // TODO(Rodrigo): Apply color mask here.
    const vk::ImageViewCreateInfo image_view_ci = color_surface->GetImageViewCreateInfo(
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    // TODO(Rodrigo): Resource manage this.
    const vk::ImageView image_view = device.createImageView(image_view_ci);

    const vk::FramebufferCreateInfo framebuffer_ci({}, renderpass, 1, &image_view,
                                                   color_params.width, color_params.height, 1);

    // TODO(Rodrigo): Resource manage this.
    const vk::Framebuffer framebuffer = device.createFramebuffer(framebuffer_ci);

    const vk::CommandBuffer cmdbuf = sync.BeginRecord();

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

    screen_info.image = surface->GetImage();

    return true;
}

} // namespace Vulkan