// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include <set>

#include <vulkan/vulkan.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "core/settings.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_helper.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_sync.h"
#include "video_core/utils.h"

namespace Vulkan {

RendererVulkan::RendererVulkan(Core::Frontend::EmuWindow& window) : RendererBase(window) {}

RendererVulkan::~RendererVulkan() {
    ShutDown();
}

void RendererVulkan::SwapBuffers(
    std::optional<std::reference_wrapper<const Tegra::FramebufferConfig>> framebuffer) {

    Core::System::GetInstance().GetPerfStats().EndSystemFrame();

    const auto& layout = render_window.GetFramebufferLayout();
    if (framebuffer && layout.width > 0 && layout.height > 0 && render_window.IsShown()) {
        if (swapchain->HasFramebufferChanged(layout)) {
            swapchain->Create(layout.width, layout.height);
        }

        swapchain->AcquireNextImage(present_semaphore);
        VulkanFence& fence = DrawScreen(*framebuffer);

        const vk::Semaphore render_semaphore = sync->QuerySemaphore();
        swapchain->Present(present_semaphore, render_semaphore, fence);

        render_window.SwapBuffers();
    }

    render_window.PollEvents();

    Core::System::GetInstance().FrameLimiter().DoFrameLimiting(CoreTiming::GetGlobalTimeUs());
    Core::System::GetInstance().GetPerfStats().BeginSystemFrame();
}

bool RendererVulkan::Init() {
    CreateRasterizer();
    return InitVulkanObjects();
}

void RendererVulkan::ShutDown() {
    if (!device_handler) {
        return;
    }

    device.waitIdle();

    device.destroy(screen_info.staging_image);

    sync.reset();
    swapchain.reset();
    resource_manager.reset();
    memory_manager.reset();

    device.destroy(present_semaphore);
    device_handler.reset();
}

void RendererVulkan::CreateRasterizer() {
    if (rasterizer) {
        return;
    }
    rasterizer = std::make_unique<RasterizerVulkan>(render_window, screen_info);
}

bool RendererVulkan::InitVulkanObjects() {
    render_window.RetrieveVulkanHandlers(reinterpret_cast<void**>(&instance),
                                         reinterpret_cast<void**>(&surface));

    if (!PickDevices()) {
        return false;
    }
    device = device_handler->GetLogical();
    physical_device = device_handler->GetPhysical();

    memory_manager = std::make_unique<VulkanMemoryManager>(*device_handler);

    resource_manager = std::make_unique<VulkanResourceManager>(*device_handler);

    const auto& framebuffer = render_window.GetFramebufferLayout();
    swapchain = std::make_unique<VulkanSwapchain>(surface, *device_handler);
    swapchain->Create(framebuffer.width, framebuffer.height);

    sync = std::make_unique<VulkanSync>(*resource_manager, *device_handler);

    if (device.createSemaphore(&vk::SemaphoreCreateInfo(), nullptr, &present_semaphore) !=
        vk::Result::eSuccess) {
        return false;
    }

    return true;
}

bool RendererVulkan::PickDevices() {
    const auto devices = instance.enumeratePhysicalDevices();

    // TODO(Rodrigo): Choose device from config file
    const s32 device_index = Settings::values.vulkan_device;
    if (device_index < 0 || device_index >= static_cast<s32>(devices.size())) {
        LOG_ERROR(Render_Vulkan, "Invalid device index {}!", device_index);
        return false;
    }
    physical_device = devices[device_index];

    if (!VulkanDevice::IsSuitable(physical_device, surface, true)) {
        LOG_ERROR(Render_Vulkan, "Device is not suitable!");
        return false;
    }

    device_handler = std::make_unique<VulkanDevice>(physical_device, surface, true);
    return device_handler->CreateLogical();
}

VulkanFence& RendererVulkan::DrawScreen(const Tegra::FramebufferConfig& framebuffer) {
    const u32 bytes_per_pixel{Tegra::FramebufferConfig::BytesPerPixel(framebuffer.pixel_format)};
    const u64 size_in_bytes{framebuffer.stride * framebuffer.height * bytes_per_pixel};
    const VAddr framebuffer_addr{framebuffer.address + framebuffer.offset};
    const vk::Extent2D& framebuffer_size{swapchain->GetSize()};

    if (rasterizer->AccelerateDisplay(framebuffer, framebuffer_addr, framebuffer.stride)) {
        UNREACHABLE();
    }

    const bool recreate{!screen_info.staging_image || framebuffer.width != screen_info.width ||
                        framebuffer.height != screen_info.height};
    if (recreate) {
        if (screen_info.staging_image || screen_info.staging_memory) {
            ASSERT(screen_info.staging_image && screen_info.staging_memory);

            // Wait to avoid using staging memory while it's being transfered.
            device.waitIdle();
            device.destroy(screen_info.staging_image);
            memory_manager->Free(screen_info.staging_memory);
        }

        const vk::ImageCreateInfo image_ci(
            {}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm,
            {framebuffer.width, framebuffer.height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eLinear, vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive, 0, nullptr, vk::ImageLayout::ePreinitialized);
        screen_info.staging_image = device.createImage(image_ci);

        const vk::MemoryRequirements mem_reqs =
            device.getImageMemoryRequirements(screen_info.staging_image);

        screen_info.staging_memory = memory_manager->Commit(mem_reqs, true);

        device.bindImageMemory(screen_info.staging_image, screen_info.staging_memory->GetMemory(),
                               screen_info.staging_memory->GetOffset());

        screen_info.width = framebuffer.width;
        screen_info.height = framebuffer.height;
        screen_info.size_in_bytes = mem_reqs.size;
    }

    Memory::RasterizerFlushVirtualRegion(framebuffer_addr, size_in_bytes, Memory::FlushMode::Flush);

    u8* data = screen_info.staging_memory->GetData();

    VideoCore::MortonCopyPixels128(framebuffer.width, framebuffer.height, bytes_per_pixel, 4,
                                   Memory::GetPointer(framebuffer_addr), data, true);

    // Record blitting
    VulkanFence& fence = sync->PrepareExecute(false);
    vk::CommandBuffer cmdbuf{sync->BeginRecord()};

    if (recreate) {
        SetImageLayout(cmdbuf, screen_info.staging_image, vk::ImageAspectFlagBits::eColor,
                       vk::ImageLayout::ePreinitialized, vk::ImageLayout::eGeneral,
                       vk::PipelineStageFlagBits::eHost,
                       vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer);
    }
    SetImageLayout(cmdbuf, swapchain->GetImage(), vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                   vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);

    // TODO(Rodrigo): Use clip values
    const bool flip_y =
        framebuffer.transform_flags == Tegra::FramebufferConfig::TransformFlags::FlipV;
    const s32 y0 = flip_y ? static_cast<s32>(framebuffer_size.height) : 0;
    const s32 y1 = flip_y ? 0 : static_cast<s32>(framebuffer_size.height);
    const vk::ImageSubresourceLayers subresource(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
    std::array<vk::Offset3D, 2> src_offsets, dst_offsets;
    src_offsets[0] = {0, 0, 0};
    src_offsets[1] = {static_cast<s32>(screen_info.width), static_cast<s32>(screen_info.height), 1};
    dst_offsets[0] = {0, y0, 0};
    dst_offsets[1] = {static_cast<s32>(framebuffer_size.width), y1, 1};
    const vk::ImageBlit blit(subresource, src_offsets, subresource, dst_offsets);

    cmdbuf.blitImage(screen_info.staging_image, vk::ImageLayout::eGeneral, swapchain->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, {blit}, vk::Filter::eLinear);

    SetImageLayout(cmdbuf, swapchain->GetImage(), vk::ImageAspectFlagBits::eColor,
                   vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
                   vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer);

    sync->EndRecord(cmdbuf);
    sync->Execute();

    return fence;
}

} // namespace Vulkan