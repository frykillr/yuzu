// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <set>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "core/settings.h"
#include "video_core/gpu.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_helper.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/utils.h"

#pragma optimize("", off)

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
            blit_screen->Recreate();
        }

        swapchain->AcquireNextImage(*present_semaphore);
        VulkanFence& fence = blit_screen->Draw(*rasterizer, *sched, *framebuffer);

        sched->Flush();

        const vk::Semaphore render_semaphore = sched->QuerySemaphore();
        swapchain->Present(*present_semaphore, render_semaphore, fence);

        render_window.SwapBuffers();
    }

    render_window.PollEvents();

    Core::System::GetInstance().FrameLimiter().DoFrameLimiting(CoreTiming::GetGlobalTimeUs());
    Core::System::GetInstance().GetPerfStats().BeginSystemFrame();
}

bool RendererVulkan::Init() {
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

    sched = std::make_unique<VulkanScheduler>(*resource_manager, *device_handler);

    present_semaphore = device.createSemaphoreUnique({});

    blit_screen =
        std::make_unique<VulkanBlitScreen>(render_window, *device_handler, *resource_manager,
                                           *memory_manager, *swapchain, screen_info);

    rasterizer = std::make_unique<RasterizerVulkan>(render_window, screen_info, *device_handler,
                                                    *resource_manager, *memory_manager, *sched);

    return true;
}

void RendererVulkan::ShutDown() {
    if (!device_handler) {
        return;
    }
    device.waitIdle();

    rasterizer.reset();
    blit_screen.reset();
    sched.reset();
    swapchain.reset();
    memory_manager.reset();
    present_semaphore.reset();
    resource_manager.reset();

    device_handler.reset();
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

} // namespace Vulkan