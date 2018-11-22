// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

static vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& formats) {
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    const auto& found = std::find_if(formats.begin(), formats.end(), [](const auto& format) {
        return format.format == vk::Format::eB8G8R8A8Unorm &&
               format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    return found != formats.end() ? *found : formats[0];
}

static vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& modes) {
    // Mailbox doesn't lock the application like fifo (vsync), prefer it
    const auto& found = std::find_if(modes.begin(), modes.end(), [](const auto& mode) {
        return mode == vk::PresentModeKHR::eMailbox;
    });
    return found != modes.end() ? *found : vk::PresentModeKHR::eFifo;
}

static vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, u32 width,
                                     u32 height) {
    if (capabilities.currentExtent.width != UNDEFINED_SIZE) {
        return capabilities.currentExtent;
    }
    vk::Extent2D extent = {width, height};
    extent.width = std::max(capabilities.minImageExtent.width,
                            std::min(capabilities.maxImageExtent.width, extent.width));
    extent.height = std::max(capabilities.minImageExtent.height,
                             std::min(capabilities.maxImageExtent.height, extent.height));
    return extent;
}

VKSwapchain::VKSwapchain(vk::SurfaceKHR surface, const VKDevice& device_handler)
    : surface(surface), device(device_handler.GetLogical()),
      physical_device(device_handler.GetPhysical()),
      present_queue(device_handler.GetPresentQueue()),
      graphics_family(device_handler.GetGraphicsFamily()),
      present_family(device_handler.GetPresentFamily()) {}

VKSwapchain::~VKSwapchain() {
    Destroy();
}

void VKSwapchain::Create(u32 width, u32 height) {
    const vk::SurfaceCapabilitiesKHR capabilities{
        physical_device.getSurfaceCapabilitiesKHR(surface)};
    if (capabilities.maxImageExtent.width == 0 || capabilities.maxImageExtent.height == 0) {
        return;
    }

    device.waitIdle();
    Destroy();

    CreateSwapchain(width, height, capabilities);
    CreateImageViews();

    fences.resize(image_count, nullptr);
}

void VKSwapchain::AcquireNextImage(vk::Semaphore present_complete) {
    device.acquireNextImageKHR(*handle, WAIT_UNLIMITED, present_complete, {}, &image_index);

    if (auto& fence = fences[image_index]; fence) {
        fence->Wait();
        fence->Release();
        fence = nullptr;
    }
}

void VKSwapchain::Present(vk::Semaphore present_semaphore, vk::Semaphore render_semaphore,
                          VKFence& fence) {
    std::array<vk::Semaphore, 2> semaphores{present_semaphore, render_semaphore};
    const u32 wait_semaphore_count{render_semaphore ? 2u : 1u};

    const vk::PresentInfoKHR present_info(wait_semaphore_count, semaphores.data(), 1, &handle.get(),
                                          &image_index, {});

    switch (present_queue.presentKHR(&present_info)) {
    case vk::Result::eErrorOutOfDateKHR:
        if (current_width > 0 && current_height > 0) {
            Create(current_width, current_height);
        }
        break;
    case vk::Result::eSuccess:
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Vulkan failed to present swapchain!");
        UNREACHABLE();
    }

    ASSERT(fences[image_index] == nullptr);
    fences[image_index] = &fence;
}

bool VKSwapchain::HasFramebufferChanged(const Layout::FramebufferLayout& framebuffer) const {
    // TODO(Rodrigo): Handle framebuffer pixel format changes
    return framebuffer.width != current_width || framebuffer.height != current_height;
}

void VKSwapchain::CreateSwapchain(u32 width, u32 height,
                                  const vk::SurfaceCapabilitiesKHR& capabilities) {
    std::vector<vk::SurfaceFormatKHR> formats{physical_device.getSurfaceFormatsKHR(surface)};

    std::vector<vk::PresentModeKHR> present_modes{
        physical_device.getSurfacePresentModesKHR(surface)};

    vk::SurfaceFormatKHR surface_format{ChooseSwapSurfaceFormat(formats)};
    vk::PresentModeKHR present_mode{ChooseSwapPresentMode(present_modes)};
    extent = ChooseSwapExtent(capabilities, width, height);

    current_width = extent.width;
    current_height = extent.height;

    u32 requested_image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && requested_image_count > capabilities.maxImageCount) {
        requested_image_count = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchain_ci(
        {}, surface, requested_image_count, surface_format.format, surface_format.colorSpace,
        extent, 1, vk::ImageUsageFlagBits::eColorAttachment, {}, {}, {},
        capabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, present_mode, false,
        {});

    std::array<u32, 2> queue_indices{graphics_family, present_family};
    if (graphics_family != present_family) {
        swapchain_ci.imageSharingMode = vk::SharingMode::eConcurrent;
        swapchain_ci.queueFamilyIndexCount = static_cast<u32>(queue_indices.size());
        swapchain_ci.pQueueFamilyIndices = queue_indices.data();
    } else {
        swapchain_ci.imageSharingMode = vk::SharingMode::eExclusive;
    }
    handle = device.createSwapchainKHRUnique(swapchain_ci);

    images = device.getSwapchainImagesKHR(*handle);
    image_count = static_cast<u32>(images.size());
    image_format = surface_format.format;
}

void VKSwapchain::CreateImageViews() {
    image_views.resize(image_count);
    for (u32 i = 0; i < image_count; i++) {
        vk::ImageViewCreateInfo image_view_ci({}, images[i], vk::ImageViewType::e2D, image_format,
                                              {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        image_views[i] = device.createImageViewUnique(image_view_ci);
    }
}

void VKSwapchain::Destroy() {
    framebuffers.clear();
    image_views.clear();
    handle.reset();
}

} // namespace Vulkan