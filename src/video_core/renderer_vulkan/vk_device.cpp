// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_device.h"

namespace Vulkan {

VulkanDevice::VulkanDevice(vk::PhysicalDevice physical, vk::SurfaceKHR surface, bool is_renderer)
    : physical(physical), is_renderer(is_renderer) {

    int i = 0;
    for (const auto& queue_family : physical.getQueueFamilyProperties()) {
        if (queue_family.queueCount == 0) {
            ++i;
            continue;
        }
        if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
            graphics_family = i;
        }
        if (physical.getSurfaceSupportKHR(i, surface)) {
            present_family = i;
        }
    }
    ASSERT(graphics_family != UndefinedFamily && present_family != UndefinedFamily);

    const vk::PhysicalDeviceProperties props = physical.getProperties();
    device_type = props.deviceType;

    uniform_buffer_alignment = props.limits.minUniformBufferOffsetAlignment;
}

VulkanDevice::~VulkanDevice() {
    if (logical) {
        logical.destroy();
    }
}

bool VulkanDevice::CreateLogical() {
    const auto queue_cis = GetDeviceQueueCreateInfos();
    vk::PhysicalDeviceFeatures device_features{};

    std::vector<const char*> extensions{};
    if (is_renderer) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo device_ci({}, static_cast<u32>(queue_cis.size()), queue_cis.data(), 0,
                                   nullptr, static_cast<u32>(extensions.size()), extensions.data(),
                                   &device_features);
    if (physical.createDevice(&device_ci, nullptr, &logical) != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Logical device_handler failed to be created!");
        return false;
    }

    graphics_queue = logical.getQueue(graphics_family, 0);
    present_queue = logical.getQueue(present_family, 0);
    return true;
}

bool VulkanDevice::IsSuitable(vk::PhysicalDevice physical, vk::SurfaceKHR surface,
                              bool is_renderer) {
    bool has_swapchain{};
    const std::string swapchain_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    for (const vk::ExtensionProperties& prop : physical.enumerateDeviceExtensionProperties()) {
        if (prop.extensionName == swapchain_extension) {
            has_swapchain = true;
        }
    }

    bool has_graphics{};
    bool has_present{};
    u32 i{};
    for (const vk::QueueFamilyProperties& family : physical.getQueueFamilyProperties()) {
        if (family.queueCount == 0) {
            ++i;
            continue;
        }
        has_graphics |=
            (family.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlagBits>(0);
        has_present |= is_renderer && has_swapchain && physical.getSurfaceSupportKHR(i, surface);
        ++i;
    }

    // For now, non-renderer devices do not need extra features.
    if (!is_renderer) {
        return has_graphics;
    }

    if (!has_swapchain) {
        return false;
    }

    // Check for the device to match Tegra needs.
    // TODO(Rodrigo): Add the rest of the requeriments.
    const vk::PhysicalDeviceProperties props = physical.getProperties();
    if (!(props.limits.maxUniformBufferRange >= 65536 && props.limits.maxColorAttachments >= 8 &&
          props.limits.maxViewports >= 16)) {
        return false;
    }

    return true;
}

std::vector<vk::DeviceQueueCreateInfo> VulkanDevice::GetDeviceQueueCreateInfos() const {
    static const float QUEUE_PRIORITY = 1.f;

    std::vector<vk::DeviceQueueCreateInfo> queue_cis;
    std::set<u32> unique_queue_families = {graphics_family, present_family};

    for (u32 queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_ci{};
        queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_ci.queueFamilyIndex = queue_family;
        queue_ci.queueCount = 1;
        queue_ci.pQueuePriorities = &QUEUE_PRIORITY;
        queue_cis.push_back(queue_ci);
    }
    return queue_cis;
}

} // namespace Vulkan