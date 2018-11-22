// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include <optional>
#include <set>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_device.h"

namespace Vulkan {

constexpr auto UNDEFINED_FAMILY = std::numeric_limits<u32>::max();

VKDevice::VKDevice(vk::PhysicalDevice physical, vk::SurfaceKHR surface) : physical{physical} {
    SetupFamilies(surface);
    SetupProperties();
}

VKDevice::~VKDevice() = default;

bool VKDevice::CreateLogical() {
    const auto queue_cis = GetDeviceQueueCreateInfos();
    vk::PhysicalDeviceFeatures device_features{};

    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    vk::DeviceCreateInfo device_ci({}, static_cast<u32>(queue_cis.size()), queue_cis.data(), 0,
                                   nullptr, static_cast<u32>(extensions.size()), extensions.data(),
                                   &device_features);
    vk::Device dummy_logical;
    if (physical.createDevice(&device_ci, nullptr, &dummy_logical) != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Logical device failed to be created!");
        return false;
    }

    logical = vk::UniqueDevice(dummy_logical);
    graphics_queue = logical->getQueue(graphics_family, 0);
    present_queue = logical->getQueue(present_family, 0);
    return true;
}

bool VKDevice::IsSuitable(vk::PhysicalDevice physical, vk::SurfaceKHR surface) {
    const std::string swapchain_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;

    bool has_swapchain{};
    for (const vk::ExtensionProperties& prop : physical.enumerateDeviceExtensionProperties()) {
        has_swapchain |= prop.extensionName == swapchain_extension;
    }
    if (!has_swapchain) {
        // The device doesn't support creating swapchains.
        return false;
    }

    bool has_graphics{}, has_present{};
    const auto queue_family_properties = physical.getQueueFamilyProperties();
    for (u32 i = 0; i < static_cast<u32>(queue_family_properties.size()); ++i) {
        const auto& family = queue_family_properties[i];
        if (family.queueCount == 0)
            continue;

        has_graphics |=
            (family.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlagBits>(0);
        has_present |= physical.getSurfaceSupportKHR(i, surface) != 0;
    }
    if (!has_graphics || !has_present) {
        // The device doesn't have a graphics and present queue.
        return false;
    }

    // TODO(Rodrigo): Check if the device matches all requeriments.
    const vk::PhysicalDeviceProperties props = physical.getProperties();
    if (props.limits.maxUniformBufferRange < 65536) {
        return false;
    }

    // Device is suitable.
    return true;
}

void VKDevice::SetupFamilies(vk::SurfaceKHR surface) {
    std::optional<u32> graphics_family, present_family;

    const auto queue_family_properties = physical.getQueueFamilyProperties();
    for (u32 i = 0; i < static_cast<u32>(queue_family_properties.size()); ++i) {
        if (graphics_family && present_family)
            break;

        const auto& queue_family = queue_family_properties[i];
        if (queue_family.queueCount == 0)
            continue;

        if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
            graphics_family = i;
        if (physical.getSurfaceSupportKHR(i, surface))
            present_family = i;
    }
    ASSERT(graphics_family && present_family);

    this->graphics_family = *graphics_family;
    this->present_family = *present_family;
}

void VKDevice::SetupProperties() {
    const vk::PhysicalDeviceProperties props = physical.getProperties();
    device_type = props.deviceType;
    uniform_buffer_alignment = static_cast<u64>(props.limits.minUniformBufferOffsetAlignment);
}

std::vector<vk::DeviceQueueCreateInfo> VKDevice::GetDeviceQueueCreateInfos() const {
    static const float QUEUE_PRIORITY = 1.f;

    std::set<u32> unique_queue_families = {graphics_family, present_family};
    std::vector<vk::DeviceQueueCreateInfo> queue_cis;

    for (u32 queue_family : unique_queue_families)
        queue_cis.push_back({{}, queue_family, 1, &QUEUE_PRIORITY});

    return queue_cis;
}

} // namespace Vulkan