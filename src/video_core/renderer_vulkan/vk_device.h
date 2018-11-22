// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_helper.h"

namespace Vulkan {

class VKDevice final {
public:
    explicit VKDevice(vk::PhysicalDevice physical, vk::SurfaceKHR surface);
    ~VKDevice();

    bool CreateLogical();

    vk::Device GetLogical() const {
        return logical.get();
    }

    vk::PhysicalDevice GetPhysical() const {
        return physical;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    u32 GetGraphicsFamily() const {
        return graphics_family;
    }

    u32 GetPresentFamily() const {
        return present_family;
    }

    bool IsIntegrated() const {
        return device_type == vk::PhysicalDeviceType::eIntegratedGpu;
    }

    u64 GetUniformBufferAlignment() const {
        return uniform_buffer_alignment;
    }

    /// Checks if the physical device is suitable.
    static bool IsSuitable(vk::PhysicalDevice physical, vk::SurfaceKHR surface);

private:
    void SetupFamilies(vk::SurfaceKHR surface);

    void SetupProperties();

    std::vector<vk::DeviceQueueCreateInfo> GetDeviceQueueCreateInfos() const;

    const vk::PhysicalDevice physical;
    vk::UniqueDevice logical{};
    vk::Queue graphics_queue{};
    vk::Queue present_queue{};
    u32 graphics_family{};
    u32 present_family{};
    vk::PhysicalDeviceType device_type{};
    u64 uniform_buffer_alignment{};
};

} // namespace Vulkan