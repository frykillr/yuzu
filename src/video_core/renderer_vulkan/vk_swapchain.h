// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <limits>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

class VulkanDevice;
class VulkanFence;

class VulkanSwapchain {
public:
    explicit VulkanSwapchain(vk::SurfaceKHR surface, const VulkanDevice& device_handler);
    ~VulkanSwapchain();

    void Create(u32 width, u32 height);

    void AcquireNextImage(vk::Semaphore present_complete);

    void Present(vk::Semaphore present_semaphore, vk::Semaphore render_semaphore,
                 VulkanFence& fence);

    bool HasFramebufferChanged(const Layout::FramebufferLayout& framebuffer) const;

    const vk::Extent2D& GetSize() const {
        return extent;
    }

    u32 GetImageCount() const {
        return image_count;
    }

    u32 GetImageIndex() const {
        return image_index;
    }

    vk::Image GetImageIndex(u32 index) const {
        return images[index];
    }

    vk::ImageView GetImageViewIndex(u32 index) const {
        return *image_views[index];
    }

    vk::Format GetImageFormat() const {
        return image_format;
    }

private:
    void CreateSwapchain(u32 width, u32 height, const vk::SurfaceCapabilitiesKHR& capabilities);
    void CreateImageViews();

    void Destroy();

    const vk::SurfaceKHR surface;
    const vk::Device device;
    const vk::PhysicalDevice physical_device;
    const vk::Queue present_queue;
    const u32 graphics_family;
    const u32 present_family;

    vk::UniqueSwapchainKHR handle;

    u32 image_count{};
    std::vector<vk::Image> images;
    std::vector<vk::UniqueImageView> image_views;
    std::vector<vk::UniqueFramebuffer> framebuffers;
    std::vector<VulkanFence*> fences;

    u32 image_index{};

    vk::Format image_format{};
    vk::Extent2D extent{};

    u32 current_width{};
    u32 current_height{};
};

} // namespace Vulkan