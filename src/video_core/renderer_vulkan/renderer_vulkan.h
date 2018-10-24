// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <limits>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/vk_helper.h"

namespace Vulkan {

class VulkanMemoryCommit;
class VulkanDevice;
class VulkanFence;
class VulkanMemoryManager;
class VulkanResourceManager;
class VulkanSwapchain;
class VulkanSync;

class VulkanSwapchain;
class VulkanSync;

struct VulkanScreenInfo {
    u32 width{};
    u32 height{};
    u64 size_in_bytes{};
    vk::Image staging_image;
    const VulkanMemoryCommit* staging_memory;
};

class RendererVulkan : public VideoCore::RendererBase {
public:
    explicit RendererVulkan(Core::Frontend::EmuWindow& window);
    ~RendererVulkan() override;

    /// Swap buffers (render frame)
    void SwapBuffers(
        std::optional<std::reference_wrapper<const Tegra::FramebufferConfig>> framebuffer) override;

    /// Initialize the renderer
    bool Init() override;

    /// Shutdown the renderer
    void ShutDown() override;

private:
    void CreateRasterizer();
    bool InitVulkanObjects();
    bool PickDevices();

    VulkanFence& DrawScreen(const Tegra::FramebufferConfig& framebuffer);

    vk::Instance instance;
    vk::SurfaceKHR surface;

    VulkanScreenInfo screen_info{};

    std::unique_ptr<VulkanDevice> device_handler;
    vk::Device device;
    vk::PhysicalDevice physical_device;

    std::unique_ptr<VulkanSwapchain> swapchain;
    std::unique_ptr<VulkanMemoryManager> memory_manager;
    std::unique_ptr<VulkanResourceManager> resource_manager;
    std::unique_ptr<VulkanSync> sync;

    vk::Semaphore present_semaphore;
};

} // namespace Vulkan