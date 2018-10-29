// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>
#include "video_core/rasterizer_interface.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace Vulkan {

struct VulkanScreenInfo;
class VulkanSync;
class VulkanRasterizerCache;
class VulkanResourceManager;
class VulkanMemoryManager;
class VulkanDevice;

class RasterizerVulkan : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& render_window,
                              VulkanScreenInfo& screen_info, VulkanDevice& device_handler,
                              VulkanResourceManager& resource_manager,
                              VulkanMemoryManager& memory_manager, VulkanSync& sync);
    ~RasterizerVulkan() override;

    void DrawArrays() override;
    void Clear() override;
    void FlushAll() override;
    void FlushRegion(Tegra::GPUVAddr addr, u64 size) override;
    void InvalidateRegion(Tegra::GPUVAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) override;
    bool AccelerateDisplay(const Tegra::FramebufferConfig& config, VAddr framebuffer_addr,
                           u32 pixel_stride) override;

private:
    Core::Frontend::EmuWindow& render_window;
    VulkanScreenInfo& screen_info;
    VulkanDevice& device_handler;
    const vk::Device device;
    VulkanResourceManager& resource_manager;
    VulkanMemoryManager& memory_manager;
    VulkanSync& sync;

    std::unique_ptr<VulkanRasterizerCache> res_cache;
};

} // namespace Vulkan