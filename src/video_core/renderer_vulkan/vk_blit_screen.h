// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>
#include "video_core/renderer_vulkan/vk_resource_manager.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace Tegra {
struct FramebufferConfig;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Vulkan {

struct ScreenInfo;
class RasterizerVulkan;
class VulkanDevice;
class VulkanFence;
class VulkanMemoryCommit;
class VulkanMemoryManager;
class VulkanSwapchain;
class VulkanScheduler;
class VulkanImage;

class VulkanBlitScreen final {
public:
    explicit VulkanBlitScreen(Core::Frontend::EmuWindow& render_window,
                              VulkanDevice& device_handler, VulkanResourceManager& resource_manager,
                              VulkanMemoryManager& memory_manager, VulkanSwapchain& swapchain,
                              VulkanScreenInfo& screen_info);
    ~VulkanBlitScreen();

    void Recreate();

    VulkanFence& Draw(VideoCore::RasterizerInterface& rasterizer, VulkanScheduler& sched,
                      const Tegra::FramebufferConfig& framebuffer);

private:
    void CreateShaders();
    void CreateDescriptorPool();
    void CreateRenderPass();
    void CreateDescriptorSetLayout();
    void CreateDescriptorSets();
    void CreatePipelineLayout();
    void CreateGraphicsPipeline();

    void CreateFramebuffers();

    void UpdateDescriptorSet(u32 image_index, vk::ImageView image_view);
    void RefreshResources(const Tegra::FramebufferConfig& framebuffer);
    void SetUniformData(const Tegra::FramebufferConfig& framebuffer);
    void SetVertexData(const Tegra::FramebufferConfig& framebuffer);

    u64 CalculateBufferSize(const Tegra::FramebufferConfig& framebuffer) const;
    u64 GetUniformDataOffset() const;
    u64 GetVertexDataOffset() const;
    u64 GetRawImageOffset(const Tegra::FramebufferConfig& framebuffer, u32 image_index) const;

    Core::Frontend::EmuWindow& render_window;
    const vk::Device device;
    VulkanResourceManager& resource_manager;
    VulkanMemoryManager& memory_manager;
    VulkanSwapchain& swapchain;
    const u32 image_count;
    VulkanScreenInfo& screen_info;

    vk::UniqueShaderModule vertex_shader;
    vk::UniqueShaderModule fragment_shader;
    vk::UniqueDescriptorPool descriptor_pool;
    vk::UniqueDescriptorSetLayout descriptor_set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    vk::UniqueRenderPass renderpass;
    std::vector<vk::UniqueFramebuffer> framebuffers;
    std::vector<vk::DescriptorSet> descriptor_sets;
    vk::UniqueSampler sampler;

    vk::UniqueBuffer buffer;
    const VulkanMemoryCommit* buffer_commit{};

    std::vector<std::unique_ptr<VulkanFenceWatch>> watches;

    std::vector<std::unique_ptr<VulkanImage>> raw_images;
    std::vector<const VulkanMemoryCommit*> raw_buffer_commits;
    u32 raw_width{}, raw_height{};
};

} // namespace Vulkan