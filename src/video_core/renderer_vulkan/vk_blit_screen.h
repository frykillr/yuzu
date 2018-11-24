// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
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
class VKDevice;
class VKFence;
class VKMemoryCommit;
class VKMemoryManager;
class VKSwapchain;
class VKScheduler;
class VKImage;

class VKBlitScreen final {
public:
    explicit VKBlitScreen(Core::Frontend::EmuWindow& render_window,
                          VideoCore::RasterizerInterface& rasterizer, VKDevice& device_handler,
                          VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                          VKSwapchain& swapchain, VKScheduler& sched,
                          const VKScreenInfo& screen_info);
    ~VKBlitScreen();

    void Recreate();

    std::tuple<VKFence&, vk::Semaphore> Draw(const Tegra::FramebufferConfig& framebuffer);

private:
    void CreateShaders();
    void CreateSemaphores();
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
    VideoCore::RasterizerInterface& rasterizer;
    const vk::Device device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKSwapchain& swapchain;
    VKScheduler& sched;
    const u32 image_count;
    const VKScreenInfo& screen_info;

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
    const VKMemoryCommit* buffer_commit{};

    std::vector<std::unique_ptr<VKFenceWatch>> watches;

    std::vector<vk::UniqueSemaphore> semaphores;
    std::vector<std::unique_ptr<VKImage>> raw_images;
    std::vector<const VKMemoryCommit*> raw_buffer_commits;
    u32 raw_width{}, raw_height{};
};

} // namespace Vulkan