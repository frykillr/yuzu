// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <utility>
#include <vulkan/vulkan.hpp>
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace Vulkan {

struct VulkanScreenInfo;
class VulkanFence;
class VulkanSync;
class VulkanRasterizerCache;
class VulkanResourceManager;
class VulkanMemoryManager;
class VulkanDevice;
class VulkanShaderCache;
class VulkanBufferCache;

class PipelineState;
struct FramebufferInfo;

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
    bool AccelerateDrawBatch(bool is_indexed) override;

    /// Maximum supported size that a constbuffer can have in bytes.
    static constexpr std::size_t MaxConstbufferSize = 0x10000;
    static_assert(MaxConstbufferSize % (4 * sizeof(float)) == 0,
                  "The maximum size of a constbuffer must be a multiple of the size of GLvec4");

private:
    static constexpr u64 STREAM_BUFFER_SIZE = 16 * 1024 * 1024;

    FramebufferInfo ConfigureFramebuffers(VulkanFence& fence, vk::RenderPass renderpass,
                                          bool using_color_fb = true, bool use_zeta_fb = true,
                                          bool preserve_contents = true);

    void SetupVertexArrays(PipelineParams& params, PipelineState& state);

    void SetupConstBuffers(PipelineState& state, Shader shader, Maxwell::ShaderStage stage,
                           vk::DescriptorSet descriptor_set);

    std::size_t CalculateVertexArraysSize() const;

    void SyncDepthStencilState(PipelineParams& params);

    Core::Frontend::EmuWindow& render_window;
    VulkanScreenInfo& screen_info;
    VulkanDevice& device_handler;
    const vk::Device device;
    const vk::Queue graphics_queue;
    VulkanResourceManager& resource_manager;
    VulkanMemoryManager& memory_manager;
    VulkanSync& sync;
    const u64 uniform_buffer_alignment;

    std::unique_ptr<VulkanRasterizerCache> res_cache;
    std::unique_ptr<VulkanShaderCache> shader_cache;
    std::unique_ptr<VulkanBufferCache> buffer_cache;

    enum class AccelDraw { Disabled, Arrays, Indexed };
    AccelDraw accelerate_draw = AccelDraw::Disabled;
};

} // namespace Vulkan