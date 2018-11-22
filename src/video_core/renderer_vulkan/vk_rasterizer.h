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

struct VKScreenInfo;
class VKFence;
class VKScheduler;
class VKRasterizerCache;
class VKResourceManager;
class VKMemoryManager;
class VKDevice;
class VKShaderCache;
class VKBufferCache;

class PipelineState;
struct FramebufferInfo;
struct FramebufferCacheKey;

class RasterizerVulkan : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& render_window, VKScreenInfo& screen_info,
                              VKDevice& device_handler, VKResourceManager& resource_manager,
                              VKMemoryManager& memory_manager, VKScheduler& sched);
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

    FramebufferInfo ConfigureFramebuffers(VKFence& fence, vk::RenderPass renderpass,
                                          bool using_color_fb = true, bool use_zeta_fb = true,
                                          bool preserve_contents = true);

    void SetupVertexArrays(PipelineParams& params, PipelineState& state);

    void SetupConstBuffers(PipelineState& state, Shader shader, Maxwell::ShaderStage stage,
                           vk::DescriptorSet descriptor_set);

    std::size_t CalculateVertexArraysSize() const;

    void SyncDepthStencilState(PipelineParams& params);

    Core::Frontend::EmuWindow& render_window;
    VKScreenInfo& screen_info;
    VKDevice& device_handler;
    const vk::Device device;
    const vk::Queue graphics_queue;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;
    const u64 uniform_buffer_alignment;

    std::unique_ptr<VKRasterizerCache> res_cache;
    std::unique_ptr<VKShaderCache> shader_cache;
    std::unique_ptr<VKBufferCache> buffer_cache;

    // FIXME(Rodrigo): Invalidate me on shader flush
    std::map<FramebufferCacheKey, vk::UniqueFramebuffer> framebuffer_cache;

    enum class AccelDraw { Disabled, Arrays, Indexed };
    AccelDraw accelerate_draw = AccelDraw::Disabled;
};

} // namespace Vulkan