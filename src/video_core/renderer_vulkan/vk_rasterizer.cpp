// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>
#include <vulkan/vulkan.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/static_vector.h"
#include "core/core.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using ImageViewsPack = StaticVector<Maxwell::NumRenderTargets + 1, vk::ImageView>;

struct FramebufferInfo {
    vk::Framebuffer framebuffer;
    std::array<Surface, Maxwell::NumRenderTargets> color_surfaces;
    Surface zeta_surface;
};

struct FramebufferCacheKey {
    vk::RenderPass renderpass;
    ImageViewsPack views;
    u32 width;
    u32 height;

    auto Tie() const {
        return std::tie(renderpass, views, width, height);
    }

    bool operator<(const FramebufferCacheKey& rhs) const {
        return Tie() < rhs.Tie();
    }
};

class PipelineState {
public:
    void AddDescriptorSet(vk::DescriptorSet descriptor_set) {
        // A null descriptor set means that the stage is not using descriptors, it must be skipped.
        if (descriptor_set) {
            descriptor_sets.Push(descriptor_set);
        }
    }

    void AddVertexBinding(vk::Buffer buffer, vk::DeviceSize offset) {
        vertex_bindings.Push(buffer, offset);
    }

    void SetIndexBinding(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
        index_buffer = buffer;
        index_offset = offset;
        index_type = type;
    }

    std::tuple<vk::WriteDescriptorSet&, vk::DescriptorBufferInfo&> GetWriteDescriptorSet() {
        const u32 index = descriptor_bindings_count++;
        ASSERT(index < static_cast<u32>(MAX_DESCRIPTOR_BINDINGS));

        return {descriptor_bindings[index], buffer_infos[index]};
    }

    void UpdateDescriptorSets(vk::Device device) const {
        device.updateDescriptorSets(descriptor_bindings_count, descriptor_bindings.data(), 0,
                                    nullptr);
    }

    void BindDescriptors(vk::CommandBuffer cmdbuf, vk::PipelineLayout layout) const {
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0,
                                  static_cast<u32>(descriptor_sets.Size()), descriptor_sets.data(),
                                  0, nullptr);
    }

    void BindVertexBuffers(vk::CommandBuffer cmdbuf) const {
        // TODO(Rodrigo): Sort data and bindings to do this in a single call.
        for (u32 index = 0; index < vertex_bindings.Size(); ++index) {
            cmdbuf.bindVertexBuffers(index, {vertex_bindings.Data<vk::Buffer>()[index]},
                                     {vertex_bindings.Data<vk::DeviceSize>()[index]});
        }
    }

    void BindIndexBuffer(vk::CommandBuffer cmdbuf) const {
        DEBUG_ASSERT(index_buffer && index_offset != 0);
        cmdbuf.bindIndexBuffer(index_buffer, index_offset, index_type);
    }

private:
    static constexpr std::size_t MAX_DESCRIPTOR_BINDINGS =
        Maxwell::MaxShaderStage * Maxwell::MaxConstBuffers;

    StaticVector<Maxwell::NumVertexArrays, vk::Buffer, vk::DeviceSize> vertex_bindings;

    vk::Buffer index_buffer{};
    vk::DeviceSize index_offset{};
    vk::IndexType index_type{};

    StaticVector<Maxwell::MaxShaderStage, vk::DescriptorSet> descriptor_sets;

    u32 descriptor_bindings_count{};
    std::array<vk::WriteDescriptorSet, MAX_DESCRIPTOR_BINDINGS> descriptor_bindings;
    std::array<vk::DescriptorBufferInfo, MAX_DESCRIPTOR_BINDINGS> buffer_infos;
};

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& renderer, VKScreenInfo& screen_info,
                                   VKDevice& device_handler, VKResourceManager& resource_manager,
                                   VKMemoryManager& memory_manager, VKScheduler& sched)
    : VideoCore::RasterizerInterface(), render_window(renderer), screen_info(screen_info),
      device_handler(device_handler), device(device_handler.GetLogical()),
      graphics_queue(device_handler.GetGraphicsQueue()), resource_manager(resource_manager),
      memory_manager(memory_manager), sched(sched),
      uniform_buffer_alignment(device_handler.GetUniformBufferAlignment()) {

    res_cache = std::make_unique<VKRasterizerCache>(*this, device_handler, resource_manager,
                                                    memory_manager);
    shader_cache = std::make_unique<VKShaderCache>(*this, device_handler);
    buffer_cache = std::make_unique<VKBufferCache>(*this, resource_manager, device_handler,
                                                   memory_manager, sched, STREAM_BUFFER_SIZE);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {
    if (accelerate_draw == AccelDraw::Disabled)
        return;

    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();
    const auto& regs = gpu.regs;
    const bool is_indexed = accelerate_draw == AccelDraw::Indexed;

    VKFence& fence = sched.BeginPass(true);
    PipelineParams params;
    PipelineState state;

    SyncDepthStencilState(params);

    // TODO(Rodrigo): Function this
    params.input_assembly.topology = regs.draw.topology;

    // TODO(Rodrigo): Function this
    params.renderpass.preserve_contents = true;
    if (params.renderpass.has_zeta = regs.zeta_enable) {
        params.renderpass.zeta_component_type =
            VideoCore::Surface::ComponentTypeFromDepthFormat(regs.zeta.format);
        params.renderpass.zeta_pixel_format =
            VideoCore::Surface::PixelFormatFromDepthFormat(regs.zeta.format);
    }
    {
        // TODO(Rodrigo): Dehardcode this and put it in a function
        PipelineParams::ColorAttachment attachment;
        attachment.index = 0;
        attachment.pixel_format =
            VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[0].format);
        attachment.component_type =
            VideoCore::Surface::ComponentTypeFromRenderTarget(regs.rt[0].format);
        params.renderpass.color_map.Push(attachment);
    }

    // Calculate buffer size.
    std::size_t buffer_size = CalculateVertexArraysSize();
    if (is_indexed) {
        buffer_size = Common::AlignUp<std::size_t>(buffer_size, 4) + CalculateIndexBufferSize();
    }
    buffer_size += Maxwell::MaxConstBuffers * (MaxConstbufferSize + uniform_buffer_alignment);

    buffer_cache->Reserve(buffer_size);

    SetupVertexArrays(params, state);
    SetupIndexBuffer(state);
    Pipeline pipeline = shader_cache->GetPipeline(params);

    for (std::size_t stage = 0; stage < pipeline.shaders.size(); ++stage) {
        const Shader& shader = pipeline.shaders[stage];
        if (shader == nullptr)
            continue;

        const auto descriptor_set = shader->CommitDescriptorSet(fence);
        SetupConstBuffers(state, shader, static_cast<Maxwell::ShaderStage>(stage), descriptor_set);
        state.AddDescriptorSet(descriptor_set);
    }

    const FramebufferInfo fb_info = ConfigureFramebuffers(fence, pipeline.renderpass);
    const Surface& color_surface = fb_info.color_surfaces[0];
    const Surface& zeta_surface = fb_info.zeta_surface;

    state.UpdateDescriptorSets(device);

    const vk::CommandBuffer cmdbuf = sched.BeginRecord();

    buffer_cache->Send(fence, cmdbuf);

    color_surface->Transition(
        cmdbuf, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

    if (zeta_surface != nullptr) {
        zeta_surface->Transition(
            cmdbuf, vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::PipelineStageFlagBits::eLateFragmentTests,
            vk::AccessFlagBits::eDepthStencilAttachmentRead |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    }

    // TODO(Rodrigo): Dehardcode renderpass size
    const vk::RenderPassBeginInfo renderpass_bi(pipeline.renderpass, fb_info.framebuffer,
                                                {{0, 0}, {1280, 720}}, 0,
                                                nullptr /* <-- this is clear values */);
    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline);
    {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.handle);
        state.BindDescriptors(cmdbuf, pipeline.layout);
        state.BindVertexBuffers(cmdbuf);
        if (is_indexed)
            state.BindIndexBuffer(cmdbuf);

        if (is_indexed) {
            constexpr u32 vertex_offset = 0;
            cmdbuf.drawIndexed(regs.index_array.count, 1, regs.index_array.first, vertex_offset, 0);
        } else {
            cmdbuf.draw(regs.vertex_buffer.count, 1, regs.vertex_buffer.first, 0);
        }
    }
    cmdbuf.endRenderPass();

    sched.EndRecord(cmdbuf);
    sched.EndPass();
}

void RasterizerVulkan::Clear() {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    const bool use_color = regs.clear_buffers.R || regs.clear_buffers.G || regs.clear_buffers.B ||
                           regs.clear_buffers.A;
    const bool use_depth = regs.clear_buffers.Z;
    const bool use_stencil = regs.clear_buffers.S;

    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    ASSERT_MSG(use_color, "Unimplemented");

    VKFence& fence = sched.BeginPass(true);
    const vk::CommandBuffer cmdbuf = sched.BeginRecord();

    if (use_color) {
        Surface color_surface =
            res_cache->GetColorBufferSurface(regs.clear_buffers.RT.Value(), false);

        color_surface->Transition(
            cmdbuf, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);

        const vk::ClearColorValue clear(std::array<float, 4>{
            regs.clear_color[0], regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]});
        cmdbuf.clearColorImage(
            color_surface->GetHandle(), vk::ImageLayout::eTransferDstOptimal, clear,
            {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)});
    }
    if (use_depth || use_stencil) {
        Surface zeta_surface = res_cache->GetDepthBufferSurface(false);

        // TODO(Rodrigo): Dehardcode this.
        const vk::ImageAspectFlags aspect =
            vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        zeta_surface->Transition(cmdbuf, aspect, vk::ImageLayout::eTransferDstOptimal,
                                 vk::PipelineStageFlagBits::eTransfer,
                                 vk::AccessFlagBits::eTransferWrite);

        const vk::ClearDepthStencilValue clear(regs.clear_depth,
                                               static_cast<u32>(regs.clear_stencil));
        cmdbuf.clearDepthStencilImage(zeta_surface->GetHandle(),
                                      vk::ImageLayout::eTransferDstOptimal, clear,
                                      {vk::ImageSubresourceRange(aspect, 0, 1, 0, 1)});
    }

    sched.EndRecord(cmdbuf);
    sched.EndPass();
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::InvalidateRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) {}

bool RasterizerVulkan::AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                         VAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }

    const auto& surface{res_cache->TryFindFramebufferSurface(framebuffer_addr)};
    if (!surface) {
        return {};
    }

    // Verify that the cached surface is the same size and format as the requested framebuffer
    const auto& params{surface->GetSurfaceParams()};
    const auto& pixel_format{
        VideoCore::Surface::PixelFormatFromGPUPixelFormat(config.pixel_format)};
    ASSERT_MSG(params.width == config.width, "Framebuffer width is different");
    ASSERT_MSG(params.height == config.height, "Framebuffer height is different");
    ASSERT_MSG(params.pixel_format == pixel_format, "Framebuffer pixel_format is different");

    screen_info.image = surface.get();

    return true;
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    accelerate_draw = is_indexed ? AccelDraw::Indexed : AccelDraw::Arrays;
    DrawArrays();
    return true;
}

FramebufferInfo RasterizerVulkan::ConfigureFramebuffers(VKFence& fence, vk::RenderPass renderpass,
                                                        bool using_color_fb, bool using_zeta_fb,
                                                        bool preserve_contents) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    Surface color_surface, zeta_surface;
    if (using_color_fb) {
        color_surface = res_cache->GetColorBufferSurface(0, preserve_contents);
    }
    if (using_zeta_fb) {
        zeta_surface = res_cache->GetDepthBufferSurface(preserve_contents);
    }

    FramebufferCacheKey fbkey;
    fbkey.renderpass = renderpass;
    fbkey.width = 1280;
    fbkey.height = 720;

    if (color_surface != nullptr) {
        fbkey.views.Push(color_surface->GetImageView());
    }
    if (zeta_surface != nullptr) {
        fbkey.views.Push(zeta_surface->GetImageView());
    }

    const auto [fbentry, is_cache_miss] = framebuffer_cache.try_emplace(fbkey);
    auto& framebuffer = fbentry->second;
    if (is_cache_miss) {
        const vk::FramebufferCreateInfo framebuffer_ci(
            {}, fbkey.renderpass, static_cast<u32>(fbkey.views.Size()), fbkey.views.data(),
            fbkey.width, fbkey.height, 1);
        framebuffer = device.createFramebufferUnique(framebuffer_ci);
    }

    FramebufferInfo info;
    info.framebuffer = *framebuffer;
    if (color_surface != nullptr) {
        info.color_surfaces[0] = std::move(color_surface);
    }
    if (zeta_surface != nullptr) {
        info.zeta_surface = std::move(zeta_surface);
    }
    return info;
}

void RasterizerVulkan::SetupVertexArrays(PipelineParams& params, PipelineState& state) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    for (u32 index = 0; index < static_cast<u32>(Maxwell::NumVertexAttributes); ++index) {
        const auto& attrib = regs.vertex_attrib_format[index];

        // Ignore invalid attributes.
        if (!attrib.IsValid())
            continue;

        const auto& buffer = regs.vertex_array[attrib.buffer];
        LOG_TRACE(HW_GPU, "vertex attrib {}, count={}, size={}, type={}, offset={}, normalize={}",
                  index, attrib.ComponentCount(), attrib.SizeString(), attrib.TypeString(),
                  attrib.offset.Value(), attrib.IsNormalized());

        ASSERT(buffer.IsEnabled());

        PipelineParams::VertexAttribute attribute;
        attribute.index = index;
        attribute.buffer = attrib.buffer;
        attribute.type = attrib.type;
        attribute.size = attrib.size;
        attribute.offset = attrib.offset;
        params.vertex_input.attributes.Push(attribute);
    }

    for (u32 index = 0; index < static_cast<u32>(Maxwell::NumVertexArrays); ++index) {
        const auto& vertex_array = regs.vertex_array[index];
        if (!vertex_array.IsEnabled())
            continue;

        Tegra::GPUVAddr start = vertex_array.StartAddress();
        const Tegra::GPUVAddr end = regs.vertex_array_limit[index].LimitAddress();

        ASSERT(end > start);
        const std::size_t size = end - start + 1;
        const auto [offset, buffer] = buffer_cache->UploadMemory(start, size);

        PipelineParams::VertexBinding binding;
        binding.index = index;
        binding.stride = vertex_array.stride;
        binding.divisor = vertex_array.divisor;
        params.vertex_input.bindings.Push(binding);

        state.AddVertexBinding(buffer, offset);
    }
}

void RasterizerVulkan::SetupIndexBuffer(PipelineState& state) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    const auto [offset, buffer] =
        buffer_cache->UploadMemory(regs.index_array.IndexStart(), CalculateIndexBufferSize());
    state.SetIndexBinding(buffer, offset, MaxwellToVK::IndexFormat(regs.index_array.format));
}

void RasterizerVulkan::SetupConstBuffers(PipelineState& state, Shader shader,
                                         Maxwell::ShaderStage stage,
                                         vk::DescriptorSet descriptor_set) {
    const auto& gpu = Core::System::GetInstance().GPU();
    const auto& maxwell3d = gpu.Maxwell3D();
    const auto& shader_stage = maxwell3d.state.shader_stages[static_cast<std::size_t>(stage)];
    const auto& entries = shader->GetEntries().const_buffers;

    for (const auto& used_buffer : entries) {
        const auto& buffer = shader_stage.const_buffers[used_buffer.GetIndex()];

        std::size_t size = 0;

        if (used_buffer.IsIndirect()) {
            // Buffer is accessed indirectly, so upload the entire thing
            size = buffer.size;

            if (size > MaxConstbufferSize) {
                LOG_CRITICAL(HW_GPU, "indirect constbuffer size {} exceeds maximum {}", size,
                             MaxConstbufferSize);
                size = MaxConstbufferSize;
            }
        } else {
            // Buffer is accessed directly, upload just what we use
            size = used_buffer.GetSize() * sizeof(float);
        }

        // Align the actual size so it ends up being a multiple of vec4 to meet the OpenGL
        // std140 UBO alignment requirements.
        size = Common::AlignUp(size, 4 * sizeof(float));
        ASSERT_MSG(size <= MaxConstbufferSize, "Constbuffer too big");

        const auto [offset, buffer_handle] =
            buffer_cache->UploadMemory(buffer.address, size, uniform_buffer_alignment);

        auto [write, buffer_info] = state.GetWriteDescriptorSet();
        buffer_info =
            vk::DescriptorBufferInfo(buffer_handle, offset, static_cast<vk::DeviceSize>(size));
        write = vk::WriteDescriptorSet(descriptor_set, used_buffer.GetBinding(), 0, 1,
                                       vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info,
                                       nullptr);
    }
}

std::size_t RasterizerVulkan::CalculateVertexArraysSize() const {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    std::size_t size = 0;
    for (u32 index = 0; index < Maxwell::NumVertexArrays; ++index) {
        if (!regs.vertex_array[index].IsEnabled())
            continue;
        // This implementation assumes that all attributes are used.

        const Tegra::GPUVAddr start = regs.vertex_array[index].StartAddress();
        const Tegra::GPUVAddr end = regs.vertex_array_limit[index].LimitAddress();

        ASSERT(end > start);
        size += end - start + 1;
    }
    return size;
}

std::size_t RasterizerVulkan::CalculateIndexBufferSize() const {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    return static_cast<std::size_t>(regs.index_array.count) *
           static_cast<std::size_t>(regs.index_array.FormatSizeInBytes());
}

void RasterizerVulkan::SyncDepthStencilState(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    auto& ds = params.depth_stencil;
    ds.depth_test_enable = regs.depth_test_enable == 1;
    ds.depth_write_enable = regs.depth_write_enabled == 1;
    ds.depth_test_function = regs.depth_test_func;
    ds.depth_bounds_enable = false;
    ds.stencil_enable = false;
    // ds.front_stencil = ;
    // ds.back_stencil = ;
    ds.depth_bounds_min = 0.f;
    ds.depth_bounds_max = 0.f;
}

} // namespace Vulkan