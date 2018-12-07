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
#include "core/memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#pragma optimize("", off)

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using ImageViewsPack = StaticVector<vk::ImageView, Maxwell::NumRenderTargets + 1>;

struct FramebufferInfo {
    vk::Framebuffer framebuffer;
    std::array<Surface, Maxwell::NumRenderTargets> color_surfaces;
    Surface zeta_surface;
    u32 width;
    u32 height;
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
    void AssignDescriptorSet(u32 stage, vk::DescriptorSet descriptor_set) {
        // A null descriptor set means that the stage is not using descriptors, it must be skipped.
        if (descriptor_set) {
            descriptor_sets[stage] = descriptor_set;
        }
    }

    void AddVertexBinding(vk::Buffer buffer, vk::DeviceSize offset) {
        vertex_bindings.Push({buffer, offset});
    }

    void SetIndexBinding(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
        index_buffer = buffer;
        index_offset = offset;
        index_type = type;
    }

    std::tuple<vk::WriteDescriptorSet&, vk::DescriptorBufferInfo&> CaptureDescriptorWriteBuffer() {
        const u32 desc_index = descriptor_bindings_count++;
        const u32 info_index = buffer_info_count++;
        ASSERT(desc_index < static_cast<u32>(MAX_DESCRIPTOR_WRITES));
        ASSERT(info_index < static_cast<u32>(MAX_DESCRIPTOR_BUFFERS));

        return {descriptor_bindings[desc_index], buffer_infos[info_index]};
    }

    std::tuple<vk::WriteDescriptorSet&, vk::DescriptorImageInfo&> CaptureDescriptorWriteImage() {
        const u32 desc_index = descriptor_bindings_count++;
        const u32 info_index = image_info_count++;
        ASSERT(desc_index < static_cast<u32>(MAX_DESCRIPTOR_WRITES));
        ASSERT(info_index < static_cast<u32>(MAX_DESCRIPTOR_IMAGES));

        return {descriptor_bindings[desc_index], image_infos[info_index]};
    }

    void UpdateDescriptorSets(vk::Device device) const {
        device.updateDescriptorSets(descriptor_bindings_count, descriptor_bindings.data(), 0,
                                    nullptr);
    }

    void BindDescriptors(vk::CommandBuffer cmdbuf, vk::PipelineLayout layout) const {
        for (std::size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            if (const auto descriptor_set = descriptor_sets[stage]; descriptor_set) {
                cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout,
                                          static_cast<u32>(stage), 1, &descriptor_set, 0, nullptr);
            }
        }
    }

    void BindVertexBuffers(vk::CommandBuffer cmdbuf) const {
        // TODO(Rodrigo): Sort data and bindings to do this in a single call.
        for (u32 index = 0; index < vertex_bindings.Size(); ++index) {
            const auto [buffer, size] = vertex_bindings.Data()[index];
            cmdbuf.bindVertexBuffers(index, {buffer}, {size});
        }
    }

    void BindIndexBuffer(vk::CommandBuffer cmdbuf) const {
        DEBUG_ASSERT(index_buffer && index_offset != 0);
        cmdbuf.bindIndexBuffer(index_buffer, index_offset, index_type);
    }

private:
    static constexpr std::size_t MAX_DESCRIPTOR_BUFFERS =
        Maxwell::MaxShaderStage * Maxwell::MaxConstBuffers;
    static constexpr std::size_t MAX_DESCRIPTOR_IMAGES = Maxwell::NumTextureSamplers;
    static constexpr std::size_t MAX_DESCRIPTOR_WRITES =
        MAX_DESCRIPTOR_BUFFERS + MAX_DESCRIPTOR_IMAGES;

    StaticVector<std::tuple<vk::Buffer, vk::DeviceSize>, Maxwell::NumVertexArrays> vertex_bindings;

    vk::Buffer index_buffer{};
    vk::DeviceSize index_offset{};
    vk::IndexType index_type{};

    std::array<vk::DescriptorSet, Maxwell::MaxShaderStage> descriptor_sets{};

    u32 descriptor_bindings_count{};
    std::array<vk::WriteDescriptorSet, MAX_DESCRIPTOR_WRITES> descriptor_bindings;

    u32 buffer_info_count{};
    std::array<vk::DescriptorBufferInfo, MAX_DESCRIPTOR_BUFFERS> buffer_infos;

    u32 image_info_count{};
    std::array<vk::DescriptorImageInfo, MAX_DESCRIPTOR_IMAGES> image_infos;
};

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& renderer, VKScreenInfo& screen_info,
                                   VKDevice& device_handler, VKResourceManager& resource_manager,
                                   VKMemoryManager& memory_manager, VKScheduler& sched)
    : VideoCore::RasterizerInterface(), render_window{renderer}, screen_info{screen_info},
      device_handler{device_handler}, device{device_handler.GetLogical()},
      graphics_queue{device_handler.GetGraphicsQueue()}, resource_manager{resource_manager},
      memory_manager{memory_manager}, sched{sched},
      uniform_buffer_alignment{device_handler.GetUniformBufferAlignment()} {

    res_cache = std::make_unique<VKRasterizerCache>(*this, device_handler, resource_manager,
                                                    memory_manager);
    shader_cache = std::make_unique<VKPipelineCache>(*this, device_handler);
    buffer_cache = std::make_unique<VKBufferCache>(*this, resource_manager, device_handler,
                                                   memory_manager, sched, STREAM_BUFFER_SIZE);
    renderpass_cache = std::make_unique<VKRenderPassCache>(device_handler);

    const vk::SamplerCreateInfo sampler_ci(
        {}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest,
        vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat, 0.f, false, 0.f, false, vk::CompareOp::eNever, 0.f, 0.f,
        vk::BorderColor::eFloatOpaqueWhite, false);
    dummy_sampler = device.createSamplerUnique(sampler_ci);
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

    // Get renderpass parameters and get a draw renderpass from the cache
    const RenderPassParams renderpass_params = GetRenderPassParams();
    const vk::RenderPass renderpass = renderpass_cache->GetDrawRenderPass(renderpass_params);

    SyncDepthStencil(params);
    SyncInputAssembly(params);
    SyncColorBlending(params);
    SyncViewportState(params);
    SyncRasterizerState(params);

    // Calculate buffer size.
    std::size_t buffer_size = CalculateVertexArraysSize();
    if (is_indexed) {
        buffer_size = Common::AlignUp<std::size_t>(buffer_size, 4) + CalculateIndexBufferSize();
    }
    buffer_size += Maxwell::MaxConstBuffers * (MaxConstbufferSize + uniform_buffer_alignment);

    buffer_cache->Reserve(buffer_size);

    SetupVertexArrays(params, state);
    if (is_indexed) {
        SetupIndexBuffer(state);
    }

    Pipeline pipeline = shader_cache->GetPipeline(params, renderpass_params, renderpass);

    const vk::CommandBuffer cmdbuf = sched.BeginRecord();

    for (std::size_t stage = 0; stage < pipeline.shaders.size(); ++stage) {
        const Shader& shader = pipeline.shaders[stage];
        if (shader == nullptr)
            continue;

        const auto descriptor_set = shader->CommitDescriptorSet(fence);
        SetupConstBuffers(state, shader, static_cast<Maxwell::ShaderStage>(stage), descriptor_set);
        SetupTextures(state, shader, static_cast<Maxwell::ShaderStage>(stage), cmdbuf,
                      descriptor_set);
        state.AssignDescriptorSet(static_cast<u32>(stage), descriptor_set);
    }

    const FramebufferInfo fb_info = ConfigureFramebuffers(fence, cmdbuf, renderpass);
    const Surface& color_surface = fb_info.color_surfaces[0];
    const Surface& zeta_surface = fb_info.zeta_surface;

    state.UpdateDescriptorSets(device);

    buffer_cache->Send(fence, cmdbuf);

    color_surface->Transition(cmdbuf, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eColorAttachmentRead |
                                  vk::AccessFlagBits::eColorAttachmentWrite);

    if (zeta_surface != nullptr) {
        zeta_surface->Transition(cmdbuf, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                 vk::PipelineStageFlagBits::eLateFragmentTests,
                                 vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                     vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    }

    const vk::RenderPassBeginInfo renderpass_bi(
        renderpass, fb_info.framebuffer, {{0, 0}, {fb_info.width, fb_info.height}}, 0, nullptr);
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

    VKFence& fence = sched.BeginPass(true);
    const vk::CommandBuffer cmdbuf = sched.BeginRecord();

    if (use_color) {
        Surface color_surface =
            res_cache->GetColorBufferSurface(regs.clear_buffers.RT.Value(), cmdbuf, false);

        color_surface->Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  vk::AccessFlagBits::eTransferWrite);

        const vk::ClearColorValue clear(std::array<float, 4>{
            regs.clear_color[0], regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]});
        cmdbuf.clearColorImage(
            color_surface->GetHandle(), vk::ImageLayout::eTransferDstOptimal, clear,
            {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)});
    }
    if (use_depth || use_stencil) {
        Surface zeta_surface = res_cache->GetDepthBufferSurface(cmdbuf, false);

        zeta_surface->Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal,
                                 vk::PipelineStageFlagBits::eTransfer,
                                 vk::AccessFlagBits::eTransferWrite);

        const vk::ClearDepthStencilValue clear(regs.clear_depth,
                                               static_cast<u32>(regs.clear_stencil));
        cmdbuf.clearDepthStencilImage(
            zeta_surface->GetHandle(), vk::ImageLayout::eTransferDstOptimal, clear,
            {vk::ImageSubresourceRange(zeta_surface->GetAspectMask(), 0, 1, 0, 1)});
    }

    sched.EndRecord(cmdbuf);
    sched.EndPass();
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::InvalidateRegion(Tegra::GPUVAddr addr, u64 size) {
    res_cache->InvalidateRegion(addr, size);
    shader_cache->InvalidateRegion(addr, size);
    buffer_cache->InvalidateRegion(addr, size);
}

void RasterizerVulkan::FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) {
    FlushRegion(addr, size);
    InvalidateRegion(addr, size);
}

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

template <typename Map, typename Interval>
static constexpr auto RangeFromInterval(Map& map, const Interval& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

void RasterizerVulkan::UpdatePagesCachedCount(Tegra::GPUVAddr addr, u64 size, int delta) {
    const u64 page_start{addr >> Memory::PAGE_BITS};
    const u64 page_end{(addr + size + Memory::PAGE_SIZE - 1) >> Memory::PAGE_BITS};

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = CachedPageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0)
        cached_pages.add({pages_interval, delta});

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const VAddr interval_start_addr = boost::icl::first(interval) << Memory::PAGE_BITS;
        const VAddr interval_end_addr = boost::icl::last_next(interval) << Memory::PAGE_BITS;
        const u64 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta)
            Memory::RasterizerMarkRegionCached(interval_start_addr, interval_size, true);
        else if (delta < 0 && count == -delta)
            Memory::RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
        else
            ASSERT(count >= 0);
    }

    if (delta < 0)
        cached_pages.add({pages_interval, delta});
}

FramebufferInfo RasterizerVulkan::ConfigureFramebuffers(VKFence& fence, vk::CommandBuffer cmdbuf,
                                                        vk::RenderPass renderpass,
                                                        bool using_color_fb, bool using_zeta_fb,
                                                        bool preserve_contents) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    Surface color_surface, zeta_surface;
    if (using_color_fb) {
        color_surface = res_cache->GetColorBufferSurface(0, cmdbuf, preserve_contents);
    }
    if (using_zeta_fb) {
        zeta_surface = res_cache->GetDepthBufferSurface(cmdbuf, preserve_contents);
    }

    FramebufferCacheKey fbkey;
    fbkey.renderpass = renderpass;
    fbkey.width = std::numeric_limits<u32>::max();
    fbkey.height = std::numeric_limits<u32>::max();

    if (color_surface != nullptr) {
        fbkey.views.Push(color_surface->GetImageView());
        fbkey.width = std::min(fbkey.width, color_surface->GetSurfaceParams().width);
        fbkey.height = std::min(fbkey.height, color_surface->GetSurfaceParams().height);
    }
    if (zeta_surface != nullptr) {
        fbkey.views.Push(zeta_surface->GetImageView());
        fbkey.width = std::min(fbkey.width, zeta_surface->GetSurfaceParams().width);
        fbkey.height = std::min(fbkey.height, zeta_surface->GetSurfaceParams().height);
    }

    const auto [fbentry, is_cache_miss] = framebuffer_cache.try_emplace(fbkey);
    auto& framebuffer = fbentry->second;
    if (is_cache_miss) {
        const vk::FramebufferCreateInfo framebuffer_ci(
            {}, fbkey.renderpass, static_cast<u32>(fbkey.views.Size()), fbkey.views.Data(),
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
    info.width = fbkey.width;
    info.height = fbkey.height;
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

void RasterizerVulkan::SetupConstBuffers(PipelineState& state, const Shader& shader,
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

        auto [write, buffer_info] = state.CaptureDescriptorWriteBuffer();
        buffer_info =
            vk::DescriptorBufferInfo(buffer_handle, offset, static_cast<vk::DeviceSize>(size));
        write = vk::WriteDescriptorSet(descriptor_set, used_buffer.GetBinding(), 0, 1,
                                       vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info,
                                       nullptr);
    }
}

void RasterizerVulkan::SetupTextures(PipelineState& state, const Shader& shader,
                                     Maxwell::ShaderStage stage, vk::CommandBuffer cmdbuf,
                                     vk::DescriptorSet descriptor_set) {
    const auto& gpu = Core::System::GetInstance().GPU();
    const auto& maxwell3d = gpu.Maxwell3D();
    const auto& entries = shader->GetEntries().samplers;

    for (u32 bindpoint = 0; bindpoint < entries.size(); ++bindpoint) {
        const auto& entry = entries[bindpoint];

        const auto texture = maxwell3d.GetStageTexture(entry.GetStage(), entry.GetOffset());
        UNIMPLEMENTED_IF(!texture.enabled);

        Surface surface = res_cache->GetTextureSurface(cmdbuf, texture, entry);
        UNIMPLEMENTED_IF(surface == nullptr);

        constexpr auto pipeline_stage = vk::PipelineStageFlagBits::eAllGraphics;
        surface->Transition(cmdbuf, vk::ImageLayout::eShaderReadOnlyOptimal, pipeline_stage,
                            vk::AccessFlagBits::eShaderRead);

        const auto [write, image_info] = state.CaptureDescriptorWriteImage();
        image_info = vk::DescriptorImageInfo(*dummy_sampler, surface->GetImageView(),
                                             vk::ImageLayout::eShaderReadOnlyOptimal);
        write = vk::WriteDescriptorSet(descriptor_set, entry.GetBinding(), 0, 1,
                                       vk::DescriptorType::eCombinedImageSampler, &image_info,
                                       nullptr, nullptr);
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

RenderPassParams RasterizerVulkan::GetRenderPassParams() const {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    RenderPassParams renderpass_params;
    if (renderpass_params.has_zeta = regs.zeta_enable) {
        renderpass_params.zeta_component_type =
            VideoCore::Surface::ComponentTypeFromDepthFormat(regs.zeta.format);
        renderpass_params.zeta_pixel_format =
            VideoCore::Surface::PixelFormatFromDepthFormat(regs.zeta.format);
    }

    // TODO(Rodrigo): Support multiple attachments
    RenderPassParams::ColorAttachment attachment;
    attachment.index = 0;
    attachment.pixel_format =
        VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[0].format);
    attachment.component_type =
        VideoCore::Surface::ComponentTypeFromRenderTarget(regs.rt[0].format);
    renderpass_params.color_map.Push(attachment);

    return renderpass_params;
}

void RasterizerVulkan::SyncDepthStencil(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    auto& ds = params.depth_stencil;

    ds.depth_test_function = regs.depth_test_func;
    ds.depth_test_enable = regs.depth_test_enable == 1;
    ds.depth_write_enable = regs.depth_write_enabled == 1;

    ds.stencil_enable = regs.stencil_enable == 1;

    auto& front = ds.front_stencil;
    front.test_func = regs.stencil_front_func_func;
    front.test_ref = regs.stencil_front_func_ref;
    front.test_mask = regs.stencil_front_func_mask;
    front.action_stencil_fail = regs.stencil_front_op_fail;
    front.action_depth_fail = regs.stencil_front_op_zfail;
    front.action_depth_pass = regs.stencil_front_op_zpass;
    front.write_mask = regs.stencil_front_mask;

    if (regs.stencil_two_side_enable) {
        auto& back = ds.back_stencil;
        back.test_func = regs.stencil_back_func_func;
        back.test_ref = regs.stencil_back_func_ref;
        back.test_mask = regs.stencil_back_func_mask;
        back.action_stencil_fail = regs.stencil_back_op_fail;
        back.action_depth_fail = regs.stencil_back_op_zfail;
        back.action_depth_pass = regs.stencil_back_op_zpass;
        back.write_mask = regs.stencil_back_mask;
    }

    // TODO(Rodrigo): Read from registers, luckily this is core in Vulkan unlike OpenGL
    ds.depth_bounds_enable = false;
    ds.depth_bounds_min = 0.f;
    ds.depth_bounds_max = 0.f;
}

void RasterizerVulkan::SyncInputAssembly(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    auto& ia = params.input_assembly;

    ia.topology = regs.draw.topology;
    // ia.primitive_restart_enable = ;
}

void RasterizerVulkan::SyncColorBlending(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    auto& cd = params.color_blending;

    cd.blend_constants = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                          regs.blend_color.a};
    cd.independent_blend = regs.independent_blend_enable == 1;

    if (!cd.independent_blend) {
        auto& blend = cd.attachments[0];
        const auto& src = regs.blend;

        blend.enable = src.enable[0] != 0;
        if (blend.enable) {
            blend.rgb_equation = src.equation_rgb;
            blend.src_rgb_func = src.factor_source_rgb;
            blend.dst_rgb_func = src.factor_dest_rgb;
            blend.a_equation = src.equation_a;
            blend.src_a_func = src.factor_source_a;
            blend.dst_a_func = src.factor_dest_a;
            // TODO(Rodrigo): Read from registers
            blend.components = {true, true, true, true};
        }
        return;
    }

    for (std::size_t i = 0; i < Tegra::Engines::Maxwell3D::Regs::NumRenderTargets; i++) {
        auto& blend = cd.attachments[i];
        const auto& src = regs.independent_blend[i];

        blend.enable = regs.blend.enable[i] != 0;
        if (!blend.enable)
            continue;
        blend.rgb_equation = src.equation_rgb;
        blend.src_rgb_func = src.factor_source_rgb;
        blend.dst_rgb_func = src.factor_dest_rgb;
        blend.a_equation = src.equation_a;
        blend.src_a_func = src.factor_source_a;
        blend.dst_a_func = src.factor_dest_a;
        // TODO(Rodrigo): Read from registers
        blend.components = {true, true, true, true};
    }
}

void RasterizerVulkan::SyncViewportState(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    auto& vs = params.viewport_state;

    vs.width = static_cast<float>(regs.viewports[0].width);
    vs.height = static_cast<float>(regs.viewports[0].height);
}

void RasterizerVulkan::SyncRasterizerState(PipelineParams& params) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    auto& rs = params.rasterizer;

    rs.cull_enable = regs.cull.enabled != 0;
    rs.cull_face = regs.cull.cull_face;

    rs.front_face = regs.cull.front_face;

    const bool flip_triangles{regs.screen_y_control.triangle_rast_flip == 0 ||
                              regs.viewport_transform[0].scale_y < 0.0f};
    // If the GPU is configured to flip the rasterizer triangles, then we need to flip the front and
    // back.
    if (flip_triangles) {
        if (rs.front_face == Maxwell::Cull::FrontFace::CounterClockWise)
            rs.front_face = Maxwell::Cull::FrontFace::ClockWise;
        else if (rs.front_face == Maxwell::Cull::FrontFace::ClockWise)
            rs.front_face = Maxwell::Cull::FrontFace::CounterClockWise;
    }
}

} // namespace Vulkan