// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>
#include <vulkan/vulkan.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_cache.h"
#include "video_core/renderer_vulkan/vk_sync.h"

#pragma optimize("", off)

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct FramebufferInfo {
    vk::RenderPass renderpass;
    vk::Framebuffer framebuffer;
    std::array<Surface, Maxwell::NumRenderTargets> color_surfaces;
};

class PipelineState {
public:
    void SetPrimitiveTopology(vk::PrimitiveTopology primitive_topology) {
        this->primitive_topology = primitive_topology;
    }

    void AddStage(vk::ShaderStageFlagBits stage, vk::ShaderModule shader_module,
                  vk::DescriptorSetLayout set_layout, vk::DescriptorSet descriptor_set) {
        const u32 index = stages_count++;
        ASSERT(index < static_cast<u32>(stages.size()));

        set_layouts[index] = set_layout;
        stages[index] = {{}, stage, shader_module, "main", nullptr};

        // A null descriptor set means that the stage is not using descriptors, it must be skipped.
        if (descriptor_set) {
            descriptor_sets[descriptor_sets_count++] = descriptor_set;
        }
    }

    void AddVertexAttribute(const vk::VertexInputAttributeDescription& attribute_description) {
        const u32 index = vertex_attributes_count++;
        ASSERT(index < static_cast<u32>(vertex_attributes.size()));
        vertex_attributes[index] = attribute_description;
    }

    void AddVertexBinding(const vk::VertexInputBindingDescription binding_description,
                          vk::Buffer buffer, vk::DeviceSize offset) {
        const u32 index = vertex_bindings_count++;
        ASSERT(index < static_cast<u32>(vertex_bindings.size()));
        vertex_bindings[index] = binding_description;
        vertex_buffers[index] = buffer;
        vertex_offsets[index] = offset;
    }

    void SetRenderPass(vk::RenderPass renderpass) {
        this->renderpass = renderpass;
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

    vk::Pipeline CreatePipeline(VulkanFence& fence, VulkanResourceManager& resource_manager) {
        const vk::PipelineLayoutCreateInfo layout_ci({}, stages_count, set_layouts.data(), 0,
                                                     nullptr);
        layout = resource_manager.CreatePipelineLayout(fence, layout_ci);

        const vk::PipelineVertexInputStateCreateInfo vertex_input(
            {}, vertex_bindings_count, vertex_bindings.data(), vertex_attributes_count,
            vertex_attributes.data());

        const vk::PipelineInputAssemblyStateCreateInfo input_assembly({}, primitive_topology, {});

        const vk::Viewport viewport(0.f, 0.f, 1280.f, 720.f, 0.f, 1.f);
        const vk::Rect2D scissor({0, 0}, {1280, 720});
        const vk::PipelineViewportStateCreateInfo viewport_state({}, 1, &viewport, 1, &scissor);

        const vk::PipelineRasterizationStateCreateInfo rasterizer(
            {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
            vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

        const vk::PipelineMultisampleStateCreateInfo multisampling(
            {}, vk::SampleCountFlagBits::e1, false, 0.0f, nullptr, false, false);

        const vk::PipelineColorBlendAttachmentState color_blend_attachment(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        const vk::PipelineColorBlendStateCreateInfo color_blending(
            {}, false, vk::LogicOp::eCopy, 1, &color_blend_attachment, {0.0f, 0.0f, 0.0f, 0.0f});

        const vk::GraphicsPipelineCreateInfo create_info(
            {}, stages_count, stages.data(), &vertex_input, &input_assembly, {}, &viewport_state,
            &rasterizer, &multisampling, nullptr, &color_blending, nullptr, layout, renderpass, 0);
        return resource_manager.CreateGraphicsPipeline(fence, create_info);
    }

    void BindDescriptors(vk::CommandBuffer cmdbuf) const {
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0,
                                  descriptor_sets_count, descriptor_sets.data(), 0, nullptr);
    }

    void BindVertexBuffers(vk::CommandBuffer cmdbuf) const {
        // TODO(Rodrigo): Sort data and bindings to do this in a single call.
        for (u32 index = 0; index < vertex_bindings_count; ++index) {
            cmdbuf.bindVertexBuffers(index, {vertex_buffers[index]}, {vertex_offsets[index]});
        }
    }

private:
    static constexpr std::size_t MAX_DESCRIPTOR_BINDINGS =
        Maxwell::MaxShaderStage * Maxwell::MaxConstBuffers;

    vk::PrimitiveTopology primitive_topology{};

    u32 vertex_attributes_count{};
    std::array<vk::VertexInputAttributeDescription, Maxwell::NumVertexAttributes> vertex_attributes;

    u32 vertex_bindings_count{};
    std::array<vk::VertexInputBindingDescription, Maxwell::NumVertexArrays> vertex_bindings;
    std::array<vk::Buffer, Maxwell::NumVertexArrays> vertex_buffers;
    std::array<vk::DeviceSize, Maxwell::NumVertexArrays> vertex_offsets;

    u32 stages_count{};
    std::array<vk::PipelineShaderStageCreateInfo, Maxwell::MaxShaderStage> stages;
    std::array<vk::DescriptorSetLayout, Maxwell::MaxShaderStage> set_layouts;

    u32 descriptor_sets_count{};
    std::array<vk::DescriptorSet, Maxwell::MaxShaderStage> descriptor_sets;

    u32 descriptor_bindings_count{};
    std::array<vk::WriteDescriptorSet, MAX_DESCRIPTOR_BINDINGS> descriptor_bindings;
    std::array<vk::DescriptorBufferInfo, MAX_DESCRIPTOR_BINDINGS> buffer_infos;

    vk::RenderPass renderpass{};

    vk::PipelineLayout layout{};
};

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& renderer,
                                   VulkanScreenInfo& screen_info, VulkanDevice& device_handler,
                                   VulkanResourceManager& resource_manager,
                                   VulkanMemoryManager& memory_manager, VulkanSync& sync)
    : VideoCore::RasterizerInterface(), render_window(renderer), screen_info(screen_info),
      device_handler(device_handler), device(device_handler.GetLogical()),
      graphics_queue(device_handler.GetGraphicsQueue()), resource_manager(resource_manager),
      memory_manager(memory_manager), sync(sync),
      uniform_buffer_alignment(device_handler.GetUniformBufferAlignment()) {

    res_cache =
        std::make_unique<VulkanRasterizerCache>(device_handler, resource_manager, memory_manager);
    shader_cache = std::make_unique<VulkanShaderCache>(device_handler);
    buffer_cache = std::make_unique<VulkanBufferCache>(resource_manager, device_handler,
                                                       memory_manager, STREAM_BUFFER_SIZE);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {
    if (accelerate_draw == AccelDraw::Disabled)
        return;

    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();
    const auto& regs = gpu.regs;

    const bool is_indexed = accelerate_draw == AccelDraw::Indexed;
    ASSERT_MSG(!is_indexed, "Unimplemented");

    VulkanFence& fence = sync.PrepareExecute(true);

    const FramebufferInfo fb_info = ConfigureFramebuffers(fence, true);
    const Surface& color_surface = fb_info.color_surfaces[0];

    const vk::PrimitiveTopology primitive_topology =
        MaxwellToVK::PrimitiveTopology(regs.draw.topology);

    PipelineState state;
    state.SetPrimitiveTopology(primitive_topology);
    state.SetRenderPass(fb_info.renderpass);

    // Calculate buffer size.
    std::size_t buffer_size = CalculateVertexArraysSize();
    buffer_size += Maxwell::MaxConstBuffers * (MaxConstbufferSize + uniform_buffer_alignment);

    buffer_cache->Reserve(buffer_size);

    SetupShaders(fence, state, primitive_topology);

    buffer_cache->Send(sync, fence);

    const vk::Pipeline pipeline = state.CreatePipeline(fence, resource_manager);
    state.UpdateDescriptorSets(device);

    const vk::CommandBuffer cmdbuf = sync.BeginRecord();

    color_surface->Transition(cmdbuf, vk::ImageAspectFlagBits::eColor,
                              vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eColorAttachmentWrite);

    const vk::RenderPassBeginInfo renderpass_bi(fb_info.renderpass, fb_info.framebuffer,
                                                {{0, 0}, {1280, 720}}, 0, nullptr);
    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline);
    {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        state.BindDescriptors(cmdbuf);
        state.BindVertexBuffers(cmdbuf);

        // TODO(Rodrigo): Implement indexed vertex buffers.
        const u32 vertex_count = regs.vertex_buffer.count;
        const u32 vertex_first = regs.vertex_buffer.first;

        cmdbuf.draw(vertex_count, 1, vertex_first, 0);
    }
    cmdbuf.endRenderPass();

    sync.EndRecord(cmdbuf);
    sync.Execute();
}

void RasterizerVulkan::Clear() {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;
    bool use_color{};
    bool use_depth{};
    bool use_stencil{};

    if (regs.clear_buffers.R || regs.clear_buffers.G || regs.clear_buffers.B ||
        regs.clear_buffers.A) {
        use_color = true;
    }
    if (regs.clear_buffers.Z) {
        //UNIMPLEMENTED_MSG("Depth clear");
    }
    if (regs.clear_buffers.S) {
        UNIMPLEMENTED_MSG("Stencil clear");
    }

    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    ASSERT_MSG(use_color, "Unimplemented");

    VulkanFence& fence = sync.PrepareExecute(true);

    const FramebufferInfo fb_info = ConfigureFramebuffers(fence, false);

    const Surface& color_surface = fb_info.color_surfaces[0];
    const auto color_params = color_surface->GetSurfaceParams();

    const vk::CommandBuffer cmdbuf = sync.BeginRecord();

    color_surface->Transition(cmdbuf, vk::ImageAspectFlagBits::eColor,
                              vk::ImageLayout::eColorAttachmentOptimal,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eColorAttachmentWrite);

    const vk::ClearValue clear_color(std::array<float, 4>{
        regs.clear_color[0], regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]});

    const vk::RenderPassBeginInfo renderpass_bi(fb_info.renderpass, fb_info.framebuffer,
                                                {{0, 0}, {color_params.width, color_params.height}},
                                                1, &clear_color);
    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline);
    cmdbuf.endRenderPass();

    sync.EndRecord(cmdbuf);
    sync.Execute();
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

FramebufferInfo RasterizerVulkan::ConfigureFramebuffers(VulkanFence& fence,
                                                        bool preserve_contents) {
    const auto& regs = Core::System::GetInstance().GPU().Maxwell3D().regs;

    Surface color_surface = res_cache->GetColorBufferSurface(regs.clear_buffers.RT.Value(), false);
    ASSERT(color_surface);
    const auto color_params = color_surface->GetSurfaceParams();

    const vk::AttachmentLoadOp load_op =
        preserve_contents ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;
    const vk::AttachmentDescription color_attachment(
        {}, color_surface->GetFormat(), vk::SampleCountFlagBits::e1, load_op,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eColorAttachmentOptimal);

    const vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);

    const vk::SubpassDescription subpass_description({}, vk::PipelineBindPoint::eGraphics, 0,
                                                     nullptr, 1, &color_attachment_ref, nullptr,
                                                     nullptr, 0, nullptr);

    const vk::SubpassDependency subpass_dependency(
        VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput, {},
        vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, {});

    const vk::RenderPassCreateInfo renderpass_ci({}, 1, &color_attachment, 1, &subpass_description,
                                                 1, &subpass_dependency);

    const vk::RenderPass renderpass = resource_manager.CreateRenderPass(fence, renderpass_ci);

    const vk::ImageViewCreateInfo image_view_ci = color_surface->GetImageViewCreateInfo(
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    const vk::ImageView image_view = resource_manager.CreateImageView(fence, image_view_ci);

    const vk::FramebufferCreateInfo framebuffer_ci({}, renderpass, 1, &image_view,
                                                   color_params.width, color_params.height, 1);
    const vk::Framebuffer framebuffer = resource_manager.CreateFramebuffer(fence, framebuffer_ci);

    FramebufferInfo info;
    info.renderpass = renderpass;
    info.framebuffer = framebuffer;
    info.color_surfaces[0] = color_surface;
    return info;
}

void RasterizerVulkan::SetupShaders(VulkanFence& fence, PipelineState& state,
                                    vk::PrimitiveTopology primitive_topology) {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();

    for (std::size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const auto& shader_config = gpu.regs.shader_config[index];
        const auto program{static_cast<Maxwell::ShaderProgram>(index)};

        // Skip stages that are not enabled
        if (!gpu.regs.IsShaderConfigEnabled(index)) {
            continue;
        }

        Shader shader = shader_cache->GetStageProgram(program);
        const vk::DescriptorSet descriptor_set = shader->CommitDescriptorSet(fence);

        // Stage indices are 0 - 5
        const auto stage = static_cast<Maxwell::ShaderStage>(index == 0 ? 0 : index - 1);
        if (stage == Maxwell::ShaderStage::Vertex) {
            SetupVertexArrays(state, shader->GetEntries().attributes);
        }

        state.AddStage(MaxwellToVK::ShaderStage(program), shader->GetHandle(primitive_topology),
                       shader->GetSetLayout(), descriptor_set);

        SetupConstBuffers(state, shader, static_cast<Maxwell::ShaderStage>(stage), descriptor_set);

        // When VertexA is enabled, we have dual vertex shaders
        if (program == Maxwell::ShaderProgram::VertexA) {
            // VertexB was combined with VertexA, so we skip the VertexB iteration
            index++;
        }
    }
}

void RasterizerVulkan::SetupVertexArrays(PipelineState& state, const std::set<u32>& attributes) {
    const auto& gpu = Core::System::GetInstance().GPU().Maxwell3D();
    const auto& regs = gpu.regs;

    for (u32 index = 0; index < static_cast<u32>(Maxwell::NumVertexAttributes); ++index) {
        const auto& attrib = regs.vertex_attrib_format[index];

        // Ignore invalid attributes and the ones not used by the vertex shader.
        if (!attrib.IsValid() || attributes.find(index) == attributes.end())
            continue;

        const auto& buffer = regs.vertex_array[attrib.buffer];
        LOG_TRACE(HW_GPU, "vertex attrib {}, count={}, size={}, type={}, offset={}, normalize={}",
                  index, attrib.ComponentCount(), attrib.SizeString(), attrib.TypeString(),
                  attrib.offset.Value(), attrib.IsNormalized());

        ASSERT(buffer.IsEnabled());

        state.AddVertexAttribute(
            {index, attrib.buffer, MaxwellToVK::VertexFormat(attrib), attrib.offset});
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

        state.AddVertexBinding({index, vertex_array.stride, vk::VertexInputRate::eVertex}, buffer,
                               offset);

        ASSERT_MSG(!regs.instanced_arrays.IsInstancingEnabled(index), "Unimplemented");
        ASSERT_MSG(vertex_array.divisor == 0, "Unimplemented");
    }
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

        // Align the actual size so it ends up being a multiple of vec4 to meet the OpenGL std140
        // UBO alignment requirements.
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

} // namespace Vulkan