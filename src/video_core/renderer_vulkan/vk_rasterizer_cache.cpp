// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <vulkan/vulkan.hpp>
#include "core/core.h"
#include "core/memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/morton.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/surface.h"
#include "video_core/textures/astc.h"

namespace Vulkan {

using VideoCore::MortonSwizzle;
using VideoCore::MortonSwizzleMode;
using VideoCore::Surface::ComponentTypeFromDepthFormat;
using VideoCore::Surface::ComponentTypeFromRenderTarget;
using VideoCore::Surface::ComponentTypeFromTexture;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::PixelFormatFromTextureFormat;
using VideoCore::Surface::SurfaceTargetFromTextureType;

static vk::ImageType SurfaceTargetToImageVK(SurfaceTarget target) {
    switch (target) {
    case SurfaceTarget::Texture2D:
        return vk::ImageType::e2D;
    }
    UNIMPLEMENTED_MSG("Unimplemented texture target={}", static_cast<u32>(target));
    return vk::ImageType::e2D;
}

static vk::ImageViewType SurfaceTargetToImageViewVK(SurfaceTarget target) {
    switch (target) {
    case SurfaceTarget::Texture2D:
        return vk::ImageViewType::e2D;
    }
    UNIMPLEMENTED_MSG("Unimplemented texture target={}", static_cast<u32>(target));
    return vk::ImageViewType::e2D;
}

static vk::ImageAspectFlags PixelFormatToImageAspect(PixelFormat pixel_format) {
    if (pixel_format < PixelFormat::MaxColorFormat) {
        return vk::ImageAspectFlagBits::eColor;
    } else if (pixel_format < PixelFormat::MaxDepthFormat) {
        return vk::ImageAspectFlagBits::eDepth;
    } else if (pixel_format < PixelFormat::MaxDepthStencilFormat) {
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    } else {
        UNREACHABLE_MSG("Invalid pixel format={}", static_cast<u32>(pixel_format));
        return vk::ImageAspectFlagBits::eColor;
    }
}

/*static*/ SurfaceParams SurfaceParams::CreateForDepthBuffer(
    u32 zeta_width, u32 zeta_height, Tegra::GPUVAddr zeta_address, Tegra::DepthFormat format,
    u32 block_width, u32 block_height, u32 block_depth,
    Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type) {

    SurfaceParams params{};
    params.is_tiled = type == Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout::BlockLinear;
    params.block_width = 1 << std::min(block_width, 5U);
    params.block_height = 1 << std::min(block_height, 5U);
    params.block_depth = 1 << std::min(block_depth, 5U);
    params.tile_width_spacing = 1;
    params.pixel_format = PixelFormatFromDepthFormat(format);
    params.component_type = ComponentTypeFromDepthFormat(format);
    params.type = GetFormatType(params.pixel_format);
    // params.srgb_conversion = false;
    params.width = zeta_width;
    params.height = zeta_height;
    params.unaligned_height = zeta_height;
    params.target = SurfaceTarget::Texture2D;
    params.depth = 1;
    // params.max_mip_level = 1;
    // params.is_layered = false;
    // params.rt = {};

    params.InitCacheParameters(zeta_address);

    return params;
}

/*static*/ SurfaceParams SurfaceParams::CreateForFramebuffer(std::size_t index) {
    const auto& config{Core::System::GetInstance().GPU().Maxwell3D().regs.rt[index]};
    SurfaceParams params{};

    params.is_tiled =
        config.memory_layout.type == Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout::BlockLinear;
    params.block_width = 1 << config.memory_layout.block_width;
    params.block_height = 1 << config.memory_layout.block_height;
    params.block_depth = 1 << config.memory_layout.block_depth;
    params.tile_width_spacing = 1;
    params.pixel_format = PixelFormatFromRenderTargetFormat(config.format);
    // params.srgb_conversion = config.format == Tegra::RenderTargetFormat::BGRA8_SRGB ||
    //                         config.format == Tegra::RenderTargetFormat::RGBA8_SRGB;
    params.component_type = ComponentTypeFromRenderTarget(config.format);
    params.type = GetFormatType(params.pixel_format);
    params.width = config.width;
    params.height = config.height;
    params.unaligned_height = config.height;
    params.target = SurfaceTarget::Texture2D;
    params.depth = 1;
    // params.max_mip_level = 0;
    // params.is_layered = false;

    // Render target specific parameters, not used for caching
    // params.rt.index = static_cast<u32>(index);
    // params.rt.array_mode = config.array_mode;
    // params.rt.layer_stride = config.layer_stride;
    // params.rt.volume = config.volume;
    // params.rt.base_layer = config.base_layer;

    params.InitCacheParameters(config.Address());

    return params;
}

/**
 * Helper function to perform software conversion (as needed) when loading a buffer from Switch
 * memory. This is for Maxwell pixel formats that cannot be represented as-is in Vulkan or with
 * typical desktop GPUs.
 */
static void ConvertFormatAsNeeded_LoadVKBuffer(u8* data, PixelFormat pixel_format, u32 width,
                                               u32 height, u32 depth) {
    switch (pixel_format) {
    case PixelFormat::ASTC_2D_4X4:
    case PixelFormat::ASTC_2D_8X8:
    case PixelFormat::ASTC_2D_8X5:
    case PixelFormat::ASTC_2D_5X4:
    case PixelFormat::ASTC_2D_5X5:
    case PixelFormat::ASTC_2D_4X4_SRGB:
    case PixelFormat::ASTC_2D_8X8_SRGB:
    case PixelFormat::ASTC_2D_8X5_SRGB:
    case PixelFormat::ASTC_2D_5X4_SRGB:
    case PixelFormat::ASTC_2D_5X5_SRGB:
    case PixelFormat::ASTC_2D_10X8:
    case PixelFormat::ASTC_2D_10X8_SRGB: {
        UNIMPLEMENTED();
        break;
    }
    case PixelFormat::S8Z24:
        UNIMPLEMENTED();
        break;
    }
}

void SurfaceParams::InitCacheParameters(Tegra::GPUVAddr gpu_addr_) {
    auto& memory_manager{Core::System::GetInstance().GPU().MemoryManager()};
    const auto cpu_addr{memory_manager.GpuToCpuAddress(gpu_addr_)};

    addr = cpu_addr ? *cpu_addr : 0;
    gpu_addr = gpu_addr_;
    size_in_bytes = SizeInBytesRaw();

    if (IsPixelFormatASTC(pixel_format)) {
        // ASTC is uncompressed in software, in emulated as RGBA8
        size_in_bytes_vk = width * height * depth * 4;
    } else {
        size_in_bytes_vk = SizeInBytesVK();
    }
}

vk::ImageCreateInfo SurfaceParams::CreateInfo() const {
    constexpr u32 mipmaps = 1;
    constexpr u32 array_layers = 1;
    constexpr auto sample_count = vk::SampleCountFlagBits::e1;
    constexpr auto tiling = vk::ImageTiling::eOptimal;

    const bool is_zeta = pixel_format >= PixelFormat::FirstDepthStencilFormat &&
                         pixel_format <= PixelFormat::MaxDepthStencilFormat;
    const auto usage = (is_zeta ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                                : vk::ImageUsageFlagBits::eColorAttachment) |
                       vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eTransferSrc;
    return {{},
            SurfaceTargetToImageVK(target),
            MaxwellToVK::SurfaceFormat(pixel_format, component_type),
            {width, height, depth},
            mipmaps,
            array_layers,
            sample_count,
            tiling,
            usage,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined};
}

static void SwizzleFunc(const MortonSwizzleMode& mode, const SurfaceParams& params, u8* vk_buffer,
                        u32 mip_level) {
    UNIMPLEMENTED_IF(params.depth != 1);

    const u64 offset = 0;
    MortonSwizzle(mode, params.pixel_format, params.width, params.block_height, params.height,
                  params.block_depth, params.depth, params.tile_width_spacing, vk_buffer, 0,
                  params.addr + offset);
}

CachedSurface::CachedSurface(VKDevice& device_handler, VKResourceManager& resource_manager,
                             VKMemoryManager& memory_manager, const SurfaceParams& params)
    : VKImage(device_handler.GetLogical(), params.CreateInfo()),
      device(device_handler.GetLogical()), resource_manager(resource_manager),
      memory_manager(memory_manager), params(params), cached_size_in_bytes(params.size_in_bytes),
      buffer_size(std::max(params.size_in_bytes, params.size_in_bytes_vk)) {

    image = GetHandle();
    image_commit = memory_manager.Commit(device.getImageMemoryRequirements(image), false);
    device.bindImageMemory(image, image_commit->GetMemory(), image_commit->GetOffset());

    const vk::BufferCreateInfo buffer_ci({}, buffer_size,
                                         vk::BufferUsageFlagBits::eTransferDst |
                                             vk::BufferUsageFlagBits::eTransferSrc,
                                         vk::SharingMode::eExclusive, 0, nullptr);
    buffer = device.createBufferUnique(buffer_ci);
    buffer_commit = memory_manager.Commit(device.getBufferMemoryRequirements(*buffer), true);
    device.bindBufferMemory(*buffer, buffer_commit->GetMemory(), buffer_commit->GetOffset());
    vk_buffer = buffer_commit->GetData();

    auto& emu_memory_manager{Core::System::GetInstance().GPU().MemoryManager()};
    const u64 max_size{emu_memory_manager.GetRegionEnd(params.gpu_addr) - params.gpu_addr};
    if (cached_size_in_bytes > max_size) {
        LOG_ERROR(HW_GPU, "Surface size {} exceeds region size {}", params.size_in_bytes, max_size);
        cached_size_in_bytes = max_size;
    }

    vk_format = MaxwellToVK::SurfaceFormat(params.pixel_format, params.component_type);
    vk_image_aspect = PixelFormatToImageAspect(params.pixel_format);
}

CachedSurface::~CachedSurface() {
    memory_manager.Free(image_commit);
    memory_manager.Free(buffer_commit);
}

vk::ImageView CachedSurface::GetImageView() {
    if (image_view) {
        return *image_view;
    }

    const auto access = [&]() -> vk::ImageAspectFlags {
        if (params.pixel_format <= PixelFormat::MaxColorFormat) {
            return vk::ImageAspectFlagBits::eColor;
        } else if (params.pixel_format <= PixelFormat::MaxDepthFormat) {
            return vk::ImageAspectFlagBits::eDepth;
        } else if (params.pixel_format <= PixelFormat::MaxDepthStencilFormat) {
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        } else {
            UNREACHABLE_MSG("Invalid pixel format={}", static_cast<u32>(params.pixel_format));
            return vk::ImageAspectFlagBits::eColor;
        }
    }();

    const vk::ImageViewCreateInfo image_view_ci(
        {}, image, SurfaceTargetToImageViewVK(params.target),
        MaxwellToVK::SurfaceFormat(params.pixel_format, params.component_type),
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {access, 0, 1, 0, 1});
    image_view = device.createImageViewUnique(image_view_ci);
    return *image_view;
}

void CachedSurface::LoadVKBuffer() {
    if (params.is_tiled) {
        ASSERT_MSG(params.block_width == 1, "Block width is defined as {} on texture type {}",
                   params.block_width, static_cast<u32>(params.target));
        SwizzleFunc(MortonSwizzleMode::MortonToLinear, params, vk_buffer, 0);
    } else {
        std::memcpy(vk_buffer, Memory::GetPointer(params.addr), params.size_in_bytes_vk);
    }
    ConvertFormatAsNeeded_LoadVKBuffer(vk_buffer, params.pixel_format, params.width, params.height,
                                       params.depth);
}

void CachedSurface::FlushVKBuffer() {
    UNIMPLEMENTED();
}

void CachedSurface::UploadVKTexture(vk::CommandBuffer cmdbuf) {
    if (params.type == SurfaceType::Fill)
        return;

    Transition(cmdbuf, vk_image_aspect, vk::ImageLayout::eTransferDstOptimal,
               vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);

    const vk::BufferImageCopy copy(0, 0, 0, {vk_image_aspect, 0, 0, 1}, {0, 0, 0},
                                   {params.width, params.height, params.depth});
    if (vk_image_aspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        vk::BufferImageCopy depth = copy;
        vk::BufferImageCopy stencil = copy;
        depth.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
        stencil.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;

        cmdbuf.copyBufferToImage(*buffer, image, vk::ImageLayout::eTransferDstOptimal,
                                 {depth, stencil});
    } else {
        cmdbuf.copyBufferToImage(*buffer, image, vk::ImageLayout::eTransferDstOptimal, {copy});
    }
}

VKRasterizerCache::VKRasterizerCache(RasterizerVulkan& rasterizer, VKDevice& device_handler,
                                     VKResourceManager& resource_manager,
                                     VKMemoryManager& memory_manager)
    : RasterizerCache{rasterizer}, device_handler{device_handler},
      resource_manager{resource_manager}, memory_manager{memory_manager} {}

VKRasterizerCache::~VKRasterizerCache() = default;

Surface VKRasterizerCache::GetDepthBufferSurface(vk::CommandBuffer cmdbuf, bool preserve_contents) {
    const auto& regs{Core::System::GetInstance().GPU().Maxwell3D().regs};
    if (!regs.zeta.Address() || !regs.zeta_enable) {
        return {};
    }

    SurfaceParams depth_params{SurfaceParams::CreateForDepthBuffer(
        regs.zeta_width, regs.zeta_height, regs.zeta.Address(), regs.zeta.format,
        regs.zeta.memory_layout.block_width, regs.zeta.memory_layout.block_height,
        regs.zeta.memory_layout.block_depth, regs.zeta.memory_layout.type)};

    return GetSurface(depth_params, cmdbuf, preserve_contents);
}

Surface VKRasterizerCache::GetColorBufferSurface(std::size_t index, vk::CommandBuffer cmdbuf,
                                                 bool preserve_contents) {
    const auto& regs{Core::System::GetInstance().GPU().Maxwell3D().regs};
    ASSERT(index < Tegra::Engines::Maxwell3D::Regs::NumRenderTargets);

    if (index >= regs.rt_control.count) {
        return {};
    }
    if (regs.rt[index].Address() == 0 || regs.rt[index].format == Tegra::RenderTargetFormat::NONE) {
        return {};
    }

    return GetSurface(SurfaceParams::CreateForFramebuffer(index), cmdbuf, preserve_contents);
}

Surface VKRasterizerCache::TryFindFramebufferSurface(VAddr addr) const {
    return TryGet(addr);
}

void VKRasterizerCache::LoadSurface(const Surface& surface, vk::CommandBuffer cmdbuf) {
    surface->LoadVKBuffer();
    surface->UploadVKTexture(cmdbuf);
    surface->MarkAsModified(false, *this);
}

Surface VKRasterizerCache::GetSurface(const SurfaceParams& params, vk::CommandBuffer cmdbuf,
                                      bool preserve_contents) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return {};
    }

    // Look up surface in the cache based on address
    Surface surface{TryGet(params.addr)};
    if (surface) {
        if (surface->GetSurfaceParams().IsCompatibleSurface(params)) {
            // Use the cached surface as-is
            return surface;
        } else if (preserve_contents) {
            // If surface parameters changed and we care about keeping the previous data, recreate
            // the surface from the old one
            Surface new_surface{RecreateSurface(surface, params)};
            Unregister(surface);
            Register(new_surface);
            return new_surface;
        } else {
            // Delete the old surface before creating a new one to prevent collisions.
            Unregister(surface);
        }
    }

    // No cached surface found - get a new one
    surface = GetUncachedSurface(params);
    Register(surface);

    // Only load surface from memory if we care about the contents
    if (preserve_contents) {
        LoadSurface(surface, cmdbuf);
    }

    return surface;
}

Surface VKRasterizerCache::GetUncachedSurface(const SurfaceParams& params) {
    Surface surface{TryGetReservedSurface(params)};
    if (!surface) {
        // No reserved surface available, create a new one and reserve it
        surface = std::make_shared<CachedSurface>(device_handler, resource_manager, memory_manager,
                                                  params);
        ReserveSurface(surface);
    }
    return surface;
}

Surface VKRasterizerCache::RecreateSurface(const Surface& old_surface,
                                           const SurfaceParams& new_params) {
    UNIMPLEMENTED();
    return {};
}

void VKRasterizerCache::ReserveSurface(const Surface& surface) {
    const auto& surface_reserve_key{SurfaceReserveKey::Create(surface->GetSurfaceParams())};
    surface_reserve[surface_reserve_key] = surface;
}

Surface VKRasterizerCache::TryGetReservedSurface(const SurfaceParams& params) {
    const auto& surface_reserve_key{SurfaceReserveKey::Create(params)};
    auto search{surface_reserve.find(surface_reserve_key)};
    if (search != surface_reserve.end()) {
        return search->second;
    }
    return {};
}

} // namespace Vulkan