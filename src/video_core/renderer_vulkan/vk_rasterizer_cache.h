// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <tuple>

#include <vulkan/vulkan.hpp>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_image.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace Vulkan {

class VulkanDevice;
class VulkanResourceManager;
class VulkanMemoryManager;
class VulkanMemoryCommit;

using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
using Surface = std::shared_ptr<CachedSurface>;
using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, MathUtil::Rectangle<u32>>;

struct SurfaceParams {
    /// Creates SurfaceParams for a depth buffer configuration
    static SurfaceParams CreateForDepthBuffer(
        u32 zeta_width, u32 zeta_height, Tegra::GPUVAddr zeta_address, Tegra::DepthFormat format,
        u32 block_width, u32 block_height, u32 block_depth,
        Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type);

    /// Creates SurfaceParams from a framebuffer configuration
    static SurfaceParams CreateForFramebuffer(std::size_t index);

    /// Returns the total size of this surface in bytes, adjusted for compression
    std::size_t SizeInBytesRaw(bool ignore_tiled = false) const {
        const u32 compression_factor{GetCompressionFactor(pixel_format)};
        const u32 bytes_per_pixel{GetBytesPerPixel(pixel_format)};
        const size_t uncompressed_size{
            Tegra::Texture::CalculateSize((ignore_tiled ? false : is_tiled), bytes_per_pixel, width,
                                          height, depth, block_height, block_depth)};

        // Divide by compression_factor^2, as height and width are factored by this
        return uncompressed_size / (compression_factor * compression_factor);
    }

    /// Returns the size of this surface as an Vulkan texture in bytes
    std::size_t SizeInBytesVK() const {
        return SizeInBytesRaw(true);
    }

    /// Checks if surfaces are compatible for caching
    bool IsCompatibleSurface(const SurfaceParams& other) const {
        return std::tie(pixel_format, type, width, height, target, depth) ==
               std::tie(other.pixel_format, other.type, other.width, other.height, other.target,
                        other.depth);
    }

    /// Initializes parameters for caching, should be called after everything has been initialized
    void InitCacheParameters(Tegra::GPUVAddr gpu_addr);

    vk::ImageCreateInfo CreateInfo() const;

    bool is_tiled;
    u32 block_width;
    u32 block_height;
    u32 block_depth;
    PixelFormat pixel_format;
    ComponentType component_type;
    SurfaceType type;
    u32 width;
    u32 height;
    u32 depth;
    u32 unaligned_height;
    SurfaceTarget target;
    // Parameters used for caching
    VAddr addr;
    Tegra::GPUVAddr gpu_addr;
    std::size_t size_in_bytes;
    std::size_t size_in_bytes_vk;
};

class CachedSurface final : public RasterizerCacheObject, public VulkanImage {
public:
    explicit CachedSurface(VulkanDevice& device_handler, VulkanResourceManager& resource_manager,
                           VulkanMemoryManager& memory_manager, const SurfaceParams& params);
    ~CachedSurface();

    vk::ImageView GetImageView();

    // Read/Write data in Switch memory to/from vk_buffer
    void LoadVKBuffer();
    void FlushVKBuffer();

    // Upload data in gl_buffer to this surface's texture
    void UploadVKTexture();

    VAddr GetAddr() const override {
        return params.addr;
    }

    std::size_t GetSizeInBytes() const override {
        return cached_size_in_bytes;
    }

    void Flush() override {
        FlushVKBuffer();
    }

    const SurfaceParams& GetSurfaceParams() const {
        return params;
    }

    vk::Format GetFormat() const {
        return vk_format;
    }

private:
    const vk::Device device;
    VulkanResourceManager& resource_manager;
    VulkanMemoryManager& memory_manager;
    const SurfaceParams params;
    const std::size_t buffer_size;

    vk::Image image;
    const VulkanMemoryCommit* image_commit{};

    vk::UniqueBuffer buffer;
    const VulkanMemoryCommit* buffer_commit{};
    u8* vk_buffer{};

    vk::UniqueImageView image_view;

    std::size_t cached_size_in_bytes;

    vk::Format vk_format;
};

} // namespace Vulkan

/// Hashable variation of SurfaceParams, used for a key in the surface cache
struct SurfaceReserveKey : Common::HashableStruct<Vulkan::SurfaceParams> {
    static SurfaceReserveKey Create(const Vulkan::SurfaceParams& params) {
        SurfaceReserveKey res;
        res.state = params;
        res.state.gpu_addr = {}; // Ignore GPU vaddr in caching
        // res.state.rt = {};       // Ignore rt config in caching
        return res;
    }
};
namespace std {
template <>
struct hash<SurfaceReserveKey> {
    std::size_t operator()(const SurfaceReserveKey& k) const {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class VulkanRasterizerCache final : public RasterizerCache<Surface> {
public:
    explicit VulkanRasterizerCache(VulkanDevice& device_handler,
                                   VulkanResourceManager& resource_manager,
                                   VulkanMemoryManager& memory_manager);
    ~VulkanRasterizerCache();

    /// Get the depth surface based on the framebuffer configuration
    Surface GetDepthBufferSurface(bool preserve_contents);

    /// Get the color surface based on the framebuffer configuration and the specified render target
    Surface GetColorBufferSurface(std::size_t index, bool preserve_contents);

    /// Tries to find a framebuffer using on the provided CPU address
    Surface TryFindFramebufferSurface(VAddr addr) const;

private:
    VulkanDevice& device_handler;
    VulkanResourceManager& resource_manager;
    VulkanMemoryManager& memory_manager;

    void LoadSurface(const Surface& surface);
    Surface GetSurface(const SurfaceParams& params, bool preserve_contents = true);

    /// Gets an uncached surface, creating it if need be
    Surface GetUncachedSurface(const SurfaceParams& params);

    /// Recreates a surface with new parameters
    Surface RecreateSurface(const Surface& old_surface, const SurfaceParams& new_params);

    /// Reserves a unique surface that can be reused later
    void ReserveSurface(const Surface& surface);

    /// Tries to get a reserved surface for the specified parameters
    Surface TryGetReservedSurface(const SurfaceParams& params);

    /// The surface reserve is a "backup" cache, this is where we put unique surfaces that have
    /// previously been used. This is to prevent surfaces from being constantly created and
    /// destroyed when used with different surface parameters.
    std::unordered_map<SurfaceReserveKey, Surface> surface_reserve;
};

} // namespace Vulkan