// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "common/static_vector.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"

namespace Vulkan {

class RasterizerVulkan;
class VKDevice;
class VKFence;

class CachedShader;
using Shader = std::shared_ptr<CachedShader>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct PipelineParams {
    struct VertexBinding {
        u32 index = 0;
        u32 stride = 0;
        u32 divisor = 0;

        auto Tie() const {
            return std::tie(index, stride, divisor);
        }

        auto operator==(const VertexBinding& rhs) const {
            return Tie() == rhs.Tie();
        }
    };

    struct VertexAttribute {
        u32 index = 0;
        u32 buffer = 0;
        Maxwell::VertexAttribute::Type type = Maxwell::VertexAttribute::Type::UnsignedNorm;
        Maxwell::VertexAttribute::Size size = Maxwell::VertexAttribute::Size::Size_8;
        u32 offset = 0;

        auto Tie() const {
            return std::tie(index, buffer, type, size, offset);
        }

        bool operator==(const VertexAttribute& rhs) const {
            return Tie() == rhs.Tie();
        }
    };

    struct StencilFace {
        u32 enable = false;
        Maxwell::StencilOp op_fail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp op_zfail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp op_zpass = Maxwell::StencilOp::Keep;
        Maxwell::ComparisonOp func_func = Maxwell::ComparisonOp::Always;
        s32 func_ref = 0;
        u32 func_mask = 0;
        u32 mask = 0;

        auto Tie() const {
            return std::tie(enable, op_fail, op_zfail, op_zpass, func_func, func_ref, func_mask,
                            mask);
        }

        bool operator==(const StencilFace& rhs) const {
            return Tie() == rhs.Tie();
        }
    };

    struct ColorAttachment {
        u32 index = 0;
        VideoCore::Surface::PixelFormat pixel_format = VideoCore::Surface::PixelFormat::Invalid;
        VideoCore::Surface::ComponentType component_type =
            VideoCore::Surface::ComponentType::Invalid;

        auto Tie() const {
            return std::tie(index, pixel_format, component_type);
        }

        bool operator==(const ColorAttachment& rhs) const {
            return Tie() == rhs.Tie();
        }
    };

    struct {
        StaticVector<Maxwell::NumVertexArrays, VertexBinding> bindings;
        StaticVector<Maxwell::NumVertexAttributes, VertexAttribute> attributes;

        auto Tie() const {
            return std::tie(bindings, attributes);
        }
    } vertex_input;

    struct {
        Maxwell::PrimitiveTopology topology = Maxwell::PrimitiveTopology::Points;
        bool primitive_restart_enable = false;

        auto Tie() const {
            return std::tie(topology, primitive_restart_enable);
        }
    } input_assembly;

    struct {
        auto Tie() const {
            return std::tie();
        }
    } viewport_state;

    struct {
        auto Tie() const {
            return std::tie();
        }
    } rasterizer;

    struct {
        auto Tie() const {
            return std::tie();
        }
    } multisampling;

    struct {
        bool depth_test_enable = false;
        bool depth_write_enable = true;
        bool depth_bounds_enable = false;
        bool stencil_enable = false;
        Maxwell::ComparisonOp depth_test_function = Maxwell::ComparisonOp::Always;
        StencilFace front_stencil;
        StencilFace back_stencil;
        float depth_bounds_min = 0.f;
        float depth_bounds_max = 0.f;

        auto Tie() const {
            return std::tie(depth_test_enable, depth_write_enable, depth_bounds_enable,
                            depth_test_function, stencil_enable, front_stencil, back_stencil,
                            depth_bounds_min, depth_bounds_max);
        }
    } depth_stencil;

    struct {
        auto Tie() const {
            return std::tie();
        }
    } color_blending;

    struct {
        StaticVector<Maxwell::NumRenderTargets, ColorAttachment> color_map = {};
        // TODO(Rodrigo): Unify has_zeta into zeta_pixel_format and zeta_component_type.
        VideoCore::Surface::PixelFormat zeta_pixel_format =
            VideoCore::Surface::PixelFormat::Invalid;
        VideoCore::Surface::ComponentType zeta_component_type =
            VideoCore::Surface::ComponentType::Invalid;
        bool has_zeta = false;
        bool preserve_contents = false;

        auto Tie() const {
            return std::tie(color_map, zeta_pixel_format, zeta_component_type, has_zeta,
                            preserve_contents);
        }
    } renderpass;

    bool operator==(const PipelineParams& rhs) const {
        return vertex_input.Tie() == rhs.vertex_input.Tie() &&
               input_assembly.Tie() == rhs.input_assembly.Tie() &&
               viewport_state.Tie() == rhs.viewport_state.Tie() &&
               rasterizer.Tie() == rhs.rasterizer.Tie() &&
               multisampling.Tie() == rhs.multisampling.Tie() &&
               depth_stencil.Tie() == rhs.depth_stencil.Tie() &&
               color_blending.Tie() == rhs.color_blending.Tie() &&
               renderpass.Tie() == rhs.renderpass.Tie();
    }

    u64 Hash() const {
        // TODO(Rodrigo): Implement a hash.
        // return Common::CityHash64(reinterpret_cast<const char*>(&params), sizeof(params));
        return 0;
    }
};

struct Pipeline {
    vk::Pipeline handle;
    vk::PipelineLayout layout;
    vk::RenderPass renderpass;
    std::array<Shader, Maxwell::MaxShaderStage> shaders;
};

class CachedShader final : public RasterizerCacheObject {
public:
    CachedShader(VKDevice& device_handler, VAddr addr, Maxwell::ShaderProgram program_type);

    /// Gets a descriptor set from the internal pool.
    vk::DescriptorSet CommitDescriptorSet(VKFence& fence);

    VAddr GetAddr() const override {
        return addr;
    }

    std::size_t GetSizeInBytes() const override {
        return VKShader::MAX_PROGRAM_CODE_LENGTH * sizeof(u64);
    }

    // We do not have to flush this cache as things in it are never modified by us.
    void Flush() override {}

    /// Gets the module handle for the shader.
    vk::ShaderModule GetHandle(vk::PrimitiveTopology primitive_mode) {
        return *shader_module;
    }

    /// Gets the descriptor set layout of the shader.
    vk::DescriptorSetLayout GetDescriptorSetLayout() const {
        return *descriptor_set_layout;
    }

    /// Gets the module entries for the shader.
    const VKShader::ShaderEntries& GetEntries() const {
        return entries;
    }

private:
    class DescriptorPool;

    void CreateDescriptorSetLayout();
    void CreateDescriptorPool();

    const VAddr addr;
    const Maxwell::ShaderProgram program_type;
    const vk::Device device;

    VKShader::ShaderSetup setup;
    VKShader::ShaderEntries entries;

    vk::UniqueShaderModule shader_module;

    vk::UniqueDescriptorSetLayout descriptor_set_layout;
    std::unique_ptr<DescriptorPool> descriptor_pool;
};

class VKShaderCache final : public RasterizerCache<Shader> {
public:
    explicit VKShaderCache(RasterizerVulkan& rasterizer, VKDevice& device_handler);

    Pipeline GetPipeline(const PipelineParams& params);

protected:
    void ObjectInvalidated(const Shader& shader) override;

private:
    using ShaderPipeline = std::array<VAddr, Maxwell::MaxShaderProgram>;
    using CacheKey = std::tuple<ShaderPipeline, PipelineParams>;

    struct CacheEntry {
        vk::UniquePipeline pipeline;
        vk::UniquePipelineLayout layout;
        vk::UniqueRenderPass renderpass;
    };

    struct HashFn {
        std::size_t operator()(const CacheKey& key) const {
            // TODO(Rodrigo): Hash shaders.
            const auto& [shaders, pipeline] = key;
            return static_cast<std::size_t>(pipeline.Hash());
        }
    };

    VKDevice& device_handler;
    const vk::Device device;

    vk::UniquePipelineLayout CreatePipelineLayout(const PipelineParams& params,
                                                  const Pipeline& pipeline) const;
    vk::UniquePipeline CreatePipeline(const PipelineParams& params, const Pipeline& pipeline) const;
    vk::UniqueRenderPass CreateRenderPass(const PipelineParams& params) const;

    std::unordered_map<CacheKey, std::unique_ptr<CacheEntry>, HashFn> cache;
    vk::UniqueDescriptorSetLayout empty_set_layout;
};

} // namespace Vulkan