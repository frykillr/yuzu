// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/shader_bytecode.h"

namespace Vulkan::VKShader {

constexpr std::size_t MAX_PROGRAM_CODE_LENGTH{0x1000};
using ProgramCode = std::vector<u64>;

struct ShaderSetup {
    explicit ShaderSetup(ProgramCode program_code) {
        program.code = std::move(program_code);
    }

    struct {
        ProgramCode code;
        ProgramCode code_b; // Used for dual vertex shaders
    } program;

    /// Used in scenarios where we have a dual vertex shaders
    void SetProgramB(ProgramCode&& program_b) {
        program.code_b = std::move(program_b);
        has_program_b = true;
    }

    bool IsDualProgram() const {
        return has_program_b;
    }

private:
    bool has_program_b{};
};

class ConstBufferEntry {
    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

public:
    u32 MarkAsUsed(u32 binding_, u64 index_, u64 offset, Maxwell::ShaderStage stage_) {
        if (!is_used) {
            binding = binding_++;
        }
        is_used = true;
        index = static_cast<u32>(index_);
        stage = stage_;
        max_offset = std::max(max_offset, static_cast<u32>(offset));
        return binding_;
    }

    u32 MarkAsUsedIndirect(u32 binding_, u64 index_, Maxwell::ShaderStage stage_) {
        if (!is_used) {
            binding = binding_++;
        }
        is_used = true;
        is_indirect = true;
        index = static_cast<u32>(index_);
        stage = stage_;
        return binding_;
    }

    bool IsUsed() const {
        return is_used;
    }

    bool IsIndirect() const {
        return is_indirect;
    }

    unsigned GetIndex() const {
        return index;
    }

    unsigned GetSize() const {
        return max_offset + 1;
    }

    u32 GetBinding() const {
        return binding;
    }

private:
    bool is_used{};
    bool is_indirect{};
    u32 index{};
    u32 max_offset{};
    u32 binding{};
    Maxwell::ShaderStage stage{};
};

class SamplerEntry {
    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

public:
    explicit SamplerEntry(Maxwell::ShaderStage stage, u32 binding, std::size_t offset,
                          std::size_t index, Tegra::Shader::TextureType type, bool is_array,
                          bool is_shadow)
        : offset(offset), stage(stage), binding(binding), sampler_index(index), type(type),
          is_array(is_array), is_shadow(is_shadow) {}

    std::size_t GetOffset() const {
        return offset;
    }

    std::size_t GetIndex() const {
        return sampler_index;
    }

    Maxwell::ShaderStage GetStage() const {
        return stage;
    }

    u32 GetBinding() const {
        return binding;
    }

    Tegra::Shader::TextureType GetType() const {
        return type;
    }

    bool IsArray() const {
        return is_array;
    }

    bool IsShadow() const {
        return is_shadow;
    }

private:
    /// Offset in TSC memory from which to read the sampler object, as specified by the sampling
    /// instruction.
    std::size_t offset;
    Maxwell::ShaderStage stage;      ///< Shader stage where this sampler was used.
    std::size_t sampler_index;       ///< Value used to index into the generated GLSL sampler array.
    u32 binding;                     ///< Descriptor binding.
    Tegra::Shader::TextureType type; ///< The type used to sample this texture (Texture2D, etc)
    bool is_array;  ///< Whether the texture is being sampled as an array texture or not.
    bool is_shadow; ///< Whether the texture is being sampled as a depth texture or not.
};

struct ShaderEntries {
    u32 descriptor_set;
    std::vector<ConstBufferEntry> const_buffers;
    std::vector<SamplerEntry> samplers;
    std::set<u32> attributes;
};

struct ProgramResult {
    std::vector<u8> code;
    ShaderEntries entries;
};

/**
 * Generates the GLSL vertex shader program source code for the given VS program
 * @returns String of the shader source code
 */
ProgramResult GenerateVertexShader(const ShaderSetup& setup);

/**
 * Generates the GLSL fragment shader program source code for the given FS program
 * @returns String of the shader source code
 */
ProgramResult GenerateFragmentShader(const ShaderSetup& setup);

} // namespace Vulkan::VKShader