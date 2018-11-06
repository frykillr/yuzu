// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"

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
    void MarkAsUsed(u32& binding, u64 index, u64 offset, Maxwell::ShaderStage stage) {
        if (!is_used)
            this->binding = binding++;
        is_used = true;
        this->index = static_cast<unsigned>(index);
        this->stage = stage;
        max_offset = std::max(max_offset, static_cast<u32>(offset));
    }

    void MarkAsUsedIndirect(u32& binding, u64 index, Maxwell::ShaderStage stage) {
        if (!is_used)
            this->binding = binding++;
        is_used = true;
        is_indirect = true;
        this->index = static_cast<u32>(index);
        this->stage = stage;
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
    Maxwell::ShaderStage stage;
};

struct ShaderEntries {
    u32 descriptor_set;
    std::vector<ConstBufferEntry> const_buffer_entries;
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