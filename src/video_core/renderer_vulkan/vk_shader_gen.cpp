// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

#pragma optimize("", off)

namespace Vulkan::VKShader {

static constexpr u32 PROGRAM_OFFSET{10};

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

#include <cstdio>

ProgramResult GenerateVertexShader(const ShaderSetup& setup) {
    // TODO(Rodrigo): Add vertex stage B.
    Decompiler::SpirvModule module(setup.program.code, PROGRAM_OFFSET,
                                   Maxwell::ShaderStage::Vertex);
    const auto vertex_entry = module.Decompile();
    const auto main =
        module.Emit(module.OpFunction(module.OpTypeVoid(), {}, module.OpTypeFunction(module.OpTypeVoid())));
    module.Emit(module.OpLabel());
    module.Emit(module.OpFunctionCall(module.OpTypeBool(), vertex_entry));
    module.Emit(module.OpReturn());
    module.Emit(module.OpFunctionEnd());
    module.AddEntryPoint(spv::ExecutionModel::Vertex, main, "main", module.GetInterfaces());

    const auto code = module.Assemble();
    /*FILE* out = fopen("D:\\vertex.spv", "wb");
    fwrite(code.data(), 1, code.size(), out);
    fclose(out);*/
    return {code, module.GetEntries()};
}

ProgramResult GenerateFragmentShader(const ShaderSetup& setup) {
    /*FILE* aa = fopen("D:\\code.bin", "wb");
    fwrite(setup.program.code.data() + 10, 1, setup.program.code.size() - 10, aa);
    fclose(aa);*/

    Decompiler::SpirvModule module(setup.program.code, PROGRAM_OFFSET,
                                   Maxwell::ShaderStage::Fragment);
    const auto fragment_entry = module.Decompile();
    const auto main = module.Emit(
        module.OpFunction(module.OpTypeVoid(), {}, module.OpTypeFunction(module.OpTypeVoid())));
    module.Emit(module.OpLabel());
    module.Emit(module.OpFunctionCall(module.OpTypeBool(), fragment_entry));
    module.Emit(module.OpReturn());
    module.Emit(module.OpFunctionEnd());
    module.AddEntryPoint(spv::ExecutionModel::Fragment, main, "main", module.GetInterfaces());

    const auto code = module.Assemble();
    /*FILE* out = fopen("D:\\fragment.spv", "wb");
    fwrite(code.data(), 1, code.size(), out);
    fclose(out);*/
    return {code, module.GetEntries()};
}

} // namespace Vulkan::VKShader