// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <sirit/sirit.h>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/shader_bytecode.h"
#include "video_core/engines/shader_header.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan::VKShader::Decompiler {

using Sirit::Id;
using Tegra::Engines::Maxwell3D;
using Tegra::Shader::Attribute;
using Tegra::Shader::Instruction;
using Tegra::Shader::Register;

struct Subroutine;

class SpirvModule final : public Sirit::Module {
public:
    explicit SpirvModule(const ProgramCode& program_code, u32 main_offset,
                         Maxwell3D::Regs::ShaderStage stage);
    ~SpirvModule();

    Id Decompile();

    const std::vector<Id>& GetInterfaces() const {
        return interfaces;
    }

    ShaderEntries GetEntries() const {
        ShaderEntries entries;
        entries.descriptor_set = descriptor_set;
        for (const auto& const_buffer_entry : declr_const_buffers) {
            if (const_buffer_entry.IsUsed()) {
                entries.const_buffers.push_back(const_buffer_entry);
            }
        }
        for (const auto& input_entry : declr_input_attribute) {
            const auto attribute = input_entry.first;
            const auto index{static_cast<u32>(attribute) -
                             static_cast<u32>(Attribute::Index::Attribute_0)};
            entries.attributes.insert(index);
        }
        return entries;
    }

private:
    struct InputAttributeEntry {
        Id id;
        Tegra::Shader::IpaMode input_mode;
    };

    Id Generate(const std::set<Subroutine> subroutines);

    /**
     * Compiles a range of instructions from Tegra to SPIR-V.
     * @param begin the offset of the starting instruction.
     * @param end the offset where the compilation should stop (exclusive).
     * @return the offset of the next instruction to compile. PROGRAM_END if the program terminates.
     */
    u32 CompileRange(u32 begin, u32 end);

    /**
     * Compiles a single instruction from Tegra to SPIR-V.
     * @param offset the offset of the Tegra shader instruction.
     * @return the offset of the next instruction to execute. Usually it is the current offset + 1.
     * If the current instruction always terminates the program, returns PROGRAM_END.
     */
    u32 CompileInstr(u32 offset);

    void DeclareVariables();

    void DeclareBuiltIns();

    /// Declares fragment shader output variables.
    void DeclareFragmentOutputs();

    /*
     * Returns whether the instruction at the specified offset is a 'sched' instruction.
     * Sched instructions always appear before a sequence of 3 instructions.
     */
    bool IsSchedInstruction(u32 offset) const;

    /**
     * Returns code that does an integer size conversion for the specified size.
     * @param type Type of the returned integer.
     * @param value Value to perform integer size conversion on.
     * @param size Register size to use for conversion instructions.
     * @returns SPIR-V id corresponding to the value converted to the specified size.
     */
    Id ConvertIntegerSize(Id type, Id value, Register::Size size);

    /// Generates code representing a temporary (GPR) register.
    Id GetRegister(const Register& reg, u32 elem);

    /**
     * Gets a register as an float.
     * @param reg The register to get.
     * @param elem The element to use for the operation.
     * @returns SPIR-V id corresponding to the register as a float.
     */
    Id GetRegisterAsFloat(const Register& reg, u32 elem = 0);

    /**
     * Gets a register as an integer.
     * @param reg The register to get.
     * @param elem The element to use for the operation.
     * @param is_signed Whether to get the register as a signed (or unsigned) integer.
     * @param size Register size to use for conversion instructions.
     * @returns SPIR-V id corresponding to the register as an integer.
     */
    Id GetRegisterAsInteger(const Register& reg, u32 elem = 0, bool is_signed = true,
                            Register::Size size = Register::Size::Word);

    /// Generates code representing an input attribute register. Returns a float4 value.
    Id GetInputAttribute(Attribute::Index attribute, const Tegra::Shader::IpaMode& input_mode,
                         std::optional<Register> vertex = {});

    /**
     * Writes code that does a register assignment to float value operation.
     * @param reg The destination register to use.
     * @param elem The element to use for the operation.
     * @param value The code representing the value to assign.
     * @param dest_num_components Number of components in the destination.
     * @param value_num_components Number of components in the value.
     * @param is_saturated Optional, when True, saturates the provided value.
     * @param dest_elem Optional, the destination element to use for the operation.
     */
    void SetRegisterToFloat(const Register& reg, u64 elem, Id value, u64 dest_num_components,
                            u64 value_num_components, bool is_saturated = false, u64 dest_elem = 0,
                            bool precise = false);

    /**
     * Writes code that does a register assignment to integer value operation.
     * @param reg The destination register to use.
     * @param elem The element to use for the operation.
     * @param value The code representing the value to assign.
     * @param dest_num_components Number of components in the destination.
     * @param value_num_components Number of components in the value.
     * @param is_saturated Optional, when True, saturates the provided value.
     * @param dest_elem Optional, the destination element to use for the operation.
     * @param size Register size to use for conversion instructions.
     */
    void SetRegisterToInteger(const Register& reg, bool is_signed, u64 elem, Id value,
                              u64 dest_num_components, u64 value_num_components,
                              bool is_saturated = false, u64 dest_elem = 0,
                              Register::Size size = Register::Size::Word, bool sets_cc = false);

    /**
     * Writes code that does a register assignment to input attribute operation. Input attributes
     * are stored as floats, so this may require conversion.
     * @param reg The destination register to use.
     * @param elem The element to use for the operation.
     * @param attribute The input attribute to use as the source value.
     * @param input_mode The input mode.
     * @param vertex The register that decides which vertex to read from (used in GS).
     */
    void SetRegisterToInputAttibute(const Register& reg, u64 elem, Attribute::Index attribute,
                                    const Tegra::Shader::IpaMode& input_mode,
                                    std::optional<Register> vertex = {});

    /**
     * Writes code that does a output attribute assignment to register operation. Output attributes
     * are stored as floats, so this may require conversion.
     * @param attribute The destination output attribute.
     * @param elem The element to use for the operation.
     * @param val_reg The register to use as the source value.
     * @param buf_reg The register that tells which buffer to write to (used in geometry shaders).
     */
    void SetOutputAttributeToRegister(Attribute::Index attribute, u64 elem, const Register& val_reg,
                                      const Register& buf_reg);

    /**
     * Writes code that does a register assignment to value operation.
     * @param reg The destination register to use.
     * @param elem The element to use for the operation.
     * @param value The code representing the value to assign.
     * @param dest_num_components Number of components in the destination.
     * @param value_num_components Number of components in the value.
     * @param dest_elem Optional, the destination element to use for the operation.
     */
    void SetRegister(const Register& reg, u64 elem, Id value, u64 dest_num_components,
                     u64 value_num_components, u64 dest_elem, bool precise);

    /*
     * Returns the condition to use in the 'if' for a predicated instruction.
     * @param instr Instruction to generate the if condition for.
     * @returns string containing the predicate condition.
     */
    Id GetPredicateCondition(u64 index, bool negate);

    /// Gets a predicate value.
    Id GetPredicate(u64 index);

    /// Generates code representing a 19-bit immediate value.
    Id GetImmediate19(const Instruction& instr);

    /// Generates code representing a 32-bit immediate value
    Id GetImmediate32(const Instruction& instr);

    /// Generates code representing a uniform (C buffer) register, interpreted as the input type.
    Id GetUniform(u64 index, u64 offset, Id type, Register::Size size = Register::Size::Word);

    /**
     * Returns a uniform buffer value in an index.
     * @param cbuf_index Constant buffer index.
     * @param offset Offset applied to retreive const buffer's subindex.
     * @param index Base index to retreive const buffer's subindex.
     * @param type Type to return.
     * @returns SPIR-V id corresponding to const buffer's value as type.
     */
    Id GetUniformIndirect(u64 cbuf_index, s64 offset, Id index, Id type);

    Id DeclareUniform(u64 cbuf_index);

    Id DeclareInputAttribute(Attribute::Index attribute, Tegra::Shader::IpaMode input_mode);

    Id DeclareOutputAttribute(u32 index);

    /**
     * Transforms the input SPIR-V id operand into one that applies the abs() function and negates
     * the output if necessary. When both abs and neg are true, the negation will be applied after
     * taking the absolute value.
     * @param operand The input operand to take the abs() of, negate, or both.
     * @param abs Whether to apply the abs() function to the input operand.
     * @param neg Whether to negate the input operand.
     * @returns SPIR-V id corresponding to the operand after being transformed by the abs() and
     * negation operations.
     */
    Id GetFloatOperandAbsNeg(Id operand, bool abs, bool neg);

    /// Writes the output values from a fragment shader to the corresponding SPIR-V output
    /// variables.
    void EmitFragmentOutputsWrite();

    static constexpr std::size_t REGISTER_COUNT = 0xff;
    static constexpr std::size_t PRED_COUNT = 0xf; // Value untested.

    static constexpr u32 MAX_CONSTBUFFER_SIZE = 0x10000;
    static constexpr u32 MAX_CONSTBUFFER_ELEMENTS = MAX_CONSTBUFFER_SIZE / (4 * sizeof(float));

    static constexpr u32 CBUF_STRIDE = 16;

    const ProgramCode& program_code;
    const u32 main_offset;
    const Maxwell3D::Regs::ShaderStage stage;
    const u32 descriptor_set;
    Tegra::Shader::Header header;

    /// Binding iterator
    u32 binding = 0;

    std::vector<Id> interfaces;

    struct {
        Id per_vertex_struct{};
        Id per_vertex{};
        Id vertex_index{};
        Id instance_index{};
    } vs;

    struct {
        Id frag_coord{};
        std::array<Id, Maxwell3D::Regs::NumRenderTargets> frag_colors;
        Id frag_depth{};
    } fs;

    std::vector<Id> regs;
    std::vector<Id> predicates;
    std::array<Id, Maxwell3D::Regs::MaxConstBuffers> cbufs;

    std::unordered_map<u32, Id> output_attrs;
    std::unordered_map<Attribute::Index, InputAttributeEntry> declr_input_attribute;
    std::array<ConstBufferEntry, Maxwell3D::Regs::MaxConstBuffers> declr_const_buffers;

    const Id t_void = Name(OpTypeVoid(), "void");
    const Id t_bool = Name(OpTypeBool(), "bool");
    const Id t_float = Name(OpTypeFloat(32), "float");
    const Id t_sint = Name(OpTypeInt(32, true), "sint");
    const Id t_uint = Name(OpTypeInt(32, false), "uint");

    const Id t_float4 = Name(OpTypeVector(t_float, 4), "float4");

    const Id t_prv_bool = Name(OpTypePointer(spv::StorageClass::Private, t_bool), "prv_bool");
    const Id t_prv_float = Name(OpTypePointer(spv::StorageClass::Private, t_float), "prv_float");

    const Id t_in_uint = Name(OpTypePointer(spv::StorageClass::Input, t_uint), "in_uint");
    const Id t_in_float4 = Name(OpTypePointer(spv::StorageClass::Input, t_float4), "in_float4");

    const Id t_out_float = Name(OpTypePointer(spv::StorageClass::Output, t_float), "out_float");
    const Id t_out_float4 = Name(OpTypePointer(spv::StorageClass::Output, t_float4), "out_float4");

    const Id t_ubo_float = Name(OpTypePointer(spv::StorageClass::Uniform, t_float), "ubo_float");

    const Id t_cbuf_array = Decorate(
        Name(OpTypeArray(t_float4, Constant(t_uint, MAX_CONSTBUFFER_ELEMENTS)), "cbuf_array"),
        spv::Decoration::ArrayStride, {CBUF_STRIDE});
    const Id t_cbuf_struct = Name(OpTypeStruct({t_cbuf_array}), "cbuf_struct");
    const Id t_cbuf_ubo =
        Name(OpTypePointer(spv::StorageClass::Uniform, t_cbuf_struct), "cbuf_ubo");

    const Id t_bool_function = OpTypeFunction(t_bool);

    const Id v_float_zero = ConstantNull(t_float);
    const Id v_float4_zero = ConstantNull(t_float4);
    const Id v_true = ConstantTrue(t_bool);
    const Id v_false = ConstantFalse(t_bool);
};

} // namespace Vulkan::VKShader::Decompiler