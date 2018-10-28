// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

vk::UniqueShaderModule BuildShader(vk::Device device, std::size_t code_size, const u8* code_data);

} // namespace Vulkan