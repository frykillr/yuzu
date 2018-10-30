// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vulkan/vulkan.hpp>
#include "video_core/renderer_vulkan/vk_image.h"

namespace Vulkan {

VulkanImage::VulkanImage(vk::Device device, const vk::ImageCreateInfo& image_ci)
    : image(device.createImageUnique(image_ci)), format(image_ci.format),
      current_layout(image_ci.initialLayout) {}

VulkanImage::~VulkanImage() = default;

void VulkanImage::Transition(vk::CommandBuffer cmdbuf, vk::ImageSubresourceRange subresource_range,
                             vk::ImageLayout new_layout, vk::PipelineStageFlags new_stage_mask,
                             vk::AccessFlags new_access, u32 new_family) {

    const vk::ImageMemoryBarrier barrier(current_access, new_access, current_layout, new_layout,
                                         current_family, new_family, *image, subresource_range);
    cmdbuf.pipelineBarrier(current_stage_mask, new_stage_mask, {}, {}, {}, {barrier});

    current_layout = new_layout;
    current_stage_mask = new_stage_mask;
    current_access = new_access;
    current_family = new_family;
}

} // namespace Vulkan