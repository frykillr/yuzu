// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vulkan/vulkan.hpp>
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_image.h"

namespace Vulkan {

VKImage::VKImage(vk::Device device, const vk::ImageCreateInfo& image_ci,
                 vk::ImageViewType view_type, vk::ImageAspectFlags aspect_mask)
    : device(device), image(device.createImageUnique(image_ci)), format(image_ci.format),
      view_type(view_type), aspect_mask(aspect_mask), current_layout(image_ci.initialLayout) {}

VKImage::~VKImage() = default;

vk::ImageView VKImage::GetImageView() {
    if (image_view) {
        return *image_view;
    }

    const vk::ImageViewCreateInfo image_view_ci(
        {}, *image, view_type, format,
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {aspect_mask, 0, 1, 0, 1});
    image_view = device.createImageViewUnique(image_view_ci);
    return *image_view;
}

void VKImage::Transition(vk::CommandBuffer cmdbuf, vk::ImageSubresourceRange subresource_range,
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