// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_sync.h"

namespace Vulkan {

constexpr u32 CALLS_RESERVE = 128;

VulkanSync::VulkanSync(VulkanResourceManager& resource_manager, const VulkanDevice& device_handler)
    : resource_manager(resource_manager), device(device_handler.GetLogical()),
      queue(device_handler.GetGraphicsQueue()) {

    calls.reserve(CALLS_RESERVE);
}

VulkanSync::~VulkanSync() = default;

VulkanFence& VulkanSync::PrepareExecute(bool take_fence_ownership) {
    this->take_fence_ownership = take_fence_ownership;

    VulkanFence& fence = resource_manager.CommitFence();
    current_call = std::make_unique<Call>();
    current_call->fence = &fence;
    current_call->semaphore = resource_manager.CommitSemaphore(fence);
    return fence;
}

void VulkanSync::Execute() {
    vk::SubmitInfo submit_info(0, nullptr, nullptr, static_cast<u32>(current_call->commands.size()),
                               current_call->commands.data(), 1, &current_call->semaphore);
    if (wait_semaphore) {
        // TODO(Rodrigo): This could be optimized with an extra argument.
        const vk::PipelineStageFlags stage_flags = vk::PipelineStageFlagBits::eAllCommands;

        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &wait_semaphore;
        submit_info.pWaitDstStageMask = &stage_flags;
    }
    queue.submit({submit_info}, *current_call->fence);
    if (take_fence_ownership) {
        current_call->fence->Release();
    }

    wait_semaphore = current_call->semaphore;
    calls.push_back(std::move(current_call));
}

vk::CommandBuffer VulkanSync::BeginRecord() {
    vk::CommandBuffer cmdbuf = resource_manager.CommitCommandBuffer(*current_call->fence);
    cmdbuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cmdbuf;
}

void VulkanSync::EndRecord(vk::CommandBuffer cmdbuf) {
    cmdbuf.end();
    current_call->commands.push_back(cmdbuf);
}

vk::Semaphore VulkanSync::QuerySemaphore() {
    vk::Semaphore semaphore = wait_semaphore;
    wait_semaphore = vk::Semaphore(nullptr);
    return semaphore;
}

} // namespace Vulkan
