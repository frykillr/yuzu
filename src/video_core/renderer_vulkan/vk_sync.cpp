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

VulkanSync::VulkanSync(VulkanResourceManager& resource_manager, const VulkanDevice& device_handler)
    : resource_manager(resource_manager), device(device_handler.GetLogical()),
      queue(device_handler.GetGraphicsQueue()) {}

VulkanSync::~VulkanSync() = default;

VulkanFence& VulkanSync::PrepareExecute(bool take_fence_ownership) {
    recording_submit = true;

    this->take_fence_ownership = take_fence_ownership;

    VulkanFence& fence = resource_manager.CommitFence();
    current_call = std::make_unique<Call>();
    current_call->fence = &fence;
    current_call->semaphore = resource_manager.CommitSemaphore(fence);
    return fence;
}

void VulkanSync::AddDependency(vk::CommandBuffer cmdbuf, vk::Semaphore semaphore,
                               vk::PipelineStageFlags pipeline_stage) {
    ASSERT(recording_submit);

    const std::size_t index = dep_cmdbufs.size();
    dep_cmdbufs.push_back(cmdbuf);
    dep_signal_semaphores.push_back(semaphore);
    dep_wait_semaphores.push_back(semaphore);
    dep_pipeline_stages.push_back(pipeline_stage);

    const vk::SubmitInfo si({}, nullptr, nullptr, 1, &dep_cmdbufs[index], 1,
                            &dep_signal_semaphores[index]);
    submit_infos.push_back(si);
}

void VulkanSync::Execute() {
    ASSERT(recording_submit);

    if (previous_semaphore) {
        // TODO(Rodrigo): Pipeline wait can be optimized with an extra argument.
        dep_wait_semaphores.push_back(previous_semaphore);
        dep_pipeline_stages.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    ASSERT_MSG(dep_wait_semaphores.size() == dep_pipeline_stages.size(), "Dependency size mismatch");

    submit_infos.push_back({static_cast<u32>(dep_wait_semaphores.size()), dep_wait_semaphores.data(),
                            dep_pipeline_stages.data(),
                            static_cast<u32>(current_call->commands.size()),
                            current_call->commands.data(), 1, &current_call->semaphore});

    queue.submit(static_cast<u32>(submit_infos.size()), submit_infos.data(), *current_call->fence);
    ClearSubmitData();

    if (take_fence_ownership) {
        current_call->fence->Release();
    }

    previous_semaphore = current_call->semaphore;
    calls.push_back(std::move(current_call));

    recording_submit = false;
}

vk::CommandBuffer VulkanSync::BeginRecord() {
    ASSERT(recording_submit);

    vk::CommandBuffer cmdbuf = resource_manager.CommitCommandBuffer(*current_call->fence);
    cmdbuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cmdbuf;
}

void VulkanSync::EndRecord(vk::CommandBuffer cmdbuf) {
    ASSERT(recording_submit);

    cmdbuf.end();
    current_call->commands.push_back(cmdbuf);
}

vk::Semaphore VulkanSync::QuerySemaphore() {
    vk::Semaphore semaphore = previous_semaphore;
    previous_semaphore = vk::Semaphore(nullptr);
    return semaphore;
}

void VulkanSync::ClearSubmitData() {
    submit_infos.clear();
    dep_cmdbufs.clear();
    dep_signal_semaphores.clear();
    dep_wait_semaphores.clear();
    dep_pipeline_stages.clear();
}

} // namespace Vulkan
