// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

static constexpr u32 TICKS_TO_FLUSH = 64;

VKScheduler::VKScheduler(VKResourceManager& resource_manager, const VKDevice& device_handler)
    : resource_manager(resource_manager), device(device_handler.GetLogical()),
      queue(device_handler.GetGraphicsQueue()) {

    next_fence = &resource_manager.CommitFence();
}

VKScheduler::~VKScheduler() = default;

VKFence& VKScheduler::BeginPass(bool take_fence_ownership) {
    DEBUG_ASSERT(!recording_submit);
    recording_submit = true;

    VKFence& now_fence = *next_fence;
    pass = std::make_unique<Pass>();
    pass->fence = &now_fence;
    pass->take_fence_ownership = take_fence_ownership;

    // Allocate the semaphore the current call will use in a future fence, to avoid it being
    // freed while it's still being waited on.
    next_fence = &resource_manager.CommitFence();

    return now_fence;
}

void VKScheduler::EndPass(vk::Semaphore semaphore) {
    DEBUG_ASSERT(recording_submit);
    pass->semaphore = semaphore;
    pass->submit_infos.push_back({0, nullptr, nullptr, static_cast<u32>(pass->commands.size()),
                                  pass->commands.data(), semaphore ? 1u : 0u, &pass->semaphore});

    scheduled_passes.push_back(std::move(pass));
    if (++flush_ticks > TICKS_TO_FLUSH) {
        Flush();
        flush_ticks = 0;
    }

    recording_submit = false;
}

vk::CommandBuffer VKScheduler::BeginRecord() {
    DEBUG_ASSERT(recording_submit);
    const vk::CommandBuffer cmdbuf = resource_manager.CommitCommandBuffer(*pass->fence);
    cmdbuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cmdbuf;
}

void VKScheduler::EndRecord(vk::CommandBuffer cmdbuf) {
    DEBUG_ASSERT(recording_submit);
    cmdbuf.end();
    pass->commands.push_back(cmdbuf);
}

void VKScheduler::Flush() {
    for (auto& pass : scheduled_passes) {
        queue.submit(static_cast<u32>(pass->submit_infos.size()), pass->submit_infos.data(),
                     *pass->fence);
        if (pass->take_fence_ownership) {
            pass->fence->Release();
        }
    }
    scheduled_passes.clear();
}

} // namespace Vulkan
