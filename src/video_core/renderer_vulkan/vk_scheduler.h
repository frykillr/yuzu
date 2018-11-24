// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VKDevice;
class VKFence;
class VKResourceManager;

class VKScheduler {
public:
    explicit VKScheduler(VKResourceManager& resource_manager, const VKDevice& device_handler);
    ~VKScheduler();

    VKFence& BeginPass(bool take_fence_ownership = true);

    vk::CommandBuffer BeginRecord();

    void EndRecord(vk::CommandBuffer cmdbuf);

    void EndPass(vk::Semaphore semaphore = nullptr);

    void Flush();

private:
    struct Pass {
        VKFence* fence;
        vk::Semaphore semaphore;
        std::vector<vk::CommandBuffer> commands;
        std::vector<vk::CommandBuffer> cmdbufs;
        std::vector<vk::SubmitInfo> submit_infos;
        bool take_fence_ownership;
    };

    VKResourceManager& resource_manager;
    const vk::Device device;
    const vk::Queue queue;

    std::unique_ptr<Pass> pass;
    std::vector<std::unique_ptr<Pass>> scheduled_passes;
    u32 flush_ticks = 0;

    VKFence* next_fence = nullptr;

    bool recording_submit = false;
};

} // namespace Vulkan