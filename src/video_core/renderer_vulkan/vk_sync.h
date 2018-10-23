// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VulkanResourceManager;
class VulkanFence;

class VulkanSync {
public:
    explicit VulkanSync(VulkanResourceManager& resource_manager, vk::Device& device,
                        vk::Queue& queue, const u32& queue_family_index);
    ~VulkanSync();

    VulkanFence& PrepareExecute(bool take_fence_ownership = true);

    void Execute();

    vk::CommandBuffer BeginRecord();

    void EndRecord(vk::CommandBuffer cmdbuf);

    vk::Semaphore QuerySemaphore();

private:
    struct Call {
        VulkanFence* fence;
        vk::Semaphore semaphore;
        std::vector<vk::CommandBuffer> commands;
    };

    VulkanResourceManager& resource_manager;
    vk::Device& device;
    vk::Queue& queue;

    std::vector<std::unique_ptr<Call>> calls;
    std::unique_ptr<Call> current_call;
    bool take_fence_ownership{};

    vk::Semaphore wait_semaphore{};
};

} // namespace Vulkan