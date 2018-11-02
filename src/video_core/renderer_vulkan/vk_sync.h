// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanResourceManager;

class VulkanSync {
public:
    explicit VulkanSync(VulkanResourceManager& resource_manager,
                        const VulkanDevice& device_handler);
    ~VulkanSync();

    VulkanFence& PrepareExecute(bool take_fence_ownership = true);

    void AddDependency(vk::CommandBuffer cmdbuf, vk::Semaphore semaphore,
                       vk::PipelineStageFlags pipeline_stage);

    vk::CommandBuffer BeginRecord();

    void EndRecord(vk::CommandBuffer cmdbuf);

    void Execute();

    vk::Semaphore QuerySemaphore();

private:
    struct Call {
        VulkanFence* fence;
        vk::Semaphore semaphore;
        std::vector<vk::CommandBuffer> commands;
    };

    void ClearSubmitData();

    VulkanResourceManager& resource_manager;
    const vk::Device device;
    const vk::Queue queue;

    std::vector<std::unique_ptr<Call>> calls;
    std::unique_ptr<Call> current_call;
    bool take_fence_ownership{};

    std::vector<vk::SubmitInfo> submit_infos;
    std::vector<vk::CommandBuffer> dep_cmdbufs;
    std::vector<vk::Semaphore> dep_signal_semaphores;
    std::vector<vk::Semaphore> dep_wait_semaphores;
    std::vector<vk::PipelineStageFlags> dep_pipeline_stages;

    vk::Semaphore previous_semaphore{};

    std::mutex mutex;
    bool recording_submit{};
};

} // namespace Vulkan