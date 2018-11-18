// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "common/threadsafe_queue.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanResourceManager;

class VulkanSync {
public:
    explicit VulkanSync(VulkanResourceManager& resource_manager,
                        const VulkanDevice& device_handler);
    ~VulkanSync();

    VulkanFence& BeginPass(bool take_fence_ownership = true);

    void AddDependency(vk::CommandBuffer cmdbuf, vk::Semaphore semaphore,
                       vk::PipelineStageFlags pipeline_stage);

    vk::CommandBuffer BeginRecord();

    void EndRecord(vk::CommandBuffer cmdbuf);

    void EndPass();

    void Flush();

    vk::Semaphore QuerySemaphore();

private:
    struct Call {
        VulkanFence* fence;
        vk::Semaphore semaphore;
        std::vector<vk::CommandBuffer> commands;
        std::vector<vk::CommandBuffer> cmdbufs;
        std::vector<vk::Semaphore> signal_semaphores;
        std::vector<vk::Semaphore> wait_semaphores;
        std::vector<vk::PipelineStageFlags> pipeline_stages;
        std::vector<vk::SubmitInfo> submit_infos;
        bool take_fence_ownership;
    };

    VulkanResourceManager& resource_manager;
    const vk::Device device;
    const vk::Queue queue;

    std::unique_ptr<Call> pass;
    Common::SPSCQueue<std::unique_ptr<Call>> scheduled_passes;

    VulkanFence* next_fence = nullptr;
    vk::Semaphore previous_semaphore = nullptr;

    std::atomic_bool executing = true;
    std::thread worker_thread;

    std::mutex work_mutex;
    std::condition_variable work_cv;

    std::atomic_bool work_done = false;
    std::mutex flush_mutex;
    std::condition_variable flush_signal;

    bool recording_submit = false;
};

} // namespace Vulkan