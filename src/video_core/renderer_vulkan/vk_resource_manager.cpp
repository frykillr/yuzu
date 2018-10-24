// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"

namespace Vulkan {

// TODO(Rodrigo): Fine tune these numbers.
constexpr u32 COMMAND_BUFFERS_COUNT = 0x1000;
constexpr u32 FENCES_COUNT = 0x2000;
constexpr u32 SEMAPHORES_COUNT = 0x1000;

VulkanFence::VulkanFence(vk::UniqueFence handle, vk::Device device, std::mutex& mutex)
    : handle(std::move(handle)), device(device), mutex(mutex) {}

VulkanFence::~VulkanFence() = default;

void VulkanFence::Wait() {
    std::unique_lock lock(mutex);
    device.waitForFences({*handle}, true, WaitTimeout);
}

void VulkanFence::Release() {
    std::unique_lock lock(mutex);
    is_owned = false;
}

void VulkanFence::Commit() {
    is_owned = true;
    is_used = true;
}

void VulkanFence::ProtectResource(VulkanResourceInterface* resource) {
    protected_resources.push_back(resource);
}

void VulkanFence::RemoveUsage() {
    for (auto* resource : protected_resources) {
        resource->RemoveUsage();
    }
    // TODO(Rodrigo): Find a way to preserve vector's allocated memory.
    protected_resources.clear();

    is_used = false;
}

bool VulkanFence::IsOwned() const {
    return is_owned;
}

bool VulkanFence::IsUsed() const {
    return is_used;
}

VulkanResourceInterface::VulkanResourceInterface(vk::Device device) : device(device) {}

VulkanResourceInterface::~VulkanResourceInterface() = default;

bool VulkanResourceInterface::TryCommit(VulkanFence& commit_fence) {
    std::unique_lock lock(mutex);
    return UnsafeTryCommit(commit_fence);
}

void VulkanResourceInterface::Commit(VulkanFence& commit_fence) {
    std::unique_lock lock(mutex);
    if (fence) {
        device.waitForFences({*fence}, true, WaitTimeout);
    }
    const bool available = UnsafeTryCommit(commit_fence);
    ASSERT_MSG(available, "Unexpected race condition");
}

void VulkanResourceInterface::RemoveUsage() {
    std::unique_lock lock(mutex);
    ASSERT(fence);
    ASSERT(is_claimed);
    fence = nullptr;
    is_claimed = false;
}

bool VulkanResourceInterface::UnsafeTryCommit(VulkanFence& commit_fence) {
    if (is_claimed) {
        return false;
    }
    is_claimed = true;
    fence = &commit_fence;
    commit_fence.ProtectResource(this);
    return true;
}

VulkanResourceManager::VulkanResourceManager(const VulkanDevice& device_handler)
    : device(device_handler.GetLogical()), graphics_family(device_handler.GetGraphicsFamily()) {
    CreateFences();
    CreateCommands();
    CreateSemaphores();
}

VulkanResourceManager::~VulkanResourceManager() = default;

VulkanFence& VulkanResourceManager::CommitFence() {
    std::unique_lock lock(fences_mutex);
    auto it = std::find_if(fences.begin(), fences.end(), [&](auto& fence) {
        if (!fence->IsUsed()) {
            return true;
        }
        if (fence->IsOwned()) {
            return false;
        }
        // The fence has been used but it's no longer owned, check for its status.
        if (device.getFenceStatus(*fence) != vk::Result::eSuccess) {
            return false;
        }
        // In the case that it has been signaled, reset and remove its usage. The fence is free to
        // be reused.
        device.resetFences({*fence});
        fence->RemoveUsage();
        return true;
    });
    if (it != fences.end()) {
        auto& fence = *it;
        fence->Commit();
        return *fence;
    }

    // Try again, this time waiting for a non owned fence to be signaled.
    it = std::find_if(fences.begin(), fences.end(), [&](auto& fence) {
        if (fence->IsOwned()) {
            return false;
        }
        device.waitForFences({*fence}, true, WaitTimeout);
        device.resetFences({*fence});
        fence->RemoveUsage();
        return true;
    });
    if (it != fences.end()) {
        auto& fence = *it;
        fence->Commit();
        return *fence;
    }

    // All fences are owned. Panic.
    // TODO(Rodrigo): Allocate more fences
    UNREACHABLE_MSG("Reached fence limit.");
}

vk::CommandBuffer VulkanResourceManager::CommitCommandBuffer(VulkanFence& fence) {
    return CommitFreeResource(command_buffers, fence);
}

vk::Semaphore VulkanResourceManager::CommitSemaphore(VulkanFence& fence) {
    return *CommitFreeResource(semaphores, fence);
}

template <typename T>
T& VulkanResourceManager::CommitFreeResource(ResourceVector<T>& resources,
                                             VulkanFence& commit_fence) {
    // TODO(Rodrigo): Optimize searching with a last-free-index to avoid searching always from the
    // beginning of the vector.
    for (std::size_t i = 0; i < resources.size(); ++i) {
        auto& resource = resources[i];
        if (resource->TryCommit(commit_fence)) {
            return resource->Get();
        }
    }
    auto& resource = resources[0];
    resource->Commit(commit_fence);
    return resource->Get();
}

void VulkanResourceManager::CreateFences() {
    const vk::FenceCreateInfo fence_ci;

    fences.resize(FENCES_COUNT);
    std::generate(fences.begin(), fences.end(), [&]() {
        return std::make_unique<VulkanFence>(device.createFenceUnique(fence_ci), device,
                                             fences_mutex);
    });
}

void VulkanResourceManager::CreateCommands() {
    const vk::CommandPoolCreateInfo pool_ci(vk::CommandPoolCreateFlagBits::eTransient |
                                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                            graphics_family);
    command_pool = device.createCommandPoolUnique(pool_ci);

    const vk::CommandBufferAllocateInfo cmdbuf_ai(*command_pool, vk::CommandBufferLevel::ePrimary,
                                                  COMMAND_BUFFERS_COUNT);
    const auto cmdbufs = device.allocateCommandBuffers(cmdbuf_ai);

    command_buffers.resize(COMMAND_BUFFERS_COUNT);
    for (u32 i = 0; i < COMMAND_BUFFERS_COUNT; ++i) {
        command_buffers[i] =
            std::make_unique<VulkanResourceEntry<vk::CommandBuffer>>(cmdbufs[i], device);
    }
}

void VulkanResourceManager::CreateSemaphores() {
    semaphores.resize(SEMAPHORES_COUNT);
    for (u32 i = 0; i < SEMAPHORES_COUNT; ++i) {
        const vk::SemaphoreCreateInfo semaphore_ci;
        semaphores[i] = std::make_unique<VulkanResourceEntry<vk::UniqueSemaphore>>(
            device.createSemaphoreUnique(semaphore_ci), device);
    }
}

} // namespace Vulkan
