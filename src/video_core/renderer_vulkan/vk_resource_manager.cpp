// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"

namespace Vulkan {

// TODO(Rodrigo): Fine tune these numbers.
constexpr u32 COMMAND_BUFFERS_COUNT = 0x1000;
constexpr u32 SEMAPHORES_COUNT = 0x1000;

constexpr u32 FENCES_COUNT = 0x2000;
constexpr u32 FENCES_GROW_STEP = 0x1000;

VulkanResource::VulkanResource() = default;

VulkanResource::~VulkanResource() = default;

VulkanResourcePersistent::VulkanResourcePersistent(VulkanResourceManager& resource_manager,
                                                   vk::Device device,
                                                   std::mutex& external_fences_mutex)
    : VulkanResource(), resource_manager(resource_manager), device(device),
      external_fences_mutex(external_fences_mutex) {

    const vk::SemaphoreCreateInfo semaphore_ci;
    write_semaphore = device.createSemaphoreUnique(semaphore_ci);
}

VulkanResourcePersistent::~VulkanResourcePersistent() = default;

void VulkanResourcePersistent::Wait() {
    // Wait for all fences, we can't write unless all previous operations have finished.
    // Lock fences to avoid external fence changes while we loop.
    std::unique_lock external_fences_lock(external_fences_mutex);

    // If there's a softlock here you may have forgotten to call Release.
    for (auto* fence : read_fences) {
        const bool is_free = fence->Tick(true, true);
        ASSERT(is_free);
    }
    if (write_fence) {
        const bool is_free = write_fence->Tick(true, true);
        ASSERT(is_free);
    }
}

vk::Semaphore VulkanResourcePersistent::ReadProtect(VulkanFence& new_fence) {
    std::unique_lock ownership_lock(ownership_mutex);
    std::unique_lock fence_change_lock(fence_change_mutex);
    std::unique_lock external_fences_lock(external_fences_mutex);

    new_fence.UnsafeProtect(this);
    read_fences.push_back(&new_fence);

    return *write_semaphore;
}

vk::Semaphore VulkanResourcePersistent::WriteProtect(VulkanFence& new_fence) {
    std::unique_lock ownership_lock(ownership_mutex);
    Wait();

    // There's a bug if the resource is not free after waiting for all of its fences.
    ASSERT(read_fences.empty());
    ASSERT(write_fence == nullptr);

    // Add current the new fence.
    std::unique_lock lock(fence_change_mutex);
    new_fence.UnsafeProtect(this);
    write_fence = &new_fence;

    return *write_semaphore;
}

void VulkanResourcePersistent::NotifyFenceRemoval(VulkanFence* signaling_fence) {
    std::unique_lock lock(fence_change_mutex);

    if (write_fence == signaling_fence) {
        write_fence = nullptr;
    }
    const auto it = std::find(read_fences.begin(), read_fences.end(), signaling_fence);
    if (it != read_fences.end()) {
        read_fences.erase(it);
    }
}

VulkanResourceTransient::VulkanResourceTransient(vk::Device device)
    : VulkanResource(), device(device) {}

VulkanResourceTransient::~VulkanResourceTransient() = default;

bool VulkanResourceTransient::TryCommit(VulkanFence& commit_fence) {
    std::unique_lock lock(mutex);
    return UnsafeTryCommit(commit_fence);
}

void VulkanResourceTransient::Commit(VulkanFence& commit_fence) {
    std::unique_lock lock(mutex);
    if (fence) {
        device.waitForFences({*fence}, true, WaitTimeout);
    }
    const bool available = UnsafeTryCommit(commit_fence);
    ASSERT_MSG(available, "Unexpected race condition");
}

void VulkanResourceTransient::NotifyFenceRemoval(VulkanFence* signaling_fence) {
    std::unique_lock lock(mutex);
    ASSERT(fence && fence == signaling_fence);
    ASSERT(is_claimed);
    fence = nullptr;
    is_claimed = false;
}

bool VulkanResourceTransient::UnsafeTryCommit(VulkanFence& commit_fence) {
    if (is_claimed) {
        return false;
    }
    is_claimed = true;
    fence = &commit_fence;
    commit_fence.UnsafeProtect(this);
    return true;
}

VulkanFence::VulkanFence(vk::UniqueFence handle, vk::Device device, std::mutex& mutex)
    : handle(std::move(handle)), device(device), fences_mutex(mutex) {}

VulkanFence::~VulkanFence() = default;

void VulkanFence::Wait() {
    std::unique_lock lock(fences_mutex);
    device.waitForFences({*handle}, true, WaitTimeout);
}

void VulkanFence::Release() {
    std::unique_lock wait_lock(wait_mutex);
    if (is_being_waited) {
        ownership_watch.notify_all();
        // is_owned will be reseted by the waiter thread.
    } else {
        std::unique_lock fences_lock(fences_mutex);
        is_owned = false;
    }
}

void VulkanFence::Commit() {
    is_owned = true;
    is_used = true;
}

bool VulkanFence::Tick(bool gpu_wait, bool owner_wait) {
    if (!is_used) {
        // If a fence is not used it's always free.
        return true;
    }
    if (is_owned) {
        if (!owner_wait) {
            // The fence is still being owned (Release has not been called) and ownership wait has
            // not been asked.
            return false;
        }

        {
            std::unique_lock wait_lock(wait_mutex);
            is_being_waited = true;
        }

        std::unique_lock owner_lock(ownership_mutex);
        ownership_watch.wait(owner_lock);

        {
            std::unique_lock wait_lock(wait_mutex);
            is_being_waited = false;
        }
    }
    if (gpu_wait) {
        // Wait for the fence if it has been requested.
        device.waitForFences({*handle}, true, WaitTimeout);
    } else {
        if (device.getFenceStatus(*handle) != vk::Result::eSuccess) {
            // Vulkan fence is not ready, not much it can do here
            return false;
        }
    }

    // Broadcast resources their free state.
    for (auto* resource : protected_resources) {
        resource->NotifyFenceRemoval(this);
    }
    // TODO(Rodrigo): Find a way to preserve vector's allocated memory.
    protected_resources.clear();

    // Prepare fence for reusage.
    device.resetFences({*handle});
    is_used = false;
    return true;
}

void VulkanFence::Protect(VulkanResource* resource) {
    std::unique_lock lock(fences_mutex);
    UnsafeProtect(resource);
}

void VulkanFence::Unprotect(VulkanResource* resource) {
    std::unique_lock lock(fences_mutex);
    const auto it = std::find(protected_resources.begin(), protected_resources.end(), resource);
    if (it != protected_resources.end()) {
        protected_resources.erase(it);
    }
}

void VulkanFence::UnsafeProtect(VulkanResource* resource) {
    protected_resources.push_back(resource);
}

VulkanFenceWatch::VulkanFenceWatch() = default;

VulkanFenceWatch::~VulkanFenceWatch() {
    if (fence) {
        fence->Unprotect(this);
    }
}

void VulkanFenceWatch::Wait() {
    std::unique_lock lock(mutex);
    if (!fence) {
        return;
    }
    fence->Wait();
    fence = nullptr;
}

void VulkanFenceWatch::Watch(VulkanFence& new_fence) {
    Wait();

    std::unique_lock lock(mutex);
    fence = &new_fence;
    fence->Protect(this);
}

void VulkanFenceWatch::NotifyFenceRemoval(VulkanFence* signaling_fence) {
    std::unique_lock lock(mutex);
    fence = nullptr;
}

VulkanResourceManager::VulkanResourceManager(const VulkanDevice& device_handler)
    : device(device_handler.GetLogical()), graphics_family(device_handler.GetGraphicsFamily()) {

    GrowFences(FENCES_COUNT);
    CreateCommands();
    CreateSemaphores();
}

VulkanResourceManager::~VulkanResourceManager() = default;

VulkanFence& VulkanResourceManager::CommitFence() {
    std::unique_lock lock(fences_mutex);

    auto StepFences = [&](bool gpu_wait, bool owner_wait) -> VulkanFence* {
        const auto it = std::find_if(fences.begin(), fences.end(), [&](auto& fence) {
            return fence->Tick(gpu_wait, owner_wait);
        });
        if (it != fences.end()) {
            auto& fence = *it;
            fence->Commit();
            return fence.get();
        }
        return nullptr;
    };

    VulkanFence* found_fence = StepFences(false, false);
    if (!found_fence) {
        // Try again, this time waiting.
        found_fence = StepFences(true, false);

        if (!found_fence) {
            // Allocate new fences and try again.
            LOG_INFO(Render_Vulkan, "Allocating new fences {} -> {}", fences.size(),
                     fences.size() + FENCES_GROW_STEP);

            GrowFences(FENCES_GROW_STEP);
            found_fence = StepFences(true, false);
            ASSERT(found_fence != nullptr);
        }
    }
    return *found_fence;
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

void VulkanResourceManager::GrowFences(std::size_t new_fences_count) {
    const vk::FenceCreateInfo fence_ci;

    const std::size_t previous_size = fences.size();
    fences.resize(previous_size + new_fences_count);

    std::generate(fences.begin() + previous_size, fences.end(), [&]() {
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
