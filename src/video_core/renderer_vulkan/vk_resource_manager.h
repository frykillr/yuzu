// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <mutex>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_helper.h"

namespace Vulkan {

class VulkanFence;
class VulkanResourceManager;

class VulkanResourceInterface {
    friend class VulkanFence;

public:
    explicit VulkanResourceInterface(vk::Device& device);
    ~VulkanResourceInterface();

    /**
     * Tries to reserve usage of the resource.
     * @remarks Thread safe.
     * @params commit_fence Fence protecting the resource.
     * @returns True if the resource has been claimed.
     */
    bool TryCommit(VulkanFence& commit_fence);

    /**
     * Wait for the resource to be available and commits.
     * @params commit_fence Fence protecting the resource.
     * @remarks Thread safe.
     */
    void Commit(VulkanFence& commit_fence);

private:
    /**
     * Removes the usage of the resource.
     * @remarks Thread safe.
     */
    void RemoveUsage();

    /// Backend for TryCommit and Commit, thread unsafe
    bool UnsafeTryCommit(VulkanFence& commit_fence);

    vk::Device& device;
    VulkanFence* fence{};
    bool is_claimed{};
    std::mutex mutex;
};

template <typename T>
class VulkanResourceEntry final : public VulkanResourceInterface {
public:
    VulkanResourceEntry(T resource, vk::Device& device)
        : VulkanResourceInterface(device), resource(std::move(resource)) {}
    ~VulkanResourceEntry() = default;

    /// Retreives the resource.
    T& Get() {
        return resource;
    }

private:
    T resource;
};

class VulkanFence {
    friend class VulkanResourceInterface;
    friend class VulkanResourceManager;

public:
    explicit VulkanFence(vk::UniqueFence handle, vk::Device& device, std::mutex& mutex);
    ~VulkanFence();

    /**
     * Waits for the fence to be signaled.
     * @warning You must have ownership of the fence to wait.
     * @remarks Thread safe.
     */
    void Wait();

    /**
     * Releases ownership of the thread. Use after it has been sent to a command buffer.
     * @remarks Thread safe.
     */
    void Release();

    /// Retreives the fence.
    operator vk::Fence() {
        return *handle;
    }

private:
    /// Take ownership of the fence.
    void Commit();

    /// Protect a resource with this fence
    void ProtectResource(VulkanResourceInterface* resource);

    /// Remove usage of the fence, call after it has been signaled and it has no owner.
    void RemoveUsage();

    /// Query if the fence is owned.
    bool IsOwned() const;

    /// Query if the fence is being used, it does not ask for ownership.
    bool IsUsed() const;

private:
    const vk::Device device;
    std::mutex& mutex;

    vk::UniqueFence handle;
    std::vector<VulkanResourceInterface*> protected_resources;
    bool is_used{};
    bool is_owned{};
};

class VulkanResourceManager final {
public:
    explicit VulkanResourceManager(vk::Device& device, const u32& graphics_family);
    ~VulkanResourceManager();

    VulkanFence& CommitFence();

    vk::CommandBuffer CommitCommandBuffer(VulkanFence& fence);

    vk::Semaphore CommitSemaphore(VulkanFence& fence);

private:
    template <typename T>
    using ResourceVector = std::vector<std::unique_ptr<VulkanResourceEntry<T>>>;

    template <typename T>
    T& CommitFreeResource(ResourceVector<T>& resources, VulkanFence& commit_fence);

    void CreateFences();
    void CreateCommands();
    void CreateSemaphores();

    vk::Device& device;
    const u32& graphics_family;

    std::mutex fences_mutex;
    std::vector<std::unique_ptr<VulkanFence>> fences;

    vk::UniqueCommandPool command_pool;
    ResourceVector<vk::CommandBuffer> command_buffers;

    ResourceVector<vk::UniqueSemaphore> semaphores;
};

} // namespace Vulkan