// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanResourceManager;

class VulkanResource {
    friend class VulkanFence;

public:
    explicit VulkanResource();
    virtual ~VulkanResource() = 0;

protected:
    /**
     * Signals the object that an owning fence has been signaled.
     * @param signaling_fence Fence that signals its usage end.
     * @remarks Thread safe.
     */
    virtual void NotifyFenceRemoval(VulkanFence* signaling_fence) = 0;
};

/**
 * Persistent resources are those that have a prolonged lifetime and you can read and write to them.
 * An use case example for this kind of resource are images.
 */
class VulkanResourcePersistent : public VulkanResource {
    friend class VulkanResourceManager;

public:
    explicit VulkanResourcePersistent(VulkanResourceManager& resource_manager, vk::Device device,
                                      std::mutex& external_fences_mutex);
    virtual ~VulkanResourcePersistent();

    /**
     * Waits for all operations to finish.
     * @remarks Thread safe.
     */
    void Wait();

    /**
     * Takes partial ownership of the resource to read from it. Read operations have to signal it
     * when they finish.
     * @param new_fence Fence to hold the right to read from the resource.
     * @returns A semaphore that's going to be signaled (or is already signaled) when the write
     * operation finishes. Read operations have to be protected with it.
     * @remarks Thread safe.
     */
    vk::Semaphore ReadProtect(VulkanFence& new_fence);

    /**
     * Takes exclusive ownership of the resource to write to it. It will wait for all pending read
     * and write operations if they exist.
     * @param new_fence Fence to hold the write to write to the resource.
     * @returns A semaphore that has to be signaled when the write operation finishes.
     * @remarks Thread safe.
     */
    vk::Semaphore WriteProtect(VulkanFence& new_fence);

protected:
    virtual void NotifyFenceRemoval(VulkanFence* signaling_fence);

private:
    VulkanResourceManager& resource_manager;
    const vk::Device device;

    vk::UniqueSemaphore write_semaphore;

    std::vector<VulkanFence*> read_fences; ///< Fence protecting read operations.

    VulkanFence* write_fence; ///< Fence protecting write operations. Null when it's free.

    std::mutex& external_fences_mutex; ///< Protects managed fence changes.
    std::mutex ownership_mutex;        ///< Protects ownership changes.
    std::mutex fence_change_mutex;     ///< Protects internal changes in the owned fences.
};

template <typename T>
class VulkanResourcePersistentEntry : public VulkanResourcePersistent {
public:
    VulkanResourcePersistentEntry(VulkanResourceManager& resource_manager, vk::Device device,
                                  std::mutex& external_fences_mutex, vk::UniqueHandle<T> handle)
        : VulkanResourcePersistent(resource_manager, device, external_fences_mutex),
          handle(std::move(handle)) {}
    ~VulkanResourcePersistentEntry() = default;

    T GetHandle() const {
        return *handle;
    }

private:
    vk::UniqueHandle<T> handle;
};

using VulkanImage = VulkanResourcePersistentEntry<vk::Image>;

/**
 * Transient resources are those you just use and discard for reusage. A simple example of this are
 * semaphores and command buffers.
 */
class VulkanResourceTransient : public VulkanResource {
    friend class VulkanResourceManager;

public:
    explicit VulkanResourceTransient(vk::Device device);
    virtual ~VulkanResourceTransient();

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

protected:
    virtual void NotifyFenceRemoval(VulkanFence* signaling_fence) override;

private:
    /// Backend for TryCommit and Commit, thread unsafe.
    bool UnsafeTryCommit(VulkanFence& commit_fence);

    const vk::Device device;

    VulkanFence* fence{};
    bool is_claimed{};
    std::mutex mutex;
};

template <typename T>
class VulkanResourceEntry final : public VulkanResourceTransient {
public:
    VulkanResourceEntry(T resource, vk::Device device)
        : VulkanResourceTransient(device), resource(std::move(resource)) {}
    virtual ~VulkanResourceEntry() = default;

    /// Retreives the resource.
    T& Get() {
        return resource;
    }

private:
    T resource;
};

/**
 * Fences take ownership of objects, protecting them from GPU-side or driver-side race conditions.
 * They must be commited from the resouce manager. Their use flow is: commit the fence from the
 * manager, protect resources with it and use them, send the fence to a execution queue and Wait for
 * it if needed and then call Release. Used resources will automatically be signaled by the manager
 * when they are free to be reused.
 */
class VulkanFence {
    friend class VulkanResourcePersistent;
    friend class VulkanResourceTransient;
    friend class VulkanResourceManager;

public:
    explicit VulkanFence(vk::UniqueFence handle, vk::Device device, std::mutex& mutex);
    ~VulkanFence();

    /**
     * Waits for the fence to be signaled.
     * @warning You must have ownership of the fence and it has to be previously sent to a queue to
     * call this function.
     * @remarks Thread safe.
     */
    void Wait();

    /**
     * Releases ownership of the thread. Call after it has been sent to an execution queue.
     * Unmanaged usage of the fence after the call will result in undefined behavior because it may
     * be being used for something else.
     * @remarks Thread safe.
     */
    void Release();

    /**
     * Protects a resource with this fence.
     * @param resource Resource to protect.
     * @remarks Thread safe.
     */
    void Protect(VulkanResource* resource);

    /// Retreives the fence.
    operator vk::Fence() const {
        return *handle;
    }

private:
    /// Take ownership of the fence.
    void Commit();

    /**
     * Updates the fence status.
     * @warning Thread unsafe, it must be externally synchronized.
     * @warning Waiting for the owner might soft lock the execution.
     * @param gpu_wait Wait for the fence to be signaled by the driver.
     * @param owner_wait Wait for the owner to signal its freedom.
     * @returns True if the fence is free. Waiting for gpu and owner will always return true.
     */
    bool Tick(bool gpu_wait, bool owner_wait);

    /// Backend for Protect
    void UnsafeProtect(VulkanResource* resource);

    const vk::Device device;
    std::mutex& fences_mutex;

    std::mutex ownership_mutex;
    std::mutex wait_mutex;
    std::condition_variable ownership_watch;

    vk::UniqueFence handle;
    std::vector<VulkanResource*> protected_resources;
    bool is_owned{}; /// The fence has been commited but not released yet.
    bool is_used{};  /// The fence has been commited but it has not been checked to be free.

    bool is_being_waited{};
};

/**
 * The resource manager handles all resources that can be protected with a fence avoiding
 * driver-side or GPU-side race conditions. Use flow is documented in VulkanFence.
 * All public methods are thread safe.
 */
class VulkanResourceManager final {
public:
    explicit VulkanResourceManager(const VulkanDevice& device_handler);
    ~VulkanResourceManager();

    /// Commits a fence. It has to be sent to a queue and released.
    VulkanFence& CommitFence();

    vk::CommandBuffer CommitCommandBuffer(VulkanFence& fence);

    vk::Semaphore CommitSemaphore(VulkanFence& fence);

    std::unique_ptr<VulkanImage> CreateImage(const vk::ImageCreateInfo& image_ci);

private:
    template <typename T>
    using ResourceVector = std::vector<std::unique_ptr<VulkanResourceEntry<T>>>;

    template <typename T>
    T& CommitFreeResource(ResourceVector<T>& resources, VulkanFence& commit_fence);

    void GrowFences(std::size_t new_fences_count);

    void CreateCommands();
    void CreateDescriptors();
    void CreateSemaphores();

    const vk::Device device;
    const u32 graphics_family;

    std::mutex fences_mutex;
    std::vector<std::unique_ptr<VulkanFence>> fences;

    vk::UniqueCommandPool command_pool;
    ResourceVector<vk::CommandBuffer> command_buffers;

    vk::UniqueDescriptorPool descriptor_pool;
    ResourceVector<vk::DescriptorSet> descriptor_set;

    ResourceVector<vk::UniqueSemaphore> semaphores;
};

} // namespace Vulkan