// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanFencedPool;
class VulkanResourceManager;
class VulkanSemaphorePool;
class VulkanCommandBufferPool;

namespace Resource {

class Base {
    friend class VulkanFence;

public:
    explicit Base();
    virtual ~Base() = 0;

protected:
    /**
     * Signals the object that an owning fence has been signaled.
     * @param signaling_fence Fence that signals its usage end.
     */
    virtual void OnFenceRemoval(VulkanFence* signaling_fence) = 0;
};

/**
 * Persistent resources are those that have a prolonged lifetime and you can read and write to them.
 * An use case example for this kind of resource are images.
 */
class Persistent : public Base {
    friend class VulkanResourceManager;

public:
    explicit Persistent(VulkanResourceManager& resource_manager, vk::Device device);
    virtual ~Persistent();

    /// Waits for all operations to finish.
    void Wait();

    /**
     * Takes partial ownership of the resource to read from it. Read operations have to signal it
     * when they finish.
     * @param new_fence Fence to hold the right to read from the resource.
     * @returns A semaphore that's going to be signaled (or is already signaled) when the write
     * operation finishes. Read operations have to be protected with it.
     */
    vk::Semaphore ReadProtect(VulkanFence& new_fence);

    /**
     * Takes exclusive ownership of the resource to write to it. It will wait for all pending read
     * and write operations if they exist.
     * @param new_fence Fence to hold the write to write to the resource.
     * @returns A semaphore that has to be signaled when the write operation finishes.
     */
    vk::Semaphore WriteProtect(VulkanFence& new_fence);

protected:
    void OnFenceRemoval(VulkanFence* signaling_fence) override;

private:
    VulkanResourceManager& resource_manager;
    const vk::Device device;

    vk::UniqueSemaphore write_semaphore;

    std::vector<VulkanFence*> read_fences; ///< Fence protecting read operations.

    VulkanFence* write_fence; ///< Fence protecting write operations. Null when it's free.
};

/**
 * One shot resources are those that are created and discarded after usage. An example of these are
 * renderpasses.
 */
class OneShot : public Base {
public:
    explicit OneShot();
    virtual ~OneShot();

    bool IsSignaled() const;

protected:
    void OnFenceRemoval(VulkanFence* signaling_fence) override;

private:
    bool is_signaled{};
};

template <typename T>
class OneShotEntry final : public OneShot {
public:
    OneShotEntry(vk::UniqueHandle<T> resource) : resource(std::move(resource)) {}
    virtual ~OneShotEntry() = default;

    /// Retreives the resource.
    T GetHandle() const {
        return *resource;
    }

private:
    vk::UniqueHandle<T> resource;
};

} // namespace Resource

using VulkanResource = Resource::Base;

/**
 * Fences take ownership of objects, protecting them from GPU-side or driver-side race conditions.
 * They must be commited from the resouce manager. Their use flow is: commit the fence from the
 * manager, protect resources with it and use them, send the fence to a execution queue and Wait for
 * it if needed and then call Release. Used resources will automatically be signaled by the manager
 * when they are free to be reused.
 */
class VulkanFence {
    friend class Resource::Persistent;
    friend class VulkanResourceManager;

public:
    explicit VulkanFence(vk::UniqueFence handle, vk::Device device);
    ~VulkanFence();

    /**
     * Waits for the fence to be signaled.
     * @warning You must have ownership of the fence and it has to be previously sent to a queue to
     * call this function.
     */
    void Wait();

    /**
     * Releases ownership of the fence. Call after it has been sent to an execution queue.
     * Unmanaged usage of the fence after the call will result in undefined behavior because it may
     * be being used for something else.
     */
    void Release();

    /**
     * Protects a resource with this fence.
     * @param resource Resource to protect.
     */
    void Protect(VulkanResource* resource);

    /**
     * Removes protection for a resource.
     * @param resource Resource to unprotect.
     */
    void Unprotect(VulkanResource* resource);

    /// Retreives the fence.
    operator vk::Fence() const {
        return *handle;
    }

private:
    /// Take ownership of the fence.
    void Commit();

    /**
     * Updates the fence status.
     * @warning Waiting for the owner might soft lock the execution.
     * @param gpu_wait Wait for the fence to be signaled by the driver.
     * @param owner_wait Wait for the owner to signal its freedom.
     * @returns True if the fence is free. Waiting for gpu and owner will always return true.
     */
    bool Tick(bool gpu_wait, bool owner_wait);

    const vk::Device device;

    vk::UniqueFence handle;
    std::vector<VulkanResource*> protected_resources;
    bool is_owned{}; /// The fence has been commited but not released yet.
    bool is_used{};  /// The fence has been commited but it has not been checked to be free.
};

class VulkanFenceWatch final : public VulkanResource {
public:
    explicit VulkanFenceWatch();
    ~VulkanFenceWatch();

    /// Waits for a watched fence if it is bound.
    void Wait();

    /**
     * Waits for a previous fence and watches a new one.
     * @param new_fence New fence to wait to.
     */
    void Watch(VulkanFence& new_fence);

    /**
     * Checks if it's currently being watched and starts watching it if it's available.
     * @returns True if a watch has started, false if it's being watched.
     */
    bool TryWatch(VulkanFence& new_fence);

protected:
    void OnFenceRemoval(VulkanFence* signaling_fence) override;

private:
    VulkanFence* fence{};
};

class VulkanFencedPool {
public:
    explicit VulkanFencedPool(std::size_t initial_capacity, std::size_t grow_step);
    explicit VulkanFencedPool(std::size_t capacity);
    virtual ~VulkanFencedPool();

protected:
    virtual void Allocate(std::size_t begin, std::size_t end);

    std::size_t ResourceCommit(VulkanFence& fence);

private:
    std::size_t HandleFullPool();

    void Grow(std::size_t new_entries);

    const bool does_allocation;
    const std::size_t grow_step;

    std::size_t free_iterator = 0;
    std::vector<std::unique_ptr<VulkanFenceWatch>> watches;
};

/**
 * The resource manager handles all resources that can be protected with a fence avoiding
 * driver-side or GPU-side race conditions. Use flow is documented in VulkanFence.
 */
class VulkanResourceManager final {
public:
    explicit VulkanResourceManager(const VulkanDevice& device_handler);
    ~VulkanResourceManager();

    /// Commits a fence. It has to be sent to a queue and released.
    VulkanFence& CommitFence();

    vk::CommandBuffer CommitCommandBuffer(VulkanFence& fence);

    vk::Semaphore CommitSemaphore(VulkanFence& fence);

    vk::RenderPass CreateRenderPass(VulkanFence& fence,
                                    const vk::RenderPassCreateInfo& renderpass_ci);

    vk::ImageView CreateImageView(VulkanFence& fence, const vk::ImageViewCreateInfo& image_view_ci);

    vk::Framebuffer CreateFramebuffer(VulkanFence& fence,
                                      const vk::FramebufferCreateInfo& framebuffer_ci);

    vk::Pipeline CreateGraphicsPipeline(VulkanFence& fence,
                                        const vk::GraphicsPipelineCreateInfo& graphics_pipeline_ci);

    vk::PipelineLayout CreatePipelineLayout(VulkanFence& fence,
                                            const vk::PipelineLayoutCreateInfo& pipeline_layout_ci);

private:
    using RenderPassEntry = Resource::OneShotEntry<vk::RenderPass>;
    using ImageViewEntry = Resource::OneShotEntry<vk::ImageView>;
    using FramebufferEntry = Resource::OneShotEntry<vk::Framebuffer>;
    using PipelineEntry = Resource::OneShotEntry<vk::Pipeline>;
    using PipelineLayoutEntry = Resource::OneShotEntry<vk::PipelineLayout>;

    template <typename EntryType, typename HandleType>
    HandleType CreateOneShot(VulkanFence& fence, std::vector<std::unique_ptr<EntryType>>& vector,
                             vk::UniqueHandle<HandleType> handle);

    void TickCreations();

    template <typename T>
    void RemoveEntries(std::vector<T>& entries);

    void GrowFences(std::size_t new_fences_count);

    const vk::Device device;
    const u32 graphics_family;

    std::vector<std::unique_ptr<VulkanFence>> fences;

    std::unique_ptr<VulkanCommandBufferPool> command_buffer_pool;
    std::unique_ptr<VulkanSemaphorePool> semaphore_pool;

    u32 tick_creations{};

    std::vector<std::unique_ptr<RenderPassEntry>> renderpasses;
    std::vector<std::unique_ptr<ImageViewEntry>> image_views;
    std::vector<std::unique_ptr<FramebufferEntry>> framebuffers;
    std::vector<std::unique_ptr<PipelineEntry>> pipelines;
    std::vector<std::unique_ptr<PipelineLayoutEntry>> pipeline_layouts;
};

} // namespace Vulkan