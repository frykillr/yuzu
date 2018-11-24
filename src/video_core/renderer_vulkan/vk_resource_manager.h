// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VKDevice;
class VKFence;
class VKFencedPool;
class VKResourceManager;
class CommandBufferPool;

namespace Resource {

class Base {
    friend class VKFence;

public:
    explicit Base();
    virtual ~Base() = 0;

protected:
    /**
     * Signals the object that an owning fence has been signaled.
     * @param signaling_fence Fence that signals its usage end.
     */
    virtual void OnFenceRemoval(VKFence* signaling_fence) = 0;
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
    void OnFenceRemoval(VKFence* signaling_fence) override;

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

using VKResource = Resource::Base;

/**
 * Fences take ownership of objects, protecting them from GPU-side or driver-side race conditions.
 * They must be commited from the resouce manager. Their use flow is: commit the fence from the
 * manager, protect resources with it and use them, send the fence to a execution queue and Wait for
 * it if needed and then call Release. Used resources will automatically be signaled by the manager
 * when they are free to be reused.
 */
class VKFence {
    friend class VKResourceManager;

public:
    explicit VKFence(vk::UniqueFence handle, vk::Device device);
    ~VKFence();

    /**
     * Waits for the fence to be signaled.
     * @warning You must have ownership of the fence and it has to be previously sent to a queue to
     * call this function.
     */
    void Wait();

    /**
     * Releases ownership of the fence. Pass after it has been sent to an execution queue.
     * Unmanaged usage of the fence after the call will result in undefined behavior because it may
     * be being used for something else.
     */
    void Release();

    /**
     * Protects a resource with this fence.
     * @param resource Resource to protect.
     */
    void Protect(VKResource* resource);

    /**
     * Removes protection for a resource.
     * @param resource Resource to unprotect.
     */
    void Unprotect(VKResource* resource);

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
    std::vector<VKResource*> protected_resources;
    bool is_owned = false; /// The fence has been commited but not released yet.
    bool is_used = false;  /// The fence has been commited but it has not been checked to be free.
};

class VKFenceWatch final : public VKResource {
public:
    explicit VKFenceWatch();
    ~VKFenceWatch();

    /// Waits for a watched fence if it is bound.
    void Wait();

    /**
     * Waits for a previous fence and watches a new one.
     * @param new_fence New fence to wait to.
     */
    void Watch(VKFence& new_fence);

    /**
     * Checks if it's currently being watched and starts watching it if it's available.
     * @returns True if a watch has started, false if it's being watched.
     */
    bool TryWatch(VKFence& new_fence);

protected:
    void OnFenceRemoval(VKFence* signaling_fence) override;

private:
    VKFence* fence{};
};

class VKFencedPool {
public:
    explicit VKFencedPool();
    virtual ~VKFencedPool();

    void InitResizable(std::size_t initial_capacity, std::size_t grow_step);
    void InitStatic(std::size_t capacity);

protected:
    virtual void Allocate(std::size_t begin, std::size_t end);

    std::size_t ResourceCommit(VKFence& fence);

private:
    std::size_t HandleFullPool();

    void Grow(std::size_t new_entries);

    bool does_allocation{};
    std::size_t grow_step{};

    std::size_t free_iterator = 0;
    std::vector<std::unique_ptr<VKFenceWatch>> watches;
};

/**
 * The resource manager handles all resources that can be protected with a fence avoiding
 * driver-side or GPU-side race conditions. Use flow is documented in VKFence.
 */
class VKResourceManager final {
public:
    explicit VKResourceManager(const VKDevice& device_handler);
    ~VKResourceManager();

    /// Commits a fence. It has to be sent to a queue and released.
    VKFence& CommitFence();

    vk::CommandBuffer CommitCommandBuffer(VKFence& fence);

    vk::RenderPass CreateRenderPass(VKFence& fence, const vk::RenderPassCreateInfo& renderpass_ci);

    vk::ImageView CreateImageView(VKFence& fence, const vk::ImageViewCreateInfo& image_view_ci);

    vk::Framebuffer CreateFramebuffer(VKFence& fence,
                                      const vk::FramebufferCreateInfo& framebuffer_ci);

    vk::Pipeline CreateGraphicsPipeline(VKFence& fence,
                                        const vk::GraphicsPipelineCreateInfo& graphics_pipeline_ci);

    vk::PipelineLayout CreatePipelineLayout(VKFence& fence,
                                            const vk::PipelineLayoutCreateInfo& pipeline_layout_ci);

private:
    using RenderPassEntry = Resource::OneShotEntry<vk::RenderPass>;
    using ImageViewEntry = Resource::OneShotEntry<vk::ImageView>;
    using FramebufferEntry = Resource::OneShotEntry<vk::Framebuffer>;
    using PipelineEntry = Resource::OneShotEntry<vk::Pipeline>;
    using PipelineLayoutEntry = Resource::OneShotEntry<vk::PipelineLayout>;

    template <typename EntryType, typename HandleType>
    HandleType CreateOneShot(VKFence& fence, std::vector<std::unique_ptr<EntryType>>& vector,
                             vk::UniqueHandle<HandleType> handle);

    void TickCreations();

    template <typename T>
    void RemoveEntries(std::vector<T>& entries);

    void GrowFences(std::size_t new_fences_count);

    const vk::Device device;
    const u32 graphics_family;

    std::vector<std::unique_ptr<VKFence>> fences;
    std::size_t fences_iterator = 0;

    std::unique_ptr<CommandBufferPool> command_buffer_pool;

    u32 tick_creations{};

    std::vector<std::unique_ptr<RenderPassEntry>> renderpasses;
    std::vector<std::unique_ptr<ImageViewEntry>> image_views;
    std::vector<std::unique_ptr<FramebufferEntry>> framebuffers;
    std::vector<std::unique_ptr<PipelineEntry>> pipelines;
    std::vector<std::unique_ptr<PipelineLayoutEntry>> pipeline_layouts;
};

} // namespace Vulkan