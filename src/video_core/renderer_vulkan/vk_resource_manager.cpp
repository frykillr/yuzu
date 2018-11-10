// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <optional>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"

#pragma optimize("", off)

namespace Vulkan {

// TODO(Rodrigo): Fine tune these numbers.
constexpr u32 COMMAND_BUFFERS_COUNT = 0x1000;
constexpr u32 SEMAPHORES_COUNT = 0x1000;

constexpr u32 FENCES_COUNT = 0x2000;
constexpr u32 FENCES_GROW_STEP = 0x1000;

constexpr u32 TICKS_TO_DESTROY = 0x10;
constexpr std::size_t OBJECTS_TO_DESTROY = 0x10;

using namespace Resource;

Base::Base() = default;

Base::~Base() = default;

Persistent::Persistent(VulkanResourceManager& resource_manager, vk::Device device)
    : resource_manager(resource_manager), device(device) {

    const vk::SemaphoreCreateInfo semaphore_ci;
    write_semaphore = device.createSemaphoreUnique(semaphore_ci);
}

Persistent::~Persistent() {
    for (auto& read_fence : read_fences) {
        read_fence->Unprotect(this);
    }
    if (write_fence) {
        write_fence->Unprotect(this);
    }
}

void Persistent::Wait() {
    for (auto* fence : read_fences) {
        const bool is_free = fence->Tick(true, true);
        ASSERT(is_free);
    }
    if (write_fence) {
        const bool is_free = write_fence->Tick(true, true);
        ASSERT(is_free);
    }
}

vk::Semaphore Persistent::ReadProtect(VulkanFence& new_fence) {
    new_fence.Protect(this);
    read_fences.push_back(&new_fence);

    return *write_semaphore;
}

vk::Semaphore Persistent::WriteProtect(VulkanFence& new_fence) {
    Wait();

    // There's a bug if the resource is not free after waiting for all of its fences.
    ASSERT(read_fences.empty());
    ASSERT(write_fence == nullptr);

    // Add current the new fence.
    new_fence.Protect(this);
    write_fence = &new_fence;

    return *write_semaphore;
}

void Persistent::OnFenceRemoval(VulkanFence* signaling_fence) {
    if (write_fence == signaling_fence) {
        write_fence = nullptr;
    }
    const auto it = std::find(read_fences.begin(), read_fences.end(), signaling_fence);
    if (it != read_fences.end()) {
        read_fences.erase(it);
    }
}

OneShot::OneShot() = default;

OneShot::~OneShot() {
    ASSERT_MSG(is_signaled, "Destroying a one shot resource that's still marked as used");
}

bool OneShot::IsSignaled() const {
    return is_signaled;
}

void OneShot::OnFenceRemoval(VulkanFence* signaling_fence) {
    is_signaled = true;
}

VulkanFence::VulkanFence(vk::UniqueFence handle, vk::Device device)
    : handle(std::move(handle)), device(device) {}

VulkanFence::~VulkanFence() = default;

void VulkanFence::Wait() {
    device.waitForFences({*handle}, true, WaitTimeout);
}

void VulkanFence::Release() {
    is_owned = false;
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
    if (is_owned && !owner_wait) {
        // The fence is still being owned (Release has not been called) and ownership wait has
        // not been asked.
        return false;
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
        resource->OnFenceRemoval(this);
    }
    protected_resources.clear();

    // Prepare fence for reusage.
    device.resetFences({*handle});
    is_used = false;
    return true;
}

void VulkanFence::Protect(VulkanResource* resource) {
    protected_resources.push_back(resource);
}

void VulkanFence::Unprotect(VulkanResource* resource) {
    const auto it = std::find(protected_resources.begin(), protected_resources.end(), resource);
    if (it != protected_resources.end()) {
        protected_resources.erase(it);
    }
}

VulkanFenceWatch::VulkanFenceWatch() = default;

VulkanFenceWatch::~VulkanFenceWatch() {
    if (fence) {
        fence->Unprotect(this);
    }
}

void VulkanFenceWatch::Wait() {
    if (!fence) {
        return;
    }
    fence->Wait();
    fence->Unprotect(this);
    fence = nullptr;
}

void VulkanFenceWatch::Watch(VulkanFence& new_fence) {
    Wait();
    fence = &new_fence;
    fence->Protect(this);
}

bool VulkanFenceWatch::TryWatch(VulkanFence& new_fence) {
    if (fence) {
        return false;
    }
    fence = &new_fence;
    fence->Protect(this);
    return true;
}

void VulkanFenceWatch::OnFenceRemoval(VulkanFence* signaling_fence) {
    ASSERT(signaling_fence == fence);
    fence = nullptr;
}

VulkanFencedPool::VulkanFencedPool(std::size_t initial_capacity, std::size_t grow_step)
    : does_allocation(true), grow_step(grow_step) {
    Grow(initial_capacity);
}

VulkanFencedPool::VulkanFencedPool(std::size_t capacity) : grow_step(0), does_allocation(false) {
    Grow(capacity);
}

VulkanFencedPool::~VulkanFencedPool() = default;

void VulkanFencedPool::Allocate(std::size_t begin, std::size_t end) {
    UNREACHABLE_MSG("Trying to allocate without an alloc implementation.");
}

std::size_t VulkanFencedPool::ResourceCommit(VulkanFence& fence) {
    const auto Search = [&](std::size_t begin, std::size_t end) -> std::optional<std::size_t> {
        for (std::size_t iterator = begin; iterator < end; ++iterator) {
            if (!watches[free_iterator]->TryWatch(fence)) {
                // The resource is now being watched, a free resource was successfully found.
                return iterator;
            }
        }
        return {};
    };
    // Try to find a free resource from the hinted position to the end.
    auto found = Search(free_iterator, watches.size());
    if (!found) {
        // Search from beginning to the hinted position.
        found = Search(0, free_iterator);
        if (!found) {
            // Both searches failed, the pool is full; handle it.
            const std::size_t free_resource = HandleFullPool();

            // Watch will wait for the resource to be free.
            watches[free_resource]->Watch(fence);
            found = free_resource;
        }
    }
    // Free iterator is hinted to the resource after the one that's been commited.
    free_iterator = (*found + 1) % watches.size();
    return *found;
}

std::size_t VulkanFencedPool::HandleFullPool() {
    if (!does_allocation) {
        // This pool doesn't allocate, just wait for a resource to be free.
        return free_iterator;
    }
    const std::size_t old_capacity = watches.size();
    Grow(grow_step);

    // The last entry is guaranted to be free, since it's the first element of the freshly
    // allocated resources.
    return old_capacity;
}

void VulkanFencedPool::Grow(std::size_t new_entries) {
    const auto old_capacity = watches.size();
    watches.resize(old_capacity + new_entries);
    std::generate(watches.begin() + old_capacity, watches.end(),
                  []() { return std::make_unique<VulkanFenceWatch>(); });
    if (does_allocation) {
        Allocate(old_capacity, old_capacity + new_entries);
    }
}

class VulkanCommandBufferPool final : public VulkanFencedPool {
public:
    VulkanCommandBufferPool(vk::Device device, u32 graphics_family)
        : VulkanFencedPool(COMMAND_BUFFERS_COUNT) {

        const vk::CommandPoolCreateInfo pool_ci(
            vk::CommandPoolCreateFlagBits::eTransient |
                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            graphics_family);
        pool = device.createCommandPoolUnique(pool_ci);

        const vk::CommandBufferAllocateInfo cmdbuf_ai(*pool, vk::CommandBufferLevel::ePrimary,
                                                      COMMAND_BUFFERS_COUNT);
        cmdbufs = device.allocateCommandBuffersUnique(cmdbuf_ai);
    }
    ~VulkanCommandBufferPool() = default;

    vk::CommandBuffer Commit(VulkanFence& fence) {
        return *cmdbufs[ResourceCommit(fence)];
    }

private:
    vk::UniqueCommandPool pool;
    std::vector<vk::UniqueCommandBuffer> cmdbufs;
};

class VulkanSemaphorePool final : public VulkanFencedPool {
public:
    explicit VulkanSemaphorePool(vk::Device device, u32 graphics_family)
        : VulkanFencedPool(SEMAPHORES_COUNT) {

        semaphores.resize(SEMAPHORES_COUNT);
        for (u32 i = 0; i < SEMAPHORES_COUNT; ++i) {
            const vk::SemaphoreCreateInfo semaphore_ci;
            semaphores[i] = device.createSemaphoreUnique(semaphore_ci);
        }
    }
    ~VulkanSemaphorePool() = default;

    vk::Semaphore Commit(VulkanFence& fence) {
        return *semaphores[ResourceCommit(fence)];
    }

private:
    std::vector<vk::UniqueSemaphore> semaphores;
};

VulkanResourceManager::VulkanResourceManager(const VulkanDevice& device_handler)
    : device(device_handler.GetLogical()), graphics_family(device_handler.GetGraphicsFamily()) {

    GrowFences(FENCES_COUNT);
    command_buffer_pool = std::make_unique<VulkanCommandBufferPool>(device, graphics_family);
    semaphore_pool = std::make_unique<VulkanSemaphorePool>(device, graphics_family);
}

VulkanResourceManager::~VulkanResourceManager() {
    // There may be owned fences here (e.g. present fences), forcefully release them. It's safe
    // on the GPU perspective since device must be idle before destroying the resource manager.
    std::for_each(fences.begin(), fences.end(), [](auto& fence) {
        fence->Release();
        fence->Tick(true, true);
    });
}

VulkanFence& VulkanResourceManager::CommitFence() {
    const auto StepFences = [&](bool gpu_wait, bool owner_wait) -> VulkanFence* {
        const auto it = std::find_if(fences.begin(), fences.end(), [=](auto& fence) {
            return fence->Tick(gpu_wait, owner_wait);
        });
        if (it == fences.end()) {
            return nullptr;
        }
        auto& fence = *it;
        fence->Commit();
        return fence.get();
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
    return command_buffer_pool->Commit(fence);
}

vk::Semaphore VulkanResourceManager::CommitSemaphore(VulkanFence& fence) {
    return semaphore_pool->Commit(fence);
}

vk::RenderPass VulkanResourceManager::CreateRenderPass(
    VulkanFence& fence, const vk::RenderPassCreateInfo& renderpass_ci) {

    return CreateOneShot(fence, renderpasses, device.createRenderPassUnique(renderpass_ci));
}

vk::ImageView VulkanResourceManager::CreateImageView(VulkanFence& fence,
                                                     const vk::ImageViewCreateInfo& image_view_ci) {
    return CreateOneShot(fence, image_views, device.createImageViewUnique(image_view_ci));
}

vk::Framebuffer VulkanResourceManager::CreateFramebuffer(
    VulkanFence& fence, const vk::FramebufferCreateInfo& framebuffer_ci) {

    return CreateOneShot(fence, framebuffers, device.createFramebufferUnique(framebuffer_ci));
}

vk::Pipeline VulkanResourceManager::CreateGraphicsPipeline(
    VulkanFence& fence, const vk::GraphicsPipelineCreateInfo& graphics_pipeline_ci) {

    return CreateOneShot(fence, pipelines,
                         device.createGraphicsPipelineUnique({}, graphics_pipeline_ci));
}

vk::PipelineLayout VulkanResourceManager::CreatePipelineLayout(
    VulkanFence& fence, const vk::PipelineLayoutCreateInfo& pipeline_layout_ci) {

    return CreateOneShot(fence, pipeline_layouts,
                         device.createPipelineLayoutUnique(pipeline_layout_ci));
}

template <typename EntryType, typename HandleType>
HandleType VulkanResourceManager::CreateOneShot(VulkanFence& fence,
                                                std::vector<std::unique_ptr<EntryType>>& vector,
                                                vk::UniqueHandle<HandleType> handle) {
    TickCreations();

    const auto handle_value = *handle;
    auto entry = std::make_unique<EntryType>(std::move(handle));
    fence.Protect(entry.get());
    vector.push_back(std::move(entry));
    return handle_value;
}

void VulkanResourceManager::TickCreations() {
    if (++tick_creations < TICKS_TO_DESTROY) {
        return;
    }
    tick_creations = 0;

    RemoveEntries(renderpasses);
    RemoveEntries(image_views);
    RemoveEntries(framebuffers);
}

template <typename T>
void VulkanResourceManager::RemoveEntries(std::vector<T>& entries) {
    const auto end = entries.begin() + std::min(OBJECTS_TO_DESTROY, entries.size());
    entries.erase(
        std::remove_if(entries.begin(), end, [](const auto& entry) { return entry->IsSignaled(); }),
        end);
}

void VulkanResourceManager::GrowFences(std::size_t new_fences_count) {
    const vk::FenceCreateInfo fence_ci;

    const std::size_t previous_size = fences.size();
    fences.resize(previous_size + new_fences_count);

    std::generate(fences.begin() + previous_size, fences.end(), [&]() {
        return std::make_unique<VulkanFence>(device.createFenceUnique(fence_ci), device);
    });
}

} // namespace Vulkan
