// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

namespace Vulkan {

constexpr u64 RESOURCE_RESERVE = 0x4000;
constexpr u64 RESOURCE_CHUNK = 0x1000;

class VulkanStreamBufferResource final : public VulkanResource {
public:
    explicit VulkanStreamBufferResource() = default;
    virtual ~VulkanStreamBufferResource() = default;

    void Reset(VulkanFence& new_fence) {
        fence = &new_fence;
        is_signaled = false;

        std::unique_lock lock(mutex);
        fence->Protect(this);
    }

    void Wait() {
        std::unique_lock lock(mutex);
        if (!is_signaled) {
            fence->Wait();
        }
    }

protected:
    virtual void OnFenceRemoval(VulkanFence* signaling_fence) {
        std::unique_lock lock(mutex);

        ASSERT(signaling_fence == fence);
        is_signaled = true;
    }

private:
    std::mutex mutex;

    VulkanFence* fence{};
    bool is_signaled{};
};

VulkanStreamBuffer::VulkanStreamBuffer(VulkanResourceManager& resource_manager,
                                       VulkanDevice& device_handler,
                                       VulkanMemoryManager& memory_manager, u64 size,
                                       vk::BufferUsageFlags usage)
    : resource_manager(resource_manager), device(device_handler.GetLogical()),
      memory_manager(memory_manager), has_device_memory(!memory_manager.IsMemoryUnified()),
      buffer_size(size) {

    CreateBuffers(memory_manager, usage);
    GrowResources(RESOURCE_RESERVE);
}

VulkanStreamBuffer::~VulkanStreamBuffer() {
    if (device_commit) {
        memory_manager.Free(device_commit);
    }
    if (mappeable_commit) {
        memory_manager.Free(mappeable_commit);
    }
}

std::tuple<u8*, u64, vk::Buffer, bool> VulkanStreamBuffer::Reserve(u64 size, bool keep_in_host) {
    mutex.lock();

    ASSERT(size <= buffer_size);
    mapped_size = size;

    bool invalidate = false;
    if (buffer_pos + size > buffer_size) {
        // TODO(Rodrigo): Find a better way to invalidate than waiting for all resources to finish.
        std::for_each(resources.begin(), resources.begin() + used_resources,
                      [&](const auto& resource) { resource->Wait(); });
        used_resources = 0;

        buffer_pos = 0;
        invalidate = true;
    }

    use_device = has_device_memory && !keep_in_host;
    return {mapped_ptr + buffer_pos, buffer_pos, use_device ? *device_buffer : *mappeable_buffer,
            invalidate};
}

std::optional<std::tuple<vk::SubmitInfo, vk::Semaphore>> VulkanStreamBuffer::Send(
    VulkanFence& fence, u64 size) {

    ASSERT(size <= mapped_size);

    vk::Semaphore semaphore{};
    vk::SubmitInfo submit;

    if (use_device) {
        const vk::CommandBuffer cmdbuf = resource_manager.CommitCommandBuffer(fence);
        semaphore = resource_manager.CommitSemaphore(fence);

        const vk::CommandBufferBeginInfo cmdbuf_bi(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdbuf.begin(cmdbuf_bi);

        // Buffers are mirrored.
        const vk::BufferCopy copy_region(buffer_pos, buffer_pos, size);
        cmdbuf.copyBuffer(*mappeable_buffer, *device_buffer, {copy_region});
        cmdbuf.end();

        submit = vk::SubmitInfo(0, nullptr, nullptr, 1, &cmdbuf, 1, &semaphore);
    }

    if (used_resources + 1 >= resources.size()) {
        resources.resize(resources.size() + RESOURCE_CHUNK);
    }
    auto& resource = resources[used_resources++];
    resource->Reset(fence);

    buffer_pos += size;

    mutex.unlock();

    if (use_device) {
        return std::tie(submit, semaphore);
    } else {
        return {};
    }
}

void VulkanStreamBuffer::CreateBuffers(VulkanMemoryManager& memory_manager,
                                       vk::BufferUsageFlags usage) {
    {
        vk::BufferUsageFlags mappeable_usage = usage;
        if (has_device_memory) {
            mappeable_usage |= vk::BufferUsageFlagBits::eTransferSrc;
        }
        const vk::BufferCreateInfo buffer_ci({}, buffer_size, usage, vk::SharingMode::eExclusive, 0,
                                             nullptr);

        mappeable_buffer = device.createBufferUnique(buffer_ci);

        const vk::MemoryRequirements reqs = device.getBufferMemoryRequirements(*mappeable_buffer);
        mappeable_commit = memory_manager.Commit(reqs, true);
        device.bindBufferMemory(*mappeable_buffer, mappeable_commit->GetMemory(),
                                mappeable_commit->GetOffset());

        mapped_ptr = mappeable_commit->GetData();
    }

    if (has_device_memory) {
        const vk::BufferCreateInfo buffer_ci({}, buffer_size,
                                             usage | vk::BufferUsageFlagBits::eTransferDst,
                                             vk::SharingMode::eExclusive, 0, nullptr);
        device_buffer = device.createBufferUnique(buffer_ci);

        const vk::MemoryRequirements reqs = device.getBufferMemoryRequirements(*device_buffer);
        device_commit = memory_manager.Commit(reqs, false);
        device.bindBufferMemory(*device_buffer, device_commit->GetMemory(),
                                device_commit->GetOffset());
    }
}

void VulkanStreamBuffer::GrowResources(std::size_t grow_size) {
    std::size_t previous_size = resources.size();
    resources.resize(previous_size + grow_size);
    std::generate(resources.begin() + previous_size, resources.end(),
                  [&]() { return std::make_unique<VulkanStreamBufferResource>(); });
}

} // namespace Vulkan