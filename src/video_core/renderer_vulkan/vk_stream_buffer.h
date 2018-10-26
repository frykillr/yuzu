// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanMemoryManager;
class VulkanStreamBufferResource;

class VulkanStreamBuffer {
public:
    explicit VulkanStreamBuffer(VulkanResourceManager& resource_manager,
                                VulkanDevice& device_handler, VulkanMemoryManager& memory_manager,
                                u64 size, vk::BufferUsageFlags usage);
    ~VulkanStreamBuffer();

    std::tuple<u8*, u64, vk::Buffer, bool> Reserve(u64 size, bool keep_in_host);

    std::optional<std::tuple<vk::SubmitInfo, vk::Semaphore>> Send(VulkanFence& fence, u64 size);

private:
    void CreateBuffers(VulkanMemoryManager& memory_manager, vk::BufferUsageFlags usage);

    void GrowResources(std::size_t grow_size);

    VulkanResourceManager& resource_manager;
    const vk::Device device;
    const u64 buffer_size;
    const bool has_device_memory;

    vk::UniqueBuffer mappeable_buffer;
    vk::UniqueBuffer device_buffer;

    u8* mapped_ptr;
    u64 buffer_pos;
    u64 mapped_size;
    bool use_device;

    std::vector<std::unique_ptr<VulkanStreamBufferResource>> resources;
    u32 used_resources{};

    std::mutex mutex;
};

} // namespace Vulkan
