// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/common_types.h"

namespace Vulkan {

class VulkanDevice;
class VulkanFence;
class VulkanMemoryAllocation;
class VulkanResourceManager;

class VulkanMemoryCommit final {
public:
    explicit VulkanMemoryCommit(VulkanMemoryAllocation* allocation, vk::DeviceMemory memory,
                                u8* data, u64 begin, u64 end);
    ~VulkanMemoryCommit();

    vk::DeviceMemory GetMemory() const {
        return memory;
    }

    vk::DeviceSize GetOffset() const {
        return static_cast<vk::DeviceSize>(interval.first);
    }

    u8* GetData() const {
        ASSERT_MSG(data != nullptr, "Trying to access an unmapped commit.");
        return data;
    }

    VulkanMemoryAllocation* GetAllocation() const {
        return allocation;
    }

    const std::pair<u64, u64>& GetInterval() const {
        return interval;
    }

private:
    const std::pair<u64, u64> interval;
    const vk::DeviceMemory memory;
    VulkanMemoryAllocation* allocation;
    u8* data;
};

class VulkanMemoryManager final {
public:
    explicit VulkanMemoryManager(const VulkanDevice& device_handler);
    ~VulkanMemoryManager();

    const VulkanMemoryCommit* Commit(const vk::MemoryRequirements& reqs, bool host_visible);

    void Free(const VulkanMemoryCommit* commit);

    bool IsMemoryUnified() const {
        return is_memory_unified;
    }

private:
    bool AllocDevice(u32 type_mask, u64 size);
    bool AllocHost(u32 type_mask, u64 size);

    u32 FindBestDeviceLocalType(u32 type_mask) const;
    u32 FindBestHostVisibleType(u32 type_mask) const;

    const vk::Device device;
    const vk::PhysicalDevice physical_device;
    const vk::PhysicalDeviceMemoryProperties props;
    bool is_memory_unified{};

    std::vector<std::unique_ptr<VulkanMemoryAllocation>> device_allocs;
    std::vector<std::unique_ptr<VulkanMemoryAllocation>> host_allocs;

    std::mutex mutex;
};

} // namespace Vulkan
