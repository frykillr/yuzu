// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <vulkan/vulkan.hpp>
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

    const VulkanMemoryCommit* Alloc(const vk::MemoryRequirements& mem_reqs, bool host_visible);

    void Free(const VulkanMemoryCommit* commit);

    bool IsMemoryUnified() const {
        return host_visible_alloc == nullptr;
    }

private:
    void AllocDeviceLocal(const vk::PhysicalDeviceMemoryProperties& props);
    void TryAllocHostVisible(const vk::PhysicalDeviceMemoryProperties& props);

    static u32 FindBestDeviceLocalType(const vk::PhysicalDeviceMemoryProperties& props,
                                       u32 heap_index);

    static u32 FindBestHostVisibleType(const vk::PhysicalDeviceMemoryProperties& props,
                                       u32 heap_index);

    static u32 FindDeviceLocalHeap(const vk::PhysicalDeviceMemoryProperties& props);

    static std::optional<u32> FindNonDeviceLocalHeap(
        const vk::PhysicalDeviceMemoryProperties& props);

    static std::optional<u32> FindBiggestHeap(const vk::PhysicalDeviceMemoryProperties& props,
                                              bool (*query)(const vk::MemoryHeap& heap));

    static u64 CalculateAllocationSize(const vk::MemoryHeap& heap, u64 use_percent, u64 size_limit);

    static bool IsMappeable(const vk::MemoryType& type);

    const vk::Device device;
    const vk::PhysicalDevice physical_device;

    std::unique_ptr<VulkanMemoryAllocation> device_local_alloc;
    std::unique_ptr<VulkanMemoryAllocation> host_visible_alloc;

    std::mutex mutex;
};

} // namespace Vulkan
