// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/common_types.h"

namespace Vulkan {

class VKDevice;
class VKFence;
class VKMemoryAllocation;
class VKMemoryManager;

class VKMemoryCommit final {
    friend VKMemoryAllocation;
    friend VKMemoryManager;

public:
    explicit VKMemoryCommit(VKMemoryAllocation* allocation, vk::DeviceMemory memory, u8* data,
                            u64 begin, u64 end);
    ~VKMemoryCommit();

    /// Returns the Vulkan memory handler.
    vk::DeviceMemory GetMemory() const {
        return memory;
    }

    /// Returns the start position of the commit relative to the allocation.
    vk::DeviceSize GetOffset() const {
        return static_cast<vk::DeviceSize>(interval.first);
    }

    /// Returns the writeable memory map. Do not call when the commit is not mappeable.
    u8* GetData() const {
        ASSERT_MSG(data != nullptr, "Trying to access an unmapped commit.");
        return data;
    }

private:
    /// Interval where the commit exists.
    const std::pair<u64, u64> interval;

    /// Vulkan device memory handler.
    const vk::DeviceMemory memory;

    /// Pointer to the large memory allocation.
    VKMemoryAllocation* const allocation;

    /// Pointer to the host mapped memory. It's nullptr when the commit is not mappeable.
    u8* const data;
};

using Commit = std::shared_ptr<VKMemoryCommit>;

class VKMemoryManager final {
public:
    explicit VKMemoryManager(const VKDevice& device_handler);
    ~VKMemoryManager();

    /**
     * Commits a memory with the specified requeriments.
     * @param reqs Requeriments returned from a Vulkan call.
     * @param host_visible Signals the allocator that it *must* use host visible and coherent
     * memory. When passing false, it will try to allocate device local memory.
     * @returns A pointer to the memory commitment.
     */
    const VKMemoryCommit* Commit(const vk::MemoryRequirements& reqs, bool host_visible);

    /// Frees the memory commit. It can be nullptr.
    void Free(const VKMemoryCommit* commit);

    /// Returns true if the memory allocations are done always in host visible and coherent memory.
    bool IsMemoryUnified() const {
        return is_memory_unified;
    }

private:
    bool AllocMemory(vk::MemoryPropertyFlags wanted_properties, u32 type_mask, u64 size);

    static bool GetMemoryUnified(const vk::PhysicalDeviceMemoryProperties& props);

    const vk::Device device;
    const vk::PhysicalDevice physical_device;
    const vk::PhysicalDeviceMemoryProperties props;
    const bool is_memory_unified;

    std::vector<std::unique_ptr<VKMemoryAllocation>> allocs;
};

} // namespace Vulkan
