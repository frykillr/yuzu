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

private:
    const std::pair<u64, u64> interval;
    const vk::DeviceMemory memory;
    VKMemoryAllocation* allocation;
    u8* data;
};

class VKMemoryManager final {
public:
    explicit VKMemoryManager(const VKDevice& device_handler);
    ~VKMemoryManager();

    const VKMemoryCommit* Commit(const vk::MemoryRequirements& reqs, bool host_visible);

    void Free(const VKMemoryCommit* commit);

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
