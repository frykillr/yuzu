// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"

#pragma optimize("", off)

namespace Vulkan {

constexpr u64 DEVICE_LOCAL_USE_PERCENT = 75;
constexpr u64 DEVICE_LOCAL_SIZE_LIMIT = 768 * 1024 * 1024;

constexpr u64 HOST_VISIBLE_USE_PERCENT = 50;
constexpr u64 HOST_VISIBLE_SIZE_LIMIT = 512 * 1024 * 1024;

class VulkanMemoryAllocation final {
public:
    explicit VulkanMemoryAllocation(vk::Device device, u64 alloc_size, u32 type, bool is_mappeable)
        : device(device), alloc_size(alloc_size), shifted_type(ShiftType(type)),
          is_mappeable(is_mappeable) {

        const vk::MemoryAllocateInfo memory_ai(alloc_size, type);
        memory = device.allocateMemoryUnique(memory_ai);
        if (is_mappeable) {
            base_address = static_cast<u8*>(device.mapMemory(*memory, 0, alloc_size, {}));
        }
    }

    ~VulkanMemoryAllocation() {
        if (is_mappeable) {
            device.unmapMemory(*memory);
        }
    }

    const VulkanMemoryCommit* Alloc(vk::DeviceSize commit_size, vk::DeviceSize alignment) {
        auto found = TryFindFreeSection(free_iterator, alloc_size, static_cast<u64>(commit_size),
                                        static_cast<u64>(alignment));
        if (!found) {
            found = TryFindFreeSection(0, free_iterator, static_cast<u64>(commit_size),
                                       static_cast<u64>(alignment));
            if (!found) {
                // TODO(Rodrigo): Try to do more allocations.
                LOG_CRITICAL(Render_Vulkan, "Device out of memory!");
                UNREACHABLE();
                return nullptr;
            }
        }
        u8* address = is_mappeable ? base_address + *found : nullptr;
        auto* commit = new VulkanMemoryCommit(this, *memory, address, *found, *found + commit_size);
        commits.push_back(std::unique_ptr<VulkanMemoryCommit>(commit));

        // Last commit's address is highly probable to be free.
        free_iterator = *found + commit_size;

        return commit;
    }

    void Free(const VulkanMemoryCommit* commit) {
        ASSERT(commit);
        const auto it =
            std::find_if(commits.begin(), commits.end(),
                         [&](const auto& stored_commit) { return stored_commit.get() == commit; });
        if (it == commits.end()) {
            LOG_CRITICAL(Render_Vulkan, "Freeing unallocated commit!");
            UNREACHABLE();
            return;
        }
        commits.erase(it);
    }

    bool IsCompatible(u32 type_mask) const {
        return (type_mask & shifted_type) != 0;
    }

    static constexpr u32 ShiftType(u32 type) {
        return 1 << type;
    }

private:
    inline std::optional<u64> TryFindFreeSection(u64 start, u64 end, u64 size, u64 alignment) {
        u64 iterator = start;
        while (iterator + size < end) {
            const u64 try_left = Common::AlignUp(iterator, alignment);
            const u64 try_right = try_left + size;
            const bool overlap = [&]() {
                for (const auto& commit : commits) {
                    const auto [commit_left, commit_right] = commit->GetInterval();
                    if (try_left < commit_right && commit_left < try_right) {
                        iterator = commit_right;
                        return true;
                    }
                }
                return false;
            }();
            if (!overlap) {
                return try_left;
            }
        }
        return {};
    }

    const vk::Device device;
    const u64 alloc_size;
    const u32 shifted_type;
    const bool is_mappeable;

    vk::UniqueDeviceMemory memory;
    u8* base_address{};

    u64 free_iterator{};
    std::vector<std::unique_ptr<VulkanMemoryCommit>> commits;
};

VulkanMemoryCommit::VulkanMemoryCommit(VulkanMemoryAllocation* allocation, vk::DeviceMemory memory,
                                       u8* data, u64 begin, u64 end)
    : allocation(allocation), memory(memory), data(data),
      interval(std::make_pair(begin, begin + end)) {}

VulkanMemoryCommit::~VulkanMemoryCommit() = default;

VulkanMemoryManager::VulkanMemoryManager(const VulkanDevice& device_handler)
    : device(device_handler.GetLogical()), physical_device(device_handler.GetPhysical()) {

    const vk::PhysicalDeviceMemoryProperties props = physical_device.getMemoryProperties();
    AllocDeviceLocal(props);
    TryAllocHostVisible(props);
}

VulkanMemoryManager::~VulkanMemoryManager() = default;

const VulkanMemoryCommit* VulkanMemoryManager::Alloc(const vk::MemoryRequirements& mem_reqs,
                                                     bool host_visible) {
    std::unique_lock lock(mutex);

    VulkanMemoryAllocation* allocation =
        IsMemoryUnified() || !host_visible ? device_local_alloc.get() : host_visible_alloc.get();

    if (!allocation->IsCompatible(mem_reqs.memoryTypeBits)) {
        LOG_CRITICAL(Render_Vulkan, "Incompatible memory in the memory allocator.");
        UNREACHABLE();
        return nullptr;
    }

    return allocation->Alloc(mem_reqs.size, mem_reqs.alignment);
}

void VulkanMemoryManager::Free(const VulkanMemoryCommit* commit) {
    std::unique_lock lock(mutex);
    commit->GetAllocation()->Free(commit);
}

void VulkanMemoryManager::AllocDeviceLocal(const vk::PhysicalDeviceMemoryProperties& props) {
    const u32 heap = FindDeviceLocalHeap(props);
    const u32 type = FindBestDeviceLocalType(props, heap);
    const u64 alloc_size = CalculateAllocationSize(
        props.memoryHeaps[heap], DEVICE_LOCAL_USE_PERCENT, DEVICE_LOCAL_SIZE_LIMIT);
    const bool is_mappeable = IsMappeable(props.memoryTypes[type]);

    device_local_alloc =
        std::make_unique<VulkanMemoryAllocation>(device, alloc_size, type, is_mappeable);
}

void VulkanMemoryManager::TryAllocHostVisible(const vk::PhysicalDeviceMemoryProperties& props) {
    const std::optional<u32> heap = FindNonDeviceLocalHeap(props);
    if (!heap) {
        // If there is no non-local memory the device is probably integrated. For these cases just
        // do one allocation.
        return;
    }

    const u32 type = FindBestHostVisibleType(props, *heap);
    const u64 alloc_size = CalculateAllocationSize(
        props.memoryHeaps[*heap], HOST_VISIBLE_USE_PERCENT, HOST_VISIBLE_SIZE_LIMIT);

    host_visible_alloc = std::make_unique<VulkanMemoryAllocation>(device, alloc_size, type, true);
}

u32 VulkanMemoryManager::FindBestDeviceLocalType(const vk::PhysicalDeviceMemoryProperties& props,
                                                 u32 heap_index) {
    std::optional<u32> best;
    for (u32 type_index = 0; type_index < props.memoryTypeCount; ++type_index) {
        const auto& type = props.memoryTypes[type_index];
        if (type.heapIndex != heap_index) {
            continue;
        }
        if (!(type.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            // It has to be device local (this is implicit in the heap type, but it's worth
            // checking).
            continue;
        }
        if (type.propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) {
            // Not optimal if the type is host visible, mark is as best.
            best = type_index;
        } else {
            // It's not host visible, just what we're looking for.
            return type_index;
        }
    }
    if (!best) {
        LOG_CRITICAL(Render_Vulkan, "Device has no device local memory type!");
        UNREACHABLE();
        return 0;
    }
    return *best;
}

u32 VulkanMemoryManager::FindBestHostVisibleType(const vk::PhysicalDeviceMemoryProperties& props,
                                                 u32 heap_index) {
    std::optional<u32> best;
    for (u32 type_index = 0; type_index < props.memoryTypeCount; ++type_index) {
        const auto& type = props.memoryTypes[type_index];
        if (type.heapIndex != heap_index) {
            continue;
        }
        if (!(type.propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) ||
            !(type.propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            continue;
        }
        if (type.propertyFlags & vk::MemoryPropertyFlagBits::eHostCached) {
            // Avoid cached memory if possible.
            best = type_index;
        } else {
            // Just what we're looking for, host visible and coherent only memory.
            return type_index;
        }
    }
    if (!best) {
        LOG_CRITICAL(Render_Vulkan, "Device has no host visible and coherent memory type!");
        UNREACHABLE();
        return 0;
    }
    return *best;
}

u32 VulkanMemoryManager::FindDeviceLocalHeap(const vk::PhysicalDeviceMemoryProperties& props) {
    const auto heap = FindBiggestHeap(props, [](const vk::MemoryHeap& heap) {
        return (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
               static_cast<vk::MemoryHeapFlagBits>(0);
    });
    if (!heap) {
        LOG_CRITICAL(Render_Vulkan, "Device has no device local memory heap!");
        UNREACHABLE();
        return 0;
    }
    return *heap;
}

std::optional<u32> VulkanMemoryManager::FindNonDeviceLocalHeap(
    const vk::PhysicalDeviceMemoryProperties& props) {

    return FindBiggestHeap(props, [](const vk::MemoryHeap& heap) {
        return (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) ==
               static_cast<vk::MemoryHeapFlagBits>(0);
    });
}

std::optional<u32> VulkanMemoryManager::FindBiggestHeap(
    const vk::PhysicalDeviceMemoryProperties& props, bool (*query)(const vk::MemoryHeap& heap)) {

    std::optional<u32> best_heap;
    vk::DeviceSize best_size = 0;

    for (u32 heap_index = 0; heap_index < props.memoryHeapCount; ++heap_index) {
        const auto& heap = props.memoryHeaps[heap_index];
        if (query(heap) && heap.size > best_size) {
            best_size = heap.size;
            best_heap = heap_index;
        }
    }
    return best_heap;
}

u64 VulkanMemoryManager::CalculateAllocationSize(const vk::MemoryHeap& heap, u64 use_percent,
                                                 u64 size_limit) {

    return std::min((heap.size * use_percent) / 100, size_limit);
}

bool VulkanMemoryManager::IsMappeable(const vk::MemoryType& type) {
    return (type.propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) ==
           vk::MemoryPropertyFlagBits::eHostVisible;
}

} // namespace Vulkan