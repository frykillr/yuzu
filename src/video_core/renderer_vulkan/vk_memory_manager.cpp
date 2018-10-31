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

constexpr u64 ALLOC_CHUNK_SIZE = 64 * 1024 * 1024;

class VulkanMemoryAllocation final {
public:
    explicit VulkanMemoryAllocation(vk::Device device, vk::DeviceMemory memory,
                                    vk::MemoryPropertyFlags properties, u64 alloc_size, u32 type)
        : device(device), memory(memory), properties(properties), alloc_size(alloc_size),
          shifted_type(ShiftType(type)),
          is_mappeable(properties & vk::MemoryPropertyFlagBits::eHostVisible) {

        if (is_mappeable) {
            base_address = static_cast<u8*>(device.mapMemory(memory, 0, alloc_size, {}));
        }
    }

    ~VulkanMemoryAllocation() {
        if (is_mappeable) {
            device.unmapMemory(memory);
        }
        device.free(memory);
    }

    const VulkanMemoryCommit* Commit(vk::DeviceSize commit_size, vk::DeviceSize alignment) {
        auto found = TryFindFreeSection(free_iterator, alloc_size, static_cast<u64>(commit_size),
                                        static_cast<u64>(alignment));
        if (!found) {
            found = TryFindFreeSection(0, free_iterator, static_cast<u64>(commit_size),
                                       static_cast<u64>(alignment));
            if (!found) {
                // Signal out of memory, try to do more allocations.
                return nullptr;
            }
        }
        u8* address = is_mappeable ? base_address + *found : nullptr;
        auto commit = std::make_unique<VulkanMemoryCommit>(this, memory, address, *found,
                                                           *found + commit_size);
        const auto* commit_ptr = commit.get();
        commits.push_back(std::move(commit));

        // Last commit's address is highly probable to be free.
        free_iterator = *found + commit_size;

        return commit_ptr;
    }

    void Free(const VulkanMemoryCommit* commit) {
        ASSERT(commit);
        const auto it =
            std::find_if(commits.begin(), commits.end(),
                         [&](const auto& stored_commit) { return stored_commit.get() == commit; });
        if (it == commits.end()) {
            LOG_ERROR(Render_Vulkan, "Freeing unallocated commit!");
            UNREACHABLE();
            return;
        }
        commits.erase(it);
    }

    bool IsCompatible(vk::MemoryPropertyFlags wanted_properties, u32 type_mask) const {
        return (wanted_properties & properties) != vk::MemoryPropertyFlagBits(0) &&
               (type_mask & shifted_type) != 0;
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
    const vk::DeviceMemory memory;
    vk::MemoryPropertyFlags properties;
    const u64 alloc_size;
    const u32 shifted_type;
    const bool is_mappeable;

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
    : device(device_handler.GetLogical()), physical_device(device_handler.GetPhysical()),
      props(device_handler.GetPhysical().getMemoryProperties()),
      is_memory_unified(GetMemoryUnified(props)) {}

VulkanMemoryManager::~VulkanMemoryManager() = default;

const VulkanMemoryCommit* VulkanMemoryManager::Commit(const vk::MemoryRequirements& reqs,
                                                      bool host_visible) {
    std::unique_lock lock(mutex);

    const vk::MemoryPropertyFlags wanted_properties =
        host_visible
            ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            : vk::MemoryPropertyFlagBits::eDeviceLocal;

    auto TryCommit = [&]() -> const VulkanMemoryCommit* {
        for (auto& alloc : allocs) {
            if (!alloc->IsCompatible(wanted_properties, reqs.memoryTypeBits)) {
                continue;
            }
            if (auto* commit = alloc->Commit(reqs.size, reqs.alignment); commit) {
                return commit;
            }
        }
        return nullptr;
    };

    if (const auto* commit = TryCommit(); commit != nullptr) {
        return commit;
    }

    if (!AllocMemory(wanted_properties, reqs.memoryTypeBits, ALLOC_CHUNK_SIZE)) {
        // TODO(Rodrigo): Try to use host memory.
        LOG_CRITICAL(Render_Vulkan, "Ran out of memory!");
        UNREACHABLE();
    }

    const auto* commit = TryCommit();
    ASSERT(commit != nullptr);
    return commit;
}

void VulkanMemoryManager::Free(const VulkanMemoryCommit* commit) {
    if (commit == nullptr) {
        return;
    }
    std::unique_lock lock(mutex);
    commit->GetAllocation()->Free(commit);
}

bool VulkanMemoryManager::AllocMemory(vk::MemoryPropertyFlags wanted_properties, u32 type_mask,
                                      u64 size) {
    const u32 type = [&]() {
        for (u32 type_index = 0; type_index < props.memoryTypeCount; ++type_index) {
            const auto type = props.memoryTypes[type_index];
            if ((type_mask & (1 << type_index)) == 0) {
                continue;
            }
            const auto flags = type.propertyFlags;
            if (!(flags & wanted_properties)) {
                continue;
            }
            return type_index;
        }
        LOG_CRITICAL(Render_Vulkan, "Couldn't find a compatible memory type!");
        UNREACHABLE();
        return 0u;
    }();

    const vk::MemoryAllocateInfo memory_ai(size, type);
    vk::DeviceMemory memory;
    if (vk::Result res = device.allocateMemory(&memory_ai, nullptr, &memory);
        res != vk::Result::eSuccess) {

        LOG_CRITICAL(Render_Vulkan, "Device allocation failed with code {}!",
                     static_cast<u32>(res));
        return false;
    }

    allocs.push_back(
        std::make_unique<VulkanMemoryAllocation>(device, memory, wanted_properties, size, type));
    return true;
}

/*static*/ bool VulkanMemoryManager::GetMemoryUnified(
    const vk::PhysicalDeviceMemoryProperties& props) {

    for (u32 heap_index = 0; heap_index < props.memoryHeapCount; ++heap_index) {
        if (!(props.memoryHeaps[heap_index].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            // Memory is considered unified when heaps are device local only.
            return false;
        }
    }
    return true;
}

} // namespace Vulkan