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

class VKMemoryAllocation final {
public:
    explicit VKMemoryAllocation(vk::Device device, vk::DeviceMemory memory,
                                vk::MemoryPropertyFlags properties, u64 alloc_size, u32 type)
        : device(device), memory(memory), properties(properties), alloc_size(alloc_size),
          shifted_type(ShiftType(type)),
          is_mappeable(properties & vk::MemoryPropertyFlagBits::eHostVisible) {

        if (is_mappeable) {
            base_address = static_cast<u8*>(device.mapMemory(memory, 0, alloc_size, {}));
        }
    }

    ~VKMemoryAllocation() {
        if (is_mappeable)
            device.unmapMemory(memory);
        device.free(memory);
    }

    const VKMemoryCommit* Commit(vk::DeviceSize commit_size, vk::DeviceSize alignment) {
        auto found = TryFindFreeSection(free_iterator, alloc_size, static_cast<u64>(commit_size),
                                        static_cast<u64>(alignment));
        if (!found) {
            found = TryFindFreeSection(0, free_iterator, static_cast<u64>(commit_size),
                                       static_cast<u64>(alignment));
            if (!found) {
                // Signal out of memory, it'll try to do more allocations.
                return nullptr;
            }
        }
        u8* address = is_mappeable ? base_address + *found : nullptr;
        auto commit =
            std::make_unique<VKMemoryCommit>(this, memory, address, *found, *found + commit_size);
        const auto* commit_ptr = commit.get();
        commits.push_back(std::move(commit));

        // Last commit's address is highly probable to be free.
        free_iterator = *found + commit_size;

        return commit_ptr;
    }

    void Free(const VKMemoryCommit* commit) {
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

    /// Returns whether this allocation is compatible with the arguments.
    bool IsCompatible(vk::MemoryPropertyFlags wanted_properties, u32 type_mask) const {
        return (wanted_properties & properties) != vk::MemoryPropertyFlagBits(0) &&
               (type_mask & shifted_type) != 0;
    }

private:
    static constexpr u32 ShiftType(u32 type) {
        return 1 << type;
    }

    /// This function is basically a memory allocator, it may return a free region between "start"
    /// and "end" with the solicited requeriments.
    inline std::optional<u64> TryFindFreeSection(u64 start, u64 end, u64 size, u64 alignment) {
        u64 iterator = start;
        while (iterator + size < end) {
            const u64 try_left = Common::AlignUp(iterator, alignment);
            const u64 try_right = try_left + size;

            for (const auto& commit : commits) {
                const auto [commit_left, commit_right] = commit->interval;
                if (try_left < commit_right && commit_left < try_right) {
                    // There's an overlap, continue the search where the overlapping commit
                    // ends.
                    iterator = commit_right;
                    continue;
                }
            }
            // A free address has been found.
            return try_left;
        }
        // No free regions where found, return an empty optional.
        return {};
    }

    const vk::Device device;                  ///< Vulkan logical device.
    const vk::DeviceMemory memory;            ///< Vulkan memory allocation handler.
    const vk::MemoryPropertyFlags properties; ///< Vulkan properties.
    const u64 alloc_size;                     ///< Size of this allocation.
    const u32 shifted_type;                   ///< Stored Vulkan type of this allocation, shifted.
    const bool is_mappeable;                  ///< Whether the allocation is mappeable.

    /// Base address of the mapped pointer.
    u8* base_address{};

    /// Hints where the next free region is likely going to be.
    u64 free_iterator{};

    /// Stores all commits done from this allocation.
    std::vector<std::unique_ptr<VKMemoryCommit>> commits;
};

VKMemoryCommit::VKMemoryCommit(VKMemoryAllocation* allocation, vk::DeviceMemory memory, u8* data,
                               u64 begin, u64 end)
    : allocation(allocation), memory(memory), data(data),
      interval(std::make_pair(begin, begin + end)) {}

VKMemoryCommit::~VKMemoryCommit() = default;

VKMemoryManager::VKMemoryManager(const VKDevice& device_handler)
    : device(device_handler.GetLogical()), physical_device(device_handler.GetPhysical()),
      props(device_handler.GetPhysical().getMemoryProperties()),
      is_memory_unified(GetMemoryUnified(props)) {}

VKMemoryManager::~VKMemoryManager() = default;

const VKMemoryCommit* VKMemoryManager::Commit(const vk::MemoryRequirements& reqs,
                                              bool host_visible) {
    ASSERT(reqs.size < ALLOC_CHUNK_SIZE);

    // When a host visible commit is asked, search for host visible and coherent, otherwise search
    // for a fast device local type.
    const vk::MemoryPropertyFlags wanted_properties =
        host_visible
            ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            : vk::MemoryPropertyFlagBits::eDeviceLocal;

    const auto TryCommit = [&]() -> const VKMemoryCommit* {
        for (auto& alloc : allocs) {
            if (!alloc->IsCompatible(wanted_properties, reqs.memoryTypeBits))
                continue;

            if (auto* commit = alloc->Commit(reqs.size, reqs.alignment); commit) {
                return commit;
            }
        }
        return nullptr;
    };

    if (const auto* commit = TryCommit(); commit != nullptr) {
        return commit;
    }

    // Commit has failed, allocate more memory.
    if (!AllocMemory(wanted_properties, reqs.memoryTypeBits, ALLOC_CHUNK_SIZE)) {
        // TODO(Rodrigo): Try to use host memory.
        LOG_CRITICAL(Render_Vulkan, "Ran out of memory!");
        UNREACHABLE();
    }

    // Commit again, this time it won't fail since there's a fresh allocation above. If it does,
    // there's a bug.
    const auto* commit = TryCommit();
    ASSERT(commit != nullptr);
    return commit;
}

void VKMemoryManager::Free(const VKMemoryCommit* commit) {
    if (commit != nullptr)
        commit->allocation->Free(commit);
}

bool VKMemoryManager::AllocMemory(vk::MemoryPropertyFlags wanted_properties, u32 type_mask,
                                  u64 size) {
    const u32 type = [&]() {
        for (u32 type_index = 0; type_index < props.memoryTypeCount; ++type_index) {
            const auto flags = props.memoryTypes[type_index].propertyFlags;

            if (type_mask & (1 << type_index) && flags & wanted_properties) {
                // The type matches in type and in the wanted properties.
                return type_index;
            }
        }
        LOG_CRITICAL(Render_Vulkan, "Couldn't find a compatible memory type!");
        UNREACHABLE();
        return 0u;
    }();

    // Try to allocate found type.
    const vk::MemoryAllocateInfo memory_ai(size, type);
    vk::DeviceMemory memory;
    if (vk::Result res = device.allocateMemory(&memory_ai, nullptr, &memory);
        res != vk::Result::eSuccess) {

        LOG_CRITICAL(Render_Vulkan, "Device allocation failed with code {}!",
                     static_cast<u32>(res));
        return false;
    }

    allocs.push_back(
        std::make_unique<VKMemoryAllocation>(device, memory, wanted_properties, size, type));
    return true;
}

/*static*/ bool VKMemoryManager::GetMemoryUnified(const vk::PhysicalDeviceMemoryProperties& props) {
    for (u32 heap_index = 0; heap_index < props.memoryHeapCount; ++heap_index) {
        if (!(props.memoryHeaps[heap_index].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            // Memory is considered unified when heaps are device local only.
            return false;
        }
    }
    return true;
}

} // namespace Vulkan