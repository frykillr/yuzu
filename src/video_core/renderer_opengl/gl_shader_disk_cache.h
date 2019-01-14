// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <tuple>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace OpenGL {

using ProgramCode = std::vector<u64>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct BaseBindings {
private:
    auto Tie() const {
        return std::tie(cbuf, gmem, sampler);
    }

public:
    u32 cbuf{};
    u32 gmem{};
    u32 sampler{};

    bool operator<(const BaseBindings& rhs) const {
        return Tie() < rhs.Tie();
    }

    bool operator==(const BaseBindings& rhs) const {
        return Tie() == rhs.Tie();
    }

    bool operator!=(const BaseBindings& rhs) const {
        return !this->operator==(rhs);
    }
};

} // namespace OpenGL