// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include "common/logging/log.h"
#include "core/core.h"
#include "core/settings.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_opengl/renderer_opengl.h"
#ifdef HAS_VULKAN
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#endif
#include "video_core/video_core.h"

namespace VideoCore {

std::unique_ptr<RendererBase> CreateRenderer(Core::Frontend::EmuWindow& emu_window) {
    switch (Settings::values.renderer_backend) {
    case Settings::RendererBackend::OpenGL:
        return std::make_unique<OpenGL::RendererOpenGL>(emu_window);
#ifdef HAS_VULKAN
    case Settings::RendererBackend::Vulkan:
        return std::make_unique<Vulkan::RendererVulkan>(emu_window);
#endif
    default:
        return nullptr;
    }
}

u16 GetResolutionScaleFactor(const RendererBase& renderer) {
    return !Settings::values.resolution_factor
               ? renderer.GetRenderWindow().GetFramebufferLayout().GetScalingRatio()
               : Settings::values.resolution_factor;
}

} // namespace VideoCore
