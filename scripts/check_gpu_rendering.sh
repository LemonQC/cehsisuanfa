#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
echo '=== Graphics and Vulkan diagnosis ==='
echo "DISPLAY=${DISPLAY-<unset>} WAYLAND_DISPLAY=${WAYLAND_DISPLAY-<unset>}"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES-<unset>}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
ls -la /usr/share/vulkan/icd.d/ 2>&1 || true
ldconfig -p 2>/dev/null | grep -E 'libGLX_nvidia|libvulkan' || true
if command -v vulkaninfo >/dev/null 2>&1; then
  env -u DISPLAY -u WAYLAND_DISPLAY vulkaninfo --summary 2>&1 | head -n 70 || true
else
  echo 'vulkaninfo absent (optional); do not reinstall NVIDIA drivers blindly.'
fi
echo '=== End ==='
