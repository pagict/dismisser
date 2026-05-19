from __future__ import annotations

from dataclasses import dataclass
import time

import pyautogui

from dismisser.config import PlatformTarget


@dataclass(frozen=True)
class DismissResult:
    performed: bool
    message: str


class NotificationDismisser:
    def dismiss_attention_target(self) -> DismissResult:
        raise NotImplementedError


class PyAutoGuiNotificationDismisser(NotificationDismisser):
    """Pointer-based MVP backend.

    Native replacements should keep the same public method and move the real
    implementation into Objective-C/Swift on macOS or Win32/UIAutomation on Windows.
    """

    def __init__(self, platform: PlatformTarget, enable_actions: bool) -> None:
        self.platform = platform
        self.enable_actions = enable_actions

    def dismiss_attention_target(self) -> DismissResult:
        if not self.enable_actions:
            return DismissResult(False, "dry-run: action suppressed")

        if self.platform == PlatformTarget.MAC:
            return self._dismiss_macos()
        if self.platform == PlatformTarget.WINDOWS:
            return self._dismiss_windows()
        return DismissResult(False, f"unsupported platform: {self.platform.value}")

    def _dismiss_macos(self) -> DismissResult:
        width, _ = pyautogui.size()
        banner_x = width - 48
        banner_y = 86
        pyautogui.moveTo(banner_x, banner_y, duration=0.08)
        time.sleep(0.12)
        pyautogui.dragTo(width - 4, banner_y, duration=0.22, button="left")
        return DismissResult(True, "macOS top-right notification swipe attempted")

    def _dismiss_windows(self) -> DismissResult:
        width, height = pyautogui.size()
        tray_x = width - 28
        tray_y = height - 20
        pyautogui.moveTo(tray_x, tray_y, duration=0.08)
        time.sleep(0.35)
        pyautogui.press("esc")
        return DismissResult(True, "Windows tray notification hover attempted")
