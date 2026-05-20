from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

from dismisser.config import PlatformTarget
from dismisser.types import GazePoint


@dataclass(frozen=True)
class UiElement:
    role: str
    name: str
    rect: tuple[float, float, float, float]

    @property
    def fingerprint(self) -> tuple[str, str, tuple[int, int, int, int]]:
        left, top, right, bottom = self.rect
        return (
            self.role,
            self.name,
            (round(left), round(top), round(right), round(bottom)),
        )

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.rect
        return (left + right) / 2.0, (top + bottom) / 2.0

    def distance_to(self, x: float, y: float) -> float:
        left, top, right, bottom = self.rect
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        return math.hypot(dx, dy)


class UiSnapBackend(Protocol):
    def element_at(self, x: int, y: int) -> UiElement | None:
        ...


class NullUiSnapBackend:
    def element_at(self, x: int, y: int) -> UiElement | None:
        return None


class UiElementSnapper:
    """Snap filtered gaze to nearby OS accessibility elements.

    The backend uses point hit-tests instead of walking the full accessibility
    tree every frame. This keeps the loop cheap and still catches nearby
    controls by probing around the gaze point.
    """

    def __init__(
        self,
        platform: PlatformTarget,
        screen_size: tuple[int, int],
        radius_px: int = 80,
        refresh_seconds: float = 0.10,
    ) -> None:
        self.screen_width = max(screen_size[0], 1)
        self.screen_height = max(screen_size[1], 1)
        self.radius_px = max(radius_px, 0)
        self.refresh_seconds = max(refresh_seconds, 0.0)
        self.backend = make_ui_snap_backend(platform)
        self._last_refresh = 0.0
        self._last_element: UiElement | None = None
        self._last_backend_error: str | None = None

    def update(self, gaze: GazePoint | None) -> GazePoint | None:
        if gaze is None or self.radius_px <= 0:
            self._last_element = None
            return gaze

        x = _clamp(gaze.x, 0.0, 1.0) * self.screen_width
        y = _clamp(gaze.y, 0.0, 1.0) * self.screen_height
        now = time.monotonic()

        if self._last_element is None or now - self._last_refresh >= self.refresh_seconds:
            self._last_refresh = now
            self._last_element = self._find_best_element(x, y)

        if self._last_element is None:
            return gaze

        distance = self._last_element.distance_to(x, y)
        if distance > self.radius_px * 1.35:
            self._last_element = None
            return gaze

        snapped_x, snapped_y = self._last_element.center
        return GazePoint(
            _clamp(snapped_x / self.screen_width, 0.0, 1.0),
            _clamp(snapped_y / self.screen_height, 0.0, 1.0),
            gaze.confidence,
        )

    def _find_best_element(self, x: float, y: float) -> UiElement | None:
        candidates: dict[tuple[str, str, tuple[int, int, int, int]], UiElement] = {}
        for sample_x, sample_y in self._sample_points(x, y):
            try:
                element = self.backend.element_at(round(sample_x), round(sample_y))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != self._last_backend_error:
                    print(f"UI snap disabled for this sample: {message}")
                    self._last_backend_error = message
                continue
            if element is None or not _is_snap_candidate(element, self.screen_width, self.screen_height):
                continue
            candidates[element.fingerprint] = element

        best_element: UiElement | None = None
        best_score = float("inf")
        for element in candidates.values():
            distance = element.distance_to(x, y)
            if distance > self.radius_px:
                continue
            center_x, center_y = element.center
            center_distance = math.hypot(center_x - x, center_y - y)
            score = distance * 3.0 + center_distance * 0.2
            if score < best_score:
                best_score = score
                best_element = element
        return best_element

    def _sample_points(self, x: float, y: float) -> list[tuple[float, float]]:
        radius = float(self.radius_px)
        half = radius / 2.0
        raw_points = [
            (x, y),
            (x - half, y),
            (x + half, y),
            (x, y - half),
            (x, y + half),
            (x - radius, y),
            (x + radius, y),
            (x, y - radius),
            (x, y + radius),
            (x - half, y - half),
            (x + half, y - half),
            (x - half, y + half),
            (x + half, y + half),
        ]
        return [
            (_clamp(px, 0.0, self.screen_width - 1), _clamp(py, 0.0, self.screen_height - 1))
            for px, py in raw_points
        ]


def make_ui_snap_backend(platform: PlatformTarget) -> UiSnapBackend:
    if platform == PlatformTarget.MAC:
        return MacAccessibilitySnapBackend()
    if platform == PlatformTarget.WINDOWS:
        return WindowsUiAutomationSnapBackend()
    return NullUiSnapBackend()


class MacAccessibilitySnapBackend:
    def __init__(self) -> None:
        try:
            from ApplicationServices import (
                AXIsProcessTrustedWithOptions,
                AXUIElementCopyAttributeValue,
                AXUIElementCopyElementAtPosition,
                AXUIElementCreateSystemWide,
                AXValueGetValue,
                kAXDescriptionAttribute,
                kAXEnabledAttribute,
                kAXPositionAttribute,
                kAXRoleAttribute,
                kAXSizeAttribute,
                kAXTitleAttribute,
                kAXTrustedCheckOptionPrompt,
                kAXValueCGPointType,
                kAXValueCGSizeType,
            )
        except Exception as exc:
            self._available = False
            self._error = f"macOS ApplicationServices unavailable: {exc}"
            return

        self._copy_attribute = AXUIElementCopyAttributeValue
        self._element_at = AXUIElementCopyElementAtPosition
        self._ax_value_get_value = AXValueGetValue
        self._position_attr = kAXPositionAttribute
        self._size_attr = kAXSizeAttribute
        self._role_attr = kAXRoleAttribute
        self._title_attr = kAXTitleAttribute
        self._description_attr = kAXDescriptionAttribute
        self._enabled_attr = kAXEnabledAttribute
        self._point_type = kAXValueCGPointType
        self._size_type = kAXValueCGSizeType
        self._system = AXUIElementCreateSystemWide()

        try:
            trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        except Exception:
            trusted = False
        self._available = bool(trusted)
        self._error = "macOS Accessibility permission is not granted"

    def element_at(self, x: int, y: int) -> UiElement | None:
        if not self._available:
            raise RuntimeError(self._error)
        err, element = _unpack_ax_result(self._element_at(self._system, float(x), float(y), None))
        if err not in (0, None) or element is None:
            return None

        role = str(self._attribute(element, self._role_attr) or "")
        name = str(
            self._attribute(element, self._title_attr)
            or self._attribute(element, self._description_attr)
            or ""
        )
        enabled = self._attribute(element, self._enabled_attr)
        if enabled is False:
            return None
        position = self._ax_pair(self._attribute(element, self._position_attr), self._point_type)
        size = self._ax_pair(self._attribute(element, self._size_attr), self._size_type)
        if position is None or size is None:
            return None
        left, top = position
        width, height = size
        return UiElement(role=role, name=name, rect=(left, top, left + width, top + height))

    def _attribute(self, element, attribute):
        err, value = _unpack_ax_result(self._copy_attribute(element, attribute, None))
        if err not in (0, None):
            return None
        return value

    def _ax_pair(self, value, value_type) -> tuple[float, float] | None:
        if value is None:
            return None
        if hasattr(value, "x") and hasattr(value, "y"):
            return float(value.x), float(value.y)
        if hasattr(value, "width") and hasattr(value, "height"):
            return float(value.width), float(value.height)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        result = self._ax_value_get_value(value, value_type, None)
        _, pair = _unpack_ax_result(result)
        if pair is None:
            return None
        if hasattr(pair, "x") and hasattr(pair, "y"):
            return float(pair.x), float(pair.y)
        if hasattr(pair, "width") and hasattr(pair, "height"):
            return float(pair.width), float(pair.height)
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            return float(pair[0]), float(pair[1])
        return None


class WindowsUiAutomationSnapBackend:
    def __init__(self) -> None:
        try:
            import comtypes.client
            from ctypes.wintypes import POINT

            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen.UIAutomationClient import CUIAutomation
        except Exception as exc:
            self._available = False
            self._error = f"Windows UI Automation unavailable: {exc}"
            return

        self._point_type = POINT
        self._uia = comtypes.client.CreateObject(CUIAutomation)
        self._available = True
        self._error = ""

    def element_at(self, x: int, y: int) -> UiElement | None:
        if not self._available:
            raise RuntimeError(self._error)
        element = self._uia.ElementFromPoint(self._point_type(x, y))
        if element is None:
            return None
        try:
            rect = element.CurrentBoundingRectangle
        except Exception:
            return None
        if getattr(element, "CurrentIsOffscreen", False):
            return None
        if getattr(element, "CurrentIsEnabled", True) is False:
            return None
        role = _windows_control_type_name(getattr(element, "CurrentControlType", 0))
        name = str(getattr(element, "CurrentName", "") or "")
        return UiElement(
            role=role,
            name=name,
            rect=(float(rect.left), float(rect.top), float(rect.right), float(rect.bottom)),
        )


def _unpack_ax_result(result) -> tuple[int | None, object | None]:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], result[1]
        if len(result) == 1:
            return None, result[0]
    return None, result


def _windows_control_type_name(control_type: int) -> str:
    names = {
        50000: "Button",
        50001: "Calendar",
        50002: "CheckBox",
        50003: "ComboBox",
        50004: "Edit",
        50005: "Hyperlink",
        50006: "Image",
        50007: "ListItem",
        50008: "List",
        50009: "Menu",
        50010: "MenuBar",
        50011: "MenuItem",
        50012: "ProgressBar",
        50013: "RadioButton",
        50014: "ScrollBar",
        50015: "Slider",
        50016: "Spinner",
        50017: "StatusBar",
        50018: "Tab",
        50019: "TabItem",
        50020: "Text",
        50021: "ToolBar",
        50022: "ToolTip",
        50023: "Tree",
        50024: "TreeItem",
        50025: "Custom",
        50026: "Group",
        50027: "Thumb",
        50028: "DataGrid",
        50029: "DataItem",
        50030: "Document",
        50031: "SplitButton",
        50032: "Window",
        50033: "Pane",
        50034: "Header",
        50035: "HeaderItem",
        50036: "Table",
        50037: "TitleBar",
        50038: "Separator",
        50039: "SemanticZoom",
        50040: "AppBar",
    }
    return names.get(control_type, str(control_type))


def _is_snap_candidate(element: UiElement, screen_width: int, screen_height: int) -> bool:
    left, top, right, bottom = element.rect
    width = right - left
    height = bottom - top
    if width < 4 or height < 4:
        return False
    if right < 0 or bottom < 0 or left > screen_width or top > screen_height:
        return False
    if width > screen_width * 0.85 and height > screen_height * 0.85:
        return False
    role = element.role.lower()
    ignored_exact_roles = {
        "text",
    }
    if role in ignored_exact_roles:
        return False
    ignored_fragments = (
        "application",
        "window",
        "pane",
        "group",
        "statictext",
        "image",
        "separator",
        "toolbar",
        "menubar",
        "scrollarea",
        "scrollbar",
        "statusbar",
        "tooltip",
        "titlebar",
    )
    return not any(fragment in role for fragment in ignored_fragments)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
