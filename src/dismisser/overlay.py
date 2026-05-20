from __future__ import annotations

from collections.abc import Callable
import platform as platform_module

from dismisser.config import PlatformTarget
from dismisser.types import GazePoint


class GazeOverlay:
    """Factory wrapper for full-screen transparent gaze overlays."""

    def __new__(
        cls,
        platform: PlatformTarget,
        on_quit: Callable[[], None],
        on_capture_neutral: Callable[[], None],
        on_reset_neutral: Callable[[], None],
        passthrough: bool = False,
    ):
        if cls is not GazeOverlay:
            return super().__new__(cls)
        if platform_module.system().lower() == "darwin":
            return MacOSGazeOverlay(
                platform,
                on_quit,
                on_capture_neutral,
                on_reset_neutral,
                passthrough=passthrough,
            )
        return TkGazeOverlay(
            platform,
            on_quit,
            on_capture_neutral,
            on_reset_neutral,
            passthrough=passthrough,
        )


class MacOSGazeOverlay:
    def __init__(
        self,
        platform: PlatformTarget,
        on_quit: Callable[[], None],
        on_capture_neutral: Callable[[], None],
        on_reset_neutral: Callable[[], None],
        passthrough: bool = False,
    ) -> None:
        from AppKit import (
            NSApp,
            NSApplication,
            NSBackingStoreBuffered,
            NSBorderlessWindowMask,
            NSColor,
            NSEvent,
            NSFloatingWindowLevel,
            NSKeyDownMask,
            NSScreen,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
        )

        self.platform = platform
        self.on_quit = on_quit
        self.on_capture_neutral = on_capture_neutral
        self.on_reset_neutral = on_reset_neutral
        self._tick: Callable[[], int | None] | None = None
        self._exit_code = 0
        self._closed = False

        app = NSApplication.sharedApplication()
        frame = NSScreen.mainScreen().frame()
        self.view = _MacOSGazeView.alloc().initWithFrame_(frame)
        self.view.platform = platform
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSBorderlessWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True)
        if passthrough:
            for obj in (self.window, self.view):
                try:
                    obj.setAccessibilityElement_(False)
                except Exception:
                    pass
        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        self.window.setContentView_(self.view)
        self.window.orderFrontRegardless()
        self._key_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask,
            self._handle_key,
        )

    def run(self, tick: Callable[[], int | None]) -> int:
        from Foundation import NSTimer
        from PyObjCTools import AppHelper

        self._tick = tick
        self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1.0 / 60.0,
            True,
            self._timer_fired,
        )
        AppHelper.runConsoleEventLoop()
        return self._exit_code

    def update_gaze(self, gaze: GazePoint | None) -> None:
        self.view.gaze = gaze
        self.view.setNeedsDisplay_(True)

    def screen_size(self) -> tuple[int, int]:
        frame = self.view.bounds()
        return int(frame.size.width), int(frame.size.height)

    def close(self) -> None:
        if self._closed:
            return
        from AppKit import NSEvent
        from PyObjCTools import AppHelper

        self._closed = True
        if hasattr(self, "_timer"):
            self._timer.invalidate()
        if self._key_monitor is not None:
            NSEvent.removeMonitor_(self._key_monitor)
            self._key_monitor = None
        self.window.close()
        AppHelper.stopEventLoop()

    def _timer_fired(self, _timer) -> None:
        if self._closed or self._tick is None:
            return
        exit_code = self._tick()
        if exit_code is not None:
            self._exit_code = exit_code
            self.close()

    def _handle_key(self, event) -> None:
        chars = event.charactersIgnoringModifiers()
        if chars == "q" or event.keyCode() == 53:
            self.on_quit()
            self.close()
        elif chars == "c":
            self.on_capture_neutral()
        elif chars == "r":
            self.on_reset_neutral()


class _MacOSGazeViewBase:
    pass


try:
    from AppKit import (
        NSBezierPath,
        NSColor,
        NSFont,
        NSMakePoint,
        NSMakeRect,
        NSView,
    )
    from Foundation import NSString

    class _MacOSGazeView(NSView):
        gaze = None
        platform = PlatformTarget.MAC

        def isOpaque(self):
            return False

        def isFlipped(self):
            return True

        def drawRect_(self, _rect):
            bounds = self.bounds()
            width = bounds.size.width
            height = bounds.size.height
            self._draw_target(width, height)
            if self.gaze is None:
                self._draw_text("No face", 28, 34)
                return
            x = self.gaze.x * width
            y = self.gaze.y * height
            self._draw_gaze(x, y)

        def _draw_target(self, width: float, height: float) -> None:
            if self.platform == PlatformTarget.MAC:
                rect = NSMakeRect(width * 0.72, 0, width * 0.28 - 2, height * 0.28)
            else:
                rect = NSMakeRect(width * 0.72, height * 0.72, width * 0.28 - 2, height * 0.28 - 2)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.69, 0.0, 0.9).set()
            path = NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(3.0)
            path.stroke()

        def _draw_gaze(self, x: float, y: float) -> None:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.0, 0.0, 0.95).set()
            outer = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - 16, y - 16, 32, 32))
            outer.setLineWidth_(4.0)
            outer.stroke()
            inner = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - 5, y - 5, 10, 10))
            inner.fill()
            hline = NSBezierPath.bezierPath()
            hline.moveToPoint_(NSMakePoint(x - 28, y))
            hline.lineToPoint_(NSMakePoint(x + 28, y))
            hline.setLineWidth_(2.0)
            hline.stroke()
            vline = NSBezierPath.bezierPath()
            vline.moveToPoint_(NSMakePoint(x, y - 28))
            vline.lineToPoint_(NSMakePoint(x, y + 28))
            vline.setLineWidth_(2.0)
            vline.stroke()

        def _draw_text(self, text: str, x: float, y: float) -> None:
            attrs = {
                "NSFont": NSFont.boldSystemFontOfSize_(22),
                "NSColor": NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.3, 0.3, 0.95),
            }
            NSString.stringWithString_(text).drawAtPoint_withAttributes_(NSMakePoint(x, y), attrs)

except Exception:
    _MacOSGazeView = _MacOSGazeViewBase


class TkGazeOverlay:
    def __init__(
        self,
        platform: PlatformTarget,
        on_quit: Callable[[], None],
        on_capture_neutral: Callable[[], None],
        on_reset_neutral: Callable[[], None],
        passthrough: bool = False,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.platform = platform
        self.on_quit = on_quit
        self.on_capture_neutral = on_capture_neutral
        self.on_reset_neutral = on_reset_neutral
        self._closed = False
        self._exit_code = 0
        self._tick: Callable[[], int | None] | None = None

        self.root = tk.Tk()
        self.root.title("Dismisser Gaze Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.geometry(
            f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
        )
        self._configure_transparency()

        self.canvas = tk.Canvas(self.root, bg="#010101", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<q>", self._quit)
        self.root.bind("<Escape>", self._quit)
        self.root.bind("<c>", self._capture_neutral)
        self.root.bind("<r>", self._reset_neutral)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        if passthrough:
            self.root.after(0, self._configure_click_through)
        self.root.after(0, self._draw_static)

    def run(self, tick: Callable[[], int | None]) -> int:
        self._tick = tick
        self.root.after(1, self._loop)
        self.root.mainloop()
        return self._exit_code

    def update_gaze(self, gaze: GazePoint | None) -> None:
        self.canvas.delete("gaze")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        if gaze is None:
            self.canvas.create_text(
                28,
                32,
                text="No face",
                fill="#ff4b4b",
                anchor="nw",
                font=("Helvetica", 22, "bold"),
                tags="gaze",
            )
            return
        x = int(gaze.x * width)
        y = int(gaze.y * height)
        self.canvas.create_oval(x - 16, y - 16, x + 16, y + 16, outline="#ff2020", width=4, tags="gaze")
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#ff2020", outline="", tags="gaze")
        self.canvas.create_line(x - 28, y, x + 28, y, fill="#ff2020", width=2, tags="gaze")
        self.canvas.create_line(x, y - 28, x, y + 28, fill="#ff2020", width=2, tags="gaze")

    def screen_size(self) -> tuple[int, int]:
        return self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.root.quit()
        self.root.destroy()

    def _loop(self) -> None:
        if self._closed or self._tick is None:
            return
        exit_code = self._tick()
        if exit_code is not None:
            self._exit_code = exit_code
            self.close()
            return
        self.root.after(16, self._loop)

    def _draw_static(self) -> None:
        self.canvas.delete("static")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            self.root.after(50, self._draw_static)
            return
        if self.platform == PlatformTarget.MAC:
            x1, y1 = int(width * 0.72), 0
            x2, y2 = width - 2, int(height * 0.28)
        else:
            x1, y1 = int(width * 0.72), int(height * 0.72)
            x2, y2 = width - 2, height - 2
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ffb000", width=3, tags="static")

    def _configure_transparency(self) -> None:
        try:
            self.root.attributes("-transparentcolor", "#010101")
        except self.tk.TclError:
            self.root.attributes("-alpha", 0.55)

    def _configure_click_through(self) -> None:
        if platform_module.system().lower() != "windows":
            return
        try:
            import ctypes

            hwnd = self.root.winfo_id()
            get_window_long = ctypes.windll.user32.GetWindowLongW
            set_window_long = ctypes.windll.user32.SetWindowLongW
            exstyle = get_window_long(hwnd, -20)
            set_window_long(hwnd, -20, exstyle | 0x00000020 | 0x00080000)
        except Exception as exc:
            print(f"Unable to make overlay click-through: {exc}")

    def _quit(self, _event=None) -> None:
        self.on_quit()
        self.close()

    def _capture_neutral(self, _event=None) -> None:
        self.on_capture_neutral()

    def _reset_neutral(self, _event=None) -> None:
        self.on_reset_neutral()
