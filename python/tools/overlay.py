"""Transparent, always-on-top subtitle overlay.

Two lines: the previous chunk on top, the live chunk below. A chunk keeps its
line after its final arrives; it only moves up when the next chunk produces its
first partial, so the subtitle never blinks between chunks. Gaps are never
shown.

The window is click-through except while Ctrl is held, which is also when it
can be dragged. Quitting happens from the tray icon: the intended launcher is
``pythonw.exe``, so there is no console.

Usage:
  pythonw tools/overlay.py                 # live capture
  python tools/overlay.py --self-test 15   # runs 15 s, logs what it drew
  python tools/overlay.py --fake --wav testdata/live_capture_10s.wav
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field

import tkinter as tk

# Allow running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioflow.asr import LlamaCppAsrClient
from audioflow.asr.fake import FakeAsrClient, VirtualClock
from audioflow.session import AudioFlowSession
from audioflow.tray import TrayIcon

KEY_COLOR = "#010203"          # colour key: these pixels become transparent
TEXT_COLOR = "#FFFFFF"
TENTATIVE_COLOR = "#B9B9B9"
DEGRADED_COLOR = "#E8C46A"
STATUS_COLOR = "#FF7A7A"

MENU_TOGGLE = 1
MENU_QUIT = 2

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
VK_CONTROL = 0x11

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetParent.argtypes = [ctypes.c_void_p]
user32.GetParent.restype = ctypes.c_void_p
kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
kernel32.CreateMutexW.restype = ctypes.c_void_p

try:
    ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)   # per-monitor aware
except Exception:  # noqa: BLE001 - older Windows keeps the default
    pass


@dataclass
class Config:
    url: str = "http://127.0.0.1:8080"
    vad_model: str = "native/model/silero_vad.onnx"
    partial_ms: int = 250
    min_partial_age: int = 300
    max_chunk_ms: int = 4000
    font_family: str = "Microsoft YaHei UI"
    font_size: int = 28
    width_ratio: float = 0.8
    bottom_margin: int = 80
    hide_after_s: float = 3.0
    show_previous: bool = True
    text_color: str = TEXT_COLOR
    tentative_color: str = TENTATIVE_COLOR
    degraded_color: str = DEGRADED_COLOR

    @classmethod
    def load(cls, path: str | None) -> "Config":
        config = cls()
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                data = tomllib.load(handle)
            for section in ("asr", "ui", "vad"):
                for key, value in (data.get(section) or {}).items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            for key, value in data.items():
                if not isinstance(value, dict) and hasattr(config, key):
                    setattr(config, key, value)
        return config


class Overlay:
    def __init__(self, args) -> None:
        self.args = args
        self.config = Config.load(args.config)
        if args.url:
            self.config.url = args.url
        if args.partial_ms:
            self.config.partial_ms = args.partial_ms
        if args.min_partial_age:
            self.config.min_partial_age = args.min_partial_age
        if args.no_hide:
            self.config.hide_after_s = 0.0
        if args.console:
            self.config.hide_after_s = 0.0

        self.root = tk.Tk()
        self.root.withdraw()
        self._build_window()

        self.session = None
        self.healthy = False
        self._stopping = False
        self._last_change = time.monotonic()
        self._last_drawn = None
        self._visible = True
        self._ctrl_down = False
        self._drag_origin = None

        self.tray = self._build_tray()
        self._bind_events()
        self._start_session()
        self._start_health_loop()
        self._schedule()

    # -- window ------------------------------------------------------------

    def _build_window(self) -> None:
        config = self.config
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", KEY_COLOR)
        self.root.configure(bg=KEY_COLOR)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.width = int(screen_w * config.width_ratio)
        self.line_height = int(config.font_size * 1.45)
        self.height = self.line_height * 2 + int(config.font_size * 0.4)

        state = self._load_state()
        x = state.get("x", (screen_w - self.width) // 2)
        y = state.get("y", screen_h - self.height - config.bottom_margin)
        self.root.geometry("%dx%d+%d+%d" % (self.width, self.height, x, y))

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height,
                                bg=KEY_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack()

        self.root.deiconify()
        self._apply_click_through(True)

    def _toplevel_hwnd(self) -> int:
        hwnd = self.root.winfo_id()
        parent = user32.GetParent(ctypes.c_void_p(hwnd))
        return parent or hwnd

    def _apply_click_through(self, enabled: bool) -> None:
        hwnd = self._toplevel_hwnd()
        style = user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
        base = style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED
        if enabled:
            base |= WS_EX_TRANSPARENT
        else:
            base &= ~WS_EX_TRANSPARENT
        if base != style:
            user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), GWL_EXSTYLE, base)

    # -- tray --------------------------------------------------------------

    def _build_tray(self):
        icon_path = os.path.join(self._root_dir(), "assets", "audioflow.ico")
        commands = {
            MENU_TOGGLE: ("显示 / 隐藏", self.toggle_visible),
            MENU_QUIT: ("退出", self.quit),
        }
        try:
            return TrayIcon(icon_path, "AudioFlow 实时字幕", commands,
                            on_left_click=self.toggle_visible)
        except Exception as exc:  # noqa: BLE001 - keep running without a tray
            logging.error("tray icon unavailable: %s", exc)
            return None

    # -- session -----------------------------------------------------------

    def _root_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _start_session(self) -> None:
        if self.args.fake:
            client = FakeAsrClient(clock=VirtualClock())
            self.healthy = True
        else:
            client = LlamaCppAsrClient(self.config.url)
            self.healthy = client.health()

        model = self.config.vad_model
        if not os.path.isabs(model):
            model = os.path.join(self._root_dir(), model)

        log_dir = os.path.join(self._root_dir(), "build", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.session = AudioFlowSession(
            client,
            vad_model=model,
            partial_ms=self.config.partial_ms,
            min_partial_age=self.config.min_partial_age / 1000.0,
            max_chunk_ms=self.config.max_chunk_ms,
            log_path=os.path.join(log_dir, "overlay-%s.jsonl"
                                  % time.strftime("%Y%m%d-%H%M%S")),
        )

        if self.args.wav:
            self.session.start_file(self.args.wav, speed=self.args.speed)
        else:
            self.session.start_capture()

    def _start_health_loop(self) -> None:
        if self.args.fake:
            return

        def watch() -> None:
            while not self._stopping:
                time.sleep(5.0)
                if self._stopping:
                    return
                try:
                    healthy = LlamaCppAsrClient(self.config.url).health()
                except Exception:  # noqa: BLE001
                    healthy = False
                if healthy != self.healthy:
                    self.healthy = healthy
                    self.root.after(0, self.refresh)
                    if self.tray is not None:
                        self.tray.set_tooltip(
                            "AudioFlow 实时字幕" if healthy else "AudioFlow：ASR 未连接")

        threading.Thread(target=watch, daemon=True).start()

    # -- events ------------------------------------------------------------

    def _bind_events(self) -> None:
        self.root.bind("<Button-1>", self._on_press)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event) -> None:
        self._drag_origin = (event.x_root, event.y_root,
                             self.root.winfo_x(), self.root.winfo_y())

    def _on_drag(self, event) -> None:
        if self._drag_origin is None:
            return
        x0, y0, wx, wy = self._drag_origin
        self.root.geometry("+%d+%d" % (wx + event.x_root - x0, wy + event.y_root - y0))

    def _on_release(self, _event) -> None:
        if self._drag_origin is not None:
            self._drag_origin = None
            self._save_state()

    # -- state -------------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self._root_dir(), "build", "overlay-state.json")

    def _load_state(self) -> dict:
        try:
            with open(self._state_path(), "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            with open(self._state_path(), "w", encoding="utf-8") as handle:
                json.dump({"x": self.root.winfo_x(), "y": self.root.winfo_y()}, handle)
        except OSError as exc:
            logging.warning("could not save state: %s", exc)

    # -- rendering ---------------------------------------------------------

    def _lines(self):
        """Return (previous_segment, current_segment); either may be None."""
        try:
            segments = self.session.tail(2)
        except Exception:  # noqa: BLE001 - a torn read must not kill the UI
            return None, None
        groups: list[tuple[int, object]] = []
        for segment in segments:
            if groups and groups[-1][0] == segment.chunk_id:
                groups[-1] = (segment.chunk_id, segment)   # later wins
            else:
                groups.append((segment.chunk_id, segment))
        if not groups:
            return None, None
        current = groups[-1][1]
        previous = groups[-2][1] if len(groups) > 1 else None
        return previous, current

    def _draw_text(self, y: int, text: str, colour: str) -> None:
        if not text:
            return
        font = (self.config.font_family, self.config.font_size, "bold")
        offsets = ((-2, -2), (2, -2), (-2, 2), (2, 2), (0, -3), (0, 3), (-3, 0), (3, 0))
        for dx, dy in offsets:
            self.canvas.create_text(
                self.width // 2 + dx, y + dy, text=text, fill="#000000",
                font=font, width=self.width - 60, justify="center", anchor="n")
        self.canvas.create_text(
            self.width // 2, y, text=text, fill=colour,
            font=font, width=self.width - 60, justify="center", anchor="n")

    def _draw(self) -> None:
        self.canvas.delete("all")
        previous, current = self._lines()

        if not self.healthy and not self.args.fake:
            self._draw_text(self.line_height // 2, "[ASR 未连接]", STATUS_COLOR)
            return

        if current is None:
            return

        if previous is not None and self.config.show_previous:
            colour = (self.config.tentative_color if previous.provisional
                      else self.config.degraded_color if previous.degraded
                      else self.config.text_color)
            self._draw_text(0, previous.text, colour)

        colour = (self.config.tentative_color if current.provisional
                  else self.config.degraded_color if current.degraded
                  else self.config.text_color)
        self._draw_text(self.line_height - 6, current.text, colour)

    def refresh(self) -> None:
        self._draw()
        self._visible = True

    def toggle_visible(self) -> None:
        self._visible = not self._visible
        if self._visible:
            self.refresh()
        else:
            self.canvas.delete("all")
            self._last_drawn = None

    # -- loop --------------------------------------------------------------

    def _schedule(self) -> None:
        self.root.after(50, self._tick)

    def _tick(self) -> None:
        if self._stopping:
            return

        if self.tray is not None:
            try:
                self.tray.pump()
            except Exception as exc:  # noqa: BLE001
                logging.error("tray pump failed: %s", exc)
                self.tray = None

        ctrl = bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
        if ctrl != self._ctrl_down:
            self._ctrl_down = ctrl
            self._apply_click_through(not ctrl)

        try:
            version = self.session.version
        except Exception:  # noqa: BLE001
            version = None

        if version is not None and version != self._last_drawn:
            self._last_drawn = version
            self._last_change = time.monotonic()
            if self._visible:
                self.refresh()

        if (self.config.hide_after_s > 0 and self._visible
                and time.monotonic() - self._last_change > self.config.hide_after_s):
            self.canvas.delete("all")
            self._visible = False

        if self.args.self_test and time.monotonic() - self.started > self.args.self_test:
            logging.info("self-test finished")
            self.quit()
            return

        self._schedule()

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        self.started = time.monotonic()
        self.root.after(200, self._log_state)
        self.root.mainloop()

    def _log_state(self) -> None:
        if self._stopping:
            return
        previous, current = self._lines()
        logging.info("UI  prev=%r  cur=%r  healthy=%s",
                     previous.text if previous else None,
                     current.text if current else None, self.healthy)
        self.root.after(1000, self._log_state)

    def quit(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        logging.info("shutting down")
        try:
            self._save_state()
            if self.session is not None:
                self.session.stop()
        finally:
            if self.tray is not None:
                try:
                    self.tray.close()
                except Exception:  # noqa: BLE001
                    pass
                self.tray = None
            self.root.quit()


def single_instance() -> bool:
    handle = kernel32.CreateMutexW(None, False, "Global\\AudioFlowOverlay")
    if not handle:
        return True
    return kernel32.GetLastError() != 183      # ERROR_ALREADY_EXISTS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--partial-ms", type=int, default=None)
    parser.add_argument("--min-partial-age", type=int, default=None)
    parser.add_argument("--wav", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fake", action="store_true", help="no ASR server needed")
    parser.add_argument("--no-hide", action="store_true")
    parser.add_argument("--console", action="store_true", help="keep text on screen")
    parser.add_argument("--self-test", type=float, default=0.0)
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = args.config or os.path.join(root_dir, "config.toml")
    args.config = config_path if os.path.isfile(config_path) else None

    log_dir = os.path.join(root_dir, "build", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(
            log_dir, "overlay-%s.log" % time.strftime("%Y%m%d-%H%M%S")),
            encoding="utf-8")],
    )

    if not args.self_test and not single_instance():
        logging.warning("another instance is already running")
        ctypes.windll.user32.MessageBoxW(None, "AudioFlow 已在运行。", "AudioFlow", 0x40)
        return 0

    try:
        overlay = Overlay(args)
        overlay.run()
    except Exception:  # noqa: BLE001 - pythonw has no console to show a traceback
        logging.exception("fatal error")
        ctypes.windll.user32.MessageBoxW(None, "AudioFlow 启动失败，详见 build\\logs。",
                                         "AudioFlow", 0x10)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
