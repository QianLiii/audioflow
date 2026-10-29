"""Windows tray icon through Shell_NotifyIcon, without third-party packages.

pystray and Pillow would both be new runtime dependencies for one 16x16 icon,
so the shell API is called directly. A message-only window owns the icon and
receives the mouse callbacks; the host pumps it from its own event loop by
calling :meth:`TrayIcon.pump` periodically (tkinter cannot pump foreign windows
on its own).

Only what an overlay needs: add, tooltip update, a menu with a few commands,
remove.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as w

user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
WM_APP = 0x8000
TRAY_CALLBACK = WM_APP + 1

WM_LBUTTONUP = 0x0202
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205
WM_CONTEXTMENU = 0x007B

NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIM_SETVERSION = 4

NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
NIF_INFO = 0x00000010

NOTIFYICON_VERSION_4 = 4

MF_STRING = 0x00000000
MF_SEPARATOR = 0x00000800
TPM_RETURNCMD = 0x0100
TPM_RIGHTBUTTON = 0x0002

IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040

HWND_MESSAGE = ctypes.c_void_p(-3)

PM_REMOVE = 0x0001


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", w.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", w.HINSTANCE),
        ("hIcon", w.HICON),
        ("hCursor", w.HANDLE),
        ("hbrBackground", w.HBRUSH),
        ("lpszMenuName", w.LPCWSTR),
        ("lpszClassName", w.LPCWSTR),
    ]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", w.DWORD),
        ("hWnd", w.HWND),
        ("uID", w.UINT),
        ("uFlags", w.UINT),
        ("uCallbackMessage", w.UINT),
        ("hIcon", w.HICON),
        ("szTip", w.WCHAR * 128),
        ("dwState", w.DWORD),
        ("dwStateMask", w.DWORD),
        ("szInfo", w.WCHAR * 256),
        ("uVersion", w.UINT),
        ("szInfoTitle", w.WCHAR * 64),
        ("dwInfoFlags", w.DWORD),
        ("guidItem", ctypes.c_byte * 16),
        ("hBalloonIcon", w.HICON),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", w.LONG), ("y", w.LONG)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hWnd", w.HWND),
        ("message", w.UINT),
        ("wParam", w.WPARAM),
        ("lParam", w.LPARAM),
        ("time", w.DWORD),
        ("pt", POINT),
        ("lPrivate", w.DWORD),
    ]


WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = w.ATOM
user32.CreateWindowExW.argtypes = [
    w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID,
]
user32.CreateWindowExW.restype = w.HWND
user32.DefWindowProcW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.DestroyWindow.argtypes = [w.HWND]
user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), w.HWND, w.UINT, w.UINT, w.UINT]
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.LoadImageW.argtypes = [
    w.HINSTANCE, w.LPCWSTR, w.UINT, ctypes.c_int, ctypes.c_int, w.UINT,
]
user32.LoadImageW.restype = w.HANDLE
user32.CreatePopupMenu.restype = w.HMENU
user32.AppendMenuW.argtypes = [w.HMENU, w.UINT, ctypes.c_size_t, w.LPCWSTR]
user32.TrackPopupMenu.argtypes = [
    w.HMENU, w.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, w.LPVOID,
]
user32.TrackPopupMenu.restype = ctypes.c_int
user32.DestroyMenu.argtypes = [w.HMENU]
user32.SetForegroundWindow.argtypes = [w.HWND]
user32.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]

shell32.Shell_NotifyIconW.argtypes = [w.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
shell32.Shell_NotifyIconW.restype = w.BOOL


class TrayIcon:
    """One tray icon with a right-click menu.

    ``commands`` maps a menu id to ``(label, callable)``. The callable runs on
    the caller's thread when :meth:`pump` dispatches the menu selection.
    """

    _class_registered = False
    _class_name = "AudioFlowTrayWindow"

    def __init__(self, icon_path: str, tooltip: str,
                 commands: dict[int, tuple[str, object]] | None = None,
                 on_left_click=None) -> None:
        self.icon_path = icon_path
        self.tooltip = tooltip[:127]
        self.commands = commands or {}
        self.on_left_click = on_left_click
        self.hwnd = None
        self._icon = None
        self._proc = WNDPROC(self._wnd_proc)
        self._visible = False
        self._register_class()
        self._create_window()
        self._add_icon()

    # -- setup -------------------------------------------------------------

    def _register_class(self) -> None:
        if TrayIcon._class_registered:
            return
        wc = WNDCLASSW()
        wc.lpfnWndProc = ctypes.cast(self._proc, ctypes.c_void_p)
        wc.hInstance = kernel32.GetModuleHandleW(None)
        wc.lpszClassName = self._class_name
        if not user32.RegisterClassW(ctypes.byref(wc)):
            raise OSError(ctypes.get_last_error(), "RegisterClassW failed")
        TrayIcon._class_registered = True

    def _create_window(self) -> None:
        self.hwnd = user32.CreateWindowExW(
            0, self._class_name, "AudioFlow", 0, 0, 0, 0, 0,
            HWND_MESSAGE, None, kernel32.GetModuleHandleW(None), None,
        )
        if not self.hwnd:
            raise OSError(ctypes.get_last_error(), "CreateWindowExW failed")

    def _add_icon(self) -> None:
        if self.icon_path:
            self._icon = user32.LoadImageW(None, self.icon_path, IMAGE_ICON,
                                           0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        data = self._notify_data(NIF_MESSAGE | NIF_ICON | NIF_TIP)
        if not shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data)):
            raise OSError(ctypes.get_last_error(), "Shell_NotifyIcon(NIM_ADD) failed")
        data.uVersion = NOTIFYICON_VERSION_4
        shell32.Shell_NotifyIconW(NIM_SETVERSION, ctypes.byref(data))
        self._visible = True

    def _notify_data(self, flags: int) -> NOTIFYICONDATAW:
        data = NOTIFYICONDATAW()
        data.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        data.hWnd = self.hwnd
        data.uID = 1
        data.uFlags = flags
        data.uCallbackMessage = TRAY_CALLBACK
        data.hIcon = self._icon
        data.szTip = self.tooltip
        return data

    # -- runtime -----------------------------------------------------------

    def set_tooltip(self, text: str) -> None:
        self.tooltip = text[:127]
        if not self._visible:
            return
        data = self._notify_data(NIF_TIP)
        shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(data))

    def pump(self) -> None:
        """Dispatch pending messages for the tray window. Call from the host loop."""
        message = MSG()
        while user32.PeekMessageW(ctypes.byref(message), self.hwnd, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

    def close(self) -> None:
        if self._visible:
            data = self._notify_data(0)
            shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(data))
            self._visible = False
        if self.hwnd:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None

    # -- window procedure --------------------------------------------------

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == TRAY_CALLBACK:
            event = lparam & 0xFFFF
            if event == WM_LBUTTONUP and self.on_left_click is not None:
                self.on_left_click()
            elif event in (WM_RBUTTONUP, WM_CONTEXTMENU):
                self._show_menu()
            return 0
        if msg == WM_CLOSE:
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _show_menu(self) -> None:
        menu = user32.CreatePopupMenu()
        for menu_id, (label, _callback) in self.commands.items():
            user32.AppendMenuW(menu, MF_STRING, menu_id, label)

        point = POINT()
        user32.GetCursorPos(ctypes.byref(point))
        # TrackPopupMenu needs the owning window to be foreground, otherwise the
        # menu does not dismiss when the user clicks elsewhere.
        user32.SetForegroundWindow(self.hwnd)
        choice = user32.TrackPopupMenu(
            menu, TPM_RETURNCMD | TPM_RIGHTBUTTON,
            point.x, point.y, 0, self.hwnd, None,
        )
        user32.DestroyMenu(menu)

        if choice:
            entry = self.commands.get(choice)
            if entry is not None:
                entry[1]()
