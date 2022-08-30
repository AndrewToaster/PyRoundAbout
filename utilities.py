from __future__ import annotations

from abc import abstractmethod
from threading import Thread, RLock
from sys import byteorder, stdin
from encodings import utf_8
from typing import Callable, Any, AnyStr, Protocol
from libs.xplatgetch import kbhit, getch


class InputThread(Thread):
    # noinspection SpellCheckingInspection
    def __init__(self, clbk: Callable[[str], Any], name='input-thread'):
        self.clbk = clbk
        self.stop_flag = False
        self.tty = stdin.isatty()
        super().__init__(name=name, daemon=True)

    def run(self):
        while not self.stop_flag:
            self.clbk(stdin.read(1))

    def stop(self):
        self.stop_flag = True


class IoBuffer:
    def __init__(self, handle: TextReadable):
        self.handle = handle
        self.buffer: str | None = None

    def any(self) -> bool:
        if self.buffer is None:
            val = self.handle.read(1)
            if len(val) != 0:
                self.buffer = val
                return True
            else:
                return False
        else:
            return True

    def read_one(self) -> str:
        if self.buffer is not None:
            val = self.buffer
            self.buffer = None
            return val
        else:
            return self.handle.read(1)


class TextReadable(Protocol):
    @abstractmethod
    def read(self, __n: int = ...) -> AnyStr: ...


class TextWritable(Protocol):
    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def write(self, __s: AnyStr) -> int: ...


class GetchIO(TextReadable):
    def read(self, __n: int = ...) -> AnyStr:
        if kbhit():
            return getch()
        else:
            return ""


class LockedListIO(TextWritable, TextReadable):
    def __init__(self):
        self.list_buffer = []
        self._lock = RLock()

    def write(self, s: str) -> int:
        with self._lock:
            self.list_buffer.extend(s)
            return len(s)

    def flush(self) -> None:
        pass

    def read(self, size: int | None = ...) -> AnyStr:
        with self._lock:
            read_len = min(size, len(self.list_buffer))
            val = self.list_buffer[:read_len]
            del self.list_buffer[:read_len]
            return ''.join(val)


def number_from_bytes(data: bytes) -> int:
    return int.from_bytes(data, byteorder)


def number_to_bytes(num: int) -> bytes:
    return int.to_bytes(num, (num.bit_length() // 8) + 1, byteorder)


def to_utf8(num: int) -> str:
    return utf_8.decode(int.to_bytes(num, (num.bit_length() // 8) + 1, byteorder, signed=True))[0]


def from_utf8(__str: str) -> int:
    return int.from_bytes(utf_8.encode(__str)[0], byteorder)


def parse_map(content: str, width: int, height: int) -> list[list[str]]:
    lines = content.splitlines(False)
    map = []
    for y in range(height):
        lst = []
        for x in range(width):
            if y < len(lines) and x < len(lines[y]):
                lst.append(lines[y][x])
            else:
                lst.append(' ')
        map.append(lst)
    return map
