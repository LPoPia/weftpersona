from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class BackgroundTasks:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._sealed = False

    def try_register(self) -> bool:
        with self._condition:
            if self._sealed:
                return False
            self._active += 1
            return True

    def complete(self) -> None:
        with self._condition:
            if self._active > 0:
                self._active -= 1
            if not self._active:
                self._condition.notify_all()

    def seal_and_wait(self) -> None:
        with self._condition:
            self._sealed = True
            while self._active:
                self._condition.wait()

    def unseal(self) -> None:
        with self._condition:
            self._sealed = False
            self._condition.notify_all()

    @contextmanager
    def track(self) -> Iterator[bool]:
        registered = self.try_register()
        try:
            yield registered
        finally:
            if registered:
                self.complete()

    @contextmanager
    def pause(self, *, unseal_on_exit: bool = False) -> Iterator[None]:
        with self._condition:
            was_sealed = self._sealed
            self._sealed = True
            while self._active:
                self._condition.wait()
        unsealed = False
        try:
            yield
            unsealed = unseal_on_exit
        finally:
            with self._condition:
                self._sealed = False if unsealed else was_sealed
                if not self._sealed:
                    self._condition.notify_all()
