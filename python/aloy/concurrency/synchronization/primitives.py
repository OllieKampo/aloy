###############################################################################
# Copyright (C) 2023 Oliver Michael Kamperis
# Email: olliekampo@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Module containing custom synchronization primitives."""

import contextlib
import functools
import threading
import types
from typing import final

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.4.0"

__all__ = (
    "OwnedRLock",
)


@final
class OwnedRLock(contextlib.AbstractContextManager):
    """
    Class defining a reentrant lock that keeps track of its owner and
    recursion depth.
    """

    __slots__ = {
        "__name": "The name of the lock.",
        "__lock": "The underlying reentrant lock.",
        "__owner": "The thread that currently owns the lock.",
        "__recursion_depth": "The number of times the lock has been acquired "
                             "by the current thread.",
        "__waiting_lock": "The lock used to protect the waiting set.",
        "__waiting": "The threads that are currently waiting to acquire the "
                     "lock."
    }

    def __init__(self, lock_name: str | None = None) -> None:
        """
        Create a new owned reentrant lock with an optional name.

        Parameters
        ----------
        `lock_name : str | None = None` - The name of the lock, or None to
        give it no name.
        """
        self.__name: str | None = lock_name
        self.__lock = threading.RLock()
        self.__owner: threading.Thread | None = None
        self.__recursion_depth: int = 0
        self.__waiting_lock = threading.Lock()
        self.__waiting: set[threading.Thread] = set()

    def __str__(self) -> str:
        """Get a simple string representation of the lock."""
        return f"{'locked' if self.is_locked else 'unlocked'} " \
               f"{self.__class__.__name__} {self.__name}, " \
               f"owned by={self.__owner}, depth={self.__recursion_depth!s}"

    def __repr__(self) -> str:
        """Get a verbose string representation of the lock."""
        return f"<{'locked' if self.is_locked else 'unlocked'} " \
               f"{self.__class__.__name__}, name={self.__name}, " \
               f"owned by={self.__owner}, " \
               f"recursion depth={self.__recursion_depth!s}, " \
               f"lock={self.__lock!r}>"

    @property
    def name(self) -> str | None:
        """Get the name of the lock, or None if the lock has no name."""
        return self.__name

    @property
    def is_locked(self) -> bool:
        """Get whether the lock is currently locked."""
        return self.__owner is not None

    @property
    def owner(self) -> threading.Thread | None:
        """
        Get the thread that currently owns the lock, or None if the lock is
        not locked.
        """
        return self.__owner

    @property
    def is_owner(self) -> bool:
        """Get whether the current thread owns the lock."""
        return threading.current_thread() is self.__owner

    @property
    def recursion_depth(self) -> int:
        """
        Get the number of times the lock has been acquired by the current
        thread.
        """
        return self.__recursion_depth

    @property
    def any_threads_waiting(self) -> bool:
        """
        Get whether any threads are currently waiting to acquire the lock.
        """
        return bool(self.__waiting)

    @property
    def num_threads_waiting(self) -> int:
        """
        Get the number of threads that are currently waiting to acquire the
        lock.
        """
        return len(self.__waiting)

    @property
    def threads_waiting(self) -> frozenset[threading.Thread]:
        """
        Get the threads that are currently waiting to acquire the lock.
        """
        return frozenset(self.__waiting)

    @property
    def is_thread_waiting(self) -> bool:
        """
        Get whether the current thread is waiting to acquire the lock.
        """
        return threading.current_thread() in self.__waiting

    @functools.wraps(
        threading._RLock.acquire,  # pylint: disable=protected-access
        assigned=("__doc__",)
    )
    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool:
        """
        Acquire the lock, blocking or non-blocking, with an optional timeout.
        """
        current_thread = threading.current_thread()
        if self.__owner is not current_thread:
            with self.__waiting_lock:
                self.__waiting.add(current_thread)
        if result := self.__lock.acquire(blocking, timeout):
            if self.__owner is None:
                with self.__waiting_lock:
                    self.__waiting.remove(current_thread)
                self.__owner = current_thread
            self.__recursion_depth += 1
        if not result:
            with self.__waiting_lock:
                self.__waiting.remove(current_thread)
        return result

    @functools.wraps(
        threading._RLock.release,  # pylint: disable=protected-access
        assigned=("__doc__",)
    )
    def release(self) -> None:
        """Release the lock, if it is locked by the current thread."""
        if self.__owner is threading.current_thread():
            self.__lock.release()
            self.__recursion_depth -= 1
            if self.__recursion_depth == 0:
                self.__owner = None

    __enter__ = acquire

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        """Release the lock when the context manager exits."""
        self.release()
