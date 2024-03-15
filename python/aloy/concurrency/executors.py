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

"""Module containing thread pools and concurrent executors."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, field
import functools
import logging
import os
import queue
import threading
import time
import types
from typing import Any, Callable, Iterable, Iterator, ParamSpec, TypeVar, final

from PySide6.QtCore import (  # pylint: disable=no-name-in-module
    QTimer, QThreadPool,
    QRunnable, QObject, Signal, Slot, Qt
)

from aloy.concurrency.atomic import AtomicNumber

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.0.3"

__all__ = (
    "AloyQThreadPoolExecutor",
    "AloyQTimerExecutor",
    "AloyThreadPool",
    "AloyThreadJob",
    "AloyThreadFinishedJob"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


SP = ParamSpec("SP")
ST = TypeVar("ST")


@final
class _AloyQRunnableSignals(QObject):
    """Signals emitted by the _AloyQRunnable class."""

    start = Signal()
    result = Signal(object)
    error = Signal(Exception)

    def __init__(
        self,
        parent: QObject | None = None,
        start_callback: Callable[[], None] | None = None,
        result_callback: Callable[[ST], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None
    ) -> None:
        super().__init__(parent=parent)
        if start_callback is not None:
            self.start.connect(start_callback)
        if result_callback is not None:
            self.result.connect(result_callback)
        if error_callback is not None:
            self.error.connect(error_callback)


@final
class _AloyQRunnable(QRunnable):
    """A QRunnable that calls a function."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The positional arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "__signals": "The signals emitted by the runnable."
    }

    def __init__(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        signals: _AloyQRunnableSignals | None = None,
        **kwargs: SP.kwargs
    ) -> None:
        super().__init__()
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs
        self.__signals: _AloyQRunnableSignals | None = signals

    @Slot()
    def run(self) -> None:
        if self.__signals is None:
            try:
                self.__func(*self.__args, **self.__kwargs)
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            try:
                self.__signals.start.emit()
                result = self.__func(*self.__args, **self.__kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                self.__signals.error.emit(exc)
            else:
                self.__signals.result.emit(result)


@final
class AloyQThreadPoolExecutor:
    """An executor that calls functions on Qt threads."""

    __slots__ = {
        "__thread_pool": "The Qt thread pool."
    }

    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is None:
            max_workers = os.cpu_count()
            if max_workers is None:
                max_workers = 1
        self.__thread_pool = QThreadPool()
        self.__thread_pool.setMaxThreadCount(max_workers)

    def submit(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> None:
        runnable = _AloyQRunnable(func, *args, signals=None, **kwargs)
        self.__thread_pool.start(runnable)

    def submit_with_callbacks(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        start_callback: Callable[[], None] | None = None,
        result_callback: Callable[[ST], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        **kwargs: SP.kwargs
    ) -> None:
        signals = _AloyQRunnableSignals(
            start_callback=start_callback,
            result_callback=result_callback,
            error_callback=error_callback
        )
        runnable = _AloyQRunnable(func, *args, signals=signals, **kwargs)
        self.__thread_pool.start(runnable)


@final
class AloyQTimerExecutor:
    """
    An executor that calls functions on a QTimer.

    Unlike a AloyQThreadPoolExecutor, this executor will call functions on the
    Qt thread that started the executor. This means that functions submitted
    to this executor can set the parent or children of a Qt widget, and start
    other Qt timers.
    """

    __slots__ = {
        "__queue": "The queue of runnables.",
        "__lock": "The lock used to protect the queue.",
        "__timer": "The timer used to execute the runnables."
    }

    def __init__(self, interval: float) -> None:
        self.__queue: queue.Queue[_AloyQRunnable] = queue.Queue()
        self.__lock = threading.Lock()
        self.__timer = QTimer()
        self.__timer.setInterval(int(interval * 1000))
        self.__timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.__timer.timeout.connect(self.__execute)

    @property
    def interval(self) -> float:
        """Return the interval of the timer."""
        return self.__timer.interval() / 1000

    @interval.setter
    def interval(self, interval: float) -> None:
        """Set the interval of the timer."""
        self.__timer.setInterval(int(interval * 1000))

    def start(self) -> None:
        """Start the timer."""
        self.__timer.start()

    def __execute(self) -> None:
        start_time = time.monotonic()
        try:
            # Execute for at most half the interval time.
            while (time.monotonic() - start_time) < (self.interval * 0.5):
                with self.__lock:
                    runnable = self.__queue.get_nowait()
                runnable.run()
        except queue.Empty:
            return

    def __add_runnable(self, runnable: _AloyQRunnable) -> None:
        with self.__lock:
            self.__queue.put(runnable)

    def submit(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> None:
        runnable = _AloyQRunnable(func, *args, signals=None, **kwargs)
        self.__add_runnable(runnable)

    def submit_with_callbacks(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        start_callback: Callable[[], None] | None = None,
        result_callback: Callable[[ST], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        **kwargs: SP.kwargs
    ) -> None:
        signals = _AloyQRunnableSignals(
            start_callback=start_callback,
            result_callback=result_callback,
            error_callback=error_callback
        )
        runnable = _AloyQRunnable(func, *args, signals=signals, **kwargs)
        self.__add_runnable(runnable)


@dataclass(frozen=True)
class AloyThreadJob:
    """A class that represents a job submitted to a thread pool."""

    name: str | None
    func: Callable[SP, ST]  # type: ignore
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(hash=False)
    start_time: float = field(default=0.0, hash=False)

    def __iter__(self) -> Iterable:
        return iter(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class AloyThreadFinishedJob(AloyThreadJob):
    """A class that represents a job that has finished execution."""

    elapsed_time: float | None = field(default=None, hash=False)


@final
class AloyThreadPool:
    """
    A thread pool that allows perfornance profiling and logging of submitted
    jobs.

    The thread pool is a wrapper around the ThreadPoolExecutor class.
    """

    __POOLS: AtomicNumber[int] = AtomicNumber(0)

    __slots__ = {
        "__name": "The name of the thread pool.",
        "__logger": "The logger used to log jobs.",
        "__log": "Whether to log jobs.",
        "__submitted_jobs": "The jobs submitted to the thread pool.",
        "__finished_jobs": "The jobs that have finished execution.",
        "__queued_jobs": "The number of jobs in the queue.",
        "__max_workers": "The maximum number of workers allowed.",
        "__active_threads": "The number of active threads.",
        "__main_thread": "The main thread.",
        "__thread_pool": "The thread pool."
    }

    @classmethod
    def __get_pool_name(cls) -> str:
        """Returns a unique name for a thread pool."""
        with cls.__POOLS:
            cls.__POOLS += 1
            return f"{cls.__name__} [{cls.__POOLS.get_obj()}]"

    def __init__(
        self,
        pool_name: str | None = None,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
        log: bool = False,
        initializer: Callable[SP, None] | None = None,
        initargs: tuple[Any, ...] = ()
    ) -> None:
        """
        Create a new aloy thread pool.

        Parameters
        ----------
        `pool_name: str | None` - The name of the thread pool. If not given or
        None, a unique name is generated.

        `max_workers: int | None` - The maximum number of workers allowed in
        the thread pool. If not given or None, the number of workers is set to
        the number of CPUs on the system.

        `thread_name_prefix: str` - A prefix to give to the names of the
        threads in the thread pool.

        `profile: bool` - Whether to profile jobs submitted to the thread
        pool.

        `log: bool` - Whether to log jobs submitted to the thread pool.

        `initializer: Callable[SP, None] | None` - A callable used to
        initialize worker threads.

        `initargs: tuple[Any, ...]` - A tuple of arguments to pass to the
        initializer.

        Raises
        ------
        `ValueError` - If `max_workers` is less than or equal to 0.
        """
        if pool_name is None:
            pool_name = self.__get_pool_name()
        self.__name: str = pool_name
        self.__logger = logging.getLogger("AloyThreadPool")
        self.__logger.setLevel(logging.DEBUG)
        self.__log: bool = log

        self.__submitted_jobs: dict[Future, AloyThreadJob] = {}
        self.__finished_jobs: dict[Future, AloyThreadFinishedJob] = {}
        self.__queued_jobs = AtomicNumber[int](0)
        self.__active_threads = AtomicNumber[int](0)

        if max_workers is None:
            max_workers = os.cpu_count()
        if max_workers is None:
            max_workers = 1
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self.__max_workers: int = max_workers
        self.__main_thread: threading.Thread = threading.current_thread()
        self.__thread_pool = ThreadPoolExecutor(
            max_workers,
            thread_name_prefix,
            initializer,
            initargs
        )

    def __enter__(self) -> "AloyThreadPool":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__thread_pool.shutdown(wait=True)

    @property
    def active_workers(self) -> int:
        """Return the number of active workers in the thread pool."""
        return self.__active_threads.get_obj()

    @property
    def max_workers(self) -> int:
        """Return the maximum number of workers allowed in the thread pool."""
        return self.__max_workers

    @property
    def is_running(self) -> bool:
        """Return True if the thread pool is running, False otherwise."""
        return self.__active_threads.get_obj() > 0

    @property
    def is_busy(self) -> bool:
        """
        Return True if the thread pool is busy, False otherwise.

        A thread pool is considered busy if all of its workers are actively
        executing a job.
        """
        return self.__active_threads.get_obj() == self.__max_workers

    def submit(
        self,
        name: str,
        func: Callable[SP, ST],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> Future:
        """
        Submits a job to the thread pool and returns a Future object.

        The callable is scheduled to be executed as `func(*args, **kwargs)` and
        returns a Future instance representing the execution of the callable.
        """
        with self.__queued_jobs, self.__active_threads:
            self.__queued_jobs += 1
            active_threads = min(self.__active_threads.get_obj() + 1,
                                 self.__max_workers)
            self.__active_threads.set_obj(active_threads)

        start_time: float = time.monotonic()

        if self.__log:
            self.__logger.debug(
                "%s: Submitting job %s -> %s(*%s, **%s)",
                self.__name, name, func.__name__, args, kwargs
            )

        future = self.__thread_pool.submit(func, *args, **kwargs)
        self.__submitted_jobs[future] = AloyThreadJob(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            start_time=start_time
        )
        future.add_done_callback(self.__callback)

        return future

    def __callback(self, future: Future) -> None:
        """Callback function for when a job finishes execution."""
        job: AloyThreadJob = self.__submitted_jobs.pop(future)
        if self.__log:
            self.__logger.debug(
                "%s: Job %s -> %s(*%s, **%s) finished",
                self.__name, job.name, job.func.__name__, job.args, job.kwargs
            )

        elapsed_time: float = time.perf_counter() - job.start_time

        self.__finished_jobs[future] = AloyThreadFinishedJob(  # type: ignore
            *job,
            elapsed_time=elapsed_time
        )

        with self.__queued_jobs, self.__active_threads:
            self.__queued_jobs -= 1
            if self.__queued_jobs.get_obj() < self.__max_workers:
                self.__active_threads -= 1

    def get_job(self, future: Future) -> AloyThreadJob:
        """Return the job associated with the given future."""
        return self.__submitted_jobs.pop(future)

    def get_finished_job(self, future: Future) -> AloyThreadFinishedJob:
        """
        Return the finished job associated with the given future.

        Raises a ValueError if the future is not a finished job.
        """
        if future not in self.__finished_jobs:
            raise ValueError(f"Future {future} is not a finished job.")
        return self.__finished_jobs.pop(future)

    def is_finished(self, future: Future) -> bool:
        """
        Return True if the given future is a finished job, False otherwise.
        """
        return future in self.__finished_jobs

    @functools.wraps(ThreadPoolExecutor.map,
                     assigned=("__doc__",))
    def map(
        self,
        name: str,
        func: Callable[SP, ST],
        *iterables: Iterable[SP.args],
        timeout: float | None = None
    ) -> Iterator[ST | None]:
        if timeout is not None:
            end_time = timeout + time.monotonic()

        futures: list[Future[ST]] = [
            self.submit(name, func, *args)
            for args in zip(*iterables)
        ]
        futures.reverse()

        return AloyThreadPool.__iter_results(
            futures, end_time)  # type: ignore[arg-type]

    @staticmethod
    def __get_result(
        future_: Future[ST],
        timeout: float | None = None
    ) -> ST | None:
        try:
            try:
                return future_.result(timeout)
            finally:
                future_.cancel()
        finally:
            del future_

    @staticmethod
    def __iter_results(
        futures: list[Future[ST]],
        end_time: float | None = None
    ) -> Iterator[ST | None]:
        try:
            while futures:
                if end_time is None:
                    yield AloyThreadPool.__get_result(futures.pop())
                else:
                    yield AloyThreadPool.__get_result(
                        futures.pop(),
                        end_time - time.monotonic()
                    )
        finally:
            for future_ in futures:
                future_.cancel()

    @functools.wraps(ThreadPoolExecutor.shutdown, assigned=("__doc__",))
    def shutdown(self, wait: bool = True) -> None:
        self.__thread_pool.shutdown(wait=wait)
