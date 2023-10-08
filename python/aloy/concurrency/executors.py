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

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, field
import functools
import logging
import os
import threading
import time
import types
from typing import Any, Callable, Iterable, Iterator, ParamSpec, TypeVar, final

import urllib.request
from PySide6.QtCore import (QThreadPool,  # pylint: disable=no-name-in-module
                            QRunnable, QObject, Signal, Slot)

from aloy.concurrency.atomic import AtomicNumber

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.0.2"

__all__ = (
    "AloyQThreadPool",
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
    error = Signal(tuple[Any, ...])


@final
class _AloyQRunnable(QRunnable):
    """A QRunnable that calls a function."""

    __slots__ = {
        "__func": "The function to call.",
        "__args": "The positional arguments to pass to the function.",
        "__kwargs": "The keyword arguments to pass to the function.",
        "_signals": "The signals emitted by the runnable."
    }

    def __init__(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        add_signals: bool,
        **kwargs: SP.kwargs
    ) -> None:
        super().__init__()
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs
        if add_signals:
            self._signals = _AloyQRunnableSignals()
        else:
            self._signals = None

    @Slot()
    def run(self) -> None:
        if self._signals is None:
            self.__func(*self.__args, **self.__kwargs)
        else:
            try:
                self._signals.start.emit()
                result = self.__func(*self.__args, **self.__kwargs)
            except Exception as exc:  # pylint: disable=broad-except
                self._signals.error.emit(exc)
            else:
                self._signals.result.emit(result)


@final
class AloyQThreadPool:
    """An executor that calls functions on a QThread."""

    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is None:
            max_workers = os.cpu_count()
        self.__thread_pool = QThreadPool()
        self.__thread_pool.setMaxThreadCount(max_workers)

    def submit(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> None:
        runnable = _AloyQRunnable(func, *args, add_signals=False, **kwargs)
        self.__thread_pool.start(runnable)

    def submit_with_callbacks(
        self,
        func: Callable[SP, ST],
        *args: SP.args,
        start_callback: Callable[[], None] | None = None,
        result_callback: Callable[[ST], None] | None = None,
        error_callback: Callable[[tuple[Any, ...]], None] | None = None,
        **kwargs: SP.kwargs
    ) -> None:
        runnable = _AloyQRunnable(func, *args, add_signals=True, **kwargs)
        # pylint: disable=protected-access
        if start_callback is not None:
            runnable._signals.start.connect(start_callback)
        if result_callback is not None:
            runnable._signals.result.connect(result_callback)
        if error_callback is not None:
            runnable._signals.error.connect(error_callback)
        # pylint: enable=protected-access
        self.__thread_pool.start(runnable)


@dataclass(frozen=True)
class AloyThreadJob:
    """A class that represents a job submitted to a thread pool."""

    name: str | None
    func: Callable[SP, ST]  # type: ignore
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(hash=False)
    start_time: float | None = field(default=None, hash=False)

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
        "__profile": "Whether to profile jobs.",
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
        profile: bool = False,
        log: bool = False,
        initializer: Callable[SP, None] | None = None,
        initargs: tuple[Any, ...] = ()
    ) -> None:
        """
        max_workers: The maximum number of threads that can be used to
        execute the given calls.
        thread_name_prefix: An optional name prefix to give our threads.
        initializer: A callable used to initialize worker threads.
        initargs: A tuple of arguments to pass to the initializer.
        """
        self.__name: str = pool_name or self.__get_pool_name()
        self.__logger = logging.getLogger("AloyThreadPool")
        self.__logger.setLevel(logging.DEBUG)
        self.__log: bool = log
        self.__profile: bool = profile

        self.__submitted_jobs: dict[Future, AloyThreadJob] = {}
        self.__finished_jobs: dict[Future, AloyThreadFinishedJob] = {}
        self.__queued_jobs = AtomicNumber[int](0)
        self.__active_threads = AtomicNumber[int](0)

        if max_workers is None:
            max_workers = os.cpu_count()
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

        start_time: float | None = None
        if self.__profile:
            start_time = time.perf_counter()

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

        elapsed_time: float | None = None
        if self.__profile:
            elapsed_time = time.perf_counter() - job.start_time  # type: ignore

        self.__finished_jobs[future] = AloyThreadFinishedJob(  # type: ignore
            *job,
            elapsed_time=elapsed_time
        )

        with self.__queued_jobs, self.__active_threads:
            self.__queued_jobs -= 1
            if self.__queued_jobs.get_obj() < self.__max_workers:
                self.__active_threads -= 1

    def get_job(self, future: Future) -> AloyThreadJob | AloyThreadFinishedJob:
        """Returns the job associated with the given future."""
        if future in self.__finished_jobs:
            return self.__finished_jobs[future]
        return self.__submitted_jobs.pop(future)

    @functools.wraps(ThreadPoolExecutor.map,
                     assigned=("__doc__",))
    def map(
        self,
        name: str,
        func: Callable[SP, ST],
        *iterables: Iterable[SP.args],
        timeout: float | None = None
    ) -> Iterator[Future[ST]]:
        if timeout is not None:
            end_time = timeout + time.perf_counter()

        futures = [
            self.submit(name, func, *args)
            for args in zip(*iterables)
        ]
        futures.reverse()

        return AloyThreadPool.__iter_results(futures, timeout, end_time)

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
        futures: list[Future],
        timeout: float | None = None,
        end_time: float | None = None
    ) -> Iterator[ST]:
        try:
            while futures:
                if timeout is None:
                    yield AloyThreadPool.__get_result(futures.pop())
                else:
                    yield AloyThreadPool.__get_result(
                        futures.pop(), end_time - time.monotonic())
        finally:
            for future_ in futures:
                future_.cancel()

    @functools.wraps(ThreadPoolExecutor.shutdown, assigned=("__doc__",))
    def shutdown(self, wait: bool = True) -> None:
        self.__thread_pool.shutdown(wait=wait)


class WebScraper:
    """
    https://realpython.com/beautiful-soup-web-scraper-python/
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.__thread_pool = AloyThreadPool(
            "WebScraper",
            max_workers,
            profile=True,
            log=True
        )

    def scrape(self, urls: Iterable[str]) -> None:
        with self.__thread_pool as executor:
            futures: dict[Future, str] = {
                executor.submit(url, self.load_url, url, 60): url
                for url in urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    print('%r page is %d bytes' % (url, len(data)))
                print(executor.get_job(future))

    @staticmethod
    def load_url(url: str, timeout: float) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as connection:
            return connection.read()


def __main() -> None:
    # Setup logging.
    format_ = "[%(asctime)s] %(levelname)-4s :: %(name)-8s >> %(message)s\n"

    logging.basicConfig(
        level=logging.DEBUG,
        format=format_,
        datefmt="%H:%M:%S",
        handlers=[
            # logging.FileHandler("debug.log"),
            # console_handler
        ],
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(format_))
    logging.getLogger("").addHandler(console_handler)

    urls = [
        "https://www.google.com",
        "https://www.yahoo.com",
        "https://www.bing.com",
        "https://www.duckduckgo.com",
        "https://www.aol.com",
        "https://www.wolframalpha.com"
    ]
    scraper = WebScraper(max_workers=4)
    scraper.scrape(urls)


if __name__ == "__main__":
    __main()
