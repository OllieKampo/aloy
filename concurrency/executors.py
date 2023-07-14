
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, field
import functools
import logging
import threading
import time
import types
from typing import Any, Callable, Iterable, ParamSpec, TypeVar, final

import urllib.request

from concurrency.atomic import AtomicNumber


SP = ParamSpec("SP")
ST = TypeVar("ST")


@dataclass(frozen=True)
class Job:
    """A class that represents a job submitted to a thread pool."""

    name: str | None
    func: Callable[SP, ST]  # type: ignore
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(hash=False)
    start_time: float | None = field(default=None, hash=False)

    def __iter__(self) -> Iterable:
        return iter(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class FinishedJob(Job):
    """A class that represents a job that has finished execution."""
    elapsed_time: float | None = field(default=None, hash=False)


@final
class JinxThreadPool:
    """
    A thread pool that allows perfornance profiling and logging of submitted
    jobs.

    The thread pool is a wrapper around the ThreadPoolExecutor class.
    """

    __POOLS: AtomicNumber[int] = AtomicNumber(0)

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
        self.__logger = logging.getLogger("JinxThreadPool")
        self.__logger.setLevel(logging.DEBUG)
        self.__log: bool = log

        self.__max_workers: int = 1
        if max_workers is not None:
            self.__max_workers = max_workers
        self.__submitted_jobs: dict[Future, Job] = {}
        self.__finished_jobs: dict[Future, FinishedJob] = {}
        self.__queued_jobs = AtomicNumber[int](0)
        self.__active_threads = AtomicNumber[int](0)

        self.__profile: bool = profile

        self.__main_thread: threading.Thread = threading.current_thread()
        self.__thread_pool = ThreadPoolExecutor(
            max_workers,
            thread_name_prefix,
            initializer,
            initargs
        )

    def __enter__(self) -> "JinxThreadPool":
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
        return self.__active_threads.get_obj() == self.__max_workers

    @functools.wraps(ThreadPoolExecutor.submit,
                     assigned=("__doc__",))
    def submit(
        self,
        name: str,
        func: Callable[SP, ST],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> Future:
        """Submits a job to the thread pool and returns a Future object."""
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
        self.__submitted_jobs[future] = Job(
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
        job: Job = self.__submitted_jobs.pop(future)

        elapsed_time: float | None = None
        if self.__profile:
            elapsed_time = time.perf_counter() - job.start_time  # type: ignore

        self.__finished_jobs[future] = FinishedJob(  # type: ignore
            *job,
            elapsed_time=elapsed_time
        )

        with self.__queued_jobs, self.__active_threads:
            self.__queued_jobs -= 1
            if self.__queued_jobs.get_obj() < self.__max_workers:
                self.__active_threads -= 1

    def get_job(self, future: Future) -> Job | FinishedJob:
        """Returns the job associated with the given future."""
        if future in self.__finished_jobs:
            return self.__finished_jobs[future]
        return self.__submitted_jobs.pop(future)

    @staticmethod
    def __result_or_cancel(future_: Future, timeout: float | None = None) -> None:
        try:
            try:
                return future_.result(timeout)
            finally:
                future_.cancel()
        finally:
            del future_

    @functools.wraps(ThreadPoolExecutor.map,
                     assigned=("__doc__",))
    def map(
        self,
        name: str,
        func: Callable[SP, ST],
        *iterables: Iterable[SP.args],
        timeout: float | None = None
    ) -> Iterable[Future]:
        if timeout is not None:
            end_time = timeout + time.perf_counter()

        fs = [self.submit(name, func, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield self.__result_or_cancel(fs.pop())
                    else:
                        yield self.__result_or_cancel(fs.pop(), end_time - time.monotonic())
            finally:
                for future in fs:
                    future.cancel()
        return result_iterator()

    @functools.wraps(ThreadPoolExecutor.shutdown, assigned=("__doc__",))
    def shutdown(self, wait: bool = True) -> None:
        self.__thread_pool.shutdown(wait=wait)


class WebScraper:
    """
    https://realpython.com/beautiful-soup-web-scraper-python/
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.__thread_pool = JinxThreadPool(
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
