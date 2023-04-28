
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, field
import functools
import logging
import threading
import time
import types
from typing import Any, Callable, Iterable, ParamSpec, final

import urllib.request

from concurrency.atomic import AtomicNumber


SP = ParamSpec("SP")


@dataclass(frozen=True)
class Job:
    """A class that represents a job submitted to a thread pool."""

    name: str | None
    fn: Callable[SP, None]
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(hash=False)
    start_time: float | None

    def __iter__(self) -> Iterable:
        return iter(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class FinishedJob(Job):
    """A class that represents a job that has finished execution."""
    elapsed_time: float


@final
class JinxThreadPool:
    """
    A thread pool that allows perfornance profiling and logging of submitted jobs.

    The thread pool is a wrapper around the ThreadPoolExecutor class.
    """

    __POOLS: AtomicNumber[int] = AtomicNumber(0)

    @classmethod
    def __get_pool_name(cls) -> str:
        """Returns a unique name for a thread pool."""
        with cls.__POOLS:
            cls.__POOLS += 1
            return f"{cls.__name__} [{cls.__POOLS.value}]"

    def __init__(
        self,
        pool_name: str | None = None,
        max_workers: int | None = None,
        thread_name_prefix: str | None = None,
        profile: bool = False,
        log: bool = False,
        initializer: Callable[SP, None] | None = None,
        initargs: tuple | None = None
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

        self.__max_workers: int = max_workers
        self.__submitted_jobs: dict[Future, Job] = {}
        self.__finished_jobs: dict[Future, FinishedJob] = {}
        self.__active_threads: AtomicNumber[int] = AtomicNumber(0)

        self.__profile: bool = profile

        self.__main_thread: threading.Thread = threading.current_thread()
        self.__thread_pool = ThreadPoolExecutor(
            max_workers, thread_name_prefix,
            initializer, initargs
        )

    def __enter__(self) -> ThreadPoolExecutor:
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
        return self.__active_threads.value

    @property
    def max_workers(self) -> int:
        """Return the maximum number of workers allowed in the thread pool."""
        return self.__max_workers

    @property
    def is_running(self) -> bool:
        """Return True if the thread pool is running, False otherwise."""
        return self.__active_threads.value > 0

    @property
    def is_busy(self) -> bool:
        return self.__active_threads.value == self.__max_workers

    @functools.wraps(ThreadPoolExecutor.submit,
                     assigned=("__doc__",))
    def submit(
        self,
        name: str,
        fn: Callable[SP, None],
        *args: SP.args,
        **kwargs: SP.kwargs
    ) -> Future:
        """Submits a job to the thread pool and returns a Future object."""
        with self.__active_threads:
            self.__active_threads += 1

        start_time: float | None = None
        if self.__profile:
            start_time = time.perf_counter()

        if self.__log:
            self.__logger.debug(
                "%s: Submitting job %s -> %s(*%s, **%s)",
                self.__name, name, fn.__name__, args, kwargs
            )

        future = self.__thread_pool.submit(fn, *args, **kwargs)
        self.__submitted_jobs[future] = Job(
            name, fn,
            args, kwargs,
            start_time
        )
        future.add_done_callback(self.__callback)

        return future

    def __callback(self, future: Future) -> None:
        """Callback function for when a job finishes execution."""
        job: Job = self.__submitted_jobs.pop(future)

        elapsed_time: float | None = None
        if self.__profile:
            elapsed_time = time.perf_counter() - job.start_time

        self.__finished_jobs[future] = FinishedJob(*job, elapsed_time)

        with self.__active_threads:
            self.__active_threads -= 1

    def get_job(self, future: Future) -> Job:
        """Returns the job associated with the given future."""
        if future in self.__finished_jobs:
            return self.__finished_jobs[future]
        return self.__submitted_jobs[future]

    @functools.wraps(ThreadPoolExecutor.map,
                     assigned=("__doc__",))
    def map(
        self,
        name: str,
        fn: Callable[SP, None],
        *iterables: Iterable[SP.args],
        timeout: float | None = None,
        chunksize: int = 1
    ) -> Iterable[Future]:
        return self.__thread_pool.map(
            fn,
            *iterables,
            timeout=timeout,
            chunksize=chunksize
        )

    @functools.wraps(ThreadPoolExecutor.shutdown, assigned=("__doc__",))
    def shutdown(self, wait: bool = True) -> None:
        self.__thread_pool.shutdown(wait=wait)


class WebScraper:
    """
    https://realpython.com/beautiful-soup-web-scraper-python/
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.__thread_pool = JinxThreadPool("WebScraper", max_workers, profile=True, log=True)

    def scrape(self, urls: Iterable[str]) -> None:
        with self.__thread_pool as executor:
            futures: dict[Future, str] = {executor.submit(url, self.load_url, url, 60): url for url in urls}
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


if __name__ == "__main__":
    ## Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-4s :: %(name)-8s >> %(message)s\n",
        datefmt="%H:%M:%S",
        handlers=[
            # logging.FileHandler("debug.log"),
            # console_handler
        ],
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-4s :: %(name)-8s >> %(message)s\n"))
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
