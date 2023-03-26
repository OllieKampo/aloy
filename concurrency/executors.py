
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import datetime
import functools
import logging
import threading
import time
import types
from typing import Any, Callable, Generic, Iterable, NamedTuple, ParamSpec, TypeVar, final

import urllib.request


SP = ParamSpec("SP")
NT = TypeVar("NT", int, float, complex)


@final
class AtomicNumber(Generic[NT]):
    """
    A thread-safe number whose updates are atomic.
    
    Updates to the number are only allowed within a context manager.
    """

    __slots__ = ("__lock", "__value", "__allow_updates")

    def __init__(self, value: NT = 0) -> None:
        """
        Create a new atomic number with given initial value.

        The number type can be int, float, or complex.
        """
        self.__lock = threading.Lock()
        self.__value: NT = value
        self.__allow_updates: bool = False
    
    def __str__(self) -> str:
        return str(self.__value)
    
    def __repr__(self) -> str:
        return f"AtomicNumber({self.__value})"
    
    def __enter__(self) -> None:
        self.__lock.acquire()
        self.__allow_updates = True
    
    def __exit__(self,
                 exc_type: type | None,
                 exc_val: BaseException | None,
                 exc_tb: types.TracebackType | None
                 ) -> None:
        self.__allow_updates = False
        self.__lock.release()
    
    def __int__(self) -> int:
        return int(self.__value)
    
    def __float__(self) -> float:
        return float(self.__value)
    
    def __complex__(self) -> complex:
        return complex(self.__value)
    
    @property
    def value(self) -> NT:
        return self.__value
    
    @value.setter
    def value(self, value: NT) -> None:
        if not self.__allow_updates:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager")
        self.__value = value
    
    def __iadd__(self, value: NT) -> None:
        if not self.__allow_updates:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager")
        self.__value += value 
    
    def __add__(self, value: NT) -> NT:
        return self.__value + value
    
    def __isub__(self, value: NT) -> None:
        if not self.__allow_updates:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager")
        self.__value -= value
    
    def __sub__(self, value: NT) -> NT:
        return self.__value - value
    
    def __imul__(self, value: NT) -> None:
        if not self.__allow_updates:
            raise RuntimeError("Cannot update AtomicNumber outside of a context manager")
        self.__value *= value
    
    def __mul__(self, value: NT) -> NT:
        return self.__value * value


@dataclass(frozen=True)
class Job:
    """A class that represents a job submitted to a thread pool."""

    name: str | None
    fn: Callable[SP, None]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    start_time: float | None


@dataclass(frozen=True)
class FinishedJob(Job):
    """A class that represents a job that has finished execution."""
    elapsed_time: float


__pools: AtomicNumber[int] = AtomicNumber(0)
def __get_pool_name() -> str:
    """Returns a unique name for a thread pool."""
    global __pools
    with __pools:
        __pools += 1
        return f"JinxThreadPool [{__pools.value}]"

@final
class JinxThreadPool:
    """
    A thread pool that allows perfornance profiling and logging of submitted jobs.

    The thread pool is a wrapper around the ThreadPoolExecutor class.
    """

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
        self.__name: str = pool_name or __get_pool_name()
        self.__logger = logging.getLogger("JinxThreadPool")
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
        return self.__thread_pool
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        self.__thread_pool.shutdown(wait=True)
    
    @property
    def active_workers(self) -> int:
        return self.__active_threads.value
    
    @property
    def max_workers(self) -> int:
        return self.__max_workers
    
    @property
    def is_running(self) -> bool:
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
            self.__logger.debug(f"{self.__name}: Submitting job {name} -> "
                                f"{fn.__name__}(*{args}, **{kwargs})")
        
        future = self.__thread_pool.submit(fn, *args, **kwargs)
        self.__submitted_jobs[future] = Job(
            name, future,
            fn, args, kwargs,
            start_time
        )
        future.add_done_callback(self.__callback)

        return future
    
    def __callback(self, future: Future) -> None:
        job: Job = self.__submitted_jobs.pop(future)

        elapsed_time: float | None = None
        if self.__profile:
            elapsed_time = time.perf_counter() - job.start_time
        
        self.__finished_jobs[job] = FinishedJob(*job, elapsed_time)
        
        with self.__active_threads:
            self.__active_threads -= 1
    
    def get_job(self, future: Future) -> Job:
        """Returns the job associated with the given future."""
        if future in self.__finished_jobs:
            return self.__finished_jobs[future]
        return self.__submitted_jobs[future]

    @functools.wraps(ThreadPoolExecutor.map,
                     assigned=("__doc__",))
    def map(self,
            name: str,
            fn: Callable[SP, None],
            *iterables: Iterable[SP.args],
            timeout: float | None = None,
            chunksize: int = 1
            ) -> Iterable[Future]:
        return self.__thread_pool.map(fn, *iterables,
                                      timeout=timeout,
                                      chunksize=chunksize)
    
    @functools.wraps(ThreadPoolExecutor.shutdown, assigned=("__doc__",))
    def shutdown(self, wait: bool = True) -> None:
        self.__thread_pool.shutdown(wait=wait)

class WebScraper:
    """
    
    https://realpython.com/beautiful-soup-web-scraper-python/
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self.__thread_pool = JinxThreadPool("WebScraper", max_workers)
    
    def scrape(self, urls: Iterable[str]) -> None:
        with self.__thread_pool as executor:
            futures: dict[Future, str] = {executor.submit(self.load_url, url, 60): url for url in urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    print('%r page is %d bytes' % (url, len(data)))
    
    @staticmethod
    def load_url(url: str, timeout: float) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as connection:
            return connection.read()


if __name__ == "__main__":
    urls = [
        "https://www.google.com",
        "https://www.yahoo.com",
        "https://www.bing.com",
        "https://www.ask.com",
        "https://www.duckduckgo.com",
        "https://www.aol.com",
        "https://www.wolframalpha.com"
    ]
    scraper = WebScraper(max_workers=4)
    scraper.scrape(urls)
