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

"""
Module defining the subject-listener design pattern.

The subject-listener pattern is a middle ground between the observable-observer
and publisher-subscriber patterns. It is similar to the observable-observer in
that listeners are registered with a subject and are notified when the subject
changes. It is similar to the publisher-subscriber pattern in that listeners
can subscribe to specific fields of the subject. Unlike the observable-observer
pattern, listeners are notified only when a field they are subscribed to
changes, and are only sent the relevant information, greatly increasing the
sacalability of the pattern. Unlike the publisher-subscriber pattern, a
subject is not a global singleton, and the intended method is to sub-class
the subject class to create a new subject type containing the fields that
listeners can subscribe to.

Because the subject-listener pattern only updates listeners when a field
they are listening to changes, there is less generality to the pattern than
the observerable-observer pattern.
"""

import functools
import inspect
import logging
from collections import defaultdict, deque
import queue
import time
from typing import (Any, Callable, Concatenate, Final, NamedTuple, ParamSpec,
                    TypeVar, final)
from aloy.concurrency.atomic import AtomicBool
from aloy.concurrency.clocks import ClockThread

from aloy.concurrency.executors import AloyThreadPool
from aloy.concurrency.synchronization import SynchronizedMeta, sync
from aloy.datastructures.mappings import TwoWayMap

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.1.1"

__all__ = (
    "Listener",
    "Subject",
    "call_on_field_change",
    "field",
    "field_change"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


def call_on_field_change(
    field_name: str
) -> Callable[
    [Callable[["Listener", "Subject", str, Any, Any], None]],
    Callable[["Listener", "Subject", str, Any, Any], None]
]:
    """Decorate a method to be called when a field changes."""
    def decorator(
        func: Callable[["Listener", "Subject", str, Any, Any], None]
    ) -> Callable[["Listener", "Subject", str, Any, Any], None]:
        func.__subject_field__ = field_name  # type: ignore
        return func
    return decorator


class Listener:
    """
    Class defining listeners.

    A listener is an object that can be registered with a subject. When a
    field of the subject changes, the listener is notified by calling its
    `field_changed(source, field_name, old_value, new_value)` method.
    """

    def field_changed(
        self,
        source: "Subject",
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        """
        Called when a field of the subject changes.

        Parameters
        ----------
        `source: Subject` - The subject that changed.

        `field_name: str` - The name of the field that changed.

        `old_value: Any` - The old value of the field.

        `new_value: Any` - The new value of the field.
        """
        return NotImplemented  # type: ignore


SP = ParamSpec("SP")
ST = TypeVar("ST")


def field(
    field_name: str | None = None,
    queue_size: int | None = None
) -> Callable[
    [Callable[Concatenate["Subject", SP], ST]],
    Callable[Concatenate["Subject", SP], ST]
]:
    """
    Decorate a field to be tracked by a Subject.

    The decorated method must have be callable with no arguments and return
    the current value of the field.

    Parameters
    ----------
    `field_name: str | None = None` - The name of the field. If not given or
    None, the name of the method is used.

    `queue_size: int | None = None` - The size of the queue used to store
    values of the field. If not given or None, the field is not queued.
    """
    def decorator(
        func: Callable[Concatenate["Subject", SP], ST]
    ) -> Callable[Concatenate["Subject", SP], ST]:
        _field_name: str
        if field_name is None:
            _field_name = func.__name__
        else:
            _field_name = field_name
        func.__subject_field__ = _field_name  # type: ignore
        func.__queue_size__ = queue_size  # type: ignore
        sync_dec = sync(lock="method", group_name=f"get:({_field_name})")
        return sync_dec(func)  # type: ignore
    return decorator


def field_change(
    field_name: str | None = None
) -> Callable[
    [Callable[Concatenate["Subject", SP], ST]],
    Callable[Concatenate["Subject", SP], ST]
]:
    """
    Decorate a method to indicate that it changes a field.

    There are no restrictions on the method signature.

    A field with the same name must have been defined prior to this decorator
    being applied.

    Parameters
    ----------
    `field_name: str | None = None` - The name of the field. If not given or
    None, the name of the method is used.
    """
    def decorator(
        func: Callable[Concatenate["Subject", SP], ST]
    ) -> Callable[Concatenate["Subject", SP], ST]:
        _field_name: str
        if field_name is None:
            _field_name = func.__name__
        else:
            _field_name = field_name

        func.__subject_field_change__ = _field_name  # type: ignore

        # pylint: disable=protected-access
        @sync(lock="method", group_name=f"update:({_field_name})")
        @functools.wraps(func)
        def wrapper(self: "Subject", *args: Any, **kwargs: Any) -> Any:
            old_value = self.__get_field__(_field_name)
            func(self, *args, **kwargs)
            new_value = self.__get_field__(_field_name)
            self.__update__(_field_name, old_value, new_value)

        return wrapper
    return decorator


class _SubjectField(NamedTuple):
    """
    A field of a subject.

    Items:
    ------
    `get_attr: Callable[[Subject], Any]` - The getter function of the field.

    `queue_size: int | None` - The size of the queue used to store values of
    the field. If None, the field is not queued.
    """
    get_attr: Callable[["Subject"], Any]
    queue_size: int | None


_NO_FIELDS: Final[dict[str, _SubjectField]] = {}


@final
class _SubjectSynchronizedMeta(SynchronizedMeta):
    """
    Synchronized metaclass for Subject.

    Gathers all fields and callbacks defined in the class namespace into the
    `__subject_fields__` attribute. This is a dictionary mapping field names to
    getter functions. Properties have their `fget` attribute pre-extracted.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[str, ...],
        namespace: dict[str, Any]
    ) -> type:
        """Create a new synchronized subject class."""
        _existing_subject_fields: dict[str, _SubjectField] = \
            namespace.get("__SUBJECT_FIELDS__")  # type: ignore

        if (_existing_subject_fields is _NO_FIELDS
                or _existing_subject_fields is None):
            _existing_subject_fields = {}
            namespace["__SUBJECT_FIELDS__"] = _existing_subject_fields

        _new_subject_fields = {}
        for attr in namespace.values():

            if isinstance(attr, property):
                _get_attr = attr.fget
                _set_attr = attr.fset
            else:
                _get_attr = attr
                _set_attr = attr

            if (_get_attr is not None
                    and hasattr(_get_attr, "__subject_field__")):
                _field_name = _get_attr.__subject_field__  # type: ignore
                _queue_size = _get_attr.__queue_size__  # type: ignore
                if _field_name in _existing_subject_fields:
                    raise ValueError(
                        f"Attribute {_get_attr.__name__} defined a field "
                        f"name '{_field_name}' which was already defined in a "
                        f"base class of {name}."
                    )
                if _field_name in _new_subject_fields:
                    raise ValueError(
                        f"Attribute {_get_attr.__name__} defined a field "
                        f"name '{_field_name}' which was already defined in "
                        f"the class {name}."
                    )

                subject_field = _SubjectField(_get_attr, _queue_size)
                _new_subject_fields[_field_name] = subject_field

            if (_set_attr is not None
                    and hasattr(_set_attr, "__subject_field_change__")):
                _field_name = _set_attr \
                    .__subject_field_change__  # type: ignore
                if (_field_name not in _existing_subject_fields
                        and _field_name not in _new_subject_fields):
                    raise ValueError(
                        f"Attribute {_set_attr.__name__} defined as changing "
                        f"field '{_field_name}', but no field with that name "
                        f"was defined in (or a base class of) {name}."
                    )

        _existing_subject_fields.update(_new_subject_fields)

        return super().__new__(mcs, name, bases, namespace)


class Subject(metaclass=_SubjectSynchronizedMeta):
    """
    A subject is an object that can be observed by listeners.
    """

    __SUBJECT_LOGGER = logging.getLogger("Subject")
    __SUBJECT_FIELDS__: dict[str, _SubjectField] = _NO_FIELDS

    __slots__ = {
        "__listeners": "The listeners registered with the subject.",
        "__callbacks": "The callbacks registered with the subject.",
        "__clock": "The clock used to update all listeners.",
        "__executor": "The thread pool executor used to update listeners.",
        "__queues": "The queues used to store values of queued fields.",
        "__updated_fields": "The fields that have been updated.",
        "__updating_fields": "The fields that are currently being updated.",
        "__update_queue": "The queue used to update listeners."
    }

    def __init__(
        self,
        max_workers: int = 10,
        profile: bool = False,
        log: bool = False
    ) -> None:
        """
        Create a new subject.

        There are three ways for a listener to listen to a subject:
        - Register a listener object (a sub-class of `Listener`) with the
          `field_changed(...)` method defined,
        - Register a (set of) callback(s) to the subject, or
        - Decorate method(s) of a class with `@call_on_field_change()` and
          register instances of that class with the subject.
        """
        self.__listeners = TwoWayMap[Listener, str]()
        self.__callbacks = TwoWayMap[Callable[..., None], str]()

        self.__clock = ClockThread(
            self.__schedule_updates,
            tick_rate=20
        )
        self.__executor = AloyThreadPool(
            pool_name="Subject :: Thread Pool Executor",
            max_workers=max(max_workers, 1),
            thread_name_prefix="Subject :: Thread Pool Executor :: Thread",
            profile=bool(profile),
            log=bool(log)
        )

        self.__queues: dict[str, deque[Any]] = defaultdict(deque)
        for field_name, (_, queue_size) \
                in self.__SUBJECT_FIELDS__.items():
            if queue_size is not None:
                self.__queues[field_name] = deque(maxlen=queue_size)

        self.__updated_fields: dict[str, AtomicBool] = defaultdict(
            AtomicBool)
        self.__updating_fields: dict[str, AtomicBool] = defaultdict(
            AtomicBool)
        self.__update_queue: dict[str, queue.SimpleQueue[tuple[Any, Any]]] = \
            defaultdict(queue.SimpleQueue)

        self.__clock.start()

    def __get_field__(self, field_name: str) -> Any:
        subject_field = self.__SUBJECT_FIELDS__.get(field_name)
        if subject_field is None:
            raise AttributeError(f"Subject {self} has no field {field_name}.")
        return subject_field.get_attr(self)

    def __check_callback(self, callback: Callable[..., None]) -> None:
        """Check that the callback is a function that takes four arguments."""
        if not inspect.ismethod(callback) and not inspect.isfunction(callback):
            raise TypeError(
                "Callback must be a method or function. "
                f"Got; {callback} of type {type(callback)}."
            )
        if len(inspect.signature(callback).parameters) != 4:
            raise TypeError(
                "Callback must take four arguments. "
                f"Got: {callback} of type {type(callback)} "
                f"with signature {inspect.signature(callback)}."
            )

    @sync(lock="all")
    def register(
        self,
        listener_or_callback: (
            Listener
            | Callable[["Subject", str, Any, Any], None]
        ),
        *fields: str
    ) -> None:
        """
        Register a listener or callback with the subject. The listener or
        callback will be called when any of the given fields change.
        """
        if isinstance(listener_or_callback, Listener):
            listener = listener_or_callback
            if hasattr(listener, "field_changed"):
                self.__listeners.add_many(listener, fields)
            for attr_name in dir(listener):
                attr = getattr(listener, attr_name)
                if hasattr(attr, "__subject_field__"):
                    callback = attr
                    _field_name = callback.__subject_field__
                    try:
                        self.__check_callback(callback)
                    except TypeError as err:
                        raise TypeError(
                            f"Listener {listener} has an invalid callback."
                        ) from err
                    self.__callbacks.add(callback, _field_name)
        else:
            callback = listener_or_callback
            self.__check_callback(callback)
            self.__callbacks.add_many(callback, fields)

    @sync(lock="method", group_name="__update__")
    def update(self, *field_names: str) -> None:
        """Update all listeners and callbacks of the given fields."""
        for field_name in field_names:
            current_value = self.__get_field__(field_name)
            self.__update__(field_name, current_value, None)

    def __update__(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        subject_field: _SubjectField = \
            self.__SUBJECT_FIELDS__.get(field_name)  # type: ignore
        if old_value != new_value:
            if subject_field.queue_size is not None and new_value is not None:
                old_value = list(self.__queues[field_name])
                self.__queues[field_name].append(new_value)
            with (atomic_bool := self.__updated_fields[field_name]):
                atomic_bool.set_obj(True)
                self.__update_queue[field_name].put(
                    (old_value, new_value)
                )

    def __schedule_updates(self) -> None:
        """Schedule an update of all listeners and callbacks of a field."""
        for field_name, atomic_bool in self.__updated_fields.items():
            with atomic_bool:
                if atomic_bool:
                    with (is_updating := self.__updating_fields[field_name]):
                        if not is_updating:
                            is_updating.set_obj(True)
                            self.__executor.submit(
                                f"Subject Update ({field_name})",
                                self.__update_all_async,
                                field_name
                            )
                    atomic_bool.set_obj(False)

    def __update_all_async(self, field_name: str) -> None:
        """Asynchronously update all listeners and callbacks of a field."""
        queue_: queue.SimpleQueue[tuple[Any, Any]] = \
            self.__update_queue[field_name]
        while True:
            try:
                old_value, new_value = queue_.get_nowait()
                self.__update_async(field_name, old_value, new_value)
            except queue.Empty:
                break

    def __update_async(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        """
        Asynchronously update all listeners and callbacks of a field.

        If a listener's `field_changed()` method returns `NotImplemented`, the
        listener is unregistered from the subject. If a listener or callback
        raises an exception, the exception is logged.
        """
        listeners = self.__listeners.backwards_get(field_name)
        if listeners is not None:
            for listener in listeners:
                try:
                    value = listener.field_changed(  # type: ignore
                        source=self,
                        field_name=field_name,
                        old_value=old_value,
                        new_value=new_value
                    )
                    if value is NotImplemented:
                        with self.instance_lock:  # type: ignore
                            self.__listeners.remove(listener, field_name)
                except Exception as err:  # pylint: disable=broad-except
                    self.__log_exception(
                        listener_or_callback=listener,
                        field_name=field_name,
                        old_value=old_value,
                        new_value=new_value,
                        exc_info=err
                    )

        callbacks = self.__callbacks.backwards_get(field_name)
        if callbacks is not None:
            for callback in callbacks:
                try:
                    callback(self, field_name, old_value, new_value)
                except Exception as err:  # pylint: disable=broad-except
                    self.__log_exception(
                        listener_or_callback=callback,
                        field_name=field_name,
                        old_value=old_value,
                        new_value=new_value,
                        exc_info=err
                    )

    def __log_exception(
        self,
        listener_or_callback: Listener | Callable[..., None],
        field_name: str,
        old_value: Any,
        new_value: Any,
        exc_info: Exception | None
    ) -> None:
        name: str
        if isinstance(listener_or_callback, Listener):
            name = "Listener"
        else:
            name = "Callback"
        self.__SUBJECT_LOGGER.exception(
            "Subject %s: %s %s raised an exception, when "
            "updating field %s from %s to %s.",
            self, name, listener_or_callback, field_name, old_value, new_value,
            exc_info=exc_info
        )


def __main():
    """Entry point of the module."""
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s %(levelname)s %(name)s %(message)s",
    #     stream=sys.stdout
    # )

    class MySubject(Subject):
        def __init__(self) -> None:
            super().__init__(log=True)
            self.__my_field = 0

        @property
        @field(queue_size=3)
        def my_field(self) -> int:
            return self.__my_field

        @my_field.setter
        @field_change()
        def my_field(self, value: int) -> None:
            self.__my_field = value

    class MyListener(Listener):
        @call_on_field_change("my_field")
        def my_field_changed(
            self,
            source: Subject,
            field_name: str,
            old_value: int,
            new_value: int
        ) -> None:
            print(f"Listener {self} got notified that field {field_name} "
                  f"changed from {old_value} to {new_value}.")

    subject = MySubject()
    listener = MyListener()
    subject.register(listener)
    subject.my_field = 1
    subject.my_field = 1
    subject.my_field = 2
    subject.my_field = 2
    subject.my_field = 3
    subject.my_field = 3
    subject.my_field = 4
    subject.my_field = 4
    subject.my_field = 5
    subject.my_field = 5
    time.sleep(1)


if __name__ == "__main__":
    __main()
