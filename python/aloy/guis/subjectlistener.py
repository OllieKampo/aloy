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

A subject declares a set of fields that can be listened to. A field is a
method or property of the subject whose return value is the value of the
field. The subject can then also declare methods that change the value of
fields. A listener is an object that can be registered with a subject. A
listener declares what fields it is interested in listening to. When such a
field changes, the listener is notified automatically.

There are three ways for a listener to listen to a subject:
- Register a listener object (a sub-class of `Listener`) with the
  `field_changed(source, field_name, old_value, new_value)` method defined,
- Register a (set of) callback(s) to the subject, or
- Decorate method(s) of a class with `@call_on_field_change()` and egister
  instances of that class with the subject.

The subject-listener pattern is a middle ground between the observable-observer
and publisher-subscriber patterns. It is similar to the observable-observer in
that listeners are registered with a subject and are notified when the subject
changes. It is similar to the publisher-subscriber pattern in that listeners
can subscribe to specific fields of the subject. Unlike the observable-observer
pattern, listeners are notified only when a field they are subscribed to
changes, and are only sent the relevant information, greatly increasing the
sacalability of the pattern. Unlike the publisher-subscriber pattern, instead
of declaring named topics or parameters on a hub, the intended method is to
sub-class the subject class to create a new subject type containing the fields
that listeners can subscribe to.
"""

import functools
import inspect
import logging
import queue
import time
from collections import defaultdict, deque
from typing import (Any, Callable, Concatenate, Final, NamedTuple, ParamSpec,
                    Type, TypeVar, final)

from PySide6.QtCore import QTimer  # pylint: disable=no-name-in-module

from aloy.concurrency.atomic import AtomicBool
from aloy.concurrency.clocks import SimpleClockThread
from aloy.concurrency.executors import AloyQTimerExecutor, AloyThreadPool
from aloy.concurrency.synchronization.sync_class import (SynchronizedMeta,
                                                         get_instance_lock,
                                                         sync)
from aloy.datastructures.mappings import TwoWayMap
from aloy.guis._utils import create_clock

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
        func.__subject_field__ = field_name  # type: ignore[attr-defined]
        return func
    return decorator


class Listener:
    """
    Class defining listeners.

    A listener is an object that can be registered with a subject. When a
    field of the subject changes, the listener is notified by calling its
    `field_changed(source, field_name, old_value, new_value)` method.
    """

    # pylint: disable=unused-argument
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
        return NotImplemented  # type: ignore[return-value]
    # pylint: enable=unused-argument


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
        bases: tuple[Type, ...],
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

    A subject declares a set of fields that can be listened to. A field is a
    method or property of the subject whose return value is the value of the
    field. To declare a field, decorate a method or property with the
    `@field(name: str | None = None, queue_size: int | None = None)`
    decorator. The decorated method must have be callable with no arguments
    and return the current value of the field. If the `name` argument is given
    and not None, it is used as the name of the field. Otherwise, the name of
    the method is used. If the `queue_size` argument is given and not None, a
    queue of that size is used to store previous values of the field.
    Otherwise, the field is not queued and only the current value is available.

    The subject can then also declare methods that change the value of fields.
    To declare a method that changes a field, decorate it with the
    `@field_change(name: str | None = None)` decorator. There are no
    restrictions on the method signature. If the `name` argument is given and
    not None, it is used as the name of the field. Otherwise, the name of the
    method is used. A field with the same name must have been defined prior to
    this decorator being applied. When the method is called, the field is
    updated and all listeners and callbacks of the field are notified.
    """

    __SUBJECT_LOGGER = logging.getLogger("Subject")
    __SUBJECT_FIELDS__: dict[str, _SubjectField] = _NO_FIELDS

    __slots__ = {
        "__weakref__": "Weak references to the object.",
        "__name": "The name of the subject.",
        "__listeners": "The listeners registered with the subject.",
        "__callbacks": "The callbacks registered with the subject.",
        "__queues": "The queues used to store values of queued fields.",
        "__updated_fields": "The fields that have been updated.",
        "__updating_fields": "The fields that are currently being updated.",
        "__update_queue": "The queue used to update listeners.",
        "__clock": "The clock used to update all listeners.",
        "__executor": "The thread pool executor used to update listeners.",
        "__debug": "Whether to log debug messages."
    }

    def __init__(
        self,
        name: str | None = None,
        clock: SimpleClockThread | QTimer | None = None,
        tick_rate: int = 10,
        start_clock: bool = True,
        max_workers: int = 10,
        debug: bool = False
    ) -> None:
        """
        Create a new subject.

        Parameters
        ----------
        `max_workers: int = 10` - The maximum number of threads to use to
        update listeners.

        `profile: bool = False` - Whether to profile the threads used to update
        listeners.

        `log: bool = False` - Whether to log the calls on threads used to
        update listeners.
        """
        self.__debug: bool = debug
        if self.__debug:
            self.__SUBJECT_LOGGER.debug(
                "Creating new subject with: "
                "name=%s, clock=%s, tick_rate=%s, start_clock=%s, "
                "max_workers=%s, debug=%s",
                name, clock, tick_rate, start_clock, max_workers, debug
            )

        # Subjects are optionally named.
        self.__name: str | None = name

        # Internal data structures.
        self.__listeners = TwoWayMap[Listener, str]()
        self.__callbacks = TwoWayMap[Callable[..., None], str]()
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

        self.__clock: SimpleClockThread | QTimer = create_clock(
            self.__schedule_updates,
            name=self.__name if self.__name is not None else "New Subject",
            clock=clock,
            tick_rate=tick_rate,
            start_clock=start_clock,
            logger=self.__SUBJECT_LOGGER,
            debug=self.__debug
        )
        self.__executor: AloyThreadPool | AloyQTimerExecutor
        if isinstance(self.__clock, SimpleClockThread):
            self.__executor = AloyThreadPool(
                pool_name=f"Subject {name} :: Thread Pool Executor",
                max_workers=max(max_workers, 1),
                thread_name_prefix=f"Subject {name} :: Thread",
                log=bool(debug)
            )
        elif isinstance(self.__clock, QTimer):
            self.__executor = AloyQTimerExecutor(
                interval=1.0 / tick_rate
            )

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
            is_updated = self.__updated_fields[field_name]
            is_updating = self.__updating_fields[field_name]
            with is_updated, is_updating:
                is_updated.set_obj(True)
                self.__update_queue[field_name].put(
                    (old_value, new_value)
                )

    def __schedule_updates(self) -> None:
        """Schedule an update of all listeners and callbacks of a field."""
        for field_name, is_updated in self.__updated_fields.items():
            is_updating = self.__updating_fields[field_name]
            with is_updated, is_updating:
                if is_updated:
                    if not is_updating:
                        is_updating.set_obj(True)
                        if isinstance(self.__executor, AloyQTimerExecutor):
                            self.__executor.submit(
                                self.__update_all_async,
                                field_name
                            )
                        else:
                            self.__executor.submit(
                                f"Subject Update ({field_name})",
                                self.__update_all_async,
                                field_name
                            )
                    is_updated.set_obj(False)

    def __update_all_async(self, field_name: str) -> None:
        """Asynchronously update all listeners and callbacks of a field."""
        queue_: queue.SimpleQueue[tuple[Any, Any]] = \
            self.__update_queue[field_name]
        while True:
            try:
                old_value, new_value = queue_.get_nowait()
                self.__update_async(field_name, old_value, new_value)
            except queue.Empty:
                is_updated = self.__updated_fields[field_name]
                is_updating = self.__updating_fields[field_name]
                with is_updated, is_updating:
                    is_updating.set_obj(False)
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
                        with get_instance_lock(self):  # type: ignore
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
    import sys  # pylint: disable=import-outside-toplevel

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout
    )

    class MySubject(Subject):
        """Test subject."""
        def __init__(self) -> None:
            super().__init__(debug=True)
            self.__my_field = 0
            self.__my_queued_field = 0

        @property
        @field()
        def my_field(self) -> int:
            """Get the value of my_field."""
            return self.__my_field

        @my_field.setter
        @field_change()
        def my_field(self, value: int) -> None:
            """Set the value of my_field."""
            self.__my_field = value

        @property
        @field(queue_size=3)
        def my_queued_field(self) -> int:
            """Get the value of my_field."""
            return self.__my_queued_field

        @my_queued_field.setter
        @field_change()
        def my_queued_field(self, value: int) -> None:
            """Set the value of my_field."""
            self.__my_queued_field = value

    class MyListener(Listener):
        """Test listener."""
        @call_on_field_change("my_field")
        def my_field_changed(
            self,
            source: Subject,  # pylint: disable=unused-argument
            field_name: str,
            old_value: int,
            new_value: int
        ) -> None:
            """Called when my_field changes."""
            print(f"Listener {self} got notified that field {field_name} "
                  f"changed from {old_value} to {new_value}.")

        @call_on_field_change("my_queued_field")
        def my_queued_field_changed(
            self,
            source: Subject,  # pylint: disable=unused-argument
            field_name: str,
            old_value: int,
            new_value: int
        ) -> None:
            """Called when my_queued_field changes."""
            print(f"Listener {self} got notified that field {field_name} "
                  f"changed from {old_value} to {new_value}.")

        def field_changed(
            self,
            source: Subject,  # pylint: disable=unused-argument
            field_name: str,
            old_value: Any,
            new_value: Any
        ) -> None:
            """Called when a field changes."""
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
    subject.my_queued_field = 1
    subject.my_queued_field = 1
    subject.my_queued_field = 2
    subject.my_queued_field = 2
    subject.my_queued_field = 3
    subject.my_queued_field = 3
    subject.my_queued_field = 4
    subject.my_queued_field = 4
    subject.my_queued_field = 5
    subject.my_queued_field = 5
    time.sleep(1)


if __name__ == "__main__":
    __main()
