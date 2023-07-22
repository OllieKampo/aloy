# Copyright (C) 2023 Oliver Michael Kamperis
# Email: o.m.kamperis@gmail.com
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


import functools
import inspect
import logging
from typing import Any, Callable, Final, final
from concurrency.executors import JinxThreadPool
from concurrency.synchronization import SynchronizedMeta, sync

from datastructures.mappings import TwoWayMap


def call_on_field_change(field_name: str) -> Any:
    """Decorate a method to be called when a field changes."""
    def decorator(func: Any) -> Any:
        func.__subject_field__ = field_name
        return func
    return decorator


class Listener:
    def field_changed(
        self,
        source: "Subject",
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        pass


def field(field_name: str | None = None) -> Any:
    """Decorate a field to be tracked by a Subject."""
    def decorator(func: Any) -> Any:
        _field_name: str
        if field_name is None:
            _field_name = func.__name__
        else:
            _field_name = field_name
        func.__subject_field__ = _field_name
        return func
    return decorator


def field_change(field_name: str | None = None) -> Any:
    """Decorate a method to indicate that it changes a field."""
    def decorator(func: Any) -> Any:
        _field_name: str
        if field_name is None:
            _field_name = func.__name__
        else:
            _field_name = field_name

        @functools.wraps(func)
        def wrapper(self: "Subject", *args: Any, **kwargs: Any) -> Any:
            old_value = getattr(self, _field_name)
            func(self, *args, **kwargs)
            new_value = getattr(self, _field_name)
            if old_value != new_value:
                # pylint: disable=protected-access
                self.__update_listeners(_field_name, old_value, new_value)

        return wrapper
    return decorator


@final
class _SubjectSynchronizedMeta(SynchronizedMeta):
    """
    Synchronized metaclass for Subject.

    Gathers all fields and callbacks defined in the class namespace into the
    `__subject_fields__` attribute. This is a dictionary mapping field names to
    getter functions. Properties have their `fget` attribute pre-extracted.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any]
    ) -> type:
        """Create a new synchronized subject class."""
        _existing_subject_fields = namespace["__subject_fields__"]
        _new_subject_fields = {}
        for attr in namespace.values():
            if hasattr(attr, "__subject_field__"):
                _field_name = attr.__subject_field__
                if isinstance(attr, property):
                    _get_attr = attr.fget
                else:
                    _get_attr = attr
                if _field_name in _existing_subject_fields:
                    continue
                if _field_name in _new_subject_fields:
                    raise ValueError(
                        f"Field name {_field_name} defined more than once in "
                        f"class {name}."
                    )
                _new_subject_fields[_field_name] = _get_attr
        _existing_subject_fields.update(_new_subject_fields)

        return super().__new__(cls, name, bases, namespace)


_SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_MAX_WORKERS: Final[int] = 10
_SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_PROFILE: Final[bool] = False
_SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_LOG: Final[bool] = False


class Subject(metaclass=_SubjectSynchronizedMeta):
    """
    A subject is an object that can be observed by listeners.
    """

    __SUBJECT_LOGGER = logging.getLogger("Subject")

    __slots__ = (
        "__subject_fields__",
        "__listeners",
        "__callbacks",
        "__executor"
    )

    def __init__(
        self,
        max_workers: int = _SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_MAX_WORKERS,
        profile: bool = _SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_PROFILE,
        log: bool = _SUBJECT_DEFAULT_THREAD_POOL_EXECUTOR_LOG
    ) -> None:
        self.__listeners = TwoWayMap[Listener, str]()
        self.__callbacks = TwoWayMap[Callable[..., None], str]()
        self.__executor = JinxThreadPool(
            pool_name="PubSubHub :: Thread Pool Executor",
            max_workers=max(max_workers, 1),
            thread_name_prefix="PubSubHub :: Thread Pool Executor :: Thread",
            profile=bool(profile),
            log=bool(log)
        )

    def __get_field(self, field_name: str) -> Any:
        get_attr = self.__subject_fields__[field_name]  # type: ignore
        return get_attr(self)

    def __check_callback(self, callback: Callable[..., None]) -> None:
        """Check that the callback is a function that takes four arguments."""
        if not inspect.ismethod(callback) and not inspect.isfunction(callback):
            raise TypeError(
                f"Callback must be a method or function. Got; {callback}.")
        if len(inspect.signature(callback).parameters) != 4:
            raise TypeError(
                f"Callback must take four arguments. Got: {callback}.")

    @sync(lock="all")
    def register(
        self,
        listener_or_callback: Listener | Callable[["Subject", str, Any, Any], None],
        *fields: str
    ) -> None:
        if isinstance(listener_or_callback, Listener):
            listener = listener_or_callback
            if hasattr(listener, "field_changed"):
                self.__listeners.add_many(listener, fields)
            for attr_name in dir(listener):
                attr = getattr(listener, attr_name)
                if hasattr(attr, "__subject_field__"):
                    callback = attr
                    try:
                        self.__check_callback(callback)
                    except TypeError as err:
                        raise TypeError(
                            f"Listener {listener} has an invalid callback."
                        ) from err
                    self.__callbacks.add_many(callback, fields)
        else:
            callback = listener_or_callback
            self.__check_callback(callback)
            self.__callbacks.add_many(callback, fields)

    def update(self, *field_names: str) -> None:
        for field_name in field_names:
            current_value = self.__get_field(field_name)
            self.__update(field_name, None, current_value)

    def __update(self, field_name: str, old_value: Any, new_value: Any) -> None:
        self.__executor.submit(
            self.__update_async,
            field_name,
            old_value,
            new_value
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

    def __update_async(self, field_name: str, old_value: Any, new_value: Any) -> None:
        for listener in self.__listeners[field_name]:
            try:
                listener.field_changed(self, field_name, old_value, new_value)
            except Exception as err:  # pylint: disable=broad-except
                self.__log_exception(
                    listener, field_name, old_value, new_value, err)
        for callback in self.__callbacks[field_name]:
            try:
                callback(self, field_name, old_value, new_value)
            except Exception as err:  # pylint: disable=broad-except
                self.__log_exception(
                    callback, field_name, old_value, new_value, err)
