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
from typing import Any
from concurrency.executors import JinxThreadPool
from concurrency.synchronization import SynchronizedMeta

from datastructures.mappings import TwoWayMap


def trigger_on_field_change(field_name: str) -> Any:
    """Decorate a method to be called when a field changes."""
    pass


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
        if field_name is None:
            field_name = func.__name__
        func.__subject_field__ = field_name
        return func
    return decorator


def field_change(field_name: str | None = None) -> Any:
    """Decorate a method to indicate that it changes a field."""
    def decorator(func: Any) -> Any:
        if field_name is None:
            field_name = func.__name__

        @functools.wraps(func)
        def wrapper(self: "Subject", *args: Any, **kwargs: Any) -> Any:
            old_value = getattr(self, field_name)
            func(self, *args, **kwargs)
            new_value = getattr(self, field_name)
            if old_value != new_value:
                self.__update_listeners(field_name, old_value, new_value)

        return wrapper
    return decorator


class Subject(metaclass=SynchronizedMeta):

    def __init__(self) -> None:
        self.__listeners = TwoWayMap[Listener, str]()
        self.__executor = JinxThreadPool()

    def assign_listener(self, listener: Listener, fields: list[str]) -> None:
        self.__listeners.add_many(listener, fields)

    def remove_listener(self, listener: Listener) -> None:
        self.__listeners.forwards_remove(listener)

    def __update_listeners(self, field_name: str, old_value: Any, new_value: Any) -> None:
        self.__executor.submit(
            self.__update_listeners_async,
            field_name,
            old_value,
            new_value
        )

    def __update_listeners_async(self, field_name: str, old_value: Any, new_value: Any) -> None:
        for listener in self.__listeners[field_name]:
            listener.field_changed(self, field_name, old_value, new_value)
