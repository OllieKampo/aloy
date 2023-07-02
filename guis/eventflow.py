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

from datastructures.mappings import TwoWayMap


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
    def decorator(func: Any) -> Any:
        if field_name is None:
            field_name = func.__name__
        func.__subject_field__ = field_name
        @functools.wraps(func)
        def wrapper(self: "Subject", *args: Any, **kwargs: Any) -> Any:
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def fire_field_changed(field_name: str | None = None) -> Any:
    def decorator(func: Any) -> Any:
        if field_name is None:
            field_name = func.__name__

        @functools.wraps(func)
        def wrapper(self: "Subject", *args: Any, **kwargs: Any) -> Any:
            old_value = getattr(self, field_name)
            func(self, *args, **kwargs)
            new_value = getattr(self, field_name)
            if old_value != new_value:
                for listener, fields in self.__listeners:
                    if field_name in fields:
                        listener.field_changed(self, field_name, new_value)

        return wrapper
    return decorator


class Subject:

    def __init__(self) -> None:
        self.__listeners = TwoWayMap[Listener, str]()

    @property
    @field()
    def value(self) -> int:
        return self.__value

    @value.setter
    @fire_field_changed()
    def value(self, value: int) -> None:
        self.__value = value

    def assign_listener(self, listener: Listener, fields: list[str]) -> None:
        self.__listeners.extend(
            (listener, set(fields))
        )
