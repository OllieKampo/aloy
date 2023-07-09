###########################################################################
###########################################################################
## Module defining publisher-subscriber interface pattern.               ##
##                                                                       ##
## Copyright (C)  2023  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

"""
Module defining publisher-subscriber interface pattern.

An alternative to explicit naming of source and destination would be to name a port through which communication is to take place.

Subscribers declare which topics they are interested in, and publishers send messages to topics without knowledge of what (if any) subscribers there may be.
Subscribers are only updated when a topic or field they are subscribed to is updated or changed.

Transmit
Report
Broadcast
"""

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ()


from typing import Any

from concurrency.executors import JinxThreadPool
from datastructures.mappings import TwoWayMap


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class MessageTopic:
    """
    A topic is a particular topic of communication. Publishers publish to a topic
    by sending messages to it, and subscribers subscribe to a topic to receive
    messages from it. Publishers publish messages to topics without knowledge
    of what (if any) subscribers there may be. Similarly, subscribers receive
    messages from topics without knowledge of what (if any) publishers there
    are. This decoupling of publishers and subscribers can allow for greater
    scalability and a more dynamic network topology. Multiple publishers can
    publish messages to the same topic, and multiple subscribers can subscribe
    to the same topic, and is therefore a multi-producer multi-consumer model.
    Each publisher and subscriber is decoupled from the others, and can
    continue operating normally regardless of the status of the others.
    """
    pass


class DataField:
    """
    A data field is a particular field of data. Publishers publish data to a
    data field by sending messages to it, and subscribers subscribe to a data
    field to receive data from it. Publishers publish data to data fields
    without knowledge of what (if any) subscribers there may be. Similarly,
    subscribers receive data from data fields without knowledge of what (if
    any) publishers there are. This decoupling of publishers and subscribers
    can allow for greater scalability and a more dynamic network topology.
    Multiple publishers can publish data to the same data field, and multiple
    subscribers can subscribe to the same data field, and is therefore a
    multi-producer multi-consumer model. Each publisher and subscriber is
    decoupled from the others, and can continue operating normally regardless
    of the status of the others.
    """
    pass


class ProcedureCall:
    """
    A procedure call is a particular procedure that can be called. Publishers
    publish procedure calls to a topic by sending messages to it, and
    subscribers subscribe to a topic to receive procedure calls from it.
    Publishers publish procedure calls to topics without knowledge of what (if
    any) subscribers there may be. Similarly, subscribers receive procedure
    calls from topics without knowledge of what (if any) publishers there are.
    This decoupling of publishers and subscribers can allow for greater
    scalability and a more dynamic network topology. Multiple publishers can
    publish procedure calls to the same topic, and multiple subscribers can
    subscribe to the same topic, and is therefore a multi-producer
    multi-consumer model. Each publisher and subscriber is decoupled from the
    others, and can continue operating normally regardless of the status of the
    others.
    """
    pass


class CommandChannel:
    """
    A channel is a virtual pipe that allows a publisher to send commands and
    receive feedback and results to and from a subscriber, relayed via the hub.

    A publisher can create and connect to a named channel on the hub. The
    connection requires the publisher to expose a signal that emits a command,
    and two slots, one for feedback and one for results. A subscriber can
    connect to the same channel on the hub. The connection requires the
    subscriber to expose a slot that receives a command, and two signals, one
    for feedback and one for results. The hub will then connect the publisher's
    command signal to the subscriber's command slot, the subscriber's feedback
    signal to the publisher's feedback slot, and the subscriber's results
    signal to the publisher's results slot. The publisher can then send
    commands to the subscriber by emitting the command signal, and the
    subscriber can send feedback and results to the publisher by emitting the
    feedback and results signals, respectively. The hub will relay the signals
    from the publisher to the subscriber, and vice versa. Therefore the
    publisher and subscriber can communicate with each other without knowledge
    of each other's existence.
    """
    pass


class Publisher:
    def publish_on_topic(self, topic: MessageTopic, payload: Any) -> None:
        """
        Publish a message to a topic.

        The publisher publishes a message to a topic by sending a payload to it.
        """
        pass


def trigger_on_field_change(field_name: str) -> Any:
    """Decorate a method to be called when a field changes."""
    pass


class Subscriber:
    def field_changed(
        self,
        source: "Subject",
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        pass


def field(field_name: str | None = None) -> Any:
    """Decorate a field to be tracked by a `Subject`."""
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


class Processer:
    """
    A processer is a publisher-subscriber that processes messages from the hub
    and then re-publishes the results back to the hub.
    """
    pass


class PusSubHub:
    """
    A publisher-subscriber hub is a central place where publishers and
    subscribers can register themselves. It is responsible for routing messages
    from publishers to subscribers.
    """
    def __init__(self) -> None:
        self.__listeners = TwoWayMap[Listener, str]()
        self.__executor = JinxThreadPool()

    @property
    @field()
    def value(self) -> int:
        return self.__value

    @value.setter
    @field_change()
    def value(self, value: int) -> None:
        self.__value = value

    def assign_listener(self, listener: Listener, fields: list[str]) -> None:
        self.__listeners.extend(
            (listener, set(fields))
        )

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
