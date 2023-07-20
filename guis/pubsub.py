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
Emit
Send
"""

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ()


from typing import Any, Callable

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
    pass


class ProcedureCall:
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


def trigger_on_topic_message_added(topic: MessageTopic) -> Any:
    """Decorate a method to be called when a message is added to a topic."""
    def decorator(func: Any) -> Any:
        hub = PubSubHub()
        hub.subscribe(topic, func)
        return func
    return decorator


def trigger_on_data_field_change(field_name: str) -> Any:
    """Decorate a method to be called when a data field is changed."""
    pass


class Subscriber:
    def topic_message_added(
        self,
        source: "PubSubHub",
        topic: MessageTopic
    ) -> None:
        pass

    def data_field_changed(
        self,
        source: "PubSubHub",
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        pass


class Processer:
    """
    A processer is a publisher-subscriber that processes messages from the hub
    and then re-publishes the results back to the hub.
    """
    pass


class PubSubHub:
    """
    A publisher-subscriber hub is a central place where publishers and
    subscribers can register themselves. It is responsible for routing messages
    from publishers to subscribers.
    """

    def __new__(cls) -> "PubSubHub":
        if not hasattr(cls, "__instance"):
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        self.__subscribers_topics: dict[str, list[Callable]] = {}
        self.__topics: dict[str, list[Any]] = {}
        self.__executor = JinxThreadPool()

    def subscribe_topic(self, topic_name: str, func: Callable) -> None:
        if topic_name not in self.__subscribers_topics:
            self.__subscribers_topics[topic_name] = []
        self.__subscribers_topics[topic_name].append(func)

    def publish_topic(self, topic_name: str, messages: str) -> None:
        if topic_name not in self.__topics:
            self.__topics[topic_name] = []
        self.__topics[topic_name].extend(messages)
        self.__executor.submit(
            self.__update_subscribers_topics,
            topic_name,
            self.__topics[topic_name][:-len(messages)],
            self.__topics[topic_name][-len(messages):]
        )

    def __update_subscribers_topics(
        self,
        topic_name: str,
        existing_messages: list[str],
        new_messages: list[str]
    ) -> None:
        for func in self.__subscribers_topics[topic_name]:
            func(topic_name, existing_messages, new_messages)
