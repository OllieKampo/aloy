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

"""
Module defining publisher-subscriber design pattern.

Subscribers declare which topics they are interested in, and publishers send
messages to topics without knowledge of what (if any) subscribers there may be.
Subscribers are only updated when a topic or field they are subscribed to is
updated or changed.
"""

from typing import Any, Callable, Final, Mapping

from PySide6.QtCore import QTimer
from PySide6 import QtWidgets

from aloy.concurrency.atomic import AtomicDict, AtomicList
from aloy.concurrency.executors import AloyQThreadPool
from aloy.concurrency.synchronization import SynchronizedMeta

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ()


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class PubSubHub(metaclass=SynchronizedMeta):
    """
    A publisher-subscriber hub is a central place where publishers and
    subscribers can register themselves. It is responsible for routing messages
    from publishers to subscribers.

    Publishers publish to a topic by sending messages to it, and subscribers
    subscribe to a topic to receive messages from it. Publishers publish
    messages to topics without knowledge of what (if any) subscribers there
    may be. Similarly, subscribers receive messages from topics without
    knowledge of what publishers there are. This decoupling of publishers and
    subscribers can allow for greater scalability. Multiple publishers can
    publish messages to the same topic, and multiple subscribers can subscribe
    to the same topic.
    """

    def __new__(cls) -> "PubSubHub":
        if not hasattr(cls, "__instance"):
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        # Internal data structures.
        self.__subscribers_topics = AtomicDict[str, AtomicList[Callable]]()
        self.__topics = AtomicDict[str, AtomicList[str]]()
        self.__topics_updated = AtomicDict[str, int | None]()

        # Threads for updating subscribers.
        self.__topic_updater_timer = QTimer()
        self.__topic_updater_timer.timeout.connect(
            self.__update_all_topic_subscribers)
        self.__topic_updater_timer.start(50)
        # self.__parameter_updater_timer = QTimer()
        # self.__parameter_updater_timer.timeout.connect(
        #     self.__update_all_parameter_subscribers)
        # self.__parameter_updater_timer.start(50)
        self.__qthreadpool = AloyQThreadPool()

    def subscribe_topic(
        self,
        topic_name: str,
        callback: Callable[[str, Any, Any], None]
    ) -> None:
        """
        Subscribe a callback function to the given topic.

        The subscribed callback will be called when a message is added to the
        topic by a publisher.

        The callback will be called with the following arguments:
        - `topic_name: str` - The name of the topic.
        - `all_messages: list[str]` - All messages in the topic.
        - `new_messages: list[str]` - The most recently added new messages.

        Args:
        -----
        `topic_name: str` - The name of the topic to subscribe to.

        `callback: Callable[[str, list[str], list[str]], None]` - The callback
        function to call when a message is added to the topic.
        """
        with self.__subscribers_topics:
            if topic_name not in self.__subscribers_topics:
                self.__subscribers_topics[topic_name] = AtomicList[Callable]()
            with (callback_funcs := self.__subscribers_topics[topic_name]):
                callback_funcs.append(callback)

    def publish_topic(self, topic_name: str, *messages: str) -> None:
        """
        Publish messages to the given topic.

        The messages are added to the topic and all subscribed callback
        functions are called with the new messages.

        Args:
        -----
        `topic_name: str` - The name of the topic to publish to.

        `*messages: str` - The messages to publish to the topic.
        """
        with self.__topics:
            if topic_name not in self.__topics:
                self.__topics[topic_name] = AtomicList[str]()
                with self.__topics_updated:
                    self.__topics_updated[topic_name] = None
        with (topic_messages := self.__topics[topic_name]):
            topic_messages.extend(messages)
            with self.__topics_updated:
                if self.__topics_updated[topic_name] is None:
                    self.__topics_updated[topic_name] = len(messages)
                else:
                    self.__topics_updated[topic_name] += len(messages)

    def __update_all_topic_subscribers(self) -> None:
        """Update all subscribers of all topics."""
        with self.__topics, self.__topics_updated:
            for topic_name, messages in self.__topics.items():
                num_new_messages = self.__topics_updated[topic_name]
                self.__topics_updated[topic_name] = None
                if num_new_messages is not None:
                    self.__qthreadpool.submit(
                        self.__update_topic,
                        topic_name,
                        messages,
                        messages[-num_new_messages:]
                    )

    def __update_topic(
        self,
        topic_name: str,
        existing_messages: list[str],
        new_messages: list[str]
    ) -> None:
        with (callback_funcs := self.__subscribers_topics[topic_name]):
            for func in callback_funcs:
                func(topic_name, existing_messages, new_messages)

    def subscribe_parameter(
        self,
        parameter_name: str,
        callback: Callable[[str, Any, Any], None]
    ) -> None:
        pass

    def publish_parameter(self, parameter_name: str, value: Any) -> None:
        pass

    def register_command_channel(
        self,
        channel_name: str,
        parameters: Mapping[str, Any],
        command_function: Callable[..., None],
        event_callbacks: Mapping[str, Mapping[str, type]]
    ) -> None:
        """
        A command channel allows a publisher to send commands and receive
        feedback and results to and from a subscriber, relayed via the hub.

        A publisher can create and connect to a named channel on the hub. The
        connection requires the publisher to expose a signal that emits a
        command, and two slots, one for feedback and one for results. A
        subscriber can connect to the same channel on the hub. The connection
        requires the subscriber to expose a slot that receives a command, and
        two signals, one for feedback and one for results. The hub will then
        connect the publisher's command signal to the subscriber's command
        slot, the subscriber's feedback signal to the publisher's feedback
        slot, and the subscriber's results signal to the publisher's results
        slot. The publisher can then send commands to the subscriber by
        emitting the command signal, and the subscriber can send feedback and
        results to the publisher by emitting the feedback and results signals,
        respectively. The hub will relay the signals from the publisher to the
        subscriber, and vice versa. Therefore the publisher and subscriber can
        communicate with each other without knowledge of each other's
        existence.
        """
        pass

    def request_command(
        self,
        channel_name: str,
        arguments: Mapping[str, Any],
        event_callbacks: Callable[..., None]
    ) -> None:
        pass


qapp = QtWidgets.QApplication([])
_PUBSUBHUB: Final[PubSubHub] = PubSubHub()


def subscribe_topic(topic_name: str) -> Any:
    """Decorate a method to be called when a message is added to a topic."""
    def decorator(func: Any) -> Any:
        _PUBSUBHUB.subscribe_topic(topic_name, func)
        return func
    return decorator


def subscribe_parameter(field_name: str) -> Any:
    """Decorate a method to be called when a data field is changed."""
    def decorator(func: Any) -> Any:
        _PUBSUBHUB.subscribe_parameter(field_name, func)
        return func
    return decorator


if __name__ == "__main__":
    import time

    def callback(topic_name: str, all_messages: list[str], new_messages: list[str]) -> None:
        print(f"Topic '{topic_name}' updated with {new_messages}")

    _PUBSUBHUB.subscribe_topic("test", callback)

    qtimer = QTimer()
    qtimer.timeout.connect(lambda: _PUBSUBHUB.publish_topic("test", str(time.time())))
    qtimer.start(20)

    qapp.exec()
