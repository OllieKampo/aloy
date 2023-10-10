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
Module defining publisher-subscriber design pattern.

Subscribers declare which topics they are interested in, and publishers send
messages to topics without knowledge of what (if any) subscribers there may be.
Subscribers are only updated when a topic or field they are subscribed to is
updated or changed.
"""

from collections import defaultdict
import logging
from typing import Any, Callable, TypeVar, final, Union, overload
import weakref

from PySide6.QtCore import QTimer  # pylint: disable=no-name-in-module

from aloy.concurrency.atomic import AtomicDict, AtomicList
from aloy.concurrency.executors import AloyQThreadPool

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.0.2"

__all__ = (
    "AloyPubSubHub",
    "subscribe_topic",
    "subscribe_param"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


# Hubs are stored in a weak value dictionary so that they can be garbage
# collected when no longer in use.
_HUBS: weakref.WeakValueDictionary[str, "AloyPubSubHub"]
_HUBS = weakref.WeakValueDictionary()
_PUBSUB_INIT_STORE_TOPICS: dict[str, dict[str, list[str]]]
_PUBSUB_INIT_STORE_TOPICS = defaultdict(lambda: defaultdict(list))
_PUBSUB_INIT_STORE_PARAMS: dict[str, dict[str, Any]]
_PUBSUB_INIT_STORE_PARAMS = defaultdict(dict)
_PUBSUB_INIT_SUBSCRIBE_TOPICS: dict[str, dict[str, list[Callable]]]
_PUBSUB_INIT_SUBSCRIBE_TOPICS = defaultdict(lambda: defaultdict(list))
_PUBSUB_INIT_SUBSCRIBE_PARAMS: dict[str, dict[str, list[Callable]]]
_PUBSUB_INIT_SUBSCRIBE_PARAMS = defaultdict(lambda: defaultdict(list))


def _add_hub(hub: "AloyPubSubHub") -> None:
    """Add a hub to the global list of hubs."""
    _HUBS[hub.name] = hub


def get_hub(hub_name: str) -> Union["AloyPubSubHub", None]:
    """
    Get the hub with the given name.

    Parameters
    ----------
    `hub_name: str` - The name of the hub to get.

    Returns
    -------
    `AloyPubSubHub | None` - The hub with the given name, or `None` if no hub
    with the given name exists.
    """
    return _HUBS.get(hub_name)


@final
class AloyPubSubHub:
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

    __PUBSUB_LOGGER = logging.getLogger("AloyPubSubHub")

    __slots__ = {
        "__weakref__": "Weak references to the object.",
        "__name": "The name of the hub.",
        "__debug": "Whether to log debug messages.",
        "__subscribers_topics": "Mapping between topic names and lists of "
                                "callback functions subscribed to the topic.",
        "__topics": "Mapping between topic names and lists of messages in the "
                    "topic.",
        "__topics_updated": "Mapping between topic names and the number of "
                            "new messages in the topic.",
        "__subscribers_params": "Mapping between parameter names and lists of "
                                "callback functions subscribed to the "
                                "parameter.",
        "__params": "Mapping between parameter names and the old and new "
                    "values of the parameter.",
        "__params_updated": "Mapping between parameter names and whether the "
                            "parameter has been updated.",
        "__updater_timer": "Timer to update subscribers.",
        "__qthreadpool": "Thread pool to update subscribers."
    }

    def __init__(
        self,
        name: str,
        param_dict: dict[str, Any] | None = None,
        qtimer: QTimer | None = None,
        tick_rate: int = 20,
        start_timer: bool = True,
        debug: bool = False
    ) -> None:
        """
        Create a new publisher-subscriber hub with the given name. The hub will
        be added to the global list of hubs and is accessible via its name. It
        is an error to create a hub with the same name as an existing hub. If
        all references to the hub are lost, it will be garbage collected and
        removed from the global list of hubs.

        Parameters
        ----------
        `name: str` - The name of the hub.

        `param_dict: dict[str, Any] | None = None` - A dictionary mapping
        parameter names to their initial values to publish to the hub.
        If not given or None, no initial values are specified. Defaults to
        None.

        `qtimer: QTimer | None = None` - The timer to use to update
        subscribers. If not given or None, a new timer will be created.
        Defaults to None. Defaults to None.

        `tick_rate: int = 20` - The tick rate of the timer in Hz. This is the
        maximum number of times per second that subscribers will be updated.
        Defaults to 20 (50ms between updates).

        `start_timer: bool = True` - Whether to start the timer if an existing
        timer is given. Ignored if a new timer is created (the timer is always
        started in this case). Defaults to True.

        `debug: bool = False` - Whether to log debug messages. Defaults to
        False.
        """
        self.__debug: bool = debug
        if self.__debug:
            self.__PUBSUB_LOGGER.debug(
                "Creating new publisher-subscriber hub with: "
                "name=%s, qtimer=%s, tick_rate=%s, start_timer=%s, debug=%s",
                name, qtimer, tick_rate, start_timer, debug
            )

        # All hubs must have a name.
        self.__name: str = name
        if get_hub(name) is not None:
            raise ValueError(f"Hub with name '{name}' already exists.")

        # Internal data structures.
        self.__subscribers_topics = AtomicDict[str, AtomicList[Callable]]()
        self.__topics = AtomicDict[str, AtomicList[str]]()
        self.__topics_updated = AtomicDict[str, int | None]()
        self.__subscribers_params = AtomicDict[str, AtomicList[Callable]]()
        self.__params = AtomicDict[str, tuple[Any, Any]]()
        self.__params_updated = AtomicDict[str, bool]()

        # Threads for updating subscribers:
        #   - The timer updates all subscribers at a fixed rate,
        #   - Whenever the timer times out, each topic and parameter is updated
        #     in parallel by a separate thread in the thread pool.
        self.__updater_timer: QTimer
        if (new_timer := qtimer is None):
            self.__updater_timer = QTimer()
        else:
            self.__updater_timer = qtimer
        self.__updater_timer.timeout.connect(
            self.__update_all_topic_subscribers)
        self.__updater_timer.timeout.connect(
            self.__update_all_parameter_subscribers)
        self.__updater_timer.setInterval(1000 // tick_rate)
        if new_timer or start_timer:
            self.__updater_timer.start()
        self.__qthreadpool = AloyQThreadPool()

        # Add hub to global list of hubs.
        _add_hub(self)

        # Add subscribers.
        for topic_name, funcs in _PUBSUB_INIT_SUBSCRIBE_TOPICS[name].items():
            for func in funcs:
                self.subscribe_topic(topic_name, func)
        _PUBSUB_INIT_SUBSCRIBE_TOPICS[name].clear()
        for param_name, funcs in _PUBSUB_INIT_SUBSCRIBE_PARAMS[name].items():
            for func in funcs:
                self.subscribe_param(param_name, func)
        _PUBSUB_INIT_SUBSCRIBE_PARAMS[name].clear()

        # Add topic messages and parameters.
        for topic_name, messages in _PUBSUB_INIT_STORE_TOPICS[name].items():
            self.publish_topic(topic_name, *messages)
        _PUBSUB_INIT_STORE_TOPICS[name].clear()
        if param_dict is not None:
            for param_name, value in param_dict.items():
                self.publish_param(param_name, value)
        for param_name, value in _PUBSUB_INIT_STORE_PARAMS[name].items():
            self.publish_param(param_name, value)
        _PUBSUB_INIT_STORE_PARAMS[name].clear()

    @property
    def name(self) -> str:
        """Get the name of the hub."""
        return self.__name

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

        Parameters
        ----------
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

        Parameters
        ----------
        `topic_name: str` - The name of the topic to publish to.

        `*messages: str` - The messages to publish to the topic.
        """
        with self.__topics:
            if topic_name not in self.__topics:
                self.__topics[topic_name] = AtomicList[str]()
            with (topic_messages := self.__topics[topic_name]):
                topic_messages.extend(messages)
            with self.__topics_updated:
                if self.__topics_updated.get(topic_name) is None:
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
        """Update all subscribers of the given topic."""
        with (callback_funcs := self.__subscribers_topics[topic_name]):
            for func in callback_funcs:
                func(topic_name, existing_messages, new_messages)

    def subscribe_param(
        self,
        param_name: str,
        callback: Callable[[str, Any, Any], None]
    ) -> None:
        """
        Subscribe a callback function to the given parameter.

        The subscribed callback will be called when the parameter is changed by
        a publisher.

        The callback will be called with the following arguments:
        - `param_name: str` - The name of the parameter.
        - `old_value: Any` - The old value of the parameter.
        - `new_value: Any` - The new value of the parameter.

        Parameters
        ----------
        `param_name: str` - The name of the parameter to subscribe to.

        `callback: Callable[[str, Any, Any], None]` - The callback function to
        call when the parameter is changed.
        """
        with self.__subscribers_params:
            if param_name not in self.__subscribers_params:
                self.__subscribers_params[param_name] = AtomicList[Callable]()
            with (callback_funcs := self.__subscribers_params[param_name]):
                callback_funcs.append(callback)

    def publish_param(self, param_name: str, value: Any) -> None:
        """
        Publish a value to the given parameter.

        The value is set to the parameter value and all subscribed callback
        functions are called with the new value.

        Parameters
        ----------
        `param_name: str` - The name of the parameter to publish to.

        `value: Any` - The value to publish to the parameter.
        """
        with self.__params:
            if param_name not in self.__params:
                self.__params[param_name] = (None, value)
            else:
                _, old_value = self.__params[param_name]
                self.__params[param_name] = (old_value, value)
            with self.__params_updated:
                self.__params_updated[param_name] = True

    def __update_all_parameter_subscribers(self) -> None:
        """Update all subscribers of all parameters."""
        with self.__params, self.__params_updated:
            for param_name, value in self.__params.items():
                if self.__params_updated[param_name]:
                    self.__params_updated[param_name] = False
                    self.__qthreadpool.submit(
                        self.__update_parameter,
                        param_name,
                        value
                    )

    def __update_parameter(
        self,
        param_name: str,
        value: tuple[Any, Any]
    ) -> None:
        """Update all subscribers of the given parameter."""
        with (callback_funcs := self.__subscribers_params[param_name]):
            for func in callback_funcs:
                func(param_name, *value)


_CT = TypeVar("_CT")
_IT = TypeVar("_IT")
_FT = TypeVar("_FT", Callable, property)


@overload
def subscribe_topic(
    hub_name: str,
    topic_name: str,
    func: Callable[[str, Any, Any], None]
) -> None:
    """
    Subscribe a callback function to the given topic.

    The subscribed callback will be called when a message is added to the given
    topic on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_topic` for more details.
    """
    ...


@overload
def subscribe_topic(
    hub_name: str,
    topic_name: str
) -> Callable[
    [Callable[[str, _IT, _IT], None]],
    Callable[[str, _IT, _IT], None]
]:
    """
    Decorate a function to subscribe it to the given topic.

    The subscribed callback will be called when a message is added to the given
    topic on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_topic` for more details.
    """
    ...


@overload
def subscribe_topic(
    hub_name: str,
    topic_name: str
) -> Callable[
    [Callable[[_CT, str, _IT, _IT], None]],
    Callable[[_CT, str, _IT, _IT], None]
]:
    """
    Decorate an instance method or property to subscribe it to the given topic.

    The subscribed callback will be called when a message is added to the given
    topic on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_topic` for more details.
    """
    ...


def subscribe_topic(
    hub_name: str,
    topic_name: str,
    func: _FT | None = None
) -> _FT | None:
    """Subscribe a callback function to the given topic."""
    if func is None:
        def decorator(func_: _FT) -> _FT:
            subscribe_topic(hub_name, topic_name, func_)
            return func_
        return decorator

    if isinstance(func, property):
        if func.fset is None:
            raise TypeError("Cannot subscribe to a read-only property.")
        func = func.fset

    hub = get_hub(hub_name)
    if hub is not None:
        hub.subscribe_topic(topic_name, func)
    else:
        _PUBSUB_INIT_SUBSCRIBE_TOPICS[hub_name][topic_name].append(func)


@overload
def subscribe_param(
    hub_name: str,
    field_name: str,
    func: Callable[[str, Any, Any], None]
) -> None:
    """
    Subscribe a callback function to the given parameter.

    The subscribed callback will be called when the given parameter is changed
    on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_param` for more details.
    """
    ...


@overload
def subscribe_param(
    hub_name: str,
    field_name: str
) -> Callable[
    [Callable[[str, _IT, _IT], None]],
    Callable[[str, _IT, _IT], None]
]:
    """
    Decorate a function to subscribe it to the given parameter.

    The subscribed callback will be called when the given parameter is changed
    on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_param` for more details.
    """
    ...


@overload
def subscribe_param(
    hub_name: str,
    field_name: str
) -> Callable[
    [Callable[[_CT, str, _IT, _IT], None]],
    Callable[[_CT, str, _IT, _IT], None]
]:
    """
    Decorate an instance method or property to subscribe it to the given
    parameter.

    The subscribed callback will be called when the given parameter is changed
    on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_param` for more details.
    """
    ...


def subscribe_param(
    hub_name: str,
    field_name: str,
    func: _FT | None = None
) -> _FT | None:
    """Subscribe a callback function to the given parameter."""
    if func is None:
        def decorator(func_: _FT) -> _FT:
            subscribe_param(hub_name, field_name, func_)
            return func_
        return decorator

    if isinstance(func, property):
        if func.fset is None:
            raise TypeError("Cannot subscribe to a read-only property.")
        func = func.fset

    hub = get_hub(hub_name)
    if hub is not None:
        hub.subscribe_param(field_name, func)
    else:
        _PUBSUB_INIT_SUBSCRIBE_PARAMS[hub_name][field_name].append(func)


def publish_topic(hub_name: str, topic_name: str, *messages: str) -> None:
    """
    Publish messages to the given topic.

    The messages are added to the topic and all subscribed callback functions
    are called with the new messages.

    Parameters
    ----------
    `hub_name: str` - The name of the hub to publish to.

    `topic_name: str` - The name of the topic to publish to.

    `*messages: str` - The messages to publish to the topic.
    """
    hub = get_hub(hub_name)
    if hub is not None:
        hub.publish_topic(topic_name, *messages)
    else:
        _PUBSUB_INIT_STORE_TOPICS[hub_name][topic_name].extend(messages)


def publish_param(hub_name: str, param_name: str, value: Any) -> None:
    """
    Publish a value to the given parameter.

    The value is set to the parameter value and all subscribed callback
    functions are called with the new value.

    Parameters
    ----------
    `hub_name: str` - The name of the hub to publish to.

    `param_name: str` - The name of the parameter to publish to.

    `value: Any` - The value to publish to the parameter.
    """
    hub = get_hub(hub_name)
    if hub is not None:
        hub.publish_param(param_name, value)
    else:
        _PUBSUB_INIT_STORE_PARAMS[hub_name][param_name] = value


def __main() -> None:
    # pylint: disable=import-outside-toplevel
    # pylint: disable=no-name-in-module
    import time
    import itertools
    from PySide6.QtWidgets import QApplication

    _qapp = QApplication([])

    def topic_callback(
        topic_name: str,
        all_messages: list[str],
        new_messages: list[str]
    ) -> None:
        print(f"Topic '{topic_name}' updated with {new_messages}")
    subscribe_topic("test_hub", "test_topic", topic_callback)

    def param_callback(
        param_name: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        print(f"Parameter '{param_name}' "
              f"updated from {old_value} to {new_value}")
    subscribe_param("test_hub", "test_param", param_callback)

    for i in range(10):
        publish_topic("test_hub", "test_topic", f"Buffered message {i}.")
    publish_param("test_hub", "test_param", "Buffered parameter.")

    _test_hub = AloyPubSubHub("test_hub")  # noqa: F841

    counter = itertools.count()
    qtimer = QTimer()
    qtimer.timeout.connect(
        lambda: publish_topic("test_hub", "test_topic", str(time.time())))
    qtimer.timeout.connect(
        lambda: publish_param("test_hub", "test_param", next(counter)))
    qtimer.start(1000)

    _qapp.exec()


if __name__ == "__main__":
    __main()
