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

from collections import defaultdict, deque
import logging
from typing import Any, Callable, TypeVar, final, Union, overload
import weakref

from PySide6.QtCore import QTimer  # pylint: disable=no-name-in-module

from aloy.concurrency.atomic import AtomicDict, AtomicList
from aloy.datastructures.views import DequeView

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "0.1.0"

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
        self.__topics = AtomicDict[str, deque[str]]()
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
            self.__update_topic_subscribers)
        self.__updater_timer.timeout.connect(
            self.__update_parameter_subscribers)
        self.__updater_timer.setInterval(1000 // tick_rate)
        if new_timer or start_timer:
            self.__updater_timer.start()

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
        - `all_messages: DequeView[str]` - All messages in the topic.
        - `new_messages: list[str]` - The most recently added new messages.

        Parameters
        ----------
        `topic_name: str` - The name of the topic to subscribe to.

        `callback: Callable[[str, list[str], list[str]], None]` - The callback
        function to call when a message is added to the topic.

        Notes
        -----
        See `aloy.datastructures.views.DequeView` for more information on the
        type of the `all_messages` argument.
        """
        with self.__subscribers_topics:
            if topic_name not in self.__subscribers_topics:
                self.__subscribers_topics[topic_name] = AtomicList[Callable]()
        with (callback_funcs := self.__subscribers_topics[topic_name]):
            if self.__debug:
                self.__PUBSUB_LOGGER.debug(
                    "Subscribing callback function %s to topic %s on hub %s.",
                    callback, topic_name, self.name
                )
            callback_funcs.append(callback)

    def declare_topic(
        self,
        topic_name: str,
        queue_length: int | None = None
    ) -> None:
        """
        Declare a new topic of given name and max queue length.

        Parameters
        ----------
        `topic_name: str` - The name of the topic to declare.

        `queue_length: int | None = None` - The maximum number of messages to
        store in the topic. If not given or None, the topic will store an
        unlimited number of messages. Note that there is no guarantee that the
        subscribers of a topic that has a maximum queue length will receive all
        messages published to the topic if more messages are published than the
        maximum queue length between each update of the subscribers. Defaults
        to None.

        Raises
        ------
        `ValueError` - If a topic with the given name already exists.
        """
        with self.__topics:
            if self.__debug:
                self.__PUBSUB_LOGGER.debug(
                    "Declaring new topic %s on hub %s.", topic_name, self.name
                )
            if topic_name not in self.__topics:
                self.__topics[topic_name] = deque(maxlen=queue_length)
            else:
                raise ValueError(f"Topic '{topic_name}' already exists.")

    def publish_topic(self, topic_name: str, *messages: str) -> None:
        """
        Publish messages to the given topic.

        The messages are added to the topic and all subscribed callback
        functions are called with the new messages.

        Parameters
        ----------
        `topic_name: str` - The name of the topic to publish to.

        `*messages: str` - The messages to publish to the topic.

        Raises
        ------
        `ValueError` - If the topic does not exist.
        """
        with self.__topics, self.__topics_updated:
            if self.__debug:
                self.__PUBSUB_LOGGER.debug(
                    "Publishing %s messages to topic %s on hub %s: \n%s",
                    len(messages), topic_name, self.name, messages
                )
            if topic_name not in self.__topics:
                raise ValueError(f"Topic '{topic_name}' does not exist.")
            else:
                self.__topics[topic_name].extend(messages)
            if (len_ := self.__topics_updated.get(topic_name)) is None:
                self.__topics_updated[topic_name] = len(messages)
            else:
                self.__topics_updated[topic_name] = len_ + len(messages)

    def __update_topic_subscribers(self) -> None:
        """Update all subscribers of all topics."""
        with self.__topics, self.__topics_updated:
            for topic_name, messages in self.__topics.items():
                messages_view = DequeView[str](messages)
                num_new_messages = self.__topics_updated[topic_name]
                self.__topics_updated[topic_name] = None
                if num_new_messages is not None:
                    callback_funcs = self.__subscribers_topics.get(topic_name)
                    if callback_funcs is None or not callback_funcs:
                        continue
                    with callback_funcs:
                        new_messages = [
                            messages_view[-(num_new_messages - i)]
                            for i in range(num_new_messages)
                        ]
                        if self.__debug:
                            self.__PUBSUB_LOGGER.debug(
                                "Updating %s subscribers of topic %s on hub %s"
                                " with:\nAll messages: %s\nNew messages: \n%s",
                                len(callback_funcs), topic_name, self.name,
                                num_new_messages,
                                new_messages
                            )
                        for func in callback_funcs:
                            func(topic_name, messages_view, new_messages)

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
            if self.__debug:
                self.__PUBSUB_LOGGER.debug(
                    "Subscribing callback function %s to parameter %s on hub "
                    "%s.", callback, param_name, self.name
                )
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
        with self.__params, self.__params_updated:
            if self.__debug:
                self.__PUBSUB_LOGGER.debug(
                    "Publishing value %s to parameter %s on hub %s.",
                    value, param_name, self.name
                )
            if param_name not in self.__params:
                self.__params[param_name] = (None, value)
            else:
                _, old_value = self.__params[param_name]
                self.__params[param_name] = (old_value, value)
            self.__params_updated[param_name] = True

    def __update_parameter_subscribers(self) -> None:
        """Update all subscribers of all parameters."""
        with self.__params, self.__params_updated:
            for param_name, value in self.__params.items():
                if self.__params_updated[param_name]:
                    self.__params_updated[param_name] = False
                    callback_funcs = self.__subscribers_params.get(param_name)
                    if callback_funcs is None or not callback_funcs:
                        continue
                    with callback_funcs:
                        if self.__debug:
                            self.__PUBSUB_LOGGER.debug(
                                "Updating %s subscribers of parameter %s on "
                                "hub %s with:\nOld value: %s\nNew value: %s",
                                len(callback_funcs), param_name, self.name,
                                value[0], value[1]
                            )
                        for func in callback_funcs:
                            func(param_name, *value)


_CT = TypeVar("_CT")
_IT = TypeVar("_IT")
_FT = TypeVar("_FT", bound=Callable)


@overload
def subscribe_topic(
    hub_name: str,
    topic_name: str,
    func: Callable[[str, _IT, _IT], None]
) -> None:
    """
    Subscribe a callback function to the given topic.

    The subscribed callback will be called when a message is added to the given
    topic on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_topic` for more details.
    """


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


def subscribe_topic(
    hub_name: str,
    topic_name: str,
    func: _FT | None = None
) -> Callable[[_FT], _FT] | None:
    """Subscribe a callback function to the given topic."""
    if func is None:
        def decorator(func_: _FT) -> _FT:
            subscribe_topic(hub_name, topic_name, func_)
            return func_
        return decorator

    hub = get_hub(hub_name)
    if hub is not None:
        hub.subscribe_topic(topic_name, func)
    else:
        _PUBSUB_INIT_SUBSCRIBE_TOPICS[hub_name][topic_name].append(func)
    return None


def method_subscribe_topic(
    hub_name: str,
    topic_name: str
) -> Callable[
    [Callable[[_CT, str, _IT, _IT], None]],
    Callable[[_CT, str, _IT, _IT], None]
]:
    """
    Decorate an instance method to subscribe it to the given topic. Note that
    the class must be decorated with `@subscriber` for this to work.

    The subscribed callback will be called when a message is added to the given
    topic on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_topic` for more details.
    """
    def decorator(
        func: Callable[[_CT, str, _IT, _IT], None]
    ) -> Callable[[_CT, str, _IT, _IT], None]:
        sub_topic = (hub_name, topic_name)
        func.__sub_topic__ = sub_topic  # type: ignore[attr-defined]
        return func
    return decorator


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


def subscribe_param(
    hub_name: str,
    field_name: str,
    func: _FT | None = None
) -> Callable[[_FT], _FT] | None:
    """Subscribe a callback function to the given parameter."""
    if func is None:
        def decorator(func_: _FT) -> _FT:
            subscribe_param(hub_name, field_name, func_)
            return func_
        return decorator

    hub = get_hub(hub_name)
    if hub is not None:
        hub.subscribe_param(field_name, func)
    else:
        _PUBSUB_INIT_SUBSCRIBE_PARAMS[hub_name][field_name].append(func)
    return None


def method_subscribe_param(
    hub_name: str,
    param_name: str
) -> Callable[
    [Callable[[_CT, str, _IT, _IT], None]],
    Callable[[_CT, str, _IT, _IT], None]
]:
    """
    Decorate an instance method to subscribe it to the given parameter. Note
    that the class must be decorated with `@subscriber` for this to work.

    The subscribed callback will be called when the given parameter is changed
    on the given hub. If the hub does not yet exist, the callback will be
    subscribed when a hub with the given name is created.

    See `AloyPubSubHub.subscribe_param` for more details.
    """
    def decorator(
        func: Callable[[_CT, str, _IT, _IT], None]
    ) -> Callable[[_CT, str, _IT, _IT], None]:
        sub_param = (hub_name, param_name)
        func.__sub_param__ = sub_param  # type: ignore[attr-defined]
        return func
    return decorator


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


def subscriber(
    cls: type[_CT]
) -> type[_CT]:
    """
    Decorate a class to subscribe its methods to topics and parameters.
    """
    __original_init__ = cls.__init__

    def __init__(self: _CT, *args: Any, **kwargs: Any) -> None:
        __original_init__(self, *args, **kwargs)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr):
                if hasattr(attr, "__sub_topic__"):
                    hub_name, topic_name = attr.__sub_topic__
                    subscribe_topic(hub_name, topic_name, attr)
                if hasattr(attr, "__sub_param__"):
                    hub_name, param_name = attr.__sub_param__
                    subscribe_param(hub_name, param_name, attr)

    cls.__init__ = __init__  # type: ignore[assignment]
    return cls


def __main() -> None:
    # pylint: disable=import-outside-toplevel
    # pylint: disable=no-name-in-module
    import sys
    import time
    import itertools

    from PySide6.QtWidgets import QApplication

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout
    )

    _qapp = QApplication([])

    def topic_callback(
        topic_name: str,
        all_messages: list[str],  # pylint: disable=unused-argument
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
    # pylint: enable=import-outside-toplevel
    # pylint: enable=no-name-in-module


if __name__ == "__main__":
    __main()
