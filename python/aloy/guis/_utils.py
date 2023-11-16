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

"""Module containing utility functions for the guis sub-package."""

import logging
from typing import Any, Callable

from PySide6.QtCore import QTimer  # pylint: disable=no-name-in-module

from aloy.concurrency.clocks import ClockThread


def create_clock(
    *functions: Callable[..., Any],
    name: str,
    clock: ClockThread | QTimer | None,
    tick_rate: int,
    start_clock: bool,
    logger: logging.Logger,
    debug: bool
) -> ClockThread | QTimer:
    """
    Utility function for creating a clock of a messaging system for
    calling the given functions at a given tick rate.

    Parameters
    ----------
    `*functions: Callable[..., Any]` - The functions to call on each tick.

    `name: str` - The name of the messaging system.

    `clock: ClockThread | QTimer | None` - The clock that calls the given
    functions. If not given or None, a new ClockThread is created with the
    given tick rate. Otherwise, the given clock is used.

    `tick_rate: int` - The tick rate of the clock if a new clock is
    created. Ignored if an existing clock is given.

    `start_clock: bool` - Whether to start the clock if an existing clock
    is given. Ignored if a new clock is created (the clock is always
    started in this case).

    `logger: logging.Logger` - The logger to use for logging debug
    messages.

    `debug: bool` - Whether to log debug messages.
    """
    if clock is None:
        if debug:
            logger.debug(
                "%s: Creating new ClockThread with tick rate %s.",
                name, tick_rate
            )
        clock = ClockThread(
            *functions,
            tick_rate=tick_rate
        )
        clock.start()
    else:
        if debug:
            logger.debug(
                "%s: Using existing clock of type %s.",
                name, type(clock)
            )
        if isinstance(clock, ClockThread):
            clock.schedule(*functions)
            if start_clock:
                clock.start()
        elif isinstance(clock, QTimer):
            for function_ in functions:
                clock.timeout.connect(function_)
            if start_clock:
                clock.start()
        else:
            raise TypeError("Clock must be of type ClockThread or QTimer."
                            f"Got; {clock} of {type(clock)}.")
    return clock
