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

"""Module defining publisher-subscriber interface pattern."""

from abc import ABCMeta

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ()


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class Publisher(metaclass=ABCMeta):
    pass


class Subscriber(metaclass=ABCMeta):
    pass


class PusSubHub(metaclass=ABCMeta):
    pass


class Topic(metaclass=ABCMeta):
    pass
