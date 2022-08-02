from abc import ABCMeta, abstractmethod

import _collections_abc
from dataclasses import dataclass
from typing import Generic, Iterator, NamedTuple, Optional, TypeVar

## Generics for plans:
##      - AI: Action type,
##      - SI: State type.
AI = TypeVar("AI", str, bytes, int)
SI = TypeVar("SI", str, bytes, int)
CT = TypeVar("CT", int, float)

@dataclass(frozen=True)
class Transition(Generic[AI, SI, CT]):
    """
    Simple data class representing a state transition of the form:
        (step: int, start_state: SI, action: AI, end_state: SI, cost: CT)
    """
    step: int
    start_state: SI
    action: AI
    end_state: SI
    cost: CT

class Plan(_collections_abc.Sequence, Generic[AI, SI, CT], metaclass=ABCMeta):
    """
    Typical classical plan iterface.
    Defines plans as a finite length sequence of actions and states,
    and exposes standard methods for iterating over them.
    
    Abstract methods:
        - __iter__ -> Iterator[AI]: An iterator over the action sequence.
        - __len__ -> int: The number of state transitions in the plan.
    
    Mixin properties:
        - total_actions -> int: The number of state transitions in the plan.
        - total_states -> int: The number of states in the plan.
    
    Mixin methods:
        - actions(range) -> Iterator[AI]: An iterator over the action sequence.
        - states(range) -> Iterator[SI]: An iterator over the state sequence.
        - transitions(range) -> Iterator[Transition[AI, SI]]: An iterator over the state transition sequence.
    """
    
    def __getitem__(self, step: int) -> AI:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    @property
    def total_transitions(self) -> int:
        return len(self)
    
    @property
    def total_states(self) -> int:
        return len(self) + 1
    
    @property
    def end_state(self) -> SI:
        return self[-1]
    
    @abstractmethod
    def extend(self, action: AI, state: SI, cost: CT) -> "Plan[AI, SI, CT]":
        raise NotImplementedError
    
    def actions(self, in_range: Optional[range] = None) -> Iterator[AI]:
        if in_range is not None:
            return (self[index] for index in in_range)
        return iter(self)
    
    # @abstractmethod
    # def states(self, in_range: range) -> Iterator[SI]:
    #     raise NotImplementedError
    
    # @abstractmethod
    # def transitions(self, in_range: range) -> Iterator[Transition[AI, SI, CT]]:
    #     raise NotImplementedError

class Planner(Generic[AI, SI], metaclass=ABCMeta):
    pass