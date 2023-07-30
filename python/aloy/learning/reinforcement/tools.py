
from collections import deque
import collections
import random
from typing import Generic, SupportsFloat, TypeVar, final


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RewType = TypeVar("RewType", bound=SupportsFloat)


@final
class Transition(collections.abc.Sequence, Generic[ObsType, ActType, RewType]):
    """Represents a single state transition in a gym environment."""

    def __init__(
        self,
        state: ObsType,
        action: ActType,
        next_state: ObsType,
        reward: RewType
    ) -> None:
        """Create a new transition."""
        self.__tuple = (state, action, next_state, reward)

    def __getitem__(self, index: int) -> ObsType | ActType | RewType:
        """Return the item at the given index."""
        return self.__tuple[index]

    def __len__(self) -> int:
        """Return the length of the transition."""
        return 4

    @property
    def state(self) -> ObsType:
        """Return the state of the transition."""
        return self.__tuple[0]

    @property
    def action(self) -> ActType:
        """Return the action of the transition."""
        return self.__tuple[1]

    @property
    def next_state(self) -> ObsType:
        """Return the next state of the transition."""
        return self.__tuple[2]

    @property
    def reward(self) -> RewType:
        """Return the reward of the transition."""
        return self.__tuple[3]


class ReplayMemory(collections.abc.Sequence, Generic[ObsType, ActType, RewType]):
    """A replay memory for storing transitions of a gym environment."""

    def __init__(self, capacity: int) -> None:
        """Create a new replay memory with given maximum capacity."""
        self.__memory: deque[Transition] = deque([], maxlen=capacity)

    def __getitem__(self, index: int) -> Transition[ObsType, ActType, RewType]:
        """Return the transition at the given index."""
        return self.__memory[index]

    def __len__(self) -> int:
        """Return the current size of the memory."""
        return len(self.__memory)

    @property
    def capacity(self) -> int:
        """Return the maximum capacity of the memory."""
        return self.__memory.maxlen

    def push(
        self,
        state: ObsType,
        action: ActType,
        next_state: ObsType,
        reward: RewType
    ) -> None:
        """Add experience to the memory."""
        self.__memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int, /) -> list[Transition[ObsType, ActType, RewType]]:
        """Take a sample of the given size from the memory."""
        if batch_size > len(self):
            raise RuntimeError("Insufficient memory")
        return random.sample(self, batch_size)

    def can_sample(self, batch_size: int, /) -> bool:
        """
        Return whether or not a sample of the given size can be taken from the
        memory.
        """
        return len(self) >= batch_size
