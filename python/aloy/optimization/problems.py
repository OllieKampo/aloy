
__all__ = ("NQueensPuzzle", "EightPuzzle")

from typing import Iterable, Literal, Sequence

import networkx.classes.graph as nxgraph
import numpy as np

import aloy.datastructures.graph as aloygraph
from aloy.datastructures.disjointset import DisjointSet


class NQueensPuzzle:
    """
    Class defining the N-Queens puzzle.

    The N-Queens puzzle is a problem where N queens are placed on an NxN
    chessboard. The objective is to position the queens on the board such
    that no queen can take any other queen.
    """

    __slots__ = {
        "__board": "The chess board.",
        "__queens": "The positions of the queens."
    }

    def __init__(self, n: int = 4, seed: int | None = None) -> None:
        """
        Create a new N-Queens puzzle with the given size.

        Possible boards are created randomly with uniform probability.
        If a seed is given, the board is created deterministically.
        """
        self.__board: np.ndarray = np.zeros((n, n), dtype=bool)
        indices = np.random.default_rng(seed).choice(
            n**2, size=n, replace=False
        )
        self.__board[indices // n, indices % n] = True
        self.__queens: dict[int, tuple[int, int]] = {
            i: (indices[i] // n, indices[i] % n)
            for i in range(n)
        }

    def __str__(self) -> str:
        """Return a string representation of the board."""
        return str(self.__board)

    def __len__(self) -> int:
        """Return the size of the board."""
        return len(self.__board)

    def __iter__(self) -> np.ndarray:
        """Iterate over the board."""
        return iter(self.__board)

    def __getitem__(self, key: tuple[int, int]) -> bool:
        """Return the value at the given position."""
        return self.__board[key]

    def __array__(self) -> np.ndarray:
        """Return the board as a numpy array."""
        return self.__board

    @property
    def queens(self) -> dict[int, tuple[int, int]]:
        """Return the positions of the queens."""
        return self.__queens

    def position(self, queen: int) -> tuple[int, int]:
        """Return the position of the given queen."""
        return self.__queens[queen]

    def move(self, queen: int, row: int, col: int) -> None:
        """Move the given queen to the given position."""
        if self.__board[row, col]:
            raise ValueError("The given position is already occupied.")
        self.__board[self.__queens[queen]] = False
        self.__board[row, col] = True
        self.__queens[queen] = (row, col)

    def move_random(self, queen: int) -> None:
        """Move the given queen to a random position."""
        row, col = self.__queens[queen]
        self.__board[row, col] = False
        row, col = np.random.choice(np.where(~self.__board))
        self.__board[row, col] = True
        self.__queens[queen] = (row, col)

    def check(self, row: int, col: int) -> bool:
        """Check if the given position is safe."""
        return not (self.__board[row, col] and
                    (self.__board[row, :].sum() > 1 or
                     self.__board[:, col].sum() > 1 or
                     self.__board.diagonal(row - col).sum() > 1 or
                     self.__board[::-1].diagonal(row + col).sum() > 1))

    def check_queen(self, queen: int) -> bool:
        """Check if the given queen is safe."""
        return self.check(*self.__queens[queen])

    def check_all(self) -> bool:
        """Check if all queens are safe."""
        return all(self.check(row, col) for row, col in self.__queens.values())


class EightPuzzle:
    """Class defining the 8-puzzle."""

    __slots__ = {
        "__grid": "The puzzle grid.",
        "__positions": "The positions of each value.",
        "__goal": "The goal grid."
    }

    def __init__(
        self,
        init_seed: int | None = None,
        goal_seed: int | None = None
    ) -> None:
        """
        Create a new 8-puzzle with the given grid.

        If a seed is given, the grid is created deterministically.
        """
        self.__grid: np.ndarray = self.__create_grid(init_seed)
        self.__goal: np.ndarray = self.__create_grid(goal_seed)
        self.__positions: dict[int, tuple[int, int]] = {
            self.__grid[row, col]: (row, col)
            for row in range(3) for col in range(3)
        }

    @staticmethod
    def __create_grid(seed: int | None = None) -> np.ndarray:
        """Create a new 8-puzzle grid."""
        grid: np.ndarray = np.arange(9)
        np.random.default_rng(seed).shuffle(grid)
        return grid.reshape((3, 3))

    def __str__(self) -> str:
        """Return a string representation of the grid."""
        return str(self.__grid)

    def __len__(self) -> int:
        """Return the size of the grid."""
        return len(self.__grid)

    def __iter__(self) -> np.ndarray:
        """Iterate over the grid."""
        return iter(self.__grid)

    def __getitem__(self, key: tuple[int, int]) -> int:
        """Return the value at the given position."""
        return self.__grid[key]

    def __array__(self) -> np.ndarray:
        """Return the grid as a numpy array."""
        return self.__grid

    @property
    def grid(self) -> np.ndarray:
        """Return the grid."""
        return self.__grid

    @property
    def goal(self) -> np.ndarray:
        """Return the goal grid."""
        return self.__goal

    @property
    def empty_space(self) -> tuple[int, int]:
        """Return the position of the empty space."""
        return self.__positions[0]

    def position(self, value: int) -> tuple[int, int]:
        """Return the position of the given value."""
        return self.__positions[value]

    def slide(self, row: int, col: int) -> None:
        """Slide the tile at the given position to the empty space."""
        if self.__grid[row, col] == 0:
            raise ValueError("The given position is empty.")
        empty_space: tuple[int, int] = self.empty_space
        if not ((abs(row - empty_space[0]) + abs(col - empty_space[1])) == 1):
            raise ValueError("The given position is not adjacent to the empty space.")
        value = self.__grid[row, col]
        self.__grid[empty_space] = value
        self.__positions[value] = empty_space
        self.__grid[row, col] = 0
        self.__positions[0] = (row, col)

    def slide_value(self, value: int) -> None:
        """Slide the tile with the given value to the empty space."""
        row, col = self.position(value)
        empty_space: tuple[int, int] = self.empty_space
        if not ((abs(row - empty_space[0]) + abs(col - empty_space[1])) == 1):
            raise ValueError("The given position is not adjacent to the empty space.")
        self.__grid[empty_space] = value
        self.__positions[value] = empty_space
        self.__grid[row, col] = 0
        self.__positions[0] = (row, col)

    def slide_sequence(self, sequence: Iterable[int]) -> None:
        """Slide the tiles with the given values to the empty space."""
        for value in sequence:
            self.slide_value(value)

    def check(self, row: int, col: int) -> bool:
        """Check if the given position is correct."""
        return self.__grid[row, col] == self.__goal[row, col]

    def check_value(self, value: int) -> bool:
        """Check if the given value is correct."""
        return self.check(*self.position(value))

    def check_all(self) -> bool:
        """Check if all values are correct."""
        return np.all(self.__grid == self.__goal)


class TravellingSalesman:
    """Class defining the travelling salesman problem."""

    def __init__(self,
                 cities: int | Sequence[str],
                 start: int | str | None = None,
                 end: int | str | None = None,
                 size: tuple[float, float] = (100.0, 100.0),
                 seed: int | None = None,
                 backend: Literal["aloy", "networkx"] = "networkx"
                 ) -> None:
        """
        Create a new travelling salesman problem with the given number of cities.

        If a seed is given, the cities are created deterministically.
        """
        graph = self.__create_graph(cities, start, end, size, seed, backend)
        self.__graph: aloygraph.Graph | nxgraph.Graph = graph

    @staticmethod
    def __create_graph(cities: int | Sequence[str],
                       start: int | str | None = None,
                       end: int | str | None = None,
                       size: tuple[float, float] = (100.0, 100.0),
                       seed: int | None = None,
                       backend: Literal["aloy", "networkx"] = "aloy"
                       ) -> aloygraph.Graph | nxgraph.Graph:
        """Create a new graph for the travelling salesman problem."""
        if isinstance(cities, int):
            num_cities = cities
            cities = range(cities)
        else:
            num_cities = len(cities)

        rng = np.random.default_rng(seed)
        positions = rng.uniform(size[0], size[1], size=(num_cities, 2))

        if backend == "aloy":
            graph = aloygraph.Graph(vertices=cities)
            for city, position in zip(cities, positions):
                graph.set_vertex_data(city, "position", position)
        elif backend == "networkx":
            graph = nxgraph.Graph()
            for city, position in zip(cities, positions):
                graph.add_node(city, position=position)
        else:
            raise ValueError("Invalid backend.")

        if start is None:
            start = rng.choice(cities)
        elif start not in cities:
            raise ValueError("Invalid start city.")
        if end is None:
            end = rng.choice(cities)
        elif end not in cities:
            raise ValueError("Invalid goal city.")

        disjoint_set = DisjointSet(cities)
        while not disjoint_set.is_connected(start, end):
            i, j = rng.choice(cities, size=2, replace=False)
            if not disjoint_set.is_connected(i, j):
                disjoint_set.union(i, j)
                graph.add_edge(
                    i, j, weight=np.linalg.norm(positions[i] - positions[j])
                )

        return graph

    def __str__(self) -> str:
        """Return a string representation of the graph."""
        return str(self.__graph)

    def __len__(self) -> int:
        """Return the number of cities."""
        return len(self.__graph)

    def __iter__(self) -> Iterable[str]:
        """Iterate over the cities."""
        return iter(self.__graph)

    def __getitem__(self, key: str | int) -> set[str] | set[int]:
        """Return the neighbours of the given city."""
        return set(self.__graph[key])

    def get_position(self, city: str | int) -> np.ndarray:
        """Return the position of the given city."""
        if self.__backend == "aloy":
            return self.__graph.get_vertex_data(city, "position")
        else:
            return self.__graph.nodes[city]["position"]

    def get_distance(self, i: str, j: str) -> float:
        """Return the distance between the two given cities."""
        if self.__backend == "aloy":
            return self.__graph.get_edge_weight(i, j)
        else:
            return self.__graph.edges[i, j]["weight"]

    def get_path_length(self, path: Sequence[str]) -> float:
        """Return the length of the given path."""
        return sum(self.get_distance(i, j) for i, j in zip(path, path[1:]))
