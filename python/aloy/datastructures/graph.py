from collections import deque
import itertools
from typing import Any, Callable, Generic, Hashable, Iterable, Iterator, KeysView, Mapping, Optional, SupportsFloat, TypeVar, overload

import collections.abc
import numpy as np

from aloy.auxiliary.getters import default_get
from aloy.auxiliary.progressbars import ResourceProgressBar
from aloy.datastructures.disjointset import DisjointSet
from aloy.datastructures.mappings import TwoWayMap
from aloy.datastructures.queues import PriorityQueue, SortedQueue
from aloy.datastructures.views import SetView


def euclidean_distance_heuristic(
    start_vertex: tuple[float, ...],
    end_vertex: tuple[float, ...],
) -> float:
    """
    Euclidean distance heuristic for use with A* search.

    Parameters
    ----------
    start_vertex : tuple[float, ...]
        The starting vertex.
    end_vertex : tuple[float, ...]
        The ending vertex.

    Returns
    -------
    float
        The euclidean distance between the two vertices.
    """
    return sum((x - y) ** 2 for x, y in zip(start_vertex, end_vertex)) ** 0.5


def manhattan_distance_heuristic(
    start_vertex: tuple[float, ...],
    end_vertex: tuple[float, ...],
) -> float:
    """
    Manhattan distance heuristic for use with A* search.

    Parameters
    ----------
    start_vertex : tuple[float, ...]
        The starting vertex.
    end_vertex : tuple[float, ...]
        The ending vertex.

    Returns
    -------
    float
        The manhattan distance between the two vertices.
    """
    return sum(abs(x - y) for x, y in zip(start_vertex, end_vertex))


def chebyshev_distance_heuristic(
    start_vertex: tuple[float, ...],
    end_vertex: tuple[float, ...],
) -> float:
    """
    Chebyshev distance heuristic for use with A* search.

    Parameters
    ----------
    start_vertex : tuple[float, ...]
        The starting vertex.
    end_vertex : tuple[float, ...]
        The ending vertex.

    Returns
    -------
    float
        The chebyshev distance between the two vertices.
    """
    return max(abs(x - y) for x, y in zip(start_vertex, end_vertex))


VT = TypeVar("VT", bound=Hashable)
WT = TypeVar("WT", bound=SupportsFloat)


class Graph(collections.abc.MutableMapping, Generic[VT, WT]):
    """
    Represents a graph as a vertex set and an adjacency mapping.

    Internally it is a mapping, where the keys are the vertex set
    of the graph, and values are the adjacency sets for each vertex.

    Example Usage
    -------------
    ```
    ## Create a graph whose nodes are represented by integers where;
    ##      - Vertex 1 is adjacent to vertices 2 and 3,
    ##      - Vertex 2 is adjacent to vertex 4.
    >>> graph = Graph[int, int]({1: {2, 3},
    ...                          2: 4})
    >>> graph
    {1: {2, 3}, 2: {1, 4}, 3: {1}, 4: {2}}

    ## All the nodes are connected to one graph.
    >>> graph.is_connected()
    True

    ## Delete vertex 1 from the graph.
    >>> del graph[1]
    >>> graph
    {2: {4}, 3: set(), 4: {2}}

    ## Vertex 3 is now disconnected.
    >>> graph.is_connected()
    False
    ```
    """

    __slots__ = {
        "__adjacency_mapping": "Dictionary containing the adjacency mapping.",
        "__vertex_values": "Dictionary containing vertex values.",
        "__vertex_tags": "Two-Way Mapping containing vertex tags.",
        "__edge_weights": "Dictionary containing edge weights.",
        "__edge_tags": "Two-Way Mapping containing edge tags.",
        "__vertex_data": "Dictionary containing vertex data attributes.",
        "__edge_data": "Dictionary containing edge data attributes.",
        "__directed": "Whether the graph is directed or not.",
        "__allow_loops": "Whether the graph allows loops or not.",
        # "__subgraphs": "Dictionary containing subgraphs."
    }

    def __init__(
        self,
        graph: Mapping[VT, VT | set[VT] | None] = {}, *,
        vertices: Iterable[VT] | Mapping[Any, Iterable[VT]] = [],
        edges:
            Iterable[tuple[VT, VT | set[VT] | None]]
            | Mapping[Any, Iterable[tuple[VT, VT | set[VT] | None]]] = [],
        vertex_data: Mapping[VT, dict[str, Any]] = {},
        vertex_values: Mapping[VT, WT] = {},
        edge_data: Mapping[tuple[VT, VT], dict[str, Any]] = {},
        edge_weights: Mapping[tuple[VT, VT], WT] = {},
        directed: bool = False,
        allow_loops: bool = False
    ) -> None:
        """
        Create a new graph.

        Parameters
        ----------
        `graph: Mapping[VT, VT | set[VT] | None] | None` - A graph-like
        mapping. A key is a vertex, and value is either; one adjacent vertex,
        a set of adjacent vertices, or None. If None, the key vertex is
        disconnected.

        `vertices: Iterable[VT] | None` - An iterable of disconnected vertices.

        `edges: Iterable[tuple[VT, VT | set[VT] | None]] | None` - An iterable
        of edge tuples. Each tuple is a pair of vertices, where the first is
        the start vertex and second is either; one adjacent vertex, set of
        adjacent vertices, or None. If None, the start vertex is disconnected.

        `vertex_data: Mapping[VT, dict[str, Any]]` - A mapping of vertex data.
        Each key is a vertex, and the value is a data attribute dictionary.
        The data attribute dictionary maps of attribute names to arbitrary
        values.

        `edge_data: Mapping[tuple[VT, VT], dict[str, Any]]` - A mapping of
        edge data. Each key is an edge tuple, and the value is a data
        attribute dictionary. The data attribute dictionary maps of attribute
        names to arbitrary values.

        `edge_weights: Mapping[tuple[VT, VT], WT]` - A mapping of edge weights.
        Each key is an edge tuple, and the value is the weight of the edge.

        `directed: bool` - Whether the graph is directed or not. If True, the
        graph is directed, and the adjacency mapping is asymmetric. Otherwise,
        the graph is undirected, and the adjacency mapping is symmetric.

        `allow_loops: bool` - Whether the graph allows loops or not.
        If True, the graph allows loops, and a vertex can be adjacent to
        itself.
        """
        self.__directed: bool = directed
        self.__allow_loops: bool = allow_loops

        self.__adjacency_mapping = dict[VT, set[VT]]()
        self.__vertex_tags = TwoWayMap[VT, str]()
        self.__edge_tags = TwoWayMap[tuple[VT, VT], str]()

        self.__vertex_values = dict[VT, WT]()
        self.__edge_weights = dict[tuple[VT, VT], WT]()

        self.__vertex_data = dict[VT, dict[str, Any]]()
        self.__edge_data = dict[tuple[VT, VT], dict[str, Any]]()

        for node, connections in graph.items():
            self[node] = connections
        if isinstance(vertices, collections.abc.Mapping):
            for tag, vertices in vertices.items():
                self.add_vertices(*vertices, tags=[tag])
        else:
            for vertex in vertices:
                self.add_vertex(vertex)
        if isinstance(edges, collections.abc.Mapping):
            for tag, edges in edges.items():
                for start, ends in edges:
                    self.add_edges(*zip(itertools.repeat(start), ends), tags=[tag])
        else:
            for start, ends in edges:
                self[start] = ends

        for vertex, value in vertex_values.items():
            self.set_vertex_value(vertex, value)
        for edge, weight in edge_weights.items():
            self.set_edge_weight(*edge, weight)

        for vertex, data in vertex_data.items():
            self.update_vertex_data(vertex, data)
        for edge, data in edge_data.items():
            self.update_edge_data(*edge, data)

    def __str__(self) -> str:
        return str(self.__adjacency_mapping)

    def __repr__(self) -> str:
        return f"Graph(graph={self.__adjacency_mapping!r}, " \
               f"vertex_data={self.__vertex_data!r}, " \
               f"edge_data={self.__edge_data!r}, " \
               f"directed={self.__directed}, allow_loops={self.__allow_loops})"

    def __getitem__(
        self,
        vertex: VT
    ) -> set[VT]:
        """Get the adjacent vertices to the given vertex."""
        return self.__adjacency_mapping[vertex]

    def __setitem__(
        self,
        vertex: VT,
        connections: VT | set[VT] | frozenset[VT] | None = None
    ) -> None:
        """
        Add connections between a given vertex, and another vertex (or set of
        vertices).

        If any of the vertices are not in the graph, add them to the graph as
        well.
        """
        # If connections is None then the vertex is disconnected from the
        # graph.
        adj_map = self.__adjacency_mapping
        edge_ws = self.__edge_weights
        if connections is None:
            if vertex not in adj_map:
                adj_map[vertex] = set()
            return

        # Convert the connections to a set.
        _connections: set[VT]
        if not isinstance(connections, (set, frozenset)):
            _connections = {connections}
        else:
            _connections = connections  # type: ignore

        # If loops are not allowed, ensure the vertex is not in the set of
        # connections.
        if not self.__allow_loops:
            if isinstance(_connections, frozenset):
                _connections = _connections - {vertex}
            else:
                _connections.discard(vertex)

        # Find the set of existing connections from the specified vertex, and
        # the set of new connections (those in the set to add that are not
        # already in the existing set).
        existing_connections: set[VT] = adj_map.setdefault(vertex, set())
        new_connections: set[VT] = _connections - existing_connections

        # Add the new connections both to and from the specified node
        existing_connections.update(new_connections)
        if not self.__directed:
            for connected_vertex in new_connections:
                adj_map.setdefault(connected_vertex, set()).add(vertex)
                edge_ws.setdefault(
                    (connected_vertex, vertex),
                    0.0  # type: ignore
                )
                edge_ws.setdefault(
                    (vertex, connected_vertex),
                    0.0  # type: ignore
                )
        else:
            for connected_vertex in new_connections:
                adj_map.setdefault(connected_vertex, set())
                edge_ws.setdefault(
                    (vertex, connected_vertex),
                    0.0  # type: ignore
                )

    def __delitem__(
        self,
        vertex_or_edge: VT | tuple[VT, VT]
    ) -> None:
        """Remove a vertex or an edge from the graph."""
        if isinstance(vertex_or_edge, tuple):
            # If the specified item is an edge, remove the connection from the
            # first vertex to the second.
            vertex, connected_vertex = vertex_or_edge
            self.__adjacency_mapping[vertex].discard(connected_vertex)
            self.__edge_data.pop(vertex_or_edge, None)
            self.__edge_weights.pop(vertex_or_edge, None)
            if not self.__directed:
                self.__adjacency_mapping[connected_vertex].discard(vertex)
                self.__edge_data.pop((connected_vertex, vertex), None)
                self.__edge_weights.pop((connected_vertex, vertex), None)
        else:
            # Get all other nodes connected to the specified node, removing
            # any loops
            vertex = vertex_or_edge
            connected_vertices: set[VT] = self.__adjacency_mapping[vertex]
            connected_vertices.discard(vertex)

            # Remove all the connections from the other vertices to the
            # specified vertex.
            for connected_vertex in connected_vertices:
                self.__adjacency_mapping[connected_vertex].discard(vertex)
                self.__edge_data.pop((connected_vertex, vertex), None)
                self.__edge_weights.pop((connected_vertex, vertex), None)

            # Remove the specified vertex and all its connections to the
            # other vertices.
            del self.__adjacency_mapping[vertex]
            self.__vertex_data.pop(vertex, None)

    def __iter__(self) -> Iterator[VT]:
        """Iterate over the vertex set of this graph."""
        yield from self.__adjacency_mapping

    def __len__(self) -> int:
        """The number of vertices in the graph."""
        return len(self.__adjacency_mapping)

    def __contains__(
        self,
        vertex: object
    ) -> bool:
        """Check if the given vertex is in the graph."""
        return vertex in self.__adjacency_mapping

    def add_vertex(
        self,
        vertex: VT, /,
        tag: Any | None = None,
        value: WT | None = None,
        **data: Any
    ) -> None:
        """
        Add a vertex to the graph.

        Parameters
        ----------
        `vertex: VT` - The vertex identifier.

        `tag: Any | None` - Any additional tag to associate with the vertex.

        `value: WT | None` - Any additional value to associate with the vertex.

        `**data: Any` - Any additional data to associate with the vertex.
        """
        adj_map = self.__adjacency_mapping
        if vertex not in adj_map:
            adj_map[vertex] = set()
        if tag is not None:
            self.__vertex_tags[vertex] = tag
        if value is not None:
            self.__vertex_values[vertex] = value
        if data:
            self.update_vertex_data(vertex, data)

    def add_vertices(
        self,
        vertices: Iterable[VT], /,
        tags: Iterable[Any | None] | None = None,
        values: Iterable[WT | None] | None = None,
        data: Iterable[Mapping[str, Any]] | None = None
    ) -> None:
        """
        Add multiple vertices to the graph.

        Parameters
        ----------
        `vertices: Iterable[VT]` - The vertex identifiers.

        `tags: Iterable[Any | None] | None` - Any additional tags to associate
        with the vertices.

        `values: Iterable[WT | None] | None` - Any additional values to
        associate with the vertices.

        `data: Iterable[Mapping[str, Any]] | None` - Any additional data to
        associate with the vertices.
        """
        if tags is None:
            tags = itertools.repeat(None)
        if values is None:
            values = itertools.repeat(None)
        if data is None:
            data = itertools.repeat({})
        for vertex, tag, value, data_dict in zip(vertices, tags, values, data):
            self.add_vertex(vertex, tag=tag, value=value, **data_dict)

    def add_edge(
        self,
        start: VT,
        end: VT, /,
        tag: Any | None = None,
        weight: WT | None = None,
        **data: Any
    ) -> None:
        """
        Add an edge to the graph.

        Parameters
        ----------
        `start: VT` - The vertex identifier at the start of the edge.

        `end: VT` - The vertex identifier at the end of the edge.

        `weight: WT | None` - The weight of the edge. If `None`, the edge is
        unweighted.

        `**data: Any` - Any additional data to associate with the edge.
        """
        self[start] = end
        if tag is not None:
            self.__edge_tags[(start, end)] = tag
            if not self.__directed:
                self.__edge_tags[(end, start)] = tag
        if weight is not None:
            self.__edge_weights[(start, end)] = weight
            if not self.__directed:
                self.__edge_weights[(end, start)] = weight
        if data:
            self.update_edge_data(start, end, data)

    def add_edges(
        self,
        edges: Iterable[tuple[VT, VT]],
        tags: Iterable[Any | None] | None = None,
        weights: Iterable[WT | None] | None = None,
        data: Iterable[dict[str, Any]] | None = None
    ) -> None:
        """
        Add multiple edges to the graph.

        Parameters
        ----------
        `edges: Iterable[tuple[VT, VT]]` - The edges to add.

        `tags: Iterable[Any | None] | None` - Any additional tags to associate
        with the edges.

        `weights: Iterable[WT | None] | None` - Any additional weights to
        associate with the edges.

        `data: Iterable[dict[str, Any]] | None` - Any additional data to
        associate with the edges.
        """
        if tags is None:
            tags = itertools.repeat(None)
        if weights is None:
            weights = itertools.repeat(None)
        if data is None:
            data = itertools.repeat({})
        zip_ = zip(edges, tags, weights, data)
        for edge, tag, weight, _data in zip_:
            self.add_edge(*edge, tag, weight, **_data)

    def tag_vertex(
        self,
        vertex: VT, /,
        *tags: str
    ) -> None:
        """Tag the given vertex."""
        if vertex not in self:
            raise ValueError(f"Vertex {vertex} not in graph {self}.")
        self.__vertex_tags.add_many(vertex, tags)

    def untag_vertex(  # pylint: disable=keyword-arg-before-vararg
        self,
        vertex: VT,
        tag: str | None = None, /,
        *tags: str,
    ) -> None:
        """Remove the given tag from the given vertex."""
        if vertex not in self:
            raise ValueError(f"Vertex {vertex} not in graph {self}.")
        if tag is None:
            self.__vertex_tags.forwards_remove(vertex)
        else:
            self.__vertex_tags.remove_many(vertex, tags)

    def get_vertex_tags(
        self,
        vertex: VT, /
    ) -> SetView[str] | None:
        """Get the set of tags for the given vertex."""
        return self.__vertex_tags.forwards_get(vertex)

    def get_tagged_vertices(
        self,
        tag: str, /
    ) -> SetView[VT] | None:
        """Get the set of vertices with the given tag."""
        return self.__vertex_tags.backwards_get(tag)

    def tag_edge(
        self,
        start: VT,
        end: VT, /,
        *tags: str
    ) -> None:
        """Tag the given edge."""
        self.__edge_tags.add_many((start, end), tags) 

    def untag_edge(  # pylint: disable=keyword-arg-before-vararg
        self,
        start: VT,
        end: VT,
        tag: str | None = None, /,
        *tags: str,
    ) -> None:
        """Remove the given tag from the given edge."""
        if tag is None:
            self.__edge_tags.forwards_remove((start, end))
        else:
            self.__edge_tags.remove_many((start, end), tags)

    def get_edge_tags(
        self,
        start: VT,
        end: VT, /
    ) -> SetView[str] | None:
        """Get the set of tags for the given edge."""
        return self.__edge_tags.forwards_get((start, end))

    def get_tagged_edges(
        self,
        tag: str, /
    ) -> SetView[tuple[VT, VT]] | None:
        """Get the set of edges with the given tag."""
        return self.__edge_tags.backwards_get(tag)

    def set_vertex_data(
        self,
        vertex: VT,
        name: str,
        value: Any, /
    ) -> None:
        """Set the given data attribute of the given vertex."""
        self.__vertex_data.setdefault(vertex, {})[name] = value

    def update_vertex_data(
        self,
        vertex: VT,
        data: dict[str, Any]
    ) -> None:
        """Update the data attributes of the given vertex from a mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "The data must be a mapping. "
                f"Got; {data!r} of type {type(data)!r}.")
        self.__vertex_data.setdefault(vertex, {}).update(data)

    @overload
    def get_vertex_data(
        self,
        vertex: VT, /, *,
        default: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        ...

    @overload
    def get_vertex_data(
        self,
        vertex: VT, /,
        name: str | None = None, *,
        default: Any | None = None
    ) -> Any | None:
        ...

    def get_vertex_data(
        self,
        vertex: VT, /,
        name: str | None = None, *,
        default: Any | dict[str, Any] | None = None
    ) -> Any | None:
        """
        Get the data attribute(s) of the given vertex.

        If an attribute name is given, return the value of that attribute,
        otherwise return a dictionary of all the attributes of the vertex.
        If the vertex is not in the graph or the vertex does not have the
        given attribute, return the default value.
        """
        if name is None:
            return self.__vertex_data.get(vertex, default)
        return self.__vertex_data[vertex].get(name, default)

    def set_edge_data(
        self,
        vertex: VT,
        adjacent: VT,
        name: str,
        value: Any
    ) -> None:
        """Set the given data attribute of the given edge."""
        self.__edge_data.setdefault((vertex, adjacent), {})[name] = value
        if not self.__directed:
            self.__edge_data.setdefault((adjacent, vertex), {})[name] = value

    def update_edge_data(
        self,
        vertex: VT,
        adjacent: VT,
        data: Mapping[str, Any]
    ) -> None:
        """Update the data attributes of the given edge from a mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "The data must be a mapping. "
                f"Got; {data!r} of type {type(data)!r}.")
        self.__edge_data.setdefault((vertex, adjacent), {}).update(data)

    @overload
    def get_edge_data(
        self,
        vertex: VT,
        adjacent: VT, /, *,
        default: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        ...

    @overload
    def get_edge_data(
        self,
        vertex: VT,
        adjacent: VT, /,
        name: str, *,
        default: Any | None = None
    ) -> Any | None:
        ...

    def get_edge_data(
        self,
        vertex: VT,
        adjacent: VT, /,
        name: str | None = None, *,
        default: Any | dict[str, Any] | None = None
    ) -> Any | None:
        """
        Get the data attribute(s) of the given edge.

        If an attribute name is given, return the value of that attribute,
        otherwise return a dictionary of all the attributes of the edge.
        If the edge is not in the graph or the edge does not have the
        given attribute, return the default value.
        """
        if name is None:
            return self.__edge_data.get((vertex, adjacent), default)
        return self.__edge_data[(vertex, adjacent)].get(name, default)

    def iter_edge_data(
        self,
        edges: Iterable[tuple[VT, VT]] | None = None,
        names: Iterable[str] | None = None
    ) -> Iterator[tuple[tuple[VT, VT], dict[str, Any]]]:
        """
        Yield and iterator over the data of the given edges.

        Parameters
        ----------
        `edges: Iterable[tuple[VT, VT]] | None = None` - The edges to iterate
        over. If `None`, iterate over all the edges in the graph.

        `names: Iterable[str] | None = None` - The names of the data attributes
        to yield. If `None`, yield all the data attributes.

        Yields
        ------
        `tuple[tuple[VT, VT], dict[str, Any]]` - The edge and the data
        attributes associated with it.
        """
        edge_data = self.__edge_data
        if edges is None:
            edges = self.edges
        if names is None:
            for edge in edges:
                yield edge, edge_data[edge]
        else:
            for edge in edges:
                data = {
                    name: edge_data[edge][name]
                    for name in names
                }
                yield edge, data

    def iter_vertex_data(
        self,
        vertices: Iterable[VT],
        names: Iterable[str] | None = None
    ) -> Iterator[tuple[VT, dict[str, Any]]]:
        """Iterate over the data of the given vertices."""
        if names is None:
            for vertex in vertices:
                yield vertex, self.__vertex_data[vertex]
        else:
            for vertex in vertices:
                data = {
                    name: self.__vertex_data[vertex][name]
                    for name in names
                }
                yield vertex, data

    def set_vertex_value(
        self,
        vertex: VT,
        value: WT
    ) -> None:
        """Set the value of the given vertex."""
        self.__vertex_values[vertex] = value

    def get_vertex_value(
        self,
        vertex: VT,
        default: WT = 0.0  # type: ignore
    ) -> WT:
        """Get the value of the given vertex."""
        return self.__vertex_values.get(vertex, default)

    @property
    def edges(self) -> KeysView[tuple[VT, VT]]:
        """The set of edges in the graph."""
        return self.__edge_weights.keys()

    def set_edge_weight(
        self,
        start: VT,
        end: VT,
        weight: WT
    ) -> None:
        """Set the weight of the given edge."""
        self.__edge_weights[(start, end)] = weight
        if not self.__directed:
            self.__edge_weights[(end, start)] = weight

    def get_edge_weight(
        self,
        start: VT,
        end: VT,
        default: WT = 0.0  # type: ignore
    ) -> WT:
        """Get the weight of the given edge."""
        return self.__edge_weights.get((start, end), default)

    def as_disjoint_set(self) -> DisjointSet[VT]:
        """Get the disjoint set representation of this graph."""
        dset = DisjointSet[VT]()
        frontier = set(self)

        while frontier:
            element: VT = frontier.pop()

            connected_to: set[VT] = self[element]
            frontier -= connected_to

            dset.add(element, *connected_to)

        return DisjointSet(self)

    def get_sub_graph(
        self,
        start_vertex: Optional[VT] = None
    ) -> "Graph[VT, WT]":
        """Get the sub-graph containing the given vertex."""
        if start_vertex is None:
            start_vertex = next(iter(self))
        elif start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is "
                             f"not in the graph {self}.")

        frontier: set[VT] = {start_vertex}
        expanded: set[VT] = set()
        sub_graph: Graph[VT, WT] = Graph()

        while frontier:
            vertex: VT = frontier.pop()
            expanded.add(vertex)

            connected_to: set[VT] = self[vertex]
            sub_graph[vertex] = connected_to
            frontier |= (connected_to - expanded)

        return sub_graph

    def get_path(
        self,
        start_vertex: VT,
        end_vertex: VT, /,
        find_cycle: bool = False,
        dfs: bool = False,
        raise_: bool = True
    ) -> list[VT] | None:
        """
        Get a discrete path between the two given vertices (ignoring edge
        weights). By default, finds the shortest path by breadth-first search,
        but if `dfs` is True, then finds the first path by depth-first search
        (which may not be the shortest path).

        If the start and end vertices are the same, then the path will be
        empty, unless `find_cycle` is True, in which case the path will the
        shortest cycle whose start and end vertices are the given vertices.

        Parameters
        ----------
        `start_vertex: VT` - The vertex to start the path from.

        `end_vertex: VT` - The vertex to end the path at.

        `find_cycle: bool = False` - Whether to find a cycle if the start and
        end vertices are the same. Otherwise, the path will be empty.

        `dfs: bool = False` - Whether to use depth-first search instead of
        breadth-first search.

        `raise_: bool` - Whether to raise a ValueError if no path exists.
        Otherwise, None will be returned if no path exists.

        Returns
        -------
        `path: list[VT] | None` - The path between the two vertices, or None
        if no path exists and `raise_` is False.
        """
        if start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is "
                             f"not in the graph {self}.")
        elif end_vertex not in self:
            raise ValueError(f"The vertex {end_vertex} is "
                             f"not in the graph {self}.")

        if (start_vertex == end_vertex
                and not find_cycle):
            return []

        # Loop variables for tracking search;
        #   - `visited` is the set of vertices that have been visited,
        #   - `frontier` is the queue of vertices left to expand,
        #   - `path` is the dictionary of vertices to its adjacent vertex
        #     currently known to the closest to the start vertex.
        visited: set[VT] = {start_vertex}
        frontier: deque[VT] | list[VT]
        if dfs:
            frontier = [start_vertex]
            pop = frontier.pop
        else:
            frontier = deque([start_vertex])
            pop = frontier.popleft  # type: ignore
        path: dict[VT, VT | None] = {start_vertex: None}

        # Whilst there are still vertices to expand;
        while frontier:
            # Pop the first vertex from the frontier (FIFO),
            # then get all the (adjacent) vertices connected to it,
            # and remove those that have already been visited.
            vertex: VT = pop()
            connected_to: set[VT] = self[vertex]
            new_connections: set[VT] = connected_to - visited

            # Visit all new connections;
            for new_vertex in new_connections:
                path[new_vertex] = vertex

                # Path is found when end vertex is visited.
                # Traverse the path back to the start vertex,
                # then reverse it, giving the shortest path
                # from start to end.
                if new_vertex == end_vertex:
                    final_path: list[VT] = [new_vertex]
                    other_vertex: VT | None = vertex
                    while other_vertex is not None:
                        final_path.append(other_vertex)
                        other_vertex = path[other_vertex]
                    final_path.reverse()
                    return final_path

                # Otherwise, add the new vertex to the frontier,
                # ready to be expanded.
                frontier.append(new_vertex)

            # All new connections have been visited.
            visited |= new_connections

        if raise_:
            raise ValueError(f"No path from {start_vertex} to {end_vertex}.")
        return None

    def get_minimal_path(
        self,
        start_vertex: VT,
        end_vertex: VT, /,
        heuristic: Callable[[VT, VT], WT] | None = None,
        weight: Callable[[VT, VT], WT] | str = "weight",
        find_cycle: bool = False,
        greedy: bool = False,
        enable_progress_bar: bool = False,
        raise_: bool = True
    ) -> list[VT] | None:
        """
        Get the minimal weighted path between the two given vertices using the
        A* algorithm. For the shortest discrete path, use `Graph.get_path()`.

        Parameters
        ----------
        `start_vertex: VT` - The vertex to start the path from.

        `end_vertex: VT` - The vertex to end the path at.

        `heuristic: Callable[[VT, VT], WT] | None = None` - The heuristic
        function to use to estimate the distance between two vertices. The
        heuristic function must take two vertices as arguments and return a
        number. If not given or None, then the heuristic will always be 0.0.
        This is equivalent to a uniform-cost search.

        `weight: Callable[[VT, VT], WT] | str = "weight"` - The weight
        function to use to get the weight of an edge. If a string, then
        the edge data for the given key will be used as the weight, unless
        the key is "weight", in which case the edge weight will be used.
        If not given or None, then the edge weight will be used.

        `find_cycle: bool = False` - Whether to find a cycle if the start and
        end vertices are the same. Otherwise, the path will be empty.

        `greedy: bool = False` - Whether to use a greedy search instead of
        A*. This ignores the edge weights and only uses the heuristic during
        search. The path returned is no longer guaranteed to be minimal weight.

        `enable_progress_bar: bool = False` - Whether to enable a progress
        bar to show the progress of the search.

        `raise_: bool = True` - Whether to raise a ValueError if no path
        exists. Otherwise, None will be returned if no path exists.

        Returns
        -------
        `path: list[VT] | None` - The path between the two vertices, or None
        if no path exists and `raise_` is False.

        Example Usage
        -------------
        >>> graph = Graph[str, int]()
        >>> graph.add_edge("A", "B", 1)
        >>> graph.add_edge("B", "C", 2)
        >>> graph.add_edge("A", "C", 5)
        # Shortest path is Z -> C, with a cost of 5.
        >>> graph.get_path("A", "C")
        ["A", "C"]
        # Minimal weighted path is A -> B -> C, with a cost of 3.
        # (no heuristic used here, equivalent to uniform-cost search).
        >>> graph.get_minimal_path("A", "C")
        ["A", "B", "C"]

        # TODO: set so can be initially disconnected
        >>> graph = Graph.grid_graph(dim=(3, 3), connect_diagonally=True)
        >>> graph.get_minimal_path((0, 0), (2, 2),
                heuristic=Graph.manhattan_distance)
        [(0, 0), (1, 1), (2, 2)]

        >>> graph = Graph.world_map_graph(
        ...     names=("London", "Dublin", "Paris", "Berlin", "Rome", "Madrid"),
        ...     coords=(
        ...         (51.5074, 0.1278), (53.3498, 6.2603), (48.8566, 2.3522),
        ...         (52.5200, 13.4050), (41.9028, 12.4964), (40.4168, 3.7038)
        ...     ),
        ...     # Calculate all edge weights using Euclidean distance.
        ...     set_weights=Graph.euclidean_distance(data="coords")
        ... )
        >>> # Calculate heuristic between any two (possibly non-adjacent)
        >>> # vertices using Euclidean distance.
        >>> graph.get_minimal_path(
        ...     "London", "Rome",
        ...     heuristic=Graph.euclidean_distance(data="coords")
        ... )
        ['London', 'Paris', 'Rome']
        """
        if start_vertex not in self:
            raise ValueError(f"Vertex {start_vertex} is not in {self}.")
        elif end_vertex not in self:
            raise ValueError(f"Vertex {end_vertex} is not in {self}.")

        if (start_vertex == end_vertex
                and not find_cycle):
            return []

        if heuristic is None:
            def heuristic_(v1: VT, v2: VT) -> WT:
                return 0.0
        else:
            heuristic_ = heuristic

        if isinstance(weight, str):
            if weight == "weight":
                def weight_(v1: VT, v2: VT) -> WT:
                    return self.get_edge_weight(v1, v2, 0.0)
            else:
                def weight_(v1: VT, v2: VT) -> WT:
                    return self.get_edge_data(v1, v2, weight)

        counter = itertools.count()
        frontier = PriorityQueue[
            tuple[VT, VT | None],
            tuple[float, float, int]
        ](
            ((start_vertex, None),       # (vertex, parent)
             (0.0, 0.0, next(counter)))  # (heuristic value, backwards cost, n)
        )
        expanded: dict[VT, tuple[WT, WT]] = {}
        path: dict[VT, VT | None] = {}
        vertices_visited: int = 0
        vertices_expanded: int = 0

        if enable_progress_bar:
            progress_bar = ResourceProgressBar(total=len(self))

        while frontier:
            item, prio = frontier.pop_prio()
            _, cost, _ = prio
            vertex, parent = item

            vertices_visited += 1
            if vertex == end_vertex:
                final_path = [vertex]
                other_vertex = parent
                while other_vertex is not None:
                    final_path.append(other_vertex)
                    other_vertex = path[other_vertex]
                final_path.reverse()
                return final_path

            if vertex in path and path[vertex] is None:
                continue

            path[vertex] = parent

            for adjacent in self[vertex]:
                adjacent_cost = cost + weight_(vertex, adjacent)
                if adjacent not in expanded:
                    if greedy:
                        adjacent_hvalue = heuristic_(adjacent, end_vertex)
                    else:
                        adjacent_hvalue = (
                            adjacent_cost + heuristic_(adjacent, end_vertex)
                        )
                    expanded[adjacent] = (adjacent_hvalue, adjacent_cost)
                    frontier.push(
                        (adjacent, vertex),
                        (adjacent_hvalue, adjacent_cost, next(counter))
                    )
                    vertices_expanded += 1
                else:
                    adjacent_hvalue, existing_cost = expanded[adjacent]
                    if adjacent_cost < existing_cost:
                        adjacent_hvalue = (
                            (adjacent_hvalue - existing_cost) + adjacent_cost)
                        frontier.push(
                            (adjacent, vertex),
                            (adjacent_hvalue, adjacent_cost, next(counter))
                        )
                        vertices_expanded += 1

            if enable_progress_bar:
                progress_bar.update(
                    vertices_visited - progress_bar.n,
                    data={
                        "frontier": len(frontier),
                        "visited": vertices_visited,
                        "expanded": vertices_expanded
                    }
                )

        if raise_:
            raise ValueError(f"No path between {start_vertex} and {end_vertex}.")
        return None

    def minimum_spanning_tree(self) -> "Graph[VT]":
        """Get the minimum spanning tree of the graph."""
        mst = Graph[VT](directed=False)
        frontier = set(self)
        edges = SortedQueue(
            *self.edges,
            key=lambda x: self.get_edge_weight(*x)
        )

        while edges:
            start_vertex, end_vertex = edges.pop()

            if start_vertex not in mst or end_vertex not in mst:
                mst.add_edge(start_vertex, end_vertex)

                frontier.discard(start_vertex)
                frontier.discard(end_vertex)
                if not frontier:
                    break

        return mst

    # def hamiltonian_path(self) -> list[VT] | None:
    #     """Get a Hamiltonian path of the graph."""
    #     if not self.__directed:
    #         return None

    #     if len(self) == 1:
    #         return list(self)

    #     for vertex in self:
    #         path = self.get_path(vertex, vertex, find_cycle=True)
    #         if path is not None:
    #             return path

    #     return None

    # def hamiltonian_cycle(self) -> list[VT] | None:
    #     """Get a Hamiltonian cycle of the graph."""
    #     if not self.__directed:
    #         return None

    #     if len(self) == 1:
    #         return list(self)

    #     for vertex in self:
    #         path = self.get_path(vertex, vertex, find_cycle=True)
    #         if path is not None:
    #             return path

    #     return None

    def contains_cycle(self, start_vertex: Optional[VT] = None) -> bool:
        """Determine if the graph contains a cycle."""
        if not self.__directed:
            return True

        if start_vertex is None:
            start_vertex = next(iter(self))
        elif start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is not in {self!s}.")

        frontier: set[VT] = {start_vertex}
        expanded: set[VT] = set()

        while frontier:
            vertex: VT = frontier.pop()
            expanded.add(vertex)

            connected_to: set[VT] = self[vertex]
            if connected_to & expanded:
                return True

            # No need to take difference with expanded since
            # the intersection with connected_to must be empty.
            frontier |= connected_to

        return False

    def is_connected(self, start_node: VT | None = None) -> bool:
        """
        Determine if this undirected graph is connected.

        A graph is connected, if there is a path from all vertices to all
        other vertices, where a path may be formed by an arbitrary length
        sequence of edges, i.e. a graph is connected if every vertex is
        reachable from all others.

        Parameters
        ----------
        `start_node: VT | None` - A vertex, from which to start searching.
        Intuitively, a vertex is connected to an undirected graph iff;
            - It is the start node,
            - It is connected to the start node via some path.

        Returns
        -------
        `bool` - A Boolean, True if the graph is connected, False otherwise.
        """
        # Variables for tracking progress of the search;
        #       - The set of vertices that have been found to be connected to
        #         the graph,
        #       - The frontier is the set of nodes that have been found to be
        #         connected to the graph, but have not yet been expanded,
        #         where expansion of a vertex is the process of marking the
        #         vertices it directly connects (via a single edge) as
        #         connected.
        connected_nodes: set[VT] = set()
        frontier: set[VT] = set()

        # Start from the given vertex if it was given, otherwise start from
        # any vertex.
        _start_node: VT = default_get(start_node, next(iter(self)))
        connected_nodes.add(_start_node)

        new_nodes: set[VT] = self[_start_node].difference(connected_nodes)
        connected_nodes.update(new_nodes)
        frontier.update(new_nodes)

        # Whilst there are nodes to search, and we have not already found a
        # connection to all nodes;
        while (frontier and len(self) != len(connected_nodes)):

            # Get and remove a node from the frontier randomly
            # (order in which they are visited does not matter)
            vertex: VT = frontier.pop()

            # Get all the nodes adjacent to this node that are not already
            # connected to the graph as a whole.
            new_nodes = self[vertex].difference(connected_nodes)

            # Mark those nodes as connected, and add them to the frontier.
            connected_nodes.update(new_nodes)
            frontier.update(new_nodes)

        # The graph is connected, if all vertices are in the connected set.
        return len(self) == len(connected_nodes)


if __name__ == "__main__":
    import random
    graph = Graph[int, float](directed=False)
    for i in range(1000):
        for j in range(1000):
            graph.add_edge(i, j, random.random())
    path = graph.get_minimal_path(1, 999, enable_progress_bar=True)
    print(path)
