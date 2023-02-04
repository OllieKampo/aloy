from collections import deque
import dataclasses
import enum
import itertools
from typing import Any, Generic, Hashable, Iterable, Iterator, KeysView, Mapping, Optional, TypeVar, overload

import collections.abc
import numpy as np

from auxiliary.getters import default_get
from datastructures.disjointset import DisjointSet

## https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_map_structure

CT = TypeVar("CT", bound=Hashable)
IT = TypeVar("IT", bound=Hashable)
WT = TypeVar("WT", bound=Hashable)

@enum.unique
class DistanceMeasure(enum.Enum):
    Euclidean = "Euclidean Distance"
    Manhatten = "Manhatten Distance"
    Chebyshev = "Chebyshev Distance"

@dataclasses.dataclass(frozen=True)
class Vertex(Generic[IT]):
    """
    Class defining a graph vertex.

    A standard vertex is defined simply by an identifier.
    The hash of the vertex is the hash of the identifier.

    Fields
    ------
    `id : IT` - The vertex identifier.
    """
    id: IT

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def as_tuple(self) -> tuple[IT]:
        return dataclasses.astuple(self)

@dataclasses.dataclass(frozen=True)
class ValVertex(Vertex[IT],
                  Generic[IT, WT]):
    """
    A class defining a value vertex.

    A value vertex is defined by an identifier and a value.

    Fields
    ------
    `id : IT` - The vertex identifier.

    `val : WT` - The vertex value.
    """
    val: WT = 0
    
    @property
    def as_tuple(self) -> tuple[IT, WT]:
        return dataclasses.astuple(self)

@dataclasses.dataclass(frozen=True)
class PosVertex(Vertex[IT],
                       Generic[IT, CT]):
    """
    Class defining a position vertex.

    A position vertex is defined by an identifier and a position.

    Fields
    ------
    `id : IT` - The vertex identifier.

    `pos : numpy.ndarray[CT]` - The vertex position.
    Accepts any iterable as input, but will be converted to a numpy array.

    Example Usage
    -------------
    ```
    from jinx.graph import PosVertex

    ## Define such a vector in a 2-dimensional vector space.
    >>> vertex = PosVertex(id="a", pos=(1, 2))
    >>> vertex
    PosVertex(id='a', pos=array([1., 2.]))
    """

    pos: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    def __post_init__(self) -> None:
        """Converts the position to a numpy array if it is not already one."""
        if not isinstance(self.at, np.ndarray):
            object.__setattr__(self, "at", np.array(self.at))
    
    @property
    def as_tuple(self) -> tuple[IT, CT]:
        return dataclasses.astuple(self)
    
    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions of the coordinate vertex."""
        return self.at.shape[0]
    
    @property
    def magnitude(self) -> float:
        """Returns the magnitude of the coordinate vertex."""
        return np.linalg.norm(self.at)
    
    def distance(self, other: "PosVertex[IT, CT]", measure: DistanceMeasure = DistanceMeasure.Euclidean) -> float:
        """Returns the distance between this vertex and another vertex."""
        if self.dimensions != other.dimensions:
            raise ValueError("Cannot measure distance between vertices in different dimensional vector spaces.")
        if measure == DistanceMeasure.Euclidean:
            return np.linalg.norm(self.at - other.at)
        elif measure == DistanceMeasure.Manhatten:
            return np.sum(np.abs(self.at - other.at))
        elif measure == DistanceMeasure.Chebyshev:
            return np.max(np.abs(self.at - other.at))
        else:
            raise ValueError(f"Unknown distance measure: {measure}")

@dataclasses.dataclass(frozen=True)
class PosValVertex(PosVertex[IT, CT],
                   ValVertex[IT, WT],
                   Generic[IT, CT, WT]):
    """
    Class defining a position value vertex.

    A position value vertex is defined by an identifier, a position, and a value.
    
    Fields
    ------
    `id : IT` - The vertex identifier.

    `val : WT` - The vertex value.

    `pos : numpy.ndarray[CT]` - The vertex position.
    Accepts any iterable as input, but will be converted to a numpy array.

    Example Usage:
    --------------
    ```
    from jinx.graph import PosValVertex
    
    ## Define such a vector in a 2-dimensional vector space.
    >>> vertex = PosValVertex(id="a", val=0.5, at=(1, 2))
    >>> vertex
    PosValVertex(id='a', val=0.5, pos=array([1., 2.]))
    ```
    """
    
    @property
    def as_tuple(self) -> tuple[IT, CT, WT]:
        return dataclasses.astuple(self)

VT = TypeVar("VT", bound=Hashable, covariant=True)

@dataclasses.dataclass(frozen=True)
class Edge(Generic[VT]):
    start: VT
    end: VT
    
    @property
    def as_tuple(self) -> tuple[VT, VT]:
        return dataclasses.astuple(self)

@dataclasses.dataclass(frozen=True)
class WeightedEdge(Edge[VT], Generic[VT, WT]):
    weight: WT
    
    @property
    def as_tuple(self) -> tuple[VT, VT, WT]:
        return dataclasses.astuple(self)

class Graph(collections.abc.MutableMapping, Generic[VT]):
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
    >>> graph: Graph[int]
    >>> graph = Graph({1 : {2, 3},
    ...                2 : 4})
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
    
    __slots__ = {"__adjacency_mapping" : "The dictionary containing the graph's adjacency mapping.",
                 "__vertex_values" : "The dictionary containing the graph's vertex values.",
                 "__edge_weights" : "The dictionary containing the graph's edge weights.",
                 "__vertex_data" : "The data attributes of the vertices in the graph.",
                 "__edge_data" : "The data attributes of the edges in the graph.",
                 "__directed" : "Whether the graph is directed or not.",
                 "__allow_loops" : "Whether the graph allows loops or not."}
    
    def __init__(self,
                 graph: Mapping[VT, VT | set[VT] | None] | None = None, *,
                 vertices: Iterable[VT] | None = None,
                 edges: Iterable[tuple[VT, VT | set[VT] | None]] | None = None,
                 vertex_data: Mapping[VT, dict[str, Any]] = {},
                 edge_data: Mapping[tuple[VT, VT], dict[str, Any]] = {},
                 directed: bool = False,
                 allow_loops: bool = False
                 ) -> None:
        """
        Create a new graph.

        Parameters
        ----------
        `graph : Mapping[VT, VT | set[VT] | None] | None` - A graph-like mapping.
        A key is a vertex, and value is either;
        one adjacent vertex, set of adjacent vertices, or None.
        If None, the key vertex is disconnected.

        `vertices : Iterable[VT] | None` - An iterable of vertices.
        All these vertices will be disconnected.

        `edges : Iterable[tuple[VT, VT | set[VT] | None]] | None` - An iterable of edge tuples.
        Each tuple is a pair of vertices, where the first is the start vertex and second is either;
        one adjacent vertex, set of adjacent vertices, or None.
        If None, the start vertex is disconnected.

        `vertex_data : Mapping[VT, dict[str, Any]]` - A mapping of vertex data.
        Each key is a vertex, and the value is a data attribute dictionary.
        The data attribute dictionary maps of attribute names to arbitrary values.

        `edge_data : Mapping[tuple[VT, VT], dict[str, Any]]` - A mapping of edge data.
        Each key is an edge tuple, and the value is a data attribute dictionary.
        The data attribute dictionary maps of attribute names to arbitrary values.

        `directed : bool` - Whether the graph is directed or not.
        If True, the graph is directed, and the adjacency mapping is not symmetric.
        Otherwise, the graph is undirected, and the adjacency mapping is symmetric.

        `allow_loops : bool` - Whether the graph allows loops or not.
        If True, the graph allows loops, and a vertex can be adjacent to itself.
        """
        self.__adjacency_mapping: dict[VT, set[VT]] = {}
        for node, connections in itertools.chain(graph.items(), zip(vertices, itertools.repeat(None)), edges):
            self[node] = connections
        self.__edge_weights: dict[tuple[VT, VT], WT] = {}
        
        self.__vertex_data: dict[VT, dict[str, Any]] = {}
        for vertex, data in vertex_data.items():
            self.update_vertex_data(vertex, data)
        self.__edge_data: dict[tuple[VT, VT], dict[str, Any]] = {}
        for edge, data in edge_data.items():
            self.update_edge_data(edge, data)
        
        self.__directed: bool = directed
        self.__allow_loops: bool = allow_loops
    
    def __str__(self) -> str:
        return str(self.__adjacency_mapping)
    
    def __repr__(self) -> str:
        return f"Graph(graph={self.__adjacency_mapping!r}, directed={self.__directed}, allow_loops={self.__allow_loops})"
    
    def __getitem__(self,
                    vertex: VT
                    ) -> set[VT]:
        """Get the adjacent vertices to the given vertex."""
        return self.__adjacency_mapping[vertex]
    
    def __setitem__(self,
                    vertex: VT,
                    connections: VT | set[VT] | frozenset[VT] = None
                    ) -> None:
        """
        Add connections between a given vertex, and another vertex (or set of vertices).
        If any of the vertices are not in the graph, add them to the graph as well.
        """
        ## If connections is None then the vertex is disconnected from the graph.
        if connections is None:
            if vertex not in self.__adjacency_mapping:
                self.__adjacency_mapping[vertex] = set()
            return
        
        ## Convert the connections to a set.
        _connections: set[VT]
        if not isinstance(connections, (set, frozenset)):
            _connections = {connections}
        else: _connections = connections
        
        ## If loops are not allowed, ensure the vertex is not in the set of connections.
        if not self.__allow_loops:
            if isinstance(_connections, frozenset):
                _connections = _connections - {vertex}
            else: _connections.discard(vertex)
        
        ## Find the set of existing connections from the specified vertex, and the set
        ## of new connections (those in the set to add that are not already in the existing set).
        existing_connections: set[VT] = self.__adjacency_mapping.setdefault(vertex, set())
        new_connections: set[VT] = _connections.difference(existing_connections)
        
        ## Add the new connections both to and from the specified node
        existing_connections.update(new_connections)
        if not self.__directed:
            for connected_vertex in new_connections:
                self.__adjacency_mapping.setdefault(connected_vertex, set()).add(vertex)
    
    def __delitem__(self,
                    vertex_or_edge: VT | tuple[VT, VT]
                    ) -> None:
        """Remove a vertex or an edge from the graph."""
        if isinstance(vertex_or_edge, tuple):
            ## If the specified item is an edge, remove the connection from the first vertex to the second.
            vertex, connected_vertex = vertex_or_edge
            self.__adjacency_mapping[vertex].discard(connected_vertex)
            if not self.__directed:
                self.__adjacency_mapping[connected_vertex].discard(vertex)
        else:
            ## Get all other nodes connected to the specified node, removing any loops
            vertex = vertex_or_edge
            connected_vertices: set[VT] = self.__adjacency_mapping[vertex]
            connected_vertices.discard(vertex)
            
            ## Remove all the connections from the other vertices to the specified vertex.
            for connected_vertex in connected_vertices:
                self.__adjacency_mapping[connected_vertex].discard(vertex)
            
            ## Remove the specified vertex and all its connections to the other vertices.
            del self.__adjacency_mapping[vertex]
    
    def __iter__(self) -> Iterator[VT]:
        "Iterate over the vertex set of this graph."
        yield from self.__adjacency_mapping
    
    def __len__(self) -> int:
        "The number of vertices in the graph."
        return len(self.__adjacency_mapping)
    
    def __contains__(self,
                     vertex: VT
                     ) -> bool:
        "Check if the given vertex is in the graph."
        return vertex in self.__adjacency_mapping
    
    def add_vertex(self,
                   vertex: VT, /,
                   connections: VT | set[VT] | frozenset[VT] = None,
                   **data: Any
                   ) -> None:
        """Add a vertex to the graph."""
        self[vertex] = connections
        self.update_vertex_data(vertex, data)
    
    def add_edge(self,
                 start: VT,
                 end: VT, /,
                 weight: float | None = None,
                 **data: Any
                 ) -> None:
        """Add an edge to the graph."""
        self[start] = end
        if weight is not None:
            self.__edge_weights[(start, end)] = weight
        self.update_edge_data((start, end), data)
    
    def set_vertex_data(self,
                        vertex: VT,
                        name: str,
                        value: Any
                        ) -> None:
        """Set the given data attribute of the given vertex."""
        self.__vertex_data.setdefault(vertex, {})[name] = value
    
    def update_vertex_data(self,
                           vertex: VT,
                           data: dict[str, Any]
                           ) -> None:
        """Update the data attributes of the given vertex from a mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(f"The data must be a mapping. Got; {data!r} of type {type(data)}.")
        self.__vertex_data.setdefault(vertex, {}).update(data)
    
    def get_vertex_data(self,
                        vertex: VT,
                        name: str,
                        default: Any | None = None
                        ) -> Any | None:
        """Get the given data attribute of the given vertex."""
        return self.__vertex_data[vertex].get(name, default)
    
    def set_edge_data(self,
                      vertex: VT,
                      adjacent: VT,
                      name: str,
                      value: Any
                      ) -> None:
        """Set the given data attribute of the given edge."""
        self.__edge_data.setdefault((vertex, adjacent), {})[name] = value
    
    def update_edge_data(self,
                         vertex: VT,
                         adjacent: VT,
                         data: Mapping[str, Any]
                         ) -> None:
        """Update the data attributes of the given edge from a mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(f"The data must be a mapping. Got; {data!r} of type {type(data)}.")
        self.__edge_data.setdefault((vertex, adjacent), {}).update(data)
    
    def get_edge_data(self,
                      vertex: VT,
                      adjacent: VT,
                      name: str,
                      default: Any | None = None
                      ) -> Any | None:
        """Get the given data attribute of the given edge."""
        return self.__edge_data[(vertex, adjacent)].get(name, default)
    
    def edges(self) -> KeysView[Edge[VT]]:
        """The set of edges in the graph."""
        return self.__edge_weights.keys()

    def set_edge_weight(self,
                        start: VT,
                        end: VT,
                        weight: float
                        ) -> None:
        """Set the weight of the given edge."""
        self.__edge_weights[(start, end)] = weight
    
    def get_edge_weight(self,
                        start: VT,
                        end: VT
                        ) -> float | None:
        """Get the weight of the given edge."""
        return self.__edge_weights[(start, end)]
    
    def as_disjoint_set(self) -> DisjointSet[VT]:
        """Get the disjoint set representation of this graph."""
        dset = DisjointSet()
        frontier = set(self)
        
        while frontier:
            element: VT = frontier.pop()
            
            connected_to: frozenset[VT] = self[element]
            frontier -= connected_to
            
            dset.add(element, connected_to)
        
        return DisjointSet(self)
    
    def get_sub_graph(self,
                      start_vertex: Optional[VT] = None
                      ) -> "Graph[VT]":
        """Get the sub-graph containing the given vertex."""
        if start_vertex is None:
            start_vertex = next(iter(self))
        elif start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is not in the graph {self}.")
        
        frontier: set[VT] = {start_vertex}
        expanded: set[VT] = set()
        sub_graph: Graph[VT] = Graph()
        
        while frontier:
            vertex: VT = frontier.pop()
            expanded.add(vertex)
            
            connected_to: set[VT] = self[vertex]
            sub_graph[vertex] = connected_to
            frontier |= (connected_to - expanded)
        
        return sub_graph
    
    def get_path(self,
                 start_vertex: VT,
                 end_vertex: VT, /,
                 find_cycle: bool = False,
                 raise_: bool = True
                 ) -> list[VT] | None:
        """
        Get the shortest path between the two given vertices.
        
        If the start and end vertices are the same, then the path will be empty,
        unless `find_cycle` is True, in which case the path will the shortest cycle
        whose start and end vertices are the given vertices.

        If `raise_` is True, then a ValueError will be raised if no path exists.
        Otherwise, if `raise_` is False, then None will be returned if no path exists.
        """
        if start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is not in the graph {self}.")
        elif end_vertex not in self:
            raise ValueError(f"The vertex {end_vertex} is not in the graph {self}.")
        
        if (start_vertex == end_vertex
            and not find_cycle):
            return []
        
        frontier: deque[VT] = deque([start_vertex])
        path: dict[VT, VT | None] = {start_vertex : None}
        visited: set[VT] = set()
        
        while frontier:
            vertex: VT = frontier.popleft()
            
            connected_to: set[VT] = self[vertex]
            new_connections: set[VT] = (connected_to - visited)
            
            for new_vertex in new_connections:
                path[new_vertex] = vertex

                if new_vertex == end_vertex:
                    final_path = [new_vertex]
                    node = vertex
                    while node is not None:
                        final_path.append(node)
                        node = path[node]
                    final_path.reverse()
                    return final_path
                
                frontier.append(new_vertex)
            
            visited |= new_connections
        
        if raise_:
            raise ValueError(f"No path between {start_vertex} and {end_vertex}.")
        return None
    
    def contains_cycle(self, start_vertex: Optional[VT] = None) -> bool:
        """Determine if the graph contains a cycle."""
        if not self.__directed:
            return True
        
        if start_vertex is None:
            start_vertex = next(iter(self))
        elif start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is not in the graph {self}.")
        
        frontier: set[VT] = {start_vertex}
        expanded: set[VT] = set()
        
        while frontier:
            vertex: VT = frontier.pop()
            expanded.add(vertex)
            
            connected_to: set[VT] = self[vertex]
            if connected_to & expanded:
                return True
            ## No need to take difference with expanded since
            ## the intersection with connected_to must be empty.
            frontier |= connected_to
        
        return False
    
    def is_connected(self, start_node: Optional[VT] = None) -> bool:
        """
        Determine if this undirected graph is connected.
        
        A graph is connected, if there is a path from all vertices to all other vertices,
        where are path may be formed by an arbitrary length sequence of edges.
        In other words, a graph is connected if every vertex is reachable from all other vertices.
        
        Parameters
        ----------
        `start_node: VT | None` - A vertex, from which to start searching.
        Intuitively, a vertex is connected to an undirected graph iff;
            - It is the start node,
            - It is connected to the start node via an arbitrary length path.
        
        Returns
        -------
        `bool` - A Boolean, True if the graph is connected, False otherwise.
        """
        ## Variables for tracking progress of the search;
        ##      - The set of vertices that have been found to be connected to the graph,
        ##      - The frontier is the set of nodes that have been found to be connected
        ##        to the graph, but have not yet been expanded, where expansion of a vertex
        ##        is the process of marking the vertices it directly connects (via a single edge) as connected.
        connected_nodes: set[VT] = set()
        frontier: set[VT] = set()
        
        ## Start from the given vertex if it was given, otherwise start from any vertex.
        _start_node: VT = default_get(start_node, next(iter(self)))
        connected_nodes.add(_start_node)
        
        new_nodes: set[VT] = self[_start_node].difference(connected_nodes)
        connected_nodes.update(new_nodes)
        frontier.update(new_nodes)
        
        ## Whilst there are nodes to search,
        ## and we have not already found a connection to all nodes...
        while (frontier and not len(self) == len(connected_nodes)):
            
            ## Get and remove a node from the frontier randomly
            ## (order in which they are visited does not matter)
            vertex: VT = frontier.pop()
            
            ## Get all the nodes adjacent to this node that are not already connected to the graph as a whole,
            ## we are essentially looking to see if this node is adjacent to any node that is not connected yet.
            new_nodes = self[vertex].difference(connected_nodes)
            
            ## Mark those nodes as connected, and add them to the frontier.
            connected_nodes.update(new_nodes)
            frontier.update(new_nodes)
        
        ## The graph is connected, if all vertices are in the connected set.
        return len(self) == len(connected_nodes)
