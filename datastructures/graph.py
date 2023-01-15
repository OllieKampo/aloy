from collections import deque
import dataclasses
import enum
from random import randint
from typing import Generic, Hashable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, overload

import _collections_abc

from auxiliary.getters import default_get
from datastructures.disjointset import DisjointSet
# from Systems.Errors import Jinx_AttemptToModifyImmutableObjectError
from Geometry.Vectors import CT, Vector

## https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)
## https://en.wikipedia.org/wiki/Directed_graph
## https://docs.python.org/3/library/graphlib.html
## https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_map_structure

IT = TypeVar("IT", bound=Hashable)
WT = TypeVar("WT", bound=Hashable)

@enum.unique
class Distance(enum.Enum):
    Euclidean = "Euclidean Distance"
    Manhatten = "Manhatten Distance"

#################################################################################################
##### Vertices and Edges

@dataclasses.dataclass(frozen=True)
class Vertex(Generic[IT]):
    id: IT
    
    @property
    def as_tuple(self) -> tuple[IT]:
        return dataclasses.astuple(self)

class ValueVertex(Vertex[IT],
                  Generic[IT, WT]):
    val: WT
    
    @property
    def as_tuple(self) -> tuple[IT, WT]:
        return dataclasses.astuple(self)

class CoordinateVertex(Vertex[IT],
                       Generic[IT, CT]):
    at: Vector[CT]
    
    @property
    def as_tuple(self) -> tuple[IT, CT]:
        return dataclasses.astuple(self)
    
    @property
    def dimensions(self) -> int:
        return len(self.at)
    
    def euclidean_distance(self, from_: Optional[Vector[CT]] = None) -> float:
        if from_ is None:
            return self.at.magnitude
        return (self.at - from_).magnitude

class CoordinateValueVertex(CoordinateVertex[IT, CT],
                            ValueVertex[IT, WT],
                            Generic[IT, CT, WT]):
    """
    A coordinate value vertex, is a vertex defined by;
        - an identifier,
        - a position vector (its coordinates),
        - a value.
    
    Example Usage:
    --------------
    ```
    from Jinx.Graph import CoordinateValueVertex
    
    ## Declare a valued coordinate vertex with;
    ##  - string names,
    ##  - an integer valued position vector,
    ##  - and floating point node values.
    vertex: CoordinateValueVertex[str, int, float]
    
    ## Define such a vertex in a 2-dimensional vector space.
    vertex = CoordinateValueVertex(id="a", at=Vector2D(x=1, y=1), val=0.5)
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
    def as_tuple(self) -> tuple[VT]:
        return dataclasses.astuple(self)

#################################################################################################
##### Graph data structures

class Graph(_collections_abc.MutableMapping, Generic[VT]):
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
    
    _slots__ = ("__dict",
                "__directed",
                "__allow_loops")
    
    @overload
    def __init__(self,
                 graph: Mapping[VT, VT | set[VT]],
                 directed: bool = False,
                 allow_loops: bool = False
                 ) -> None:
        "Create a graph from a mapping, from a vertex set, to their adjacent vertices."
        ...
    
    @overload
    def __init__(self,
                 edge_set: Iterable[tuple[VT, VT | set[VT]]],
                 directed: bool = False,
                 allow_loops: bool = False
                 ) -> None:
        """
        Create a graph from an an iterable of tuples specifying edges between vertices given
        by the first tuple element and the (set of) vertice(s) in the second element(s).
        """
        ...
    
    def __init__(self,
                 graph_or_edge_set: Mapping[VT, VT | set[VT]] | Iterable[tuple[VT, VT | set[VT]]] = [],
                 directed: bool = False,
                 allow_loops: bool = False
                 ) -> None:
        
        self.__dict: dict[VT, set[VT]] = {}
        self.__directed: bool = directed
        self.__allow_loops: bool = allow_loops
        
        items: Iterable[tuple[VT, VT | set[VT]]]
        if isinstance(graph_or_edge_set, Mapping):
            items = graph_or_edge_set.items()
        else: items = graph_or_edge_set
        
        for node, connections in items:
            self[node] = connections
    
    @classmethod
    def from_vertices(cls, vertices: Iterable[VT], allow_loops: bool = False) -> "Graph":
        return cls(dict.fromkeys(vertices), allow_loops)
    
    def __str__(self) -> str:
        return str(self.__dict)
    
    def __repr__(self) -> str:
        return f"Graph({repr(self.__dict)}, directed={self.__directed}, allow_loops={self.__allow_loops})"
    
    def __getitem__(self,
                    vertex: VT
                    ) -> set[VT]:
        "Get the adjacent vertices to the given vertex."
        return self.__dict[vertex]
    
    def __setitem__(self,
                    vertex: VT,
                    connections: Optional[VT | set[VT] | frozenset[VT]]
                    ) -> None:
        """
        Add connections between a given vertex, and another vertex (or set of vertices).
        If any of the vertices are not in the graph, add them to the graph as well.
        """
        ## If connections is None then the vertex is disconnected from the graph.
        if connections is None:
            self.__dict.setdefault(vertex, set())
            return
        
        ## Convert the connections to a set.
        _connections: set[VT]
        if not isinstance(connections, (set, frozenset)):
            _connections = {connections}
        else: _connections = connections
        
        ## If loops are not allowed, ensure the vertex is not in the set of connections.
        if not self.__allow_loops:
            if isinstance(_connections, frozenset):
                _connections -= {vertex}
            else: _connections.discard(vertex)
        
        ## Find the set of existing connections from the specified vertex, and the set
        ## of new connections (those in the set to add that are not already in the existing set).
        existing_connections: set[VT] = self.__dict.setdefault(vertex, set())
        new_connections: set[VT] = _connections.difference(existing_connections)
        
        ## Add the new connections both to and from the specified node
        existing_connections.update(new_connections)
        if not self.__directed:
            for connected_vertex in new_connections:
                self.__dict.setdefault(connected_vertex, set()).add(vertex)
    
    def __delitem__(self,
                    vertex: VT
                    ) -> None:
        "Remove a vertex and all its connections from the graph."
        ## Get all other nodes connected to the specified node, removing any loops
        connected_vertices: set[VT] = self.__dict[vertex]
        connected_vertices.discard(vertex)
        
        ## Remove all the connections from the other vertices to the specified vertex.
        for connected_vertex in connected_vertices:
            self.__dict[connected_vertex].discard(vertex)
        
        ## Remove the specified vertex and all its connections to the other vertices.
        del self.__dict[vertex]
    
    def __iter__(self) -> Iterator[VT]:
        "Iterate over the vertex set of this graph."
        yield from self.__dict
    
    def __len__(self) -> int:
        "The number of vertices in the graph."
        return len(self.__dict)
    
    def vertex_set(self) -> set[VT]:
        "The set of vertices in the graph."
        return set(self.keys())
    
    def edge_set(self) -> set[Edge[VT]]:
        "The set of edges in the graph."
        return set(Edge(vertex, adjacent) for vertex in self for adjacent in self[vertex])
    
    def as_disjoint_set(self) -> DisjointSet[VT]:
        dset = DisjointSet()
        frontier = self.vertex_set()
        
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
                 end_vertex: VT,
                 raise_: bool = True
                 ) -> list[VT] | None:
        """Get the shortest path between the given vertices."""
        if start_vertex not in self:
            raise ValueError(f"The vertex {start_vertex} is not in the graph {self}.")
        elif end_vertex not in self:
            raise ValueError(f"The vertex {end_vertex} is not in the graph {self}.")
        
        Path = type("Path", (), {"__init__": lambda self, vertex: self.path.append(vertex),
                                 "append": lambda self, vertex: self.path.append(vertex),
                                 "path": list(), "__hash__": lambda self: hash(self.path[-1])})
        
        frontier: deque[VT] = deque([[start_vertex]])
        visited: set[VT] = set()
        
        while frontier:
            plan: list[VT] = frontier.popleft()
            vertex: VT = plan[-1]
            
            connected_to: set[VT] = self[vertex]
            new_connections: set[VT] = (connected_to - visited)
            
            for new_vertex in new_connections:
                new_plan = [*plan, new_vertex]
                if new_vertex == end_vertex:
                    return new_plan
                frontier.append(new_plan)
            
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
        
        ## Start from the given vertex if it was given, otherwise start from a random vertex.
        _start_node: VT = default_get(start_node, self.random_vertex())
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

# class FrozenGraph(_collections_abc.Mapping, Generic[VT]):
#     """
#     An immutable and therefore hashable version of an undirected graph.
#     """
    
#     __slots__ = ("__hash")
    
#     def __init__(self,
#                 graph: dict[VT, VT | set[VT]],
#                 allow_loops: bool = False
#                 ) -> None:
#         super().__init__({}, allow_loops)
#         for node, connections in graph.items():
#             super().__setitem__(node, connections)
#         self.__hash: Optional[int] = None
    
#     def __setitem__(self, node: VT, connections: Optional[VT | set[VT]]) -> None:
#         raise Jinx_AttemptToModifyImmutableObjectError(self.__class__)
    
#     def __delitem__(self, node: VT) -> None:
#         raise Jinx_AttemptToModifyImmutableObjectError(self.__class__)
    
#     def __hash__(self) -> int:
#         if self.__hash is None:
#             hash_: int = 0
#             for key, value in self.__dict.items():
#                 hash_ ^= hash((key, value))
#             self.__hash = hash_
#         return self.__hash

# ## Graph where the arcs have weights
# class WeightedGraph(Graph[ValueVertex[IT, WT]]):
#     pass

# ## A weighted graph where the arc weights are calculated based on the relative positions of nodes
# class DistanceGraph(WeightedGraph[CoordinateVertex[IT, CT]]):
#     pass
