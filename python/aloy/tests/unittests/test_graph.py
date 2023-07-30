
import unittest

from datastructures.graph import Graph

class TestGraph(unittest.TestCase):
    def test_undirected_graph(self):
        graph: Graph[int] = Graph({1 : {2, 3}, 2 : {4, 5}})
        self.assertDictEqual(dict(graph), {1: {2, 3}, 2: {1, 4, 5}, 3: {1}, 4: {2}, 5: {2}})

    def test_directed_graph(self):
        graph: Graph[int] = Graph({1 : {2, 3}, 2 : {4, 5}}, directed=True)
        self.assertDictEqual(dict(graph), {1: {2, 3}, 2: {4, 5}, 3: set(), 4: set(), 5: set()})

    def test_vertices(self):
        range_ = range(1, 5)
        graph: Graph[int] = Graph(vertices=range_)
        self.assertDictEqual(dict(graph), {i : set() for i in range_})

    def test_edges(self):
        graph: Graph[int] = Graph(edges=[(1, 2), (2, 3), (3, 4)])
        self.assertDictEqual(dict(graph), {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}})

    def test_get_path(self):
        graph: Graph[int] = Graph({1 : {2, 3}, 2 : {4, 5}})
        self.assertEqual(graph.get_path(1, 5), [1, 2, 5])

    def test_edges_directed(self):
        edges = [(1, 2), (2, 3), (3, 4)]
        graph: Graph[int] = Graph(edges=edges, directed=True)
        self.assertEqual(graph.edges, dict.fromkeys(edges, 1.0).keys())

    def test_edges_undirected(self):
        edges = [(1, 2), (2, 3), (3, 4)]
        reversed_edges = [edge[::-1] for edge in edges]
        graph: Graph[int] = Graph(edges=edges)
        self.assertEqual(graph.edges, dict.fromkeys([*edges, *reversed_edges], 1.0).keys())

    def test_add_vertex(self):
        graph: Graph[int] = Graph()
        graph[1] = None
        self.assertTrue(graph[1] == set())

    def test_add_edge_undirected(self):
        graph: Graph[int] = Graph()
        graph.add_edge(1, 2)
        self.assertDictEqual(dict(graph), {1 : {2}, 2 : {1}})

    def test_add_edge_directed(self):
        graph: Graph[int] = Graph(directed=True)
        graph.add_edge(1, 2)
        self.assertDictEqual(dict(graph), {1 : {2}, 2 : set()})

    def test_add_edge_with_weight(self):
        graph: Graph[int] = Graph()
        graph.add_edge(1, 2, 3)
        self.assertEqual(graph.get_edge_weight(1, 2), 3)

    def test_remove_vertex(self):
        graph: Graph[int] = Graph(vertices=[1, 2, 3])
        self.assertTrue(1 in graph)
        del graph[1]
        self.assertTrue(1 not in graph)

    def test_remove_edge(self):
        graph: Graph[int] = Graph(edges=[(1, 2), (2, 3), (3, 4)])
        self.assertTrue((1, 2) in graph.edges)
        del graph[1, 2]
        self.assertTrue((1, 2) not in graph.edges)
