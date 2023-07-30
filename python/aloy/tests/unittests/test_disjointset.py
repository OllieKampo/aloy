
import unittest

from datastructures.disjointset import DisjointSet

class TestDisjointSet(unittest.TestCase):
    def test_parents(self):
        dset: DisjointSet[int] = DisjointSet((i for i in range(0, 100, 2)))
        self.assertDictEqual(dset.parents, {i: i for i in range(0, 100, 2)})
    
    def test_find_sets(self):
        dset: DisjointSet[str] = DisjointSet({i: l for i, l in zip("lkjhgf", "abcdefg")})
        self.assertDictEqual(dset.find_all_sets(), {i: {i, l} for i, l in zip("lkjhgf", "abcdefg")})
