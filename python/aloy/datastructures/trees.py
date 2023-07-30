import collections.abc
from typing import Generic, Optional, TypeVar


TT = TypeVar("TT")


class BinaryTree(collections.abc.Sequence, Generic[TT]):
    """
    Each tree level is a (sequence of) tree instances,
    such that each tree instance simply points to two other
    tree instances, thus creating an arbitrarily depth tree.
    However, each individual tree instance does not store
    the depth of the complete tree down each of this branches.
    """
    
    __slots__ = ("__head", "__left", "__right")
    
    def __init__(self,
                 head: TT,
                 left: Optional["BinaryTree"] = None,
                 right: Optional["BinaryTree"] = None
                 ) -> None:
        self.__head: TT = head
        if ((left is None and right is not None)
            or (left is not None and right is None)):
            raise ValueError("Left and right must be both None "
                             "or both not None.")
        self.__left: Optional[BinaryTree] = left
        self.__right: Optional[BinaryTree] = right
    
    @property
    def head(self) -> TT:
        """Head value of the tree."""
        return self.__head
    
    @property
    def left(self) -> Optional["BinaryTree"]:
        """Left child tree."""
        return self.__left
    
    @property
    def right(self) -> Optional["BinaryTree"]:
        """Left child tree."""
        return self.__right
    
    @property
    def is_leaf(self) -> bool:
        """Whether the tree is a leaf (it has no children)."""
        return self.__left is None
    
    def get_inverted(self) -> "BinaryTree":
        tree = self.__class__(self.head)
        if not self.is_leaf:
            tree.left = self.right.get_inverted()
            tree.right = self.left.get_inverted()
        return tree
    
    def get_flipped(self) -> "HierarchyTree":
        """
        Return a flipped version of the tree.
        """
        return NotImplemented

    def sum_depths(self) -> int:
        """
        Return the sum of the depths of all nodes in the tree.
        """
        frontier: list[tuple[int, BinaryTree]] = [(0, self)]
        depth_sum: int = 0
        
        while frontier:
            depth, node = frontier.pop()
            depth_sum += depth
            
            if node.left is not None:
                frontier.extend([(depth + 1, node.left), (depth + 1, node.right)])
        
        return depth_sum


class Tree(collections.abc.Sequence):
    """
    An arbitrary depth tree represented by a single instance.
    Any node of the tree can be converted to a tree.
    """
    pass


class Forest(collections.abc.Sequence):
    """
    A collection of trees.
    """
    pass


class HierarchyTree(collections.abc.Sequence):
    """
    A collection of trees arranged in a multi-level hierarchy,
    in which a node can have multiple parents, and therefore
    a node can also be the child of multiple nodes.

    Unlike a normal tree, two nodes at the same level can be
    linked by some relationship. The hierarchy can also be sliced
    at, above, or below any node given level to obtain a sub-hierarchy.
    """
    pass
