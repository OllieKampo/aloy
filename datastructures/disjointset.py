###########################################################################
###########################################################################
## A disjoint-set data structure and union-find algorithm.               ##
##                                                                       ##
## Copyright (C)  2022  Oliver Michael Kamperis                          ##
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

"""Module contain a disjoint-set data structure."""

import collections.abc
import enum
from typing import (Generic, Hashable, Iterable, Iterator,
                    Literal, Mapping, Optional, TypeVar, overload)

__all__ = (
    "DisjointSet",
    "DefaultFindMethod"
)


## Disjoint-set generic element type (must be hashable).
ST = TypeVar("ST", bound=Hashable)


class DisjointSet(collections.abc.Mapping, Generic[ST]):
    """
    A disjoint-set data structure, also called union-find.

    A disjoint-set, is a set of disjoint sub-sets (sets with empty
    intersections).

    In simple terms, a disjoint-set is a set of elements (elements are
    unordered and unique), split/partitioned into multiple sub-sets, each of
    which is disjoint with all other sub-sets, such that any given element of
    the whole disjoint-set is part of one, and only one, of its sub-sets.

    It provides two main operators;
    - unioning (merging) disjoint sub-sets,
    - and finding the root of a disjoint sub-set,
        where the root is a unique representative member
        of sub-set that identifies it from other sub-sets.

    Hence the alternative name; union-find.

    Each disjoint sub-set is represented by its root, and all other elements
    in the same sub-set are stored in a tree structure that connects them to
    their root. This makes it easy to determine if any two elements are in the
    same or different sub-sets. This is because sub-set membership of any
    given element, reduces simply to finding its root, and the tree structure
    contains a trivially traceable path from each element to its sub-set's
    root. This path is (usually) compressed as the disjoint-set is modified,
    making it easier to find an element's root on future look-ups.

    The benefit of the tree-based internal representation of a disjoint-set,
    is that union operations are trivial. This is because performing the union
    of two sub-sets requires only setting the parent of the root of one
    sub-set to be the root of the other sub-set. This is a single operation
    (which is constant time), which relies on two root find operations (one to
    find the root of each sub-set). Although union operations may make root
    finding more expensive (because they can make the trees grow), if the
    union is done intelligently (whereby the tree depths do not grow linearly
    with the size of the disjoint set), and path compression is used
    appropriately, root finding becomes constant amortised time and therefore
    union operations are also constant time.

    The alternative representation, would be to explicitly keep track of the
    root of each element of the disjoint-set. Unfortunately, whilst this would
    make root finding always constant time, union operations would be linear
    time (in the size of the disjoint-set) since unioning two sets would
    involve reassigning the root of all elements of one of the two sets.

    Example Usage
    -------------
    ```
    from jinx.datastructures.disjointset import DisjointSet

    ## Construct a disjoint-set with integer elements from
    ## an iterable, its is initially fully-disjoint, such that
    ## no two elements are in the same sub-set.
    dset: DisjointSet[int] = DisjointSet([i for i in range(5)])

    >>> dset
    DisjointSet({0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}})
    >>> str(dset)
    'Disjoint-Set: total elements = 5, total disjoint sub-sets = 5'

    ## A simple union operation unions the sub-sets
    ## containing the given elements into one new sub-set,
    ## returning the root of the new sub-set.
    >>> dset.union(3, 4)
    4
    >>> dset
    DisjointSet({0: {0}, 1: {1}, 2: {2}, 4: {3, 4}})

    ## Many elements can be unioned simultaneously.
    dset.union_many(0, 1, 2)
    >>> dset
    DisjointSet({1: {0, 1, 2}, 4: {3, 4}})

    ## Elements 0, 1 and 2 are now in the same sub-set, therefore
    ## they have the same root, and are considered connected.
    >>> dset.find_root(2)
    1
    >>> dset.find_root(0)
    1
    >>> dset.is_connected(0, 2)
    True
    >>> dset.is_connected(0, 4)
    False
    ```
    """

    __slots__ = (
        "__parent_of",
        "__rank_of",
        "__is_changed",
        # "__track_sets",
        "__sets",
        "__find_method"
    )

    @overload
    def __init__(self) -> None:
        """Create an empty disjoint-set."""
        ...

    @overload
    def __init__(self,
                 sets: Mapping[ST, set[ST]]
                 # track_subsets: bool = False
                 # Useful if the disjoint sub-sets need to be accessed many times and the disjoint-set will be updated very often.
                 # This is unlike caching the disjoint sub-sets when then are accessed, because the cache is usually invalidated when the disjoint-set is updated (with an add or a union).
                 # Accessing the disjoint sub-sets when they are not tracked is computationally expensive, but it saves memory during operation and updates are less expensive.
                 # Tracking the sets uses more memory during operation and is more computationally expensive during updates, but accessing the disjoint sub-sets is trivial.
                 ) -> None:
        """Create a new disjoint-set from a root element to sub-set mapping."""
        ...

    @overload
    def __init__(self,
                 sets: Mapping[ST, set[ST]],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress",
                                                            "recurse", "recurse_compress",
                                                            "path_split", "path_halve"]
                 ) -> None:
        """
        Create a new disjoint-set from a root element to sub-set mapping.
        
        Optionally specify a default root finding method used for; disjoint
        sub-set finding, union operations, and element connectedness checking.
        """
        ...
    
    @overload
    def __init__(self,
                 elements: Iterable[ST]
                 ) -> None:
        """Create a new disjoint-set from an iterable of elements (unit sub-sets)."""
        ...
    
    @overload
    def __init__(self,
                 elements: Iterable[ST],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress",
                                                            "recurse", "recurse_compress",
                                                            "path_split", "path_halve"]
                 ) -> None:
        """
        Create a new disjoint-set from an iterable of elements (unit sub-sets).
        
        Specify a default root finding method for; disjoint sub-set
        finding, union operations, and element connectedness checking.
        """
        ...
    
    def __init__(self,
                 sets_or_elements: Optional[Mapping[ST, ST | set[ST]]
                                            | Iterable[ST]] = None,
                 find_method: "DefaultFindMethod"
                              | Literal["loop", "loop_compress",
                                        "recurse", "recurse_compress",
                                        "path_split", "path_halve"] = "loop_compress"
                 ) -> None:
        """Create a new disjoint-set."""
        ## Variables for storing the disjoint-set elements themselves.
        self.__parent_of: dict[ST, ST] = {} # Maps: child element -> its parent element
        self.__rank_of: dict[ST, int] = {} # Maps: root element -> rank of its set
        
        if isinstance(sets_or_elements, Mapping):
            ## If the disjoint-set elements are given as a mapping, create
            ## a disjoint sub-set for each item of the mapping, where the
            ## keys are the sub-sets' root and the values the set elements.
            for root, set_ in sets_or_elements.items():
                
                ## If this sub-set's given root is already in the disjoint-set, then fall back
                ## on unioning this sub-set with the existing sub-set that contains the given root.
                ## Therefore instead set the parent of all of this sub-sets elements to the root of
                ## the existing sub-set that already contains the originally given root.
                if root in self:
                    if not isinstance(set_, set):
                        set_ = {set_}
                    root = self.find_root(root, compress=False)
                else:
                    ## Otherwise this is a valid disjoint sub-set, and it is
                    ## necessary to ensure the root is in the sub-set.
                    if not isinstance(set_, set):
                        set_ = {set_, root}
                    else:
                        set_ = set_ | {root}
                
                ## Assign the parent of all elements in the sub-set to be its root.
                for element in set_:
                    self.__parent_of[element] = root
                
                ## Initialise rank as 0 if the sub-set is a single element
                ## (just the root), otherwise initialise the rank as 1 since
                ## the set starts fully compressed.
                self.__rank_of[root] = 0 if len(set_) == 1 else 1
        elif sets_or_elements is not None:
            ## If the disjoint-set elements are given as an iterable, assign
            ## each to its own disjoint sub-set and initialise rank as zero.
            self.__parent_of = {
                element: element for element in sets_or_elements
            }
            self.__rank_of = {
                element: 0 for element in self.__parent_of
            }
        else:
            self.__parent_of = {}
            self.__rank_of = {}
        
        ## Variables for caching containing disjoint sub-sets;
        ##      - Only add and union operations will change the sub-sets,
        ##      - Path compression only changes the paths within the sets.
        self.__is_changed: bool = True
        self.__sets: dict[ST, frozenset[ST]] = {}
        
        ## Determine the default find method used by this disjoint set.
        if isinstance(find_method, DefaultFindMethod):
            default_find, can_compress = find_method.value
        else:
            default_find, can_compress = DefaultFindMethod[find_method].value
        
        ## Create an alias for calling the default find method.
        if can_compress:
            if "compress" in find_method:
                def _default_find(self, element: ST, compress: bool) -> ST:
                    return default_find(
                        self, element, compress or compress is None
                    )
                self.__find_method = _default_find
            else:
                self.__find_method = default_find
        else:
            def _default_find(self, element: ST, compress: bool) -> ST:
                return default_find(self, element)
            self.__find_method = _default_find

    def __str__(self) -> str:
        """Return string summary representation describing number of elements and disjoint sub-sets."""
        sets_: dict[ST, set[ST]] = self.find_all_sets(compress=None, cache=True)
        return f"Disjoint-Set: total elements = {len(self)}, total disjoint sub-sets = {len(sets_)}"

    def __repr__(self) -> str:
        """Return an instantiable string representation of the disjoint-set."""
        return f"{self.__class__.__name__}({self.find_all_sets(compress=None, cache=True)})"

    def __getitem__(
        self,
        element: ST
    ) -> ST:
        """
        Find the root of the disjoint sub-set containing the given element
        using the default find method.
        """
        return self.__find_method(self, element, compress=None)

    def __contains__(self, element: ST) -> bool:
        """Whether an element is in the disjoint-set."""
        return element in self.__parent_of

    def __iter__(self) -> Iterator[ST]:
        """Return an iterator over all set elements."""
        yield from self.__parent_of

    def __len__(self) -> int:
        """Get the number of elements in the disjoint-set."""
        return len(self.__parent_of)

    @property
    def parents(self) -> dict[ST, ST]:
        """Get the element to parent mapping."""
        return self.__parent_of

    @property
    def ranks(self) -> dict[ST, ST]:
        """Get the element to rank mapping."""
        return self.__rank_of

    def add(self,
            parent: ST,
            *elements: ST,
            union: bool = True
            ) -> ST:
        """
        Add the element(s) to the disjoint sub-set with the given root.

        If the root is not already in the disjoint-set, then add it as a new element as well.

        If `union` is True, and both the parent and the element are already in the disjoint-set,
        then this method simply unions the disjoint sub-sets containing the given elements,
        i.e. it performs the same operation as `union(root, element)`.
        If `union` is False, then it is an error to add an element that is already in the disjoint-set.

        Parameters
        ----------
        `parent: ST@DisjointSet` - The parent of the element or set of elements to to add to the disjoint-set.
        If the parent is not already in the disjoint-set, then it is added as a new root element, and the element or set of elements to add become its children.
        If the parent is already in the disjoint-set, then the element or set of elements to add is unioned onto to the disjoint sub-set containing the parent.

        `element_or_set: ST@DisjointSet | set[ST@DisjointSet]` - The element or set of elements to add.

        `union: bool = True` - Whether to union the disjoint sub-sets containing the parent and the element or set of elements if they are already in the disjoint-set.
        If False, then an error is raised if the element or set of elements are already in the disjoint-set.

        Returns
        -------
        `ST@DisjointSet` - The root element of the disjoint sub-set the given element or set of elements was added to.

        Raises
        ------
        `ValueError` - If `union` is False and the given element(s) are already in the disjoint-set.
        """
        self.__is_changed: bool = True

        root: ST
        if parent not in self:
            self.__parent_of[parent] = parent
            self.__rank_of[parent] = 1
            root = parent
        else:
            root = self.find_root(parent)

        for element in elements:
            if element not in self:
                self.__parent_of[element] = root
            else:
                if union:
                    root = self.union(root, element)
                else:
                    raise ValueError(f"The element {element} is already in the disjoint-set.")

        return root

    def find_root(
        self,
        element: ST,
        compress: bool = True
    ) -> ST:
        """
        Find the root of the sub-set containing the given element.

        A root, is a set element, whose parent is itself.

        This method finds roots via an iterative search, and if
        compression is enabled, iteratively compresses the path
        from the given element to the root, such that the parents
        of all non-root elements on the path, will be the root.

        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.

        `compress: bool = True` - Whether to fully compress the path
        from the given node to its root. This more expensive operation
        will speed up future lookups for elements on the same path.

        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given
        element.

        Raises
        ------
        `KeyError` - If the given element is not in this disjoint-set.
        """
        ## To find the root of the sub-set containing the given element,
        ## simple iterate up the element's tree until a root is found.
        ## An element is a root if its parent is itself.
        _element: ST = element
        while (parent := self.__parent_of[_element]) != _element:
            _element = parent
        root: ST = parent

        ## Compression performed by a seperate loop,
        ## achieves maximum possible level of compression.
        ## Simply set the parent of all other elements on the path
        ## from the given element to its root to the root.
        if compress:
            while (parent := self.__parent_of[element]) != root:
                self.__parent_of[element] = root
                element = parent

        return root

    def find_root_recurse(self,
                          element: ST,
                          compress: bool = True
                          ) -> ST:
        """
        Recursive variant of the `find_root` method.

        This may be more efficient for small disjoint-sets,
        or where the average path lengths are short.
        However, python's recursion depth is quite shallow,
        so this method may fail for larger disjoint-sets.

        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.

        `compress: bool = True` - Whether to fully compress the path
        from the given node to its root. This more expensive operation
        will speed up future lookups for elements on the same path.

        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given
        element.

        Raises
        ------
        `KeyError` - If the given element is not in this disjoint-set.
        """
        if (parent := self.__parent_of[element]) != element:
            root = self.find_root(parent)
            if compress and parent != root:
                self.__parent_of[element] = root
            return root
        return element

    def find_root_path_split(self,
                             element: ST
                             ) -> ST:
        """
        Find the root of the sub-set containing the given element with path
        splitting.

        This partially compresses the path from the given element to the root
        during the iterative search for the root element, by replacing every
        element's parent on the path with its grandparent instead.

        This avoids the need for a second loop to perform the path compression
        however it can only achieve half the level of compression as
        `find_root`.

        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.

        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given
        element.

        Raises
        ------
        `KeyError` - If the given element is not in this disjoint-set.
        """
        ## Set the parent of every element on the path to
        ## the root to be the parent of the its parent.
        while (parent := self.__parent_of[element]) != element:
            self.__parent_of[element] = self.__parent_of[parent]
            element = parent
        return parent

    def find_root_path_halve(self,
                             element: ST
                             ) -> ST:
        """
        Find the root of the sub-set containing the given element with path
        halving.

        This partially compresses the path from the given element to the root
        during the iterative search for the root element, by replacing every
        other element's parent on the path with its grandparent instead.

        This is similar to `find_root_path_split` in that it avoids the
        need for a second loop to perform path compression, except it
        can half the number of elements that need to be searched along
        to reach the root (since it only checks if every other element
        is the root), however it also only achieves half the level of
        compression as path splitting.

        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.

        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given
        element.

        Raises
        ------
        `KeyError` - If the given element is not in this disjoint-set.
        """
        ## Set the parent of every checked element on the path to the root to
        ## be the parent of the its parent, and set the parent of the parent to
        ## be the next element to check (checking only every other element).
        while (parent := self.__parent_of[element]) != element:
            self.__parent_of[element] = self.__parent_of[parent]
            element = self.__parent_of[parent]
        return parent

    def find_path(self,
                  element: ST
                  ) -> list[ST]:
        """
        Find the current path from the given element to the root element of
        its disjoint sub-set.

        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose path to its sub-set's
        root element to find.

        Returns
        -------
        `list[ST@DisjointSet]` - A list of elements on the path from the given element,
        to the root element of its disjoint sub-set. The list will contain only the given
        element if and only if the given element is the root of its own sub-set.
        """
        path: list[ST] = [element]
        while (parent := self.__parent_of[element]) != element:
            path.append(parent)
            element = parent
        return path

    def find_set(self,
                 element: ST,
                 compress: Optional[bool] = None,
                 cache: bool = True,
                 default: Optional[set[ST]] = None
                 ) -> frozenset[ST]:
        """
        Get a single distinct sub-set in this disjoint-set.
        
        This operation requires finding all dictinct sub-sets,
        and as such `find_all_sets(...)` or `cache=True` should
        be used if more than one distinct sub-set needs to be found.
        
        Parameters
        ----------
        `element: ST@DisjointSet` - An element of the distinct sub-set to find.
        All other elements with the same root as the given element will be returned in the sub-set.
        
        `compress: bool = True` - Whether to compress all paths,
        from all elements, of all sets, to their respective roots.
        This will achieve the maximum possible compression of the disjoint-set.
        If the default find method is; path splitting or path halving, this parameter is ignored.
        
        `cache: bool = True` - Whether to cache the result.
        Future calls to this function will return the cached result,
        if and only if the disjoint-set has not been modified since the previous call.
        
        `default: {None | set[ST@DisjointSet]}` - If given and not None,
        acts as a default sub-set to return if the given element is not in this disjoint-set.
        
        Returns
        -------
        `set[ST@DisjointSet]` - A distinct sub-set containing the given element.
        
        Raises
        ------
        `KeyError` - If the element is not in this disjoint-set and `default` is None.
        """
        if element not in self:
            if default is None:
                raise KeyError(f"The element {element} of type {type(element)} is not in the disjoint-set {self!s}.")
            return default
        
        if not self.__is_changed:
            return self.__sets[element]
        return self.find_all_sets(compress, cache)[element]
    
    def find_all_sets(self,
                      compress: Optional[bool] = None,
                      cache: bool = True
                      ) -> dict[ST, frozenset[ST]]:
        """
        Find all distinct sub-sets in this disjoint-set.
        
        Parameters
        ----------
        `compress: bool = True` - Whether to compress all paths,
        from all elements, of all sets, to their respective roots.
        This will achieve the maximum possible compression of the disjoint-set.
        If the default find method is; path splitting or path halving, this parameter is ignored.
        
        `cache: bool = True` - Whether to cache the result.
        Future calls to this function will return the cached result,
        if and only if the disjoint-set has not been modified since the previous call.
        
        Returns
        -------
        `dict[ST@DisjointSet, set[ST@DisjointSet]]` - A dictionary,
        whose keys are root's of each distinct sub-set,
        and the values are the sub-sets themselves.
        """
        if not self.__is_changed:
            return self.__sets
        
        ## Find all disjoint sub-sets by finding the root of all
        ## elements and then grouping elements with the same root.
        sets: dict[ST, set[ST]] = {}
        for element in self:
            sets.setdefault(self.__find_method(self, element, compress), set()).add(element)
        
        ## If compression is enabled and the default find method
        ## accepts the compress parameter, the ranks can be reset,
        ## since all sub-sets are now fully compressed.
        if compress and DefaultFindMethod[self.__find_method.__name__].value[1]:
            self.__rank_of = {}
            for root, set_ in sets.items():
                self.__rank_of[root] = 0 if len(set_) == 1 else 1
        
        # frozen_sets = FrozenDict(sets) TODO: use frozen dict for the dictionary itself cannot be changed
        frozen_sets = {root : frozenset(set_) for root, set_ in sets.items()}
        
        if cache:
            self.__sets = frozen_sets
            self.__is_changed = False
        return frozen_sets
    
    def __find_root_pair(self,
                         element_1: ST,
                         element_2: ST,
                         compress: Optional[bool] = None
                         ) -> tuple[ST, ST]:
        """
        Find the roots of a pair of two different elements of this disjoint-set.
        
        Used by union methods, to simultaneously find the roots of two different
        elements of this disjoint-set, whose distinct sub-sets are to be unioned.
        """
        return (self.__find_method(self, element_1, compress),
                self.__find_method(self, element_2, compress))
    
    def union_left(self,
                   element_1: ST,
                   element_2: ST,
                   compress: Optional[bool] = None
                   ) -> ST:
        # noqa: D205, D400
        """
        Union the sub-set containing the second element (right argument) onto
        the sub-set containing the first element (left argument).
        
        This is such that the root of the new combined set, is the root
        of the original sub-set containing the first element.
        
        Parameters
        ----------
        `element_1: ST@DisjointSet` - Any element of the disjoint-set.
        
        `element_2: ST@DisjointSet` - Any element of the disjoint-set.
        
        `compress: {bool | None} = None` - Whether to compress the paths from both
        of the given elements to the original root elements of their respective original sub-sets.
        
        Returns
        -------
        `ST@DisjointSet` - The root element of the new combined set.
        """
        root_1, root_2 = self.__find_root_pair(element_1, element_2, compress)
        if root_1 == root_2: return
        self.__is_changed = True
        self.__parent_of[root_2] = root_1
        self.__rank_of[root_1] = self.__rank_of[root_1] + 1
        return root_1
    
    def union_right(self,
                    element_1: ST,
                    element_2: ST,
                    compress: Optional[bool] = None
                    ) -> ST:
        # noqa: D205, D400
        """
        Union the sub-set containing the first element (left argument) onto
        the sub-set containing the second element (right argument).
        
        This is such that the root of the new combined set, is the root
        of the original sub-set containing the second element.
        
        Parameters
        ----------
        `element_1: ST@DisjointSet` - Any element of the disjoint-set.
        
        `element_2: ST@DisjointSet` - Any element of the disjoint-set.
        
        `compress: {bool | None} = None` - Whether to compress the paths from both
        of the given elements to the original root elements of their respective original sub-sets.
        
        Returns
        -------
        `ST@DisjointSet` - The root element of the new combined set.
        """
        root_1, root_2 = self.__find_root_pair(element_1, element_2, compress)
        if root_1 == root_2: return
        self.__is_changed = True
        self.__parent_of[root_1] = root_2
        self.__rank_of[root_2] = self.__rank_of[root_2] + 1
        return root_2
    
    def union(self,
              element_1: ST,
              element_2: ST,
              compress: Optional[bool] = None
              ) -> ST:
        # noqa: D205, D400
        """
        Union the sub-sets containing the given elements together,
        using the union-by-rank algorithm.
        
        This ensures that the smaller original sub-set is unioned onto
        the larger original sub-set. This is such that the root of the
        new combined set, is the root of the larger original sub-set.
        
        This is beneficial, since it reduces the average path length
        from any given descendent element of the combined set, to it's root.
        If the larger set were unioned onto the smaller, there would be a greater
        number of set elements, that were one element further from the root node
        after the union operation. Performing such an operation many times without
        path compression, can massively increase the complexity of the find operation
        on the disjoint set (i.e. searching the sub-set's trees).
        
        Parameters
        ----------
        `element_1: ST@DisjointSet` - Any element of the disjoint-set.
        
        `element_2: ST@DisjointSet` - Any element of the disjoint-set.
        
        `compress: {bool | None} = None` - Whether to compress the paths from both
        of the given elements to the original root elements of their respective original sub-sets.
        
        Returns
        -------
        `ST@DisjointSet` - The root element of the new combined set.
        """
        root_1, root_2 = self.__find_root_pair(element_1, element_2, compress)
        if root_1 == root_2: return
        self.__is_changed = True
        
        ## Union by rank - Always union the shorter tree into the longer tree;
        ##      - If you put the longer tree onto the shorter;
        ##          - The tree may grow linearly,
        ##            a tree with a set of n elements may
        ##            grow to be up to n elements long
        ##            (a straight chain without branching).
        ##      - root_1 should always be the smaller rank,
        ##      - With this method, we always get log(n) complexity.
        if self.__rank_of[root_1] > self.__rank_of[root_2]:
           root_1, root_2 = root_2, root_1
        self.__parent_of[root_1] = root_2
        
        ## If the ranks of the roots are the same then the tree whose root was unioned onto has now grown
        ## (if the ranks are different that means that the tree unioned onto was already deeper than the other so it hasn't grown),
        ## and therefore the rank of the root that was unioned onto must increase (this will always be root_2),
        ## to keep the rank proportionate to depth growth from union operations.
        if self.__rank_of[root_1] == self.__rank_of[root_2]:
            self.__rank_of[root_2] = self.__rank_of[root_2] + 1
        
        return root_2
    
    def union_many(self,
                   elements: Iterable[ST],
                   compress: Optional[bool] = None
                   ) -> ST:
        """Union all elements in the given iterable of elements and return the root of the resulting sub-set."""
        iter_: Iterator[ST] = iter(elements)
        try: first: ST = next(iter_)
        except: return
        for element in iter_:
            root: ST = self.union(first, element, compress)
        return root
    
    def is_connected(self,
                     element_1: ST,
                     *elements: ST,
                     compress: Optional[bool] = None
                     ) -> bool:
        """
        Determine whether the two elements are in the same disjoint sub-set.
        
        If two elements are given, equivalent to: `self.find_root(element_1) == self.find_root(element_2)`.
        
        If more than two elements are given, equivalent to: `all(self.find_root(element_1) == self.find_root(other) for other in elements)`.
        """
        root_1: ST = self.__find_method(self, element_1, compress)
        return all(root_1 == self.__find_method(self, element, compress) for element in elements)

class DefaultFindMethod(enum.Enum):
    """
    The default root finding methods that can be passed as argument to the constructor of a disjoint-set.
    
    These are used by the methods;
        - Finding all disjoint sub-sets with `find_all_sets` or `find_set`,
        - Union operations with `union` or `union_many`,
        - Checking if elements are connected (are in the same disjoint sub-set) with `is_connected`.
    
    Items
    -----
    `loop` - Find roots iteratively, never compress the path.
    
    `loop_compress` - Find roots iteratively, always fully compress the path.
    
    `recurse` - Find root recursively, never compress the path.
    
    `recurse_compress` - Find roots recursively, always fully compress the path.
    
    `path_split` - Find roots iteratively, partially compress the path.
    
    `path_halve` - Find roots iteratively, skip every other element to increase speed, partially compress the path.
    """
    
    loop = (DisjointSet.find_root, True)
    loop_compress = (DisjointSet.find_root, True)
    recurse = (DisjointSet.find_root_recurse, True)
    recurse_compress = (DisjointSet.find_root_recurse, True)
    path_split = (DisjointSet.find_root_path_split, False)
    path_halve = (DisjointSet.find_root_path_halve, False)
