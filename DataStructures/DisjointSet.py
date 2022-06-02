import collections.abc
import enum
from typing import Generic, Hashable, Iterable, Iterator, Literal, Mapping, Optional, TypeVar, overload

ST = TypeVar("ST", bound=Hashable)

class DisjointSet(collections.abc.Mapping, Generic[ST]):
    """
    A disjoint-set data structure, also called union-find.
    A disjoint-set, is a set of disjoint sub-sets (sets with empty intersections).
    
    In simple terms, a disjoint-set is a set of elements (elements are unordered and unique),
    split/partitioned into multiple sub-sets, each of which is disjoint with all other sub-sets,
    such that any given element of the whole disjoint-set is part of one, and only one, of its sub-sets.
    
    It provides two main operators;
    
        - unioning (merging) disjoint sub-sets,
        - and finding the root of a disjoint sub-set,
          where the root is a unique representative member
          of sub-set that identifies it from other sub-sets.
    
    Hence the alternative name; union-find.
    
    Each disjoint sub-set is represented by its root, and all other elements in the
    same sub-set are stored in a tree structure that connects them to their root.
    This makes it easy to determine if any two elements are in the same or different sub-sets.
    This is because sub-set membership of any given element, reduces simply to finding its root,
    and the tree structure contains a trivially traceable path from each element to its sub-set's root.
    This path is (usually) compressed as the disjoint-set is modified, making it easier to find an element's root on future look-ups.
    
    Example Usage
    -------------
    ```
    from Jinx.DataStructure.DisjointSet import DisjointSet
    
    ## Construct a disjoint-set with integer elements from
    ## an iterable, its is initially fully-disjoint, such that
    ## no two elements are in the same sub-set.
    dset: DisjointSet[int] = DisjointSet([i for i in range(10)])
    
    >>> dset
    DisjointSet({0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {6}, 7: {7}, 8: {8}, 9: {9}})
    >>> str(dset) 
    'Disjoint-Set: total elements = 10, total disjoint sub-sets = 10'
    
    ## A simple union operation unions the sub-sets
    ## containing the given elements into one new sub-set,
    ## returning the root of the new sub-set.
    new_root: int = dset.union(8, 9)
    
    >>> new_root
    9
    >>> dset
    DisjointSet({0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}, 5: {5}, 6: {6}, 7: {7}, 9: {8, 9}})
    
    ## Many elements can be unioned in either of the following ways;
    for i in range(1, 6):
        dest.union(0, i)
    
    dset.union_many(0, 1, 2, 3, 4, 5)
    
    >>> dset
    DisjointSet({1: {0, 1, 2, 3, 4, 5}, 6: {6}, 7: {7}, 9: {8, 9}})
    
    ## Elements 2 and 3 are now in the same sub-set, therefore
    ## they have the same root, and are considered connected.
    >>> dset.find_root(2)
    1
    >>> dset.find_root(3)
    1
    >>> dset.is_connected(2, 3)
    True
    
    ## Element 6 is (of course) still in its own disjoint sub-set.
    >>> dset.find_root(6)
    6
    >>> dset.is_connected(1, 6)
    False
    ```
    """
    
    __slots__ = ("__parent_of",
                 "__rank_of",
                 "__is_changed",
                 "__sets",
                 "__find_method")
    
    @overload
    def __init__(self,
                 sets: Mapping[ST, set[ST]]
                 ) -> None:
        "A new disjoint-set from a root element to sub-set mapping."
        ...
    
    @overload
    def __init__(self,
                 sets: Mapping[ST, set[ST]],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress", "recurse", "recurse_compress", "path_split", "path_halve"]
                 ) -> None:
        """
        A new disjoint-set from a root element to sub-set mapping,
        specifying a default root finding method for; disjoint sub-set
        finding, union operations, and element connectedness checking.
        """
        ...
    
    @overload
    def __init__(self,
                 elements: Iterable[ST],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress", "recurse", "recurse_compress", "path_split", "path_halve"] = "loop_compress"
                 ) -> None:
        "A new disjoint-set from an iterable of elements (unit sub-sets)."
        ...
    
    @overload
    def __init__(self,
                 elements: Iterable[ST],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress", "recurse", "recurse_compress", "path_split", "path_halve"]
                 ) -> None:
        """
        A new disjoint-set from an iterable of elements (unit sub-sets),
        specifying a default root finding method for; disjoint sub-set
        finding, union operations, and element connectedness checking.
        """
        ...
    
    def __init__(self,
                 sets_or_elements: Mapping[ST, ST | set[ST]] | Iterable[ST],
                 find_method: "DefaultFindMethod" | Literal["loop", "loop_compress", "recurse", "recurse_compress", "path_split", "path_halve"] = "loop_compress"
                 ) -> None:
        
        ## Variables for storing the disjoint-set elements themselves.
        self.__parent_of: dict[ST, ST] = {} # Maps: child element -> its parent element
        self.__rank_of: dict[ST, int] = {} # Maps: root element -> rank of its set
        
        if isinstance(sets_or_elements, Mapping):
            
            for root, set_ in sets_or_elements.items():
                if not isinstance(set_, set):
                    set_ = {set_}
                else: set_ = set_
                
                ## Ensure the root is in the set
                set_ = set_ | {root}
                
                for element in set_:
                    self.__parent_of[element] = root
                
                ## Ranks must in initialised as size
                self.__rank_of[root] = len(set_)
            
        else:
            self.__parent_of = {element : element for element in sets_or_elements}
            self.__rank_of = {element : 0 for element in self.__parent_of}
        
        ## Variables for caching containing disjoint sub-sets;
        ##      - Only add and union operations will change the sub-sets,
        ##      - Path compression only changes the paths within the sets.
        self.__is_changed: bool = True
        self.__sets: dict[ST, set[ST]] = {}
        
        ## Determine the default find method used by this disjoint set.
        if isinstance(find_method, DefaultFindMethod):
            default_find_method, accepts_compress = find_method.value
        else: default_find_method, accepts_compress = DefaultFindMethod[find_method].value
        
        ## Create an alias for calling the default find method.
        if accepts_compress:
            if "compress" in find_method:
                self.__find_method = (lambda self, element, compress:
                                      default_find_method(self, element, compress or compress is None))
            else: self.__find_method = (lambda self, element, compress:
                                        default_find_method(self, element, compress))
        else: self.__find_method = (lambda self, element, compress:
                                    default_find_method(self, element))
    
    def __str__(self) -> str:
        sets_: dict[ST, set[ST]] = self.find_all_sets(compress=None, cache=True)
        return f"Disjoint-Set: total elements = {len(self)}, total disjoint sub-sets = {len(sets_)}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.find_all_sets(compress=None, cache=True)})"
    
    def __getitem__(self, element: ST) -> ST:
        return self.__parent_of[element]
    
    def add(self, root: ST, element_or_set: ST | set[ST]) -> None:
        """
        Add an element or set of elements, to the disjoint sub-set with the given root.
        If the root is not already in the disjoint-set, then add it as a new element as well.
        
        It is an error to add an element that is already in the disjoint-set.
        
        Parameters
        ----------
        `root: ST@DisjointSet` - The root element of the disjoint
        sub-set to add the element or set of elements to.
        
        `element_or_set: ST@DisjointSet | set[ST@DisjointSet]` - The
        element or set of elements to add.
        
        Raises
        ------
        `ValueError` - If the given element, or any element
        of the given set, is already in the disjoint-set.
        """
        self.__is_changed: bool = True
        
        if root not in self:
            self.__parent_of[root] = root
            self.__rank_of[root] = 1
        
        if isinstance(element_or_set, set):
            for element in element_or_set:
                if element in self:
                    raise ValueError(f"The element {element} is already in the disjoint-set.")
                self.__parent_of[element] = root
        else:
            if element in self:
                raise ValueError(f"The element {element} is already in the disjoint-set.")
            self.__parent_of[element_or_set] = root
    
    def __iter__(self) -> Iterator[ST]:
        yield from self.__parent_of
    
    def __len__(self) -> int:
        return len(self.__parent_of)
    
    @property
    def parents(self) -> dict[ST, ST]:
        "The element to parent mapping."
        return self.__parent_of
    
    @property
    def ranks(self) -> dict[ST, ST]:
        "The element to rank mapping."
        return self.__rank_of
    
    def find_root(self, element: ST, compress: bool = True) -> ST:
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
        `ST@DisjointSet` - The root of the sub-set containing the given element.
        
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
    
    def find_root_recurse(self, element: ST, compress: bool = True) -> ST:
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
        `ST@DisjointSet` - The root of the sub-set containing the given element.
        
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
    
    def find_root_path_split(self, element: ST) -> ST:
        """
        Variant of the `find_root` method which partially compresses
        the path from the given element to the root during the iterative
        search for the root element. This is done by replacing every
        element's parent on the path with its grandparent instead.
        
        This avoids the need for a second loop to perform the path compression
        however it can only achieve half the level of compression as `find_root`.
        
        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.
        
        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given element.
        
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
    
    def find_root_path_halve(self, element: ST) -> ST:
        """
        Variant of the `find_root` method which partially compresses
        the path from the given element to the root during the iterative
        search for the root element. This is done by replacing every
        other element's parent on the path with its grandparent instead.
        
        This is similar to `find_root_path_split` in that it avoids the
        need for a second loop to perform path compression, except it
        can half the number of elements that need to be searched along
        to reach the root (since it only checks if every other element
        is the root), however it also only achieve half the level of
        compression as path splitting.
        
        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose root to find.
        
        Returns
        -------
        `ST@DisjointSet` - The root of the sub-set containing the given element.
        
        Raises
        ------
        `KeyError` - If the given element is not in this disjoint-set.
        """
        ## Set the parent of every checked element on the path to the root to
        ## be the parent of the its parent, and set the parent of the parent to
        ## be the next element to check, thus checking only every other element.
        while (parent := self.__parent_of[element]) != element:
            self.__parent_of[element] = self.__parent_of[parent]
            element = self.__parent_of[parent]
        return parent
    
    def find_path(self, element: ST) -> list[ST]:
        """
        Find the current path from the given element to the root element of its disjoint sub-set.
        
        Parameters
        ----------
        `element: ST@DisjointSet` - The element whose path to its sub-set's root element to find.
        
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
                 ) -> set[ST]:
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
                      ) -> dict[ST, set[ST]]:
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
        
        sets: dict[ST, set[ST]] = {}
        for element in self:
            sets.setdefault(self.__find_method(self, element, compress), set()).add(element)
        
        if cache:
            self.__sets = sets
            self.__is_changed = False
        return sets
    
    def __find_root_pair(self,
                         element_1: ST,
                         element_2: ST,
                         compress: Optional[bool] = None
                         ) -> tuple[ST, ST]:
        """
        Used by union methods, to simultaneously find the roots of two different
        elements of this disjoint-set, whose distinct sub-sets are to be unioned.
        """
        return (self.__find_method(self, element_1, compress),
                self.__find_method(self, element_2, compress))
    
    def union_left(self, element_1: ST, element_2: ST, compress: Optional[bool] = None) -> ST:
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
        return root_1
    
    def union_right(self, element_1: ST, element_2: ST, compress: Optional[bool] = None) -> ST:
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
        return root_2
    
    def union(self, element_1: ST, element_2: ST, compress: Optional[bool] = None) -> ST:
        """
        Union the sub-sets containing the given elements together,
        using the union-by-rank algorithm. This ensures that the
        smaller original sub-set is unioned onto the larger original
        sub-set. This is such that the root of the new combined set,
        is the root of the larger original sub-set.
        
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
        
        ## If the ranks of the roots are the same,
        ## we must increase the rank of the root that was unioned onto the other
        ## (this will always be root_2), as this root has now grown in size,
        ## as such, the rank will alwys stay proportionate to size.
        if self.__rank_of[root_1] == self.__rank_of[root_2]:
            self.__rank_of[root_2] = self.__rank_of[root_2] + 1
        
        return root_2
    
    def union_many(self, elements: Iterable[ST], compress: Optional[bool] = None) -> ST:
        """
        Union all elements in the given iterable of elements,
        and return the root of the resulting sub-set.
        """
        iter_: Iterator[ST] = iter(elements)
        try: first: ST = next(iter_)
        except: return
        for element in iter_:
            root: ST = self.union(first, element, compress)
        return root
    
    def is_connected(self, element_1: ST, *elements: ST, compress: Optional[bool] = None) -> bool:
        """
        Determine whether the two elements are in the same disjoint sub-set.
        
        If two elements are given, equivalent to: `self.find_root(element_1) == self.find_root(element_2)`.
        
        If more than two elements are given, equivalent to: `all(self.find_root(element_1) == self.find_root(other) for other in elements)`.
        """
        root_1: ST = self.__find_method(self, element_1, compress)
        return all(root_1 == self.__find_method(self, element, compress) for element in elements)

class DefaultFindMethod(enum.Enum):
    """
    The root finding methods that can be passed as argument to
    the constructor of a disjoint-set, and used as the default for;
        - Finding all disjoint sub-sets,
        - Union operations,
        - Checking if elements are connected (are in the same disjoint sub-set).
    """
    loop = (DisjointSet.find_root, True)
    loop_compress = (DisjointSet.find_root, True)
    recurse = (DisjointSet.find_root_recurse, True)
    recurse_compress = (DisjointSet.find_root_recurse, True)
    path_split = (DisjointSet.find_root_path_split, False)
    path_halve = (DisjointSet.find_root_path_halve, False)