###########################################################################
###########################################################################
## A general implementation of a genetic optimisation algorithm.         ##
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

"""General implementation of a genetic optimisation algorithm."""

import dataclasses
import enum
import itertools
import math
from abc import ABCMeta, abstractmethod
from fractions import Fraction
from functools import cached_property
from numbers import Number, Real
from random import getrandbits as randbits
from typing import Any, Generic, Iterable, Iterator, Literal, Type, TypeVar

import numpy as np
from numpy.random import Generator, choice, default_rng, random_integers
from typing_extensions import override

from aloy.auxiliary.moreitertools import (arg_max_n, chunk, cycle_for,
                                          getitem_zip, index_sequence)
from aloy.auxiliary.progressbars import ResourceProgressBar
from aloy.moremath.mathutils import normalize_to_sum
from aloy.optimization.decayfunctions import get_decay_function

# Need to be able to deal with degree of mutation based on the range.
# For arbitrary bases, this will be based on the order of the possible values
# in the base.

# How will we deal with constraints like those that will be needed for the
# membership functions in the fuzzy controllers where the values must be in
# ascending order?
# Can we use multiple chromosomes to encode: membership function limits, rule
# outputs, and module gains?

# Numerical gene base type.
NT = TypeVar("NT", bound=Number)

# Generic gene base type (used for arbitrary base types).
GT = TypeVar("GT", bound=Any)

# Generic chromosome type (i.e. genotype).
CT = TypeVar("CT", str, list)

# Generic solution type (i.e. phenotype).
ST = TypeVar("ST")


class GeneBase(Generic[CT], metaclass=ABCMeta):
    """Base class for gene base types."""

    @abstractmethod
    def random_chromosomes(self, length: int, quantity: int) -> list[CT]:
        """Return the given quantity of random chromosomes of the given length."""
        ...

    @abstractmethod
    def random_genes(self, quantity: int) -> list[GT]:
        """Return the given quantity of random genes."""
        ...


# TODO: Change to a normal class, take "bin", "hex", or "oct" as argument.
@dataclasses.dataclass(frozen=True)
class BitStringBase(GeneBase[str]):
    """
    Represents a bit-string gene base type.

    Genes can take only integer values between;
    0 and 2^bits-1 encoded in the given numerical base.

    Fields
    ------
    `name: str` - The name of the base type.

    `format_: str` - The string formatter symbol for
    converting from binary to the given numerical base type.

    `bits: int` - The number of bits needed to represent one gene.

    Properties
    ----------
    `values: int` - The number of possible values for one gene.

    `values_range: tuple[str]` - The possible values for one gene.

    Methods
    -------
    `chromosome_bits: (length: int) -> int` - Return the number of bits needed
    to represent a chromosome of the given length.

    `random_chromosomes: (length: int) -> str` - Return a random chromosome of
    the given length.
    """

    name: str
    format_: str
    bits: int

    def __str__(self) -> str:
        """Return the name of the base type and the number of bits per gene."""
        return f"{self.__class__.__name__} :: {self.name}, values: {self.all_values}, bits/gene: {self.bits}"

    @cached_property
    def total_values(self) -> int:
        """Return the number of values a gene can take for the given base type."""
        return 1 << self.bits

    @cached_property
    def all_values(self) -> tuple[str, ...]:
        """
        Return the range of possible values a gene can take for the given base
        type in ascending order.
        """
        return tuple(format(v, self.format_) for v in range(self.total_values))

    @cached_property
    def as_numerical_base(self) -> "NumericalBase[int]":
        """
        Return a numerical base representation of the given bit-string base
        type.
        """
        return NumericalBase(self.name, int, 0, (2 ** self.bits) - 1)

    def chromosome_bits(self, length: int) -> int:
        """
        Return the number of bits needed to represent a chromosome of the
        given length.
        """
        return self.bits * length

    def random_chromosomes(self, length: int, quantity: int) -> list[str]:
        """
        Return the given quantity of random chromosomes of the given length.
        """
        return [
            format(
                randbits(self.chromosome_bits(length)),
                self.format_
            ).zfill(length)
            for _ in range(quantity)
        ]

    def random_genes(self, quantity: int) -> list[str]:
        """Return the given quantity of random genes."""
        return [
            format(randbits(self.bits), self.format_)
            for _ in range(quantity)
        ]


@enum.unique
class BitStringBaseTypes(enum.Enum):
    """
    The standard gene base types for 'bit-string' choromsome encodings.

    Items
    -----
    `bin = GeneBase("binary", 'b', 1)` - A binary string represenation.
    This uses "base two", i.e. there are only two values a gene can take.

    `oct = GeneBase("octal", 'o', 3)` - An octal string representation.
    This uses "base eight", i.e. there are eight possible values a gene can
    take.

    `hex = GeneBase("hexadecimal", 'h', 4)` - A hexadecimal representation.
    This uses "base sixteen", i.e. there are sixteen possible values a gene
    can take.
    """

    bin = BitStringBase("binary", 'b', 1)
    oct = BitStringBase("octal", 'o', 3)
    hex = BitStringBase("hexadecimal", 'x', 4)


@dataclasses.dataclass(frozen=True)
class NumericalBase(GeneBase[list], Generic[NT]):
    """
    Represents an numerical base type.

    Genes can take any value from a given range.
    """

    name: str
    type_: Type[Real]
    min_range: NT
    max_range: NT

    def __post_init__(self) -> None:
        """Check that the given range is valid."""
        if self.min_range >= self.max_range:
            raise ValueError(f"Minimum of range must be less than maximum of range. Got; {self.min_range=} and {self.max_range=}.")

    def random_chromosomes(self, length: int, quantity: int) -> list[list[NT]]:
        """Return the given quantity of random chromosomes of the given length."""
        return list(chunk(random_integers(self.min_range, self.max_range, length * quantity), length, quantity, as_type=list))

    def random_genes(self, quantity: int) -> list[NT]:
        """Return the given quantity of random genes."""
        return random_integers(self.min_range, self.max_range, quantity).tolist()


@dataclasses.dataclass(frozen=True)
class ArbitraryBase(GeneBase[list], Generic[GT]):
    """
    Represents an arbitrary base type.

    Genes can take any value from a given set of values.
    """

    name: str
    values: tuple[GT]

    def __str__(self) -> str:
        """Return the name of the base type and the number of bits per gene."""
        return (f"{self.__class__.__name__} :: Values (total = {len(self.values)}): "
                + ", ".join(str(v) for v in self.values[:min(len(self.values, 5))])
                + (", ..." if len(self.values) > 5 else ""))

    def random_chromosomes(self, length: int, quantity: int) -> list[list[GT]]:
        """Return the given quantity of random chromosomes of the given length."""
        return chunk(choice(self.values, length * quantity), length, quantity, as_type=list)

    def random_genes(self, quantity: int) -> list[GT]:
        """Return the given quantity of random genes."""
        return choice(self.values, quantity).tolist()


class GeneticEncoder(Generic[ST], metaclass=ABCMeta):
    """
    Base class for genetic encoders.

    A genetic encoder can encode and decode a candidate solution to and from a
    chromosome (a sequence of genes). A candidiate solution is often referred
    to as a phenotype and its encoding as a chromosome as a genotype. The
    encoder must also define a fitness evaluation function, which is specific
    to the encoding. Where a fitness value defines the quality of the
    candidate solution.

    There is an important relation between the genetic encoding used, the
    fitness evaluation function and the other genetic operators involved in
    the algorithm; the recombinator, mutator and selector. The designer must
    ensure that the operators are compatible with the encoding, such that
    operators that modify chromosomes promotes an increase in fitness and
    generate valid solutions.
    """

    def __init__(
        self,
        chromosome_length: int,
        permutation: bool = False,
        base: BitStringBaseTypes | Literal["bin", "oct", "hex"] | list[GT] | NumericalBase | ArbitraryBase = "bin"  ## TODO
    ) -> None:
        """Create a genetic encoder with a given gene length and base type."""
        self.__chromosome_length: int = chromosome_length
        if isinstance(base, str):
            self.__base = BitStringBaseTypes[base].value
        elif isinstance(base, BitStringBaseTypes):
            self.__base = base.value
        elif isinstance(base, list):
            self.__base = ArbitraryBase("arbitrary", base)
        elif isinstance(base, (NumericalBase, ArbitraryBase)):
            self.__base = base
        else:
            raise TypeError(
                F"Unknown base; {base} of type {type(base)!r}. Expected one "
                "of: BitStringBase, str, list, NumericalBase, ArbitraryBase."
            )

    @property
    def chromosome_length(self) -> int:
        """Return the length of a chromosome."""
        return self.__chromosome_length

    @property
    def base(self) -> GeneBase:
        """Return the base type of a chromosome."""
        return self.__base

    @abstractmethod
    def encode(self, solution: ST) -> CT:
        """Return the solution encoded as a chromosome."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, chromosome: CT) -> ST:
        """Return the solution represented by the given chromosome."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_fitness(self, chromosome: CT) -> Real:
        """Return the fitness of the given chromosome."""
        raise NotImplementedError



class GeneticOperator(metaclass=ABCMeta):
    """Base class for genetic operators."""

    __slots__ = ("__generator",)

    def __init__(self, rng: Generator | int | None = None) -> None:
        """
        Super constructor for genetic operators.

        Parameters
        ----------
        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        if isinstance(rng, Generator):
            self.__generator: Generator = rng
        self.__generator: Generator = default_rng(rng)

    @property
    def generator(self) -> Generator:
        """Get the random number generator used by the genetic operator."""
        return self.__generator



class GeneticRecombinator(GeneticOperator):
    """
    Base class for genetic recombination operators.

    A genetic recombinator selects and swaps genes between two 'parent'
    chromosomes, in order to create two new 'offspring' chromosomes, each of
    which is some combination of the parents.

    A genetic recombinator is agnostic to the representation scheme and
    encoding used for chromosomes to solutions. This is because they only
    consider the sequence of genes (i.e. their order/position), not the
    encoding.

    In non-permutation recombinators, the offspring chromosomes may not be
    permutations of the parents. Swapped genes or genes sub-sequences should
    be between those in the same positions in the chromosome.

    In permutation recombinators, the offspring chromosomes must be
    permutations of the parents.
    """

    def __init__(self, rng: Generator | int | None = None) -> None:
        """
        Super constructor for genetic recombinators.

        Parameters
        ----------
        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)

    @abstractmethod
    def recombine(self, chromosome_1: CT, chromosome_2: CT) -> tuple[CT, CT]:
        """
        Generate two new offspring chromosomes by recombining the given parent
        chromosomes.
        """
        ...


class SplitCrossOver(GeneticRecombinator):
    """
    Class defining split cross-over recombinators.

    A split cross-over recombinator splits each chromosome into two
    sub-sequences at a random index over their length and 'crosses-over'
    (i.e. swaps) those sub-sequences between those chromosomes.

    Split cross-over is equivalent to, but more efficient than, a one-point
    cross-over recombinator.
    """

    __slots__ = ()

    def __init__(self, rng: Generator | int | None = None) -> None:
        """
        Create a split-crossover recombinator.

        Parameters
        ----------
        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)

    def recombine(self, chromosome_1: CT, chromosome_2: CT) -> tuple[CT, CT]:
        """
        Generate two new offspring chromosomes by recombining the given
        parent chromosomes using split cross-over.
        """
        point: int = self.generator.integers(1, len(chromosome_1) - 1)
        return ((chromosome_1[:point] + chromosome_2[point:]),
                (chromosome_2[:point] + chromosome_1[point:]))


class PointCrossOver(GeneticRecombinator):
    """
    Class defining N-point cross-over recombinators.

    An N-point cross-over recombinator chooses N random points (or indicies)
    over the length of the chromosomes, splits the chromosomes into N + 1
    sub-sequences at those points, and 'crosses-over' (i.e. swaps) those
    sub-sequences between those chromosomes.
    """

    __slots__ = ("__points",)

    def __init__(
        self,
        points: int,
        rng: Generator | int | None = None
    ) -> None:
        """
        Create an N-point cross-over recombinator.

        Parameters
        ----------
        `points: int = 1` - The number (N) of points to use for the cross-over.
        The N + 1 sub-sequences are created by the split and used for the
        cross-over.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)
        if not isinstance(points, int) or points < 1:
            raise ValueError(
                "Number of points must be an integer greater than zero. "
                f"Got; {points} of type {type(points)}."
            )
        self.__points: int = points

    def __get_subsequences(
        self,
        chromosome_1: CT,
        chromosome_2: CT,
        i: int,
        lp: int,
        rp: int
    ) -> tuple[CT, CT]:
        """
        Return the sub-sequences of the given chromosomes at the given
        indices.
        """
        if i % 2 == 0:
            return (chromosome_1[lp:rp], chromosome_2[lp:rp])
        return (chromosome_2[lp:rp], chromosome_1[lp:rp])

    def recombine(self, chromosome_1: CT, chromosome_2: CT) -> Iterable[CT]:
        """
        Generate two new offspring chromosomes by recombining the given parent
        chromosomes using N-point cross-over.
        """
        points = np.zeros(self.__points)
        points[-1] = len(chromosome_1)
        points[1:self.__points-1] = self.generator.integers(0, len(chromosome_1), self.__points).sort()
        sum_func = lambda x: "".join(x) if isinstance(chromosome_1, str) else sum
        return tuple(map(sum_func, zip(self.__get_subsequences(chromosome_1, chromosome_2, i, lp, rp)
                                       for i, lp, rp in enumerate(zip(points[:-1], points[1:])))))


class UniformSwapper(GeneticRecombinator):
    """
    Class defining uniform-swapper recombinators.

    A uniform-swapper recombinator iterates over the length of the chromosomes,
    randomly choosing for each pair of genes at a given index, whether to swap
    them to the other chromosome or leave them in the original
    (with 50% probability).
    """

    __slots__ = ()

    def __init__(self, rng: Generator | int | None = None) -> None:
        """
        Create a uniform-swapper recombinator.

        Parameters
        ----------
        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)

    def recombine(self, chromosome_1: CT, chromosome_2: CT) -> tuple[CT, CT]:
        """
        Generate two new offspring chromosomes by recombining the given parent
        chromosomes using uniform swapping.
        """
        new_chromosome_1: list = []
        new_chromosome_2: list = []
        choices = self.generator.integers(0, 1, len(chromosome_1))
        for gene_1, gene_2, choice_ in zip(
            chromosome_1, chromosome_2, choices
        ):
            if choice_:
                new_chromosome_1.append(gene_1)
                new_chromosome_2.append(gene_2)
            else:
                new_chromosome_1.append(gene_2)
                new_chromosome_2.append(gene_1)
        if isinstance(chromosome_1, str):
            return ("".join(new_chromosome_1), "".join(new_chromosome_2))
        return new_chromosome_1, new_chromosome_2


class PermutationSwapper(GeneticRecombinator):
    """
    Class defining permutation-swapper recombinators.

    A permutation-swapper recombinator chooses N random points (or indicies)
    over the length of the first chromosome, then for each point it swaps the
    gene at that point in the first chromosome with the gene at the same point
    in the second chromosome. For either chromosome, if the new gene is
    already in the chromosome, then its (first) occurance is replaced with the
    old gene to keep the genes unique (this fails if more than one occurance
    exists).
    """

    __slots__ = ("__swaps",)

    def __init__(self, swaps: int, rng: Generator | int | None = None) -> None:
        """
        Create a permutation-swapper recombinator.

        Parameters
        ----------
        `swaps: int = 1` - The number of swaps to use for the cross-over.
        The N + 1 sub-sequences are created by the split and used for the
        cross-over.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)
        if not isinstance(swaps, int) or swaps < 1:
            raise ValueError(
                "Number of swaps must be an integer greater than zero. "
                f"Got; {swaps} of type {type(swaps)}."
            )
        self.__swaps: int = swaps

    def recombine(self, chromosome_1: CT, chromosome_2: CT) -> tuple[CT, CT]:
        """
        Generate two new offspring chromosomes by recombining the given parent
        chromosomes using permutation swapping.
        """
        new_chromosome_1: list = chromosome_1.copy()
        new_chromosome_2: list = chromosome_2.copy()
        swaps = self.generator.integers(0, len(chromosome_1), self.__swaps)
        for swap_from_index in swaps:
            gene_1 = chromosome_1[swap_from_index]
            gene_2 = chromosome_2[swap_from_index]
            if gene_1 != gene_2:
                if gene_2 in new_chromosome_1:
                    new_chromosome_1[chromosome_1.index(gene_2)] = gene_1
                new_chromosome_1[swap_from_index] = gene_2
                if gene_1 in new_chromosome_2:
                    new_chromosome_2[chromosome_2.index(gene_1)] = gene_2
                new_chromosome_2[swap_from_index] = gene_1
        return new_chromosome_1, new_chromosome_2


class GeneticMutator(GeneticOperator):
    """
    Base class for genetic mutation operators.

    A mutator is the genetic operator that promotes diversity in a population
    of possible candidate solutions by randomly modifying the genes of a
    chromosome. A mutator therefore encourages exploration of the search space,
    by causing the population to spread across a larger area of the search
    space, and evaluate new possible candidate solutions that may be closer to
    the optimum than the existing solutions in the population.

    In contrast to recombinators, mutators focus on local search, since they
    cause relatively small changes in the chromosome and therefore small
    movements in the search space. Optimisation based on mutation is a
    relatively slow process compared to recombination.

    It is therefore one of the fundamental operators in causing evolution
    (i.e. change) in a population, and allowing a genetic algorithm to improve
    the fitness (i.e. quality) of the candidate solutions in that population,
    towards finding the glocal optimum.

    In theory, mutation also helps prevent a genetic algorithm from getting
    stuck in local optima, by ensuring the population does not become too
    similar to each other, thus slowing convergence to the global optimum, and
    discouraging exploitation.

    Sub-classing
    ------------
    The class provides a generic method `mutate`, which if overridden in
    sub-classes must be able to handle any genetic base type.

    If the designer however wishes the mutator to treat 'bit-string' and
    arbitrary bases differently, two seperate specialised methods can be
    overridden instead;
        - `numeric_mutate` any of the standard numeric 'bit-string' bases;
          binary, octal, hexadecimal,
        - `arbitrary_mutate` any other arbitrary base over some fixed alphabet,

    By default, these just call the standard mutate method.

    As such, to sub-class genetic mutator, one must override either;
        - the generic mutate method (and possibly any of the specialised
          mutate methods),
        - or both of the specialised mutatre methods and not the generic
          mutate method.
    """

    __slots__ = ()

    def __init__(self, rng: Generator | int | None = None) -> None:
        """
        Super constructor for mutation operators.

        Parameters
        ----------
        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)

    def mutate(self, chromosome: CT, base: GeneBase) -> CT:
        """Mutate the given chromosome encoded in the given base."""
        return NotImplemented

    def bitstring_mutate(self, chromosome: str, base: BitStringBase) -> str:
        """Point mutate the given chromosome encoded in the given bit-string base."""
        return "".join([str(g) for g in self.mutate(list(chromosome), base.as_numerical_base)])

    def numerical_mutate(self, chromosome: list[NT], base: NumericalBase[NT]) -> list[NT]:
        """Mutate the given chromosome encoded in the given numeric base."""
        return self.mutate(chromosome, base)

    def arbitrary_mutate(self, chromosome: list[GT], base: ArbitraryBase[GT]) -> list[GT]:
        """Mutate the given chromosome encoded in the given arbitrary base."""
        return self.mutate(chromosome, base)


class PointMutator(GeneticMutator):
    """
    Class defining point mutators.

    A point mutator randomly selects one or more genes in a chromosome (with
    uniform probability with replacement), and changes their values to some
    random value (with uniform probability). In a binary base representation,
    chosen genes are simply `flipped` to the opposite value.

    Point mutators are simple and efficient, but are not appropriate for
    permutation problems, since they do not maintain the same set of genes in
    the chromosome.
    """

    __slots__ = ("__points",)

    def __init__(self, points: int = 1, rng: Generator | int | None = None) -> None:
        """
        Create a point mutator.

        A point mutator randomly selects one or more genes in a chromosome
        (with uniform probability with replacement), and changes their values
        to some random value (with uniform probability).

        Parameters
        ----------
        `points: int = 1` - The number of genes in the chromosome to point
        mutate.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)
        if not isinstance(points, int) or points < 1:
            raise ValueError("Number of points must be an integer greater than zero. "
                             f"Got; {points} of type {type(points)}.")
        self.__points: int = points

    def mutate(self, chromosome: CT, base: GeneBase) -> CT:
        """Point mutate the given chromosome encoded in the given base."""
        for index, gene in zip(self.generator.integers(len(chromosome), size=self.__points),
                               base.random_genes(self.__points)):
            chromosome[index] = gene
        return chromosome

class PairSwapMutator(GeneticMutator):
    """
    CLass defining pair swap mutators.

    Pick N pairs of genes at random (with uniform probability distribution),
    and swap the values of the genes between each pair. Pair swap mutators are
    appropriate for permutation problems, where the set of values need to be
    preserved, but the order can be changed.
    """

    __slots__ = ("__pairs",)

    def __init__(
        self,
        pairs: int = 1,
        rng: Generator | int | None = None
    ) -> None:
        """
        Create a pair swap mutator.

        A pair swap mutator randomly selects one or more pairs of genes in a
        chromosome (with uniform probability with replacement), and swap the
        values of the genes between each pair.

        Parameters
        ----------
        `pairs: int = 1` - The number of pairs of genes in the chromosome to
        swap mutate. If greater than one, the same gene may be chosen and
        swapped multiple times.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        super().__init__(rng)
        if not isinstance(pairs, int) or pairs < 1:
            raise ValueError(
                "Number of pairs must be an integer greater than zero. "
                f"Got; {pairs} of type {type(pairs)}."
            )
        self.__pairs: int = pairs

    def __one_pair_swap_mutate(self, chromosome: CT, base: GeneBase) -> CT:
        """Swap mutate the given chromosome encoded in the given base."""
        index_1, index_2 = self.generator.integers(
            len(chromosome),
            size=(self.__pairs * 2)
        )
        chromosome[index_1], chromosome[index_2] = (
            chromosome[index_2], chromosome[index_1]
        )
        return chromosome

    def mutate(self, chromosome: CT, base: GeneBase) -> CT:
        """Swap mutate the given chromosome encoded in the given base."""
        if self.__pairs == 1:
            return self.__one_pair_swap_mutate(chromosome, base)
        pairs = chunk(self.generator.integers(len(chromosome), size=(self.__pairs * 2)), 2)
        for index_1, index_2 in pairs:
            chromosome[index_1], chromosome[index_2] = (
                chromosome[index_2], chromosome[index_1]
            )
        return chromosome


class PoolShuffleMutator(GeneticMutator):
    """
    Class defining pool shuffle mutators.

    Pool shuffle mutators pick a random set of genes (with uniform probability), and randomly shuffle/scramble (with uniform probability) the values between them.

    Pool shuffle mutators are appropriate for permutation problems, where the set of values need to be preserved, but the order can be changed.
    """

    __slots__ = ("__pool_size",)

    def __init__(self, pool_size: int | float = 0.5, rng: Generator | int | None = None) -> None:
        """
        Create a pool shuffle mutator.

        Parameters
        ----------
        `pool_size: int | float = 0.5` - The number of genes in the chromosome to shuffle.
        Either an integer defining a fixed number of genes, or a float defining a factor
        of the total number of genes in the chromosome to shuffle.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.

        Raises
        ------
        `ValueError` - If;
            - the pool size is not a positive integer or float,
            - the pool size is an integer less than 2,
            - the pool size is a float less than 0.0 or greater than 1.0.
        """
        super().__init__(rng)
        if (not isinstance(pool_size, (int, float))
            or (isinstance(pool_size, int) and pool_size < 2)
            or (isinstance(pool_size, float) and (pool_size < 0.0 or pool_size > 1.0))):
            raise ValueError("Pool size must be an integer greater than zero, or a float between 0.0 and 1.0."
                             f"Got; {pool_size} of type {type(pool_size)}.")
        self.__pool_size: int | float = pool_size

    def mutate(self, chromosome: CT, base: GeneBase) -> CT:
        """Shuffle mutate the given chromosome encoded in the given base."""
        pool_size = self.__pool_size
        if isinstance(self.__pool_size, float):
            pool_size = math.ceil(self.__pool_size * len(chromosome))
        pool = self.generator.integers(len(chromosome), size=pool_size)
        new_pool = self.generator.permutation(pool)
        for from_index, to_index in zip(pool, new_pool):
            chromosome[from_index], chromosome[to_index] = chromosome[to_index], chromosome[from_index]
        return chromosome


# TODO: Implement PoolInversionMutator
class SequenceShuffleMutator(GeneticMutator):
    """
    Class defining sequence shuffle mutators.

    Sequence shuffle mutators pick a random sequence of genes (with uniform
    probability), and shuffle/scramble (with uniform probability) the values
    between them.

    Sequence shuffle mutators are appropriate for permutation problems, where
    the set of values need to be preserved, but the order can be changed.
    """

    __slots__ = ("__sequence_length",)

    def __init__(self, rng: Generator | int | None = None) -> None:
        super().__init__(rng)


# TODO: Implement SequenceInversionMutator
class SequenceInversionMutator(SequenceShuffleMutator):
    """
    Class defining inversion mutators.

    Pick two different genes at random (with uniform probability distribution),
    and reverse (invert) the order of the contiguous sub-sequence of values
    between them.

    Similar to shuffle, except invert (flip, or pivot around its center) the
    sub-sequence instead of performing the expensive shuffle operation.
    This still disrupts the order, but mostly preserves adjacency of gene
    values (within the sub-sequence only).
    """

    __slots__ = ("__sequence_length",)

    def __init__(self, rng: Generator | int | None = None) -> None:
        super().__init__(rng)


# scaling_scheme: Literal["lin", "sigma", "power-law"],
# selection_scheme: Literal["prop", "rank"],  ## "trans-ranked", "tournament", "SUS" : Params for tournament >> tournament_size: int, inner_selection: best | prop | rank | trans-ranked | SUS
class GeneticSelector(metaclass=ABCMeta):
    """
    Base class for genetic selection operators.

    Selection operators expose two functions;
        - A select method of selecting a subset of a population of candidate solutions
          to be used for culling and reproduction to generate the next generation,
        - A scale method of scaling the fitness of the candidate solutions in the population.
    """

    __slots__ = (
        "__requires_sorted",
        "__requires_normalised",
        "__generator"
    )

    def __init__(
        self,
        requires_sorted: bool,
        requires_normalised: bool,
        rng: Generator | int | None = None
    ) -> None:
        """
        Super constructor for selection operators.

        Parameters
        ----------
        `requires_sorted: bool` - Whether the fitness values of the
        chromosomes must be sorted in ascending order.

        `rng: Generator | int | None` - Either an random number generator
        instance, or a seed for the selector to create its own, None generates
        a random seed.
        """
        self.__requires_sorted: bool = requires_sorted
        self.__requires_normalised: bool = requires_normalised
        if isinstance(rng, Generator):
            self.__generator: Generator = rng
        self.__generator: Generator = default_rng(rng)

    @property
    def requires_sorted(self) -> bool:
        """Whether the selector requires the fitness values to be sorted in ascending order."""
        return self.__requires_sorted

    @property
    def requires_normalised(self) -> bool:
        """Whether the selector requires the fitness values to be normalised to sum to one."""
        return self.__requires_normalised

    @property
    def generator(self) -> Generator:
        """Get the random number generator used by the selector."""
        return self.__generator

    @abstractmethod
    def select(
        self,
        population: list[CT],
        fitness: list[float],
        quantity: int
    ) -> Iterable[int]:
        """
        Select the given number of chromosomes from the population.

        Must return an iterable of the indices of the selected chromosomes.
        """
        ...

    def scale(self,
              fitness: list[Real]
              ) -> list[Real]:
        """Scale the given fitness values."""
        return fitness

    @staticmethod
    def validate(
        population: list[CT],
        fitness: list[float],
        quantity: int
    ) -> None:
        """Validate arguments for the selection operator."""
        if len(population) != len(fitness):
            raise ValueError("Population and fitness values must be the same length.")
        if len(population) == 0:
            raise ValueError("Population must contain at least one chromosome to select from.")
        if quantity < 1:
            raise ValueError("Quantity of chromosomes to select must be greater than zero.")


class ProportionateSelector(GeneticSelector):
    """
    Selects chromosomes from a population with probability proportionate to
    fitness with replacement.
    """

    def __init__(self) -> None:
        """Create a new proportionate selector."""
        super().__init__(requires_sorted=False, requires_normalised=True)

    @override
    def select(
        self,
        population: list[CT],
        fitness: list[float],
        quantity: int
    ) -> Iterable[int]:
        """
        Select a given quantity of chromosomes from the population with
        probability proportionate to fitness with replacement.

        Returns an iterable of the indices of the selected chromosomes.
        """
        return self.generator.choice(len(population), size=quantity, p=fitness)


class RankedSelector(GeneticSelector):
    """
    Selects chromosomes from a population with probability proportionate to
    fitness rank with replacement.
    """

    def __init__(self) -> None:
        """Create a new ranked selector."""
        super().__init__(requires_sorted=True, requires_normalised=False)

    def select(
        self,
        population: list[CT],
        fitness: list[float],
        quantity: int
    ) -> Iterable[int]:
        """
        Select a given quantity of chromosomes from the population with
        probability proportionate to fitness rank with replacement.

        Returns an iterable of the indices of the selected chromosomes.
        """
        pop_size: int = len(population)
        rank_sum: float = (pop_size + 1) * (pop_size / 2.0) 
        ranks: list[float] = [(i / rank_sum) for i in range(pop_size)]
        return self.generator.choice(pop_size, size=quantity, p=ranks)


class TournamentSelector(GeneticSelector):
    """
    Class defining tournament selectors.

    Tournament selection selects chromosomes from a population by
    pitching them against each other in competitions called tournaments.

    Individuals are selected for inclusion in tournaments with uniform probability
    with replacement, and the tournamenet winner(s) are selected either by;
    choosing the highest fitness individual(s), or randomly with probability
    either proportionate to fitness or ranked fitness.
    """

    def __init__(self,
                 tournamenet_size: int = 3,
                 n_chosen: int = 1,
                 inner_selector: ProportionateSelector | RankedSelector | None = None
                 ) -> None:
        """
        Create a new tournament selector.
    
        Selecting either the best or proportionate to fitness or fitness rank.
        """
        if tournamenet_size < 2:
            raise ValueError(f"Tournament size must be greater than one. Got; {tournamenet_size}")
        if not n_chosen < tournamenet_size:
            raise ValueError("Number of chosen chromosomes must be less than the tournament size. "
                             f"Got; {tournamenet_size=}, {n_chosen=}.")
        super().__init__(requires_sorted=(inner_selector is not None and inner_selector.requires_sorted),
                         requires_normalised=(inner_selector is not None and inner_selector.requires_normalised))
        self.__tournament_size: int = tournamenet_size
        self.__n_chosen: int = n_chosen
        self.__inner_selector: ProportionateSelector | RankedSelector | None = inner_selector

    def select(
        self,
        population: list[CT],
        fitness: list[float],
        quantity: int
    ) -> Iterable[int]:
        """
        Select a given quantity of chromosomes from the population by pitching them against each other in tournaments.

        Chromosomes selected for tournamenets are selected with uniform probability with replacement.
        """
        tournaments: list[list[tuple[CT, float]]] = chunk(
            index_sequence(
                getitem_zip(population, fitness),
                self.generator.integers(
                    len(population),
                    self.__tournament_size * quantity
                )
            ),
            self.__tournament_size,
            quantity
        )

        winner_lists: Iterator[Iterable[int]]
        if self.__inner_selector is None:
            winner_lists = (
                arg_max_n(
                    tournament,
                    n=self.__n_chosen,
                    key=lambda x: x[1]
                )
                for tournament in tournaments
            )
        else:
            winner_lists = (
                self.__inner_selector.select(
                    *zip(*tournament),
                    quantity=self.__n_chosen
                )
                for tournament in tournaments
            )

        return itertools.chain.from_iterable(winner_lists)

    def scale(self, fitness: list[Real]) -> list[Real]:
        """
        Scale the given fitness values, by default calling the inner
        selector's scale method.
        """
        if self.__inner_selector is not None:
            return self.__inner_selector.scale(fitness)
        return super().scale(fitness)


# class SelectorCombiner(Selector):
#     """Can combine multiple selection schemes and transition between using different ones a various stages of the search."""
#     pass


@dataclasses.dataclass(frozen=True)
class GeneticAlgorithmSolution:
    """A solution to a genetic algorithm."""

    best_individual: CT
    best_fitness: Fraction
    population: list[CT] ## TODO Order the population such that the highest fitness individuals occur first.
    fitness_values: list[Fraction]
    max_generations_reached: bool = False
    max_fitness_reached: bool = False
    stagnation_limit_reached: bool = False


class GeneticSystem:
    """
    A genetic system.
    
    Genetic Operator
    ----------------
    
    1. Solution Encoder - Representation as chromosomes formed by a sequence of genes.
    
        A solution is called the phenotype,
        and its encoding that the genetic algorithm operates on and evolves is called the genotype.
        
        The decoder function is required, the encoder is optional.
        
        In order to evaluate fitness and to return a best-fit solution at the end,
        the algorithm needs to be able to decode genotypes to phenotypes.
        
        When initialising a problem, one may want to specific an initial set of solutions to evolve,
        to do this the algorithm needs to be able to encode phenotypes to genotypes.
        
        - Numeric representation; binary, octal, or hexadecimal sequence;
            - In binary, the nucleotide bases are; 1 or 2, for example.
            - For some problems, it is difficult to encode a solution in binary,
              you may need to split the chromosome up into mutliple sub-sequences
              to encode different properties of the solution. The quality of the complete
              solution is then the sum of the quality of its parts.
        
        - Identifier list representation - Any sized set of arbitrary identifiers for properties or elements of a solution;
            - The nucleotide bases are any of a set of identifiers; "london", "birmingham", "manchester", etc.
            - Useful when solution length is known, but ordering needs to be optimised.
        
        - Multiple chromosome representation
    
    2. Selection Scheme and Fitness Function
    
        - Deterministic Selection: Only best n < pop_size reproduce, in this case the fitness function is not necessary.
        - Proportional Seletion: Reproduction chances are proportional to fitness value.
        - Ranked Fitness Selection: Solutions are ranked according to fitness, reproduction chance proportional to fitness.
        - Tournament Selection: Solutions compete against each other, fittest wins and gets to reproduce.

        - Elitism in selection:
        - Convergence and diversity biases in selection: Affect selection pressure towards high values of fitness, and thus exploration/exploitation trade-off.
        - Boltzmann decay for biases:
            ...In Boltzmann selection, a continuously varying temperature controls the rate of selection according to a preset schedule.
            The temperature starts out high, which means that the selection pressure is low.
            The temperature is gradually lowered, which gradually increases the selection pressure, thereby allowing the GA to narrow in more closely to the best part of the search space while maintaining the appropriate degree of diversity...

    3. Genetic Recombinator

    4. Genetic Mutator

    Algorithm procedure/structure
    -----------------------------

    1. population initialisation

    2. population evaluation

    3. stopping condition -> best solution

    4. selection

    5. crossover

    6. mutation

    7. population evaluation and update 3
    """

    __slots__ = (
        "__encoder",
        "__selector",
        "__recombinator",
        "__mutator",
        "__random_generator"
    )

    def __init__(
        self,
        encoder: GeneticEncoder,
        selector: GeneticSelector,
        recombinator: GeneticRecombinator,
        mutator: GeneticMutator
    ) -> None:
        """
        Create a genetic system.

        Parameters
        ----------
        `encoder: GeneticEncoder` - An encoder instance.

        `selector: GeneticSelector` - A selector instance. The selector is
        used for both population culling and population growth (reproduction)
        phases. It is used to select which individuals; survive culling to be
        eligible to reproduce, survive through the reproduction phase, and
        that get to actually reproduce.

        `recombinator: GeneticRecombinator` - A recombinator instance. The
        recombinator is used only in the population growth (reproduction)
        phase. It is used to combine selected pairs of 'parent' individuals
        from an existing generation, to generate pairs of new 'child'
        individuals for the next generation, that are some mix of the parents'
        genes.

        `mutator: GeneticMutator` - A mutator instance. The mutator is used
        only in the mutation phase. It is used to mutate the genes of
        individuals in the population, to introduce new genetic material into
        the population, and to prevent the population from converging on a
        local optimum.
        """
        self.__encoder: GeneticEncoder = encoder
        self.__selector: GeneticSelector = selector
        self.__recombinator: GeneticRecombinator = recombinator
        self.__mutator: GeneticMutator = mutator
        self.__random_generator: Generator = default_rng()

    def set_operators(
        self,
        selector: GeneticSelector | None = None,
        recombinator: GeneticRecombinator | None = None,
        mutator: GeneticMutator | None = None
    ) -> None:
        """
        Set the genetic operators.

        A value of None leaves the operator unchanged.
        """
        self.__selector: GeneticSelector = selector
        self.__recombinator: GeneticRecombinator = recombinator
        self.__mutator: GeneticMutator = mutator

    def run(
        self,
        init_pop_size: int,
        max_pop_size: int,
        expansion_factor: float = 1.5,

        survival_factor: float = 0.75,
        survival_factor_final: float = 0.75,
        survival_factor_growth_type: Literal["lin", "pol", "exp", "sin"] = "lin",
        survival_factor_growth_start: int = 0,
        survival_factor_growth_end: int | Literal["threshold", "stagnation"] = None,
        survival_factor_growth_rate: float = 0.0,

        survival_elitism_factor: float | None = None,
        survival_elitism_factor_final: float | None = None,
        survival_elitism_factor_growth_type: Literal["lin", "pol", "exp", "sin"] | None = None,
        survival_elitism_factor_growth_start: int | None = None,
        survival_elitism_factor_growth_end: int | Literal["threshold", "stagnation"] | None = None,
        survival_elitism_factor_growth_rate: float | None = None,

        reproduction_elitism_factor: float = 0.0,
        reproduction_elitism_factor_final: float = 0.25,
        reproduction_elitism_factor_growth_type: Literal["lin", "pol", "exp", "sin"] = "sin",
        reproduction_elitism_factor_growth_start: int = 0,
        reproduction_elitism_factor_growth_end: int | Literal["threshold", "stagnation"] | None = None,
        reproduction_elitism_factor_growth_rate: float = 1.0,

        mutation_factor: float = 2.0,
        mutation_factor_final: float = 0.5,
        mutation_factor_growth_type: Literal["lin", "pol", "exp", "sin"] = "sin",
        mutation_factor_growth_start: int = 0,
        mutation_factor_growth_end: int | Literal["threshold", "stagnation"] | None = None,
        mutation_factor_growth_rate: float = 1.0,
        mutate_all: bool = False,

        mutation_strength: float = 1.0,
        mutation_strength_final: float = 0.05,
        mutation_strength_decay_type: Literal["lin", "pol", "exp", "sin"] = "sin",
        mutation_strength_decay_start: int = 0,
        mutation_strength_decay_end: int | Literal["threshold", "stagnation"] | None = None,
        mutation_strength_decay_rate: float = 1.0,

        max_generations: int | None = 100,  # If not none this is the maximum number of generations to run the algorithm.
        fitness_threshold: float | None = None,  # If not none this is the fitness threshold that must be reached to stop the algorithm.
        # fitness_proportion: float | int | None = None,  # If not none this proportion of the population must be above the fitness threshold to stop the algorithm.
        stagnation_limit: float | int | None = None,  # If not none this is the number of generations that the best fitness individual(s) can stay the same before stopping the algorithm.
        # stagnation_proportion: float | int | None = 0.25,  # If not none this proportion of the population must be stagnated to stop the algorithm.

        ## These are used only for proportional fitness
        diversity_bias: float = 0.95,
        diversity_bias_final: float = 0.00,
        diversity_bias_decay_type: Literal["lin", "pol", "exp", "sin"] = "exp",  # ["threshold-converge", "stagnation-diverge"]
        ##      - converge towards fitness threshold - proportional to difference between mean fitness and fitness threshold,
        ##      - converge on rate of change towards fitness threshold,
        ##      - diverge on stagnation on best fittest towards fitness threshold. Increase proportional to diversity_bias * (stagnated_generations / stagnation_limit). This should try to explore around the solution space when the best fitness gets stuck on a local minima.
        diversity_bias_decay_start: int = 0,
        diversity_bias_decay_end: int | None = None,
        diversity_bias_decay_rate: float = 1.0

    ) -> GeneticAlgorithmSolution:
        """
        Run the genetic algorithm.

        Parameters
        ----------
        `survival_factor: Fraction` - Survival factor defines how much culling (equal to 1.0 - survival factor) we have, i.e. how much of the population for a given generation does not survive to the reproduction stage, and are not even considered for selection for recombination/reproduction.
        Low survival factor encourages exploitation of better solutions and speeds up convergence, by culling all but the best individuals, and allowing only the best to reproduce and search (relatively) locally to those best.

        `survival_factor_rate: Fraction` - 

        `survival_factor_change: Literal["decrease", "increase"]` - Usually, if replacement is enabled, it is desirable to start with a high survive factor to promote early exploration of the search space and decrease the factor to promote greater exploitation
        as the search progresses, focusing search towards the very best solutions it has found.

        `stagnation_proportion: Fraction` - If given and not none, return if the average fitness of the best fitting fraction of the population is stagnated (does not increase) for a number of generations equal to the stagnation limit.
        This intuition is that the search should stop only if a "large" proportion of the best quality candidates are not making significant improvement for a "long" time.
        Otherwise, return of the fitness of the best fitting individual is stagenated for the stagnation limit.
        If only the best fitting individual is used as the test of stagnation, it may result in a premature return of the algorithm, when other high fitness individuals would have achieved better fitness that the current maximum if allowed to evolve more
        particularly by creep mutation, which can we time consuming.

        `mutation_step_size: Fraction` - The degree of mutation or "step size" (i.e. the amount a single gene can change), change between totally random mutation and creep mutation based on generations or fitness to threshold.

        `max_generations: Optional[int]` -

        `fitness_threshold: Optional[Fraction]` -

        `fitness_proportion: Optional[Fraction | int]` - Return if the average fitness of the best fitness fraction of the population is above the fitness threshold.

        `stagnation_limit: Optional[int | Fraction]` - The maximum number of stagnated generations before returning.

        `stagnation_proportion: Optional[Fraction | int] = 0.10` -

        Stop Conditions
        ---------------
        The algorithm runs until one of the following stop conditions has been reached:

            - A solution is found that reaches or exceeds the fitness threshold,
            - The maximum generation limit has been reached,
            - The maximum running time or memory usage has been reached,
            - The best fit solutions have stagnated (reached some maxima) such that more generations are not increasing fitness,
              the algorithm may have found the global maxima, or it may be stuck in a logcal maxima.

        Notes
        -----
        To perform steady state selection for reproduction set;
            - survival_factor = 1.0 - X, where X is fraction of individuals to be culled,
            - survival_elitism_factor = 1.0, such that only the best survive (and the worst are culled) deterministically.
        """
        ## TODO Add logging, data collection, and data visualisation.
        if survival_factor >= Fraction(1.0):
            raise ValueError("Survival factor must be less than 1.0."
                             f"Got; {survival_factor=}.")

        if expansion_factor * survival_factor <= 1.0:
            raise ValueError("Population size would shrink or not grow "
                             f"with; {expansion_factor=}, {survival_factor=}."
                             "Their multiple must be greater than 1.0.")

        if stagnation_limit is not None and not isinstance(stagnation_limit, int):
            if max_generations is None:
                raise TypeError("Stagnation limit must be an integer if the maximum generations is not given or None."
                                f"Got; {stagnation_limit=} of type {type(stagnation_limit)} and {max_generations=} of {type(max_generations)}.")
            stagnation_limit = int(stagnation_limit * max_generations)

        population: list[CT] = self.create_population(init_pop_size)
        fitness_values: list[float] = [
            self.__encoder.evaluate_fitness(individual)
            for individual in population
        ]

        # If elitism is enabled for either selection or mutation then the population and their fitness values need to be ordered.
        if survival_elitism_factor is not None:
            population, fitness_values = zip(*sorted(zip(population, fitness_values),
                                                        key=lambda item: item[1]))
            population = list(population)
            fitness_values = list(fitness_values)

        max_fitness_index, max_fitness = max(enumerate(fitness_values), key=lambda item: item[1])
        generation: int = 0

        # Variables for checking stagnation
        best_individual_achieved: CT = population[max_fitness_index]
        best_fitness_achieved: float = max_fitness
        stagnated_generations: int = 0

        if reproduction_elitism_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Reproduction elitism factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {reproduction_elitism_factor_growth_end=} of type {type(reproduction_elitism_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            reproduction_elitism_factor_growth_end = max_generations
        if diversity_bias_decay_end is None:
            if max_generations is None:
                raise ValueError("Diversity bias decay end must be given if the maximum generations is not given or None."
                                 f"Got; {diversity_bias_decay_end=} of type {type(diversity_bias_decay_end)} and {max_generations=} of {type(max_generations)}.")
            diversity_bias_decay_end = max_generations
        if mutation_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Mutation factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {mutation_factor_growth_end=} of type {type(mutation_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            mutation_factor_growth_end = max_generations
        if mutation_strength_decay_end is None:
            if max_generations is None:
                raise ValueError("Mutation strength growth end must be given if the maximum generations is not given or None."
                                 f"Got; {mutation_strength_decay_end=} of type {type(mutation_strength_decay_end)} and {max_generations=} of {type(max_generations)}.")
            mutation_strength_decay_end = max_generations

        def _get(_var, _default):
            if _var is None:
                return _default
            return _var

        reproduction_elitism_factor_growth_start = _get(reproduction_elitism_factor_growth_start, 0)
        diversity_bias_decay_start = _get(diversity_bias_decay_start, 0)
        mutation_factor_growth_start = _get(mutation_factor_growth_start, 0)
        mutation_strength_decay_start = _get(mutation_strength_decay_start, 0)

        if survival_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Survival factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {survival_factor_growth_end=} of type {type(survival_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            survival_factor_growth_end = max_generations
        if reproduction_elitism_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Reproduction elitism factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {reproduction_elitism_factor_growth_end=} of type {type(reproduction_elitism_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            reproduction_elitism_factor_growth_end = max_generations
        if survival_elitism_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Survival elitism factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {survival_elitism_factor_growth_end=} of type {type(survival_elitism_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            survival_elitism_factor_growth_end = max_generations
        if diversity_bias_decay_end is None:
            if max_generations is None:
                raise ValueError("Diversity bias decay end must be given if the maximum generations is not given or None."
                                 f"Got; {diversity_bias_decay_end=} of type {type(diversity_bias_decay_end)} and {max_generations=} of {type(max_generations)}.")
            diversity_bias_decay_end = max_generations
        if mutation_factor_growth_end is None:
            if max_generations is None:
                raise ValueError("Mutation factor growth end must be given if the maximum generations is not given or None."
                                 f"Got; {mutation_factor_growth_end=} of type {type(mutation_factor_growth_end)} and {max_generations=} of {type(max_generations)}.")
            mutation_factor_growth_end = max_generations

        reproduction_elitism_factor_growth_rate = _get(reproduction_elitism_factor_growth_rate, 1.0)
        diversity_bias_decay_rate = _get(diversity_bias_decay_rate, 1.0)
        mutation_factor_growth_rate = _get(mutation_factor_growth_rate, 1.0)
        mutation_strength_decay_rate = _get(mutation_strength_decay_rate, 1.0)

        get_df = get_decay_function
        survival_factor_growth_func = None
        reproduction_elitism_factor_growth_func = None
        survival_elitism_factor_growth_func = None
        diversity_bias_decay_func = None
        mutation_factor_growth_func = None
        mutation_strength_decay_func = None

        if survival_factor_growth_type is not None:
            survival_factor_growth_func = get_df(
                survival_factor_growth_type,
                survival_factor,
                survival_factor_final,
                survival_factor_growth_start,
                survival_factor_growth_end,
                survival_factor_growth_rate
            )
        if reproduction_elitism_factor_growth_type is not None:
            reproduction_elitism_factor_growth_func = get_df(
                reproduction_elitism_factor_growth_type,
                reproduction_elitism_factor,
                reproduction_elitism_factor_final,
                reproduction_elitism_factor_growth_start,
                reproduction_elitism_factor_growth_end,
                reproduction_elitism_factor_growth_rate
            )
        if survival_elitism_factor_growth_type is not None:
            survival_elitism_factor_growth_func = get_df(
                survival_elitism_factor_growth_type,
                survival_elitism_factor,
                survival_elitism_factor_final,
                survival_elitism_factor_growth_start,
                survival_elitism_factor_growth_end,
                survival_elitism_factor_growth_rate
            )
        if diversity_bias_decay_type is not None:
            diversity_bias_decay_func = get_df(
                diversity_bias_decay_type,
                diversity_bias,
                diversity_bias_final,
                diversity_bias_decay_start,
                diversity_bias_decay_end,
                diversity_bias_decay_rate
            )
        if mutation_factor_growth_type is not None:
            mutation_factor_growth_func = get_df(
                mutation_factor_growth_type,
                mutation_factor,
                mutation_factor_final,
                mutation_factor_growth_start,
                mutation_factor_growth_end,
                mutation_factor_growth_rate
            )
        if mutation_strength_decay_type is not None:
            mutation_strength_decay_func = get_df(
                mutation_strength_decay_type,
                mutation_strength,
                mutation_strength_final,
                mutation_strength_decay_start,
                mutation_strength_decay_end,
                mutation_strength_decay_rate
            )

        max_generations_reached: bool = False
        max_fitness_reached: bool = False
        stagnation_limit_reached: bool = False

        progress_bar = ResourceProgressBar(initial=1, total=max_generations)

        while not (generation >= max_generations):
            if diversity_bias is not None:
                if diversity_bias_decay_func is not None:
                    diversity_bias = diversity_bias_decay_func(generation)

                ## Apply biases to the fitness values to encourage exploration;
                ##      Individuals gain fitness directly proportional to diversity
                ##      bias and how much worse they are than the maximum fitness.
                fitness_values = [
                    fitness + (diversity_bias * (max_fitness - fitness))
                    for fitness in fitness_values
                ]

            ## Applying scaling to the fitness values.
            fitness_values = self.__selector.scale(fitness_values)
            fitness_values = normalize_to_sum(fitness_values, 1.0)

            # Select part of the existing population to survive to the next
            # generation and cull the rest;
            #     - The selection scheme is random (unless the elitism factor
            #       is 1.0) with chance of survival proportional to fitness.
            #     - This step emulates Darwin's principle of survival of the
            #       fittest.
            base_population_size: int = len(population)
            population, fitness_values = self.cull_population(
                population,
                fitness_values,
                survival_factor,
                survival_elitism_factor
            )
            fitness_values = normalize_to_sum(fitness_values, 1.0)

            # Recombine the survivors to produce offsrping and expand the
            # population to the lower of;
            #     - The max population size,
            #     - increase the size by our maximum expansion factor,
            #     - This step emulates the principle that the fittest
            #       individuals are more likely to succeed in the competition
            #       to reproduce, because they are more desirable as a partner
            #       for mating/reproduction, since their offspring are more
            #       likely to have high fitness and therefore survive longer.
            if generation != 0 and reproduction_elitism_factor_growth_func is not None:
                reproduction_elitism_factor = reproduction_elitism_factor_growth_func(generation)
            desired_population_size: int = min(
                math.ceil(base_population_size * expansion_factor),
                max_pop_size
            )
            population = self.grow_population(
                population,
                fitness_values,
                desired_population_size,
                reproduction_elitism_factor,
                survival_factor,
                survival_elitism_factor
            )

            # Randomly mutate the grown population
            if generation != 0 and mutation_factor_growth_func is not None:
                mutation_factor = mutation_factor_growth_func(generation)
            mutated_population: list[CT] = self.mutate_population(
                population,
                mutation_factor,
                mutate_all
            )

            # Update the population and fitness values with the new generation.
            population = mutated_population
            fitness_values: list[float] = [
                self.__encoder.evaluate_fitness(individual)
                for individual in population
            ]

            # If elitism is enabled the population and their fitness values
            # need to be ordered.
            if survival_elitism_factor is not None:
                population, fitness_values = zip(
                    *sorted(
                        zip(population, fitness_values),
                        key=lambda item: item[1]
                    )
                )
                population = list(population)
                fitness_values = list(fitness_values)
                max_individual = population[-1]
                max_fitness = fitness_values[-1]
            else:
                max_fitness_index, max_fitness = max(enumerate(fitness_values), key=lambda item: item[1])
                max_individual = population[max_fitness_index]
                min_fitness = min(fitness_values)

            if max_fitness > best_fitness_achieved:
                best_fitness_achieved = max_fitness
                best_individual_achieved = max_individual
            else:
                stagnated_generations += 1

            generation += 1
            progress_bar.update(data={"Best fitness": str(best_fitness_achieved)})

            # Determine whether the fitness threshold has been reached.
            if fitness_threshold is not None and best_fitness_achieved >= fitness_threshold:
                max_fitness_reached = True
            if stagnation_limit is not None and stagnated_generations == stagnation_limit:
                stagnation_limit_reached = True

        progress_bar.close()
        return GeneticAlgorithmSolution(
            best_individual_achieved,
            best_fitness_achieved,
            population,
            fitness_values,
            max_generations_reached=max_generations_reached,
            max_fitness_reached=max_fitness_reached,
            stagnation_limit_reached=stagnation_limit_reached
        )

    def create_population(self, population_size: int) -> list[CT]:
        """Create a new population of the given size."""
        return self.__encoder.base.random_chromosomes(
            self.__encoder.chromosome_length,
            population_size
        )

    def cull_population(
        self,
        population: list[CT],
        fitness_values: list[float],
        survival_factor: float,
        survival_elitism_factor: float | None
    ) -> tuple[list[CT], list[float]]:
        """
        Select individuals from the current population to survive to and
        reproduce for the next generation. Individuals that do not survive are
        said to be culled from the population and do not get a chance to
        reproduce and propagate features of their genes to the next generation.

        The intuition is that individuals with sufficiently low fitness
        (relative to the other individuals in the population) will get out
        competed by better adapted individuals and therefore will not survive
        to reproduce offspring. The assumption, is that such individuals don't
        have genes with desirable features, and therefore we don't want them
        in the gene pool at all.

        Reproduction with replacement might be considered similar to
        population culling and reproduction without replacement, however this
        is not true, since the prior allows low-fitness individuals to dilute
        the mating pool and allows potentially undesirable genes to remain in
        gene pool.

        A low survival factor results in a more exploitative search and faster
        convergence, since a large proportion of the population is culled, and
        the search is much more focused on a small set of genes/area of the
        search space. This is particuarly true if the elitism factor is high,
        because lower fitness indivuduals will have much less chance to
        reproduce and contribute their genes to future generations, and higher
        fitness individuals will dominate the mating proceedure.

        The survival factor decay exists to allow a greater number of
        individuals to survive to reproduce at earlier generations to promote
        early exploration, but increasingly reducing the number of individuals
        that survive to increasingly focus the search and exploit the best
        quality solutions.

        Parameters
        ----------
        `population: list[CT]` - The current population. If `elitism_factor`
        is not None, then the population must be sorted in ascending order.

        `fitness_values: list[float]` - The fitness values of the current
        population. If `elitism_factor` is not None, then the fitness values
        must be sorted in ascending order.

        `survival_factor: float` - The proportion of the population that
        survives. Must be in the range [0, 1].

        `elitism_factor: float | None` - The proportion of the population that
        is guaranteed to survive. Must be in the range [0, 1].
        """
        # The current population size and the quantity of them to choose to
        # survive to the next generation.
        population_size: int = len(population)
        survive_quantity: int = math.ceil(population_size * survival_factor)

        # If no individuals or all individuals in the current population
        # survive then skip the culling phase.
        if survive_quantity == 0:
            return ([], [])
        if population_size == survive_quantity:
            return (population, fitness_values)

        # If elitism factor is not given or None then always choose randomly
        # with probability proportion.
        if survival_elitism_factor is None:
            indices = self.__selector.select(
                population,
                fitness_values,
                survive_quantity
            )
            population = list(index_sequence(population, indices))
            fitness_values = list(index_sequence(fitness_values, indices))
            return (population, fitness_values)

        # The quantity of elite individuals that are guaranteed to survive.
        elite_quantity: int = math.ceil(survive_quantity * survival_elitism_factor)

        # If all surviving are elite, then skip random selection phase,
        # simply select deterministically the best individuals from the
        # previous generation (`elite_quantity` cannot be zero here).
        if survive_quantity == elite_quantity:
            return (
                population[-survive_quantity:],
                fitness_values[-survive_quantity:]
            )

        # Non-elite part of the population chosen from randomly to generate
        # competing quantity of survive quantity
        comp_quantity: int = survive_quantity - elite_quantity
        if elite_quantity != 0:
            comp_population = population[:-elite_quantity]
            comp_fitness_values = fitness_values[:-elite_quantity]
            elite_population = population[-elite_quantity:]
            elite_fitness_values = fitness_values[-elite_quantity:]
        else:
            comp_population = population
            comp_fitness_values = fitness_values
            elite_population = []
            elite_fitness_values = []
        comp_indices = self.__selector.select(
            comp_population,
            comp_fitness_values,
            comp_quantity
        )
        comp_population = list(index_sequence(comp_population, comp_indices))
        comp_fitness_values = list(
            index_sequence(comp_fitness_values, comp_indices)
        )

        return (
            comp_population + elite_population,
            comp_fitness_values + elite_fitness_values
        )

    def grow_population(
        self,
        population: list[CT],
        fitness_values: list[float],
        desired_population_size: int,
        reproduction_elitism_factor: float | None,
        survival_factor: float,
        survival_elitism_factor: float | None
    ) -> list[CT]:
        """
        Grow the population to the desired size.

        Population growth occurs by individuals in the population reproducing
        in (usually) randomly chosen pairs of parents with fitness poportional
        probability (with replacement), to produce a pair of offspring.
        Usually, the parents are replaced by their offspring in the grown
        population, but optionally (and with fitness poportional probability)
        the parents can also survive and remain in the grown population.

        Parameters
        ----------
        `population: list` - The current population. If
        `reproduction_elitism_factor` or `survival_elitism_factor` is not
        None, then the population must be sorted in ascending order of their
        fitness.

        `fitness_values: list[float]` - The fitness values of the current
        population. If `reproduction_elitism_factor` or
        `survival_elitism_factor` is not None, then the fitness values must be
        sorted in ascending order.

        `desired_population_size: int` - The desired size of the grown
        population.

        `reproduction_elitism_factor: float | None` - Factor of the highest
        fitness portion of the population that are considered the reproductive
        elite, which are guaranteed to reproduce, assuming that there is space
        in the population to do so. Parent pairs from the reproductive elite
        set are chosen to reproduce in descending order of their fitness, as a
        seperate initial stage of the population growth. Once the elite set
        is consumed, or the population is full, then return to the usual
        reproductive mechanism. If not given or None, then the reproductive
        elite is not used.

        `survival_factor: float` - The proportion of the current population
        that will survive to the next generation. The remaining proportion of
        the population will be culled and not survive to the next generation
        but are still eligible to reproduce offspring.

        `survival_elitism_factor: float | None` - Factor of the highest
        fitness portion of the population that are considered the survival
        elite, which are guaranteed to survive to the next generation.
        If not given or None, then the survival elite is not used.

        Returns
        -------
        `list[CT]` - The grown population.
        """
        # The current population size and the quantity of them to choose to
        # survive to the next generation.
        population_size: int = len(population)
        survive_quantity: int = math.ceil(population_size * survival_factor)

        # If all individuals in the current population survive then skip the
        # reproduction phase.
        if desired_population_size == survive_quantity:
            return population

        # Determine number of elite individuals guaranteed to survive, and the
        # number of individuals that must compete to survive.
        elite_survive_quantity: int = 0
        if survival_elitism_factor is not None:
            elite_survive_quantity = math.ceil(
                survive_quantity * survival_elitism_factor
            )
        comp_survive_quantity: int = survive_quantity - elite_survive_quantity

        # Add individuals that survive reproduction to the offspring.
        offspring: list[CT] = []
        if elite_survive_quantity != 0:
            offspring.extend(population[-elite_survive_quantity:])
        if comp_survive_quantity != 0:
            if elite_survive_quantity != 0:
                comp_population = population[:-elite_survive_quantity]
                comp_fitness_values = fitness_values[:-elite_survive_quantity]
                comp_fitness_values = normalize_to_sum(
                    comp_fitness_values,
                    1.0
                )
            else:
                comp_population = population
                comp_fitness_values = fitness_values
            survived_indices = self.__selector.select(
                comp_population,
                comp_fitness_values,
                comp_survive_quantity
            )
            survived = list(index_sequence(comp_population, survived_indices))
            offspring.extend(survived)

        # Determine number of offspring to produce.
        offspring_quantity: int = desired_population_size - survive_quantity
        if offspring_quantity == 0:
            return offspring

        # Calculate number of parents to choose from to reproduce.
        total_parents: int = (offspring_quantity + (offspring_quantity % 2))
        elite_reprod_quantity: int = 0
        if reproduction_elitism_factor is not None:
            elite_reprod_quantity = math.ceil(
                total_parents * reproduction_elitism_factor
            )
            elite_reprod_quantity += elite_reprod_quantity % 2
        comp_reprod_quantity: int = total_parents - elite_reprod_quantity

        # Select parent pairs to reproduce.
        selected: list[CT] = []
        if elite_reprod_quantity != 0:
            selected.extend(population[-elite_reprod_quantity:])
        if comp_reprod_quantity != 0:
            if elite_reprod_quantity != 0:
                comp_population = population[:-elite_reprod_quantity]
                comp_fitness_values = fitness_values[:-elite_reprod_quantity]
                comp_fitness_values = normalize_to_sum(
                    comp_fitness_values,
                    1.0
                )
            else:
                comp_population = population
                comp_fitness_values = fitness_values
            comp_indices = self.__selector.select(
                comp_population,
                comp_fitness_values,
                comp_reprod_quantity
            )
            comp_selected = list(index_sequence(comp_population, comp_indices))
            selected.extend(comp_selected)

        # Split selected parents into pairs.
        parent_pairs: Iterator[tuple[CT, CT]] = chunk(
            selected,
            2,
            total_parents // 2,
            as_type=tuple
        )

        # Recombine parent pairs to produce offspring.
        recomb = self.__recombinator
        for parent_1, parent_2 in parent_pairs:
            children: tuple[CT, CT] = recomb.recombine(parent_1, parent_2)
            max_children: int = min(
                len(children),
                offspring_quantity - (len(offspring) - survive_quantity)
            )
            offspring.extend(children[:max_children])

        return offspring

    def mutate_population(
        self,
        population: list[CT],
        mutation_factor: float = 1.0,
        mutate_all: bool = True
    ) -> list[CT]:
        """
        Mutate the population.

        By default, mutates each individual in the population exactly once.
        Otherwise, if `mutation_factor` is greater than one and `mutate_all`
        is True then mutate each individual a multiple of times equal to the
        integer part of the factor, i.e. `floor(mutation_factor)`, and then
        randomly choose a number of individuals from the population to mutate
        (with replacement) proportional to the non-integer part of the factor,
        i.e. `floor(len(population) * (mutation_factor % 1.0))`. Otherwise, if
        `mutate_all` is False or `mutation_factor` is less than one then only
        randomly choose a number of individuals from the population to mutate
        (with replacement) equal to `floor(len(population) * mutation_factor)`.

        Parameters
        ----------
        `population: list[CT]` - The population to mutate.

        `mutation_factor: float = 1.0` - The factor of the population to
        mutate.

        `mutate_all: bool = True` - Whether to mutate each individual in the
        population a multiple of times equal to the integer part of the
        mutation factor if it is greater than one.

        Returns
        -------
        `list[CT]` - The mutated population.
        """
        mutations_quantity: int = math.floor(len(population) * mutation_factor)
        mutated_population: list[CT] = population

        if isinstance(self.__encoder.base, BitStringBase):
            mutate = self.__mutator.bitstring_mutate
        elif isinstance(self.__encoder.base, NumericalBase):
            mutate = self.__mutator.numerical_mutate
        else:
            mutate = self.__mutator.arbitrary_mutate

        # Deterministically select each individual in the population
        # a multiple of times equal to the integer part of the factor.
        if (mutate_all and mutations_quantity >= len(population)):
            cycles: int = mutations_quantity // len(population)
            for index in cycle_for(range(len(population)), cycles):
                mutated_population[index] = mutate(
                    mutated_population[index],
                    self.__encoder.base
                )
            mutations_quantity -= len(population) * cycles

        # Randomly select individuals from the population to account for the
        # rest of the factor.
        if not mutate_all or mutations_quantity != 0:
            choices = self.__random_generator.choice(
                len(population),
                mutations_quantity
            )
            for index in choices:
                mutated_population[index] = mutate(
                    mutated_population[index],
                    self.__encoder.base
                )

        return mutated_population
