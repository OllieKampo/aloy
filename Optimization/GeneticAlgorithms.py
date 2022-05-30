from abc import ABCMeta, abstractmethod
import dataclasses
import enum
from fractions import Fraction
import math
from random import getrandbits as randbits, randint
from random import choices
from typing import Generic, Iterable, Literal, NamedTuple, Optional, TypeAlias, TypeVar

## A gene is a strand (a sequence) of bits
Gene: TypeAlias = str

ST = TypeVar("ST")

class GeneBase(NamedTuple):
    """
    Represents a gene base type as a tuple.
    This is the base used for encoding a gene.
    
    Elements
    --------
    `name: str` - The name of the base type.
    
    `format_: str` - The string formatter symbol for
    converting between binary and the base type.
    
    `bits: int` - The number of bits needed to represent one gene.
    """
    name: str
    format_: str
    bits: int

@enum.unique
class Bases(enum.Enum):
    """
    The gene base types a genetic system can use.
    This defines the representation scheme used for chromosomes,
    i.e. is the chromosome a binary sequence of bits
    
    Items
    -----
    `bin = GeneBase("binary", 'b', 1)` - A binary string represenation.
    This uses "base two", i.e. there are only two values a gene can take.
    
    `oct = GeneBase("octal", 'o', 3)` - An octal string representation.
    This uses "base eight", i.e. there are eight possibly values a gene can take.
    
    `hex = GeneBase("hexadecimal", 'h', 4)` - A hexadecimal representation.
    This uses
    """
    bin = GeneBase("binary", 'b', 1)
    oct = GeneBase("octal", 'o', 3)
    hex = GeneBase("hexadecimal", 'h', 4)

class GeneticEncoder(Generic[ST], metaclass=ABCMeta):
    """
    Base class for genetic encoders.
    
    A genetic encoder can encode a solution as a bit sequence,
    
    Also defines the genetic fitness evaluation function, which is specific to the encoding.
    """
    
    def __init__(self,
                 gene_length: int = 8,
                 base: Bases | Literal["bin", "oct", "hex"] = "bin") -> None:
        "Create an encoder with a given gene length and base type."
        self.__gene_length: int = gene_length
        self.__base: GeneBase = Bases[base].value
    
    @property
    def gene_length(self) -> int:
        return self.__gene_length
    
    @property
    def base(self) -> GeneBase:
        return self.__base
    
    @abstractmethod
    def encode(self, solution: ST) -> Gene:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, gene: Gene) -> ST:
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_fitness(self, gene: Gene) -> int | float:
        raise NotImplementedError

class GeneticRecombinator(metaclass=ABCMeta):
    """
    Base class for genetic recombination operators.
    
    A genetic recombinator essentially simply randomly selects
    and swaps genes between two "parent" chromosomes to create new "offspring".
    Swapped bits or bit sub-sequences must always be between those in the same positions in the gene.
    
    A genetic recombinator is agnostic to the representation scheme and encoding used for chromosomes to solutions.
    """
    
    ## We can use a random bit stream to decide when elements to take from the first element, and which from the second: 0 is from first, 1 is from second.
    ## Perhaps we can bit shift, or bitwise or/xor, the two genes together?
    @abstractmethod
    def recombine(self, gene_1: Gene, gene_2: Gene) -> Iterable[Gene]:
        raise NotImplementedError

class SplitCrossOver(GeneticRecombinator):
    """
    Splits a pair of chromosomes into two pieces (sub-sequences) in the same place,
    and swaps the pieces between those chromosomes.
    """
    def recombine(self, gene_1: Gene, gene_2: Gene) -> Iterable[Gene]:
        ## Split the genes in half with a sinlge "point" (in the same place), and swap the sub-sequences.
        point: int = randint(0, len(gene_1) - 1)
        return ((gene_1[:point] + gene_2[point:]), (gene_2[:point] + gene_1[point:]))

class GeneticMutator(metaclass=ABCMeta):
    """
    Base class for genetic mutation operators.
    """
    
    @abstractmethod
    def mutate(self, gene: Gene, base: GeneBase) -> Gene:
        raise NotImplementedError

class PointMutator(GeneticMutator):
    """
    A simple genetic mutator, which randomly selects one gene in a chromosome (with equal probability),
    and changes their values to a different random value (in a binary base representation, simply flip them to the opposite value).
    """
    
    def mutate(self, gene: Gene, base: GeneBase) -> Gene:
        ## Randomly generate a single digit number in the given base,
        ## this is a random new gene.
        new_gene: str = format(randbits(base.bits), base.format_) ## TODO We need to support arbitrary bases to allow for solving things like the TSP.
        ## Insert the new gene into the existing chromosome.
        index: int = randint(0, len(gene) - 1)
        return gene[:index] + new_gene + gene[index+1:]



@dataclasses.dataclass(frozen=True)
class GeneticAlgorithmSolution:
    population: list[Gene]
    fitness_values: list[Fraction]
    max_fitness_reached: bool = False
    max_generations_reached: bool = False
    stagnation_limit_reached: bool = False



class GeneticSystem:
    __slots__ = (## Functions defining the system's genetic operators.
                 "__genetic_encoder",
                 "__genetic_recombinator",
                 "__genetic_mutator")
    
    def __init__(self,
                 encoder: GeneticEncoder,
                 recombinator: GeneticRecombinator,
                 mutator: GeneticMutator
                 ) -> None:
        
        self.__genetic_encoder: GeneticEncoder = encoder
        self.__genetic_recombinator: GeneticRecombinator = recombinator
        self.__genetic_mutator: GeneticMutator = mutator
    
    def _create_population(self, population_size: int) -> list[Gene]:
        "Create a list of gene strings, representing a population of the given size."
        total_bits: int = self.__genetic_encoder.base.bits * self.__genetic_encoder.gene_length
        return [format(randbits(total_bits),
                       self.__genetic_encoder.base.format_).zfill(self.__genetic_encoder.gene_length)
                for _ in range(population_size)]
    
    def run(self,
            initial_population_size: int,
            max_generations: Optional[int],
            max_population_size: int,
            survival_factor: Fraction,
            growth_factor: Fraction,
            mutation_proportion: Fraction,
            mutation_decay: Fraction,
            fitness_threshold: Optional[Fraction],
            convergence_bias: Fraction,
            diversity_bias: Fraction,
            bias_decay: Fraction
            ) -> GeneticAlgorithmSolution:
        
        if survival_factor >= Fraction(1.0):
            raise ValueError()
        
        if growth_factor * survival_factor < 1.0:
            raise ValueError("Population size would shrink.")
        
        population: list[Gene] = self._create_population(initial_population_size)
        fitness_values: list[Fraction] = [self.__genetic_encoder.evaluate_fitness(individual)
                                          for individual in population]
        max_fitness, min_fitness = max(fitness_values), min(fitness_values)
        generation: int = 0
        
        while not (generation >= max_generations):
            ## Update the convergence and diversity biases;
            ##      - Convergence increases by the decay factor,
            ##      - Diversity reduces by the decay factor.
            if generation != 0:
                convergence_bias = convergence_bias * (1 + bias_decay)
                diversity_bias = diversity_bias * (1 - bias_decay)
            
            ## Apply biases to the fitness values;
            ##      - Convergence bias increases values with high fitness to even higher values,
            ##      - Diversity bias increases values with low fitness to higher values.
            fitness_values = [fitness
                              + ((fitness - min_fitness) * convergence_bias)
                              + ((max_fitness - fitness) * diversity_bias)
                              for fitness in fitness_values]
            
            ## Select part of the existing population to survive to the next generation;
            ##      - The selection is random but bias towards survivors with the best fitness,
            ##      - This step emulates Darwin's principle of survival of the fittest.
            survive_quantity: int = math.floor(len(population) * survival_factor)
            surviving_population: list[Gene] = self._select_individuals(population, fitness_values, survive_quantity)
            
            ## Recombine the survivors to produce offsrping and expand the population to the lower of;
            ##      - The max population size,
            ##      - increase the size by our maximum expansion factor.
            offspring_quantity: int = min(max_population_size, math.ceil(survive_quantity * growth_factor))
            grown_population: list[Gene] = self._grow_population(surviving_population, offspring_quantity)
            
            ## Randomly mutate the grown population
            if generation != 0:
                mutation_proportion *= (1.0 - mutation_decay)
            max_mutations: int = math.floor(len(grown_population) * mutation_proportion)
            mutated_population: list[Gene] = self._mutate_population(grown_population, max_mutations)
            
            population = mutated_population
            generation += 1
            
            ## Determine whether the fitness threshold has been reached.
            fitness_values: list[Fraction] = [self.__genetic_encoder.evaluate_fitness(individual)
                                              for individual in population]
            max_fitness, min_fitness = max(fitness_values), min(fitness_values)
            if max_fitness >= fitness_threshold:
                return GeneticAlgorithmSolution(population, fitness_values, max_fitness_reached=True)
        
        return GeneticAlgorithmSolution(population, fitness_values, max_generations_reached=True)
    
    def _select_individuals(self, population: list[Gene], fitness_values: list[Fraction], quantity: int) -> list[Gene]:
        return choices(population, fitness_values, k=quantity)
    
    def _grow_population(self, surviving_fittest: list[Gene], offspring_quantity: int) -> list[Gene]:
        
        added_offspring: int = 0
        offspring: list[Gene] = []
        
        while added_offspring != offspring_quantity:
            children: list[Gene] = self.__genetic_recombinator.recombine(*choices(surviving_fittest, k=2))
            max_children: int = min(2, offspring_quantity - added_offspring)
            offspring.extend(children[:max_children])
            added_offspring += max_children
        
        return offspring
    
    def _mutate_population(self, population: list[Gene], mutations_quantity: int) -> list[Gene]:
        mutated_population: list[Gene] = population
        
        for index in choices(range(len(population)), k=mutations_quantity):
            mutated_population[index] = self.__genetic_mutator.mutate(mutated_population[index],
                                                                      self.__genetic_encoder.base)
        
        return mutated_population

if __name__ == "__main__":
    
    class BasicEncoder(GeneticEncoder):
        def evaluate_fitness(self, gene: Gene) -> int | float:
            return (sum(int(gene[i]) for i in range(0, len(gene), 2))
                    + ((len(gene) // 2) - (sum(int(gene[i]) for i in range(1, len(gene), 2)))))
        def encode(self, solution: ST) -> Gene:
            return super().encode(solution)
        def decode(self, gene: Gene) -> ST:
            return super().decode(gene)
    
    genetic_encoder = BasicEncoder(gene_length=20)
    
    genetic_combiner = SplitCrossOver()
    
    genetic_mutator = PointMutator()
    
    genetic_system = GeneticSystem(genetic_encoder,
                                   genetic_combiner,
                                   genetic_mutator)
    
    solution = genetic_system.run(initial_population_size=10,
                                  max_generations=100,
                                  max_population_size=100,
                                  survival_factor=0.5,
                                  growth_factor=3.0,
                                  mutation_proportion=0.5,
                                  mutation_decay=0.025,
                                  fitness_threshold=100.0,
                                  convergence_bias=0.75,
                                  diversity_bias=0.25,
                                  bias_decay=0.05)
    
    print(solution)