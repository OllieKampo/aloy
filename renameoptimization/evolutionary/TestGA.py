from DataStructures.DisjointSet import ST
from Optimization.Evolutionary.GeneticAlgorithms import Gene, GeneticEncoder, GeneticSystem, PointMutator, SplitCrossOver

if __name__ == "__main__":
    
    class BasicEncoder(GeneticEncoder):
        def evaluate_fitness(self, gene: Gene) -> int | float:
            return (sum(int(gene[i], base=16) for i in range(0, len(gene), 2))
                    + ((len(gene) // 2) - (sum(int(gene[i], base=16) for i in range(1, len(gene), 2)))))
        def encode(self, solution: ST) -> Gene:
            return super().encode(solution)
        def decode(self, gene: Gene) -> ST:
            return super().decode(gene)
    
    genetic_encoder = BasicEncoder(gene_length=16, base="hex")
    
    genetic_combiner = SplitCrossOver()
    
    genetic_mutator = PointMutator(points=1)
    
    genetic_system = GeneticSystem(genetic_encoder,
                                   genetic_combiner,
                                   genetic_mutator)
    
    solution = genetic_system.run(init_pop_size=50,
                                  max_pop_size=500,
                                  max_generations=1000,
                                  survival_factor=0.5,
                                  survival_elitism_factor=0.05,
                                  replacement=True,
                                  expansion_factor=3.0,
                                  mutation_factor=0.95,
                                  mutation_factor_growth=0.01,
                                  mutation_factor_growth_type="exp",
                                  fitness_threshold=200.0,
                                  stagnation_limit=0.25,
                                  diversity_bias=0.95,
                                  diversity_bias_decay=0.025)
    
    print(solution)