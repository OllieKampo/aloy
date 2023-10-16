from aloy.optimization.evolutionary.geneticalgorithms import GeneticSystem, GeneticEncoder, BitStringBaseTypes, SplitCrossOver, PointMutator, ProportionateSelector

class TestEncoder(GeneticEncoder):
    def encode(self, solution):
        return solution
    def decode(self, solution):
        return solution
    def evaluate_fitness(self, chromosome):
        return sum(1 if ((int(x) % 2) == (i % 2)) else 0 for i, x in enumerate(chromosome))

encoder = TestEncoder(20, False, BitStringBaseTypes.bin)
selector = ProportionateSelector()
recombiner = SplitCrossOver()
mutator = PointMutator(1)
system = GeneticSystem(encoder, selector, recombiner, mutator)

result = system.run(100, 200, max_generations=100, stagnation_limit=25)
print(result.best_individual, result.best_fitness)

from aloy.optimization.evolutionary.geneticalgorithms import GeneticSystem, GeneticEncoder, NumericalBase, SplitCrossOver, PointMutator, ProportionateSelector

from aloy.control.pid import PIDController
from aloy.control.systems import InvertedPendulumSystem
from aloy.control.controlutils import simulate_control

pend_system = InvertedPendulumSystem()
controller = PIDController(0.0, 0.0, 0.0, initial_error=pend_system.get_control_input() - pend_system.get_setpoint())

ticks: int = 100
delta_time: float = 0.1

min_ = -10
max_ = 10

def control_evaluator(chromosome):
    # vec = [((max_ - min_) * (int(x, base=2) / ((2 ** 12) - 1))) + min_ for x in ichunk_sequence(chromosome, 12, 3)]
    vec = [((max_ - min_) * x) + min_ for x in chromosome]
    controller.reset()
    controller.set_gains(*vec)
    pend_system.reset()
    return simulate_control(pend_system, controller, ticks, delta_time)

class TestEncoder(GeneticEncoder):
    def encode(self, solution):
        return solution
    def decode(self, solution):
        return solution
    def evaluate_fitness(self, chromosome):
        return -control_evaluator(chromosome)

encoder = TestEncoder(3, False, NumericalBase("base", int, -10, 10))
selector = ProportionateSelector()
recombiner = SplitCrossOver()
mutator = PointMutator(1)
system = GeneticSystem(encoder, selector, recombiner, mutator)

result = system.run(1000, 1000, max_generations=100)

print(f"\nBest position :: {result.best_individual}")
chromosome = result.best_individual
print([((max_ - min_) * x) + min_ for x in chromosome])
