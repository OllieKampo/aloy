# Genetic Algorithms



## Applications

- Layout optimisation (electric circuits etc),
- Combinatorial optimisation problems, particularly "permutation problems";
    - Task Scheduling
    - Travelling Salesman Problem

## Main Advantages

- Global search technique,
- General purpose,
- Well understood.

## Main Disadvantages

- Slow convergence,
- Fitness evaluation expensive,
- Parameters to optimise,
- Need tayloring to problem domain,
- Can't deal with variable length solutions;
    - Can do task scheduling when number of tasks is known,
    - Can't to task planning when number of tasks needed to reach an arbitrary goal is not known.

## Main Issues

- Selection Pressure:
    - Drives population convergence,
    - High selection pressure causes exploitative search,
    - Low selection pressure causes explorative search,
    - Risk of premature sub-optimal convergence if trade-off between exploitation and exploration is not well balanced because selection pressure too high.

- Crossover:
    - Large jumps is search space,
    - Problem-specific, can violate problem constraints and create invalid solutions if incorrectly chosen/designed,
    - Different crossover operators may promote or oppose convergence depending on nature of problem.

- Mutation:
    - Small jumps in search space,
    - General purpose,
    - Opposes convergence.