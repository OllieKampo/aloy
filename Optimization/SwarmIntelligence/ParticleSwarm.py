###########################################################################
###########################################################################
## A general implementation of a particle swarm optimisation algorithm.  ##
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

"""General implementation of a particle swarm optimisation algorithm."""

import dataclasses
from fractions import Fraction
from numbers import Real
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

@dataclasses.dataclass
class Particle:
    fitness: np.float64
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    
    best_fitness: np.float64
    best_position: npt.NDArray[np.float64]
    
    @property
    def current(self) -> tuple[np.float64, npt.NDArray[np.float64]]:
        return (self.fitness, self.position)
    
    @property
    def best(self) -> tuple[np.float64, npt.NDArray[np.float64]]:
        return (self.best_fitness, self.best_position)
    
    @classmethod
    def initial(cls,
                fitness: Real,
                position_vector: npt.NDArray[np.float64],
                velocity_vector: npt.NDArray[np.float64]
                ) -> "Particle":
        """
        Create a new particle for the initial stage of a particle
        swarm optimisation process. This sets the particle's personal
        best known fitness and position to its initial fitness/position.
        """
        return cls(fitness,
                   position_vector,
                   velocity_vector,
                   fitness,
                   position_vector)

@dataclasses.dataclass(frozen=True)
class Dimension:
    """
    A dimension of the search space.
    
    Fields
    ------
    `lower_bound: float` - The lower bound of the domain of the dimension.
    
    `upper_bound: float` - The upper bound of the domain of the dimension.
    
    `max_velocity: float` - The maximum velocity of a particle moving along the dimension.
    """
    
    lower_bound: Real
    upper_bound: Real
    max_velocity: Real

@dataclasses.dataclass(frozen=True)
class ParticleSwarmSolution:
    # best: Particle
    population: list[Particle]
    # max_fitness_reached: bool = False
    max_generations_reached: bool = False
    stagnation_limit_reached: bool = False

class ParticleSwarmSystem:
    __slots__ = ("__dimensions",
                 "__fitness_function")
    
    def __init__(self,
                 dimensions: list[Dimension],
                 fitness_function: Callable[[npt.NDArray[np.float64]], np.float64]
                 ) -> None:
        """
        A dimension must be added for each variable in the search space.
        """
        self.__dimensions: list[Dimension] = dimensions
        self.__fitness_function: Callable[[npt.NDArray[np.float64]], np.float64] = fitness_function
    
    def set_problem(self,
                    dimensions: list[Dimension],
                    fitness_function: Callable[[Particle], list[Real]]
                    ) -> None:
        self.__dimensions = dimensions
        self.__fitness_function = fitness_function
    
    def run(self,
            total_particles: int,
            
            inertia: Fraction,
            inertia_decay: Fraction,
            
            personal_coefficient: Fraction,
            # local_coefficient: Fraction,
            global_coefficient: Fraction,
            coefficient_decay: Fraction,
            
            ## Stop criteria
            iterations_limit: Optional[int],
            stagnation_limit: Optional[int],
            fitness_limit: Real
            ) -> Optional[ParticleSwarmSolution]:
        """
        Run the particle swarm optimisation algorithm.
        
        `personal_coefficient: Fraction` - 
        
        `local_coefficient: Fraction` - 
        
        `global_coefficient: Fraction` - 
        
        `decay_constant: Fraction` - 
        
        """
        
        rng = np.random.default_rng()
        
        particles: list[Particle] = []
        iterations: int = 0
        stagnated_iterations: int = 0
        global_best_fitness: Real = 0.0
        global_best_position: npt.NDArray[np.float64] = np.zeros(len(self.__dimensions), np.float64)
        
        ## np.uniform to calculate over correct range immediately
        initial_position_vectors: npt.NDArray[np.float64] = np.random.random((total_particles, len(self.__dimensions)))
        initial_position_vectors *= [dimension.upper_bound - dimension.lower_bound for dimension in self.__dimensions]
        initial_position_vectors += [dimension.lower_bound for dimension in self.__dimensions]
        
        # groups: list[list[Particle]] = []
        # need group size
        
        initial_velocity_vectors: npt.NDArray[np.float64] = np.random.random((total_particles, len(self.__dimensions)))
        initial_velocity_vectors *= [dimension.max_velocity for dimension in self.__dimensions]
        initial_velocity_vectors *= 2.0
        initial_velocity_vectors -= [dimension.max_velocity for dimension in self.__dimensions]
        
        for position_vector, velocity_vector in zip(initial_position_vectors, initial_velocity_vectors):
            fitness = self.__fitness_function(position_vector)
            particles.append(Particle.initial(fitness, position_vector, velocity_vector))
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = position_vector
        
        while (iterations < iterations_limit
               and stagnated_iterations < stagnation_limit
               and global_best_fitness < fitness_limit):
            
            if iterations != 0:
                inertia = inertia * (1.0 - inertia_decay)
                personal_coefficient = personal_coefficient * (1.0 - coefficient_decay)
                global_coefficient = (1.0 - personal_coefficient)
            
            ## For each particle we need to generate two random numbers for each dimension
            update_vectors: np.ndarray = rng.random((total_particles, 2, len(self.__dimensions)))
            
            for particle, particle_update_vector in zip(particles, update_vectors):
                
                ## Velocity is updated based on inertia and random contribution of displacement from;
                ##      - personal best position,
                ##      - local "friend" group position,
                ##      - global best position.
                
                personal_displacement = particle.best_position - particle.position
                global_displacement = global_best_position - particle.position
                
                new_velocity = (inertia * particle.velocity
                                + (personal_coefficient * particle_update_vector[0] * personal_displacement)
                                + (global_coefficient * particle_update_vector[1] * global_displacement))
                particle.velocity = np.maximum(np.minimum(new_velocity, [dimension.upper_bound for dimension in self.__dimensions]),
                                               [dimension.lower_bound for dimension in self.__dimensions])
                
                ## Position is updated based on previous position and velocity.
                particle.position = np.maximum(np.minimum(particle.position + particle.velocity, [dimension.upper_bound for dimension in self.__dimensions]),
                                               [dimension.lower_bound for dimension in self.__dimensions])
                
                particle.fitness = self.__fitness_function(particle.position)
                # print(particle)
                
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position
                    if particle.fitness > global_best_fitness:
                        global_best_fitness = particle.fitness
                        global_best_position = particle.position
                
                print(f"Global fitness: coeff={global_coefficient}, fit={global_best_position}")
                
            iterations += 1
        
        return ParticleSwarmSolution(particles)

psystem = ParticleSwarmSystem([Dimension(0, 100.0, 10.0), Dimension(0, 100.0, 10.0)],
                              lambda dimensions: (np.float64(dimensions[0]) - 100.0) + (100.0 - np.float64(dimensions[1])))

result = psystem.run(100, 1.0, 0.005, 1.0, 0.0, 0.05, 1000, 100, 10000)
print(max(result.population, key=lambda p: p.best_fitness), sep="\n")
