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

__all__ = ("Dimension",
           "ParticleSwarmSolution",
           "ParticleSwarmSystem")

import dataclasses
import math
from fractions import Fraction
from numbers import Real
from typing import Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt
import sklearn.neighbors as skn
import sklearn.tree as skt
import sklearn.svm as sks
from datahandling.makedata import DataHolder
from optimization.decayfunctions import DecayFunctionType, get_decay_function
from auxiliary.progressbars import ResourceProgressBar

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
    """
    Represents the solution of a particle swarm optimisation algorithm.
    
    Elements
    --------
    `best_position: npt.NDArray[np.float64]` - The best position of the swarm.
    
    `best_fitness: np.float64` - The best fitness of the swarm.
    """
    
    ## Particles
    best_position: npt.NDArray[np.float64]
    best_fitness: np.float64
    positions: npt.NDArray[np.float64]
    fitnesses: npt.NDArray[np.float64]
    
    ## Loop variables
    total_iterations: int
    stagnated_at_iteration: int
    
    ## Stop conditions
    iterations_limit_reached: bool = False
    stagnation_limit_reached: bool = False
    fitness_limit_reached: bool = False
    
    def __str__(self) -> str:
        """Return a decscription of the best solution obtained and the stop condition that was reached."""
        best_parameters = ", ".join(f"{param:.4f}" for param in self.best_position)
        stop_condition: str = ""
        if self.iterations_limit_reached:
            stop_condition = "iterations limit reached"
        if self.stagnation_limit_reached:
            stop_condition = "stagnation limit reached"
            stop_condition += f" (stagnated at iteration {self.stagnated_at_iteration})"
        if self.fitness_limit_reached:
            stop_condition = "fitness limit reached"
        return f"PSO Solution :: best parameters = [{best_parameters}], best fitness = {self.best_fitness}, " \
               f"total iterations = {self.total_iterations}, stop condition = {stop_condition}"

class ParticleSwarmSystem:
    """
    Particle swarm optimisation algorithm.
    
    Algorithm
    ---------
    
    There are two coefficients used in particle velocity updates; the personal and global coefficients.
    
    """
    
    __slots__ = {"__dimensions" : "A list of dimensions of the search space.",
                 "__fitness_function" : "The objective function to be optimised.",
                 "__maximise" : "Whether to maximise the objective function or minimise it."}
    
    def __init__(self,
                 dimensions: Sequence[Dimension],
                 fitness_function: Callable[[npt.NDArray[np.float64]], np.float64],
                 maximise: bool = True
                 ) -> None:
        """
        Create a particle swarm optimisation system from a list of dimensions and an objective function.
        
        A dimension must be added for each variable of the objective function (i.e. the search space).
        """
        self.__dimensions: list[Dimension] = list(dimensions)
        self.__fitness_function: Callable[[npt.NDArray[np.float64]], np.float64] = fitness_function
        self.__maximise: bool = maximise
    
    def run(self,
            total_particles: int,
            init_strategy: Literal["random", "linspace"] = "random",
            
            ## Stop criteria
            iterations_limit: int | None = 1000,
            stagnation_limit: int | float | Fraction | None = 0.1,
            fitness_limit: Real | None = None,
            
            ## Particle inertia
            inertia: float = 1.0,
            final_inertia: float | None = None,
            inertia_decay: float | None = None,
            inertia_decay_type: DecayFunctionType | None = "lin",
            inertia_decay_start: int = 0,
            inertia_decay_end: int | None = None,
            
            ## Particle direction of movement coefficients
            personal_coef: float = 1.0,
            final_personal_coef: float = 0.0,
            global_coef: float = 0.0,
            final_global_coef: float = 1.0,
            coef_decay: float | None = None,
            coef_decay_type: DecayFunctionType | None = "lin",
            coef_decay_start: int = 0,
            coef_decay_end: int | None = None,
            
            use_neighbourhood: bool = False,
            neighbour_coef: float = 0.5,
            final_neighbour_coef: float = 0.0,
            neighbourhood_update: int = 10,
            neighbourhood_size: int = 5,
            neighbourhood_method: Literal["kd_tree", "ball_tree"] = "kd_tree",
            
            use_fitness_approximation: bool = False,
            fitness_approximation_update: int = 10,
            fitness_approximation_start: int = 0,
            fitness_approximation_method: Literal["dtr", "svr"] = "dtr",
            
            use_tqdm: bool = False
            
            ) -> ParticleSwarmSolution | None:
        """
        Run the particle swarm optimisation algorithm.
        
        Parameters
        ----------
        `total_particles : int` - The total number of particles in the swarm.
        
        `init_strategy : Literal["random", "linspace"]` - The strategy to use for initialising the particles.
        If "random", the particles are initialised randomly.
        If "linspace", the particles are initialised evenly spaced over the search space,
        the number of points over each dimension is equal to the N-th root of the total number of particles rounded down,
        where N is the number of dimensions (this means that the actual number of particles).
        
        `iterations_limit : int | None = 1000` - The maximum number of iterations to run the algorithm.
        If None, no limit is imposed.
        
        `stagnation_limit : int | float | Fraction | None = 0.1` - The maximum number of iterations without improvement in the fitness of the best solution.
        If a float or Fraction is given, it is interpreted as a percentage of the iterations limit.
        If None, not limit is imposed.
        
        `fitness_limit : float | None = None` - The maximum fitness of the best solution.
        
        `inertia : float = 1.0` - The inertia of the particles.
        The inertia defines the contribution of the previous velocity of the particle towards its current velocity.
        
        `inertia_decay : float | None = None` - The decay constant of the inertia parameter.
        
        `inertia_decay_type : DecayFunctionType | None = "lin"` - The type of decay function to use for the inertia parameter.
        
        `inertia_decay_start : int = 0` - The iteration at which the decay of the inertia parameter starts.
        
        `inertia_decay_end : int | None = None` - The iteration at which the decay of the inertia parameter ends.
        
        `personal_coef : float = 1.0` - The coefficient of the contribution of the personal best position of each particle towards its current velocity.
        A high personal coefficient will promote a explorative search by allowing each particle to explore their own personal best position in the search space.
        Whereas a high global coefficient will promote an exploitative search by forcing each particle to explore closer towards the global best position in the search space.
        
        `coef_decay: float | None = None` - The decay constant of the personal coefficient.
        A low decay constant will cause rapid decay of the local coefficient and gain in the global coefficient, and vice versa for a high decay constant.
        
        `coef_decay_type: DecayFunctionType | None = "lin"` - The decay rate type of the coefficient decay.
        
        `coef_decay_start : int = 0` - The iteration at which the decay of the local coefficient starts.
        
        `coef_decay_end : int | None = None` - The iteration at which the decay of the local coefficient ends.
        """
        if total_particles < 1:
            raise ValueError(f"Total particles must be at least 1. Got; {total_particles}")
        
        ## Check stop conditions.
        if all(map(lambda x: x is None, [iterations_limit, stagnation_limit, fitness_limit])):
            raise ValueError("At least one of the stop condition must be given and not None. "
                             f"Got; {iterations_limit=}, {stagnation_limit=}, {fitness_limit=}")
        if iterations_limit is not None and iterations_limit < 1:
            raise ValueError(f"Iterations limit must be at least 1. Got; {iterations_limit}")
        if stagnation_limit is not None:
            if stagnation_limit < 0.0:
                raise ValueError(f"Stagnation limit must be at least 0. Got; {stagnation_limit}")
            if isinstance(stagnation_limit, (float, Fraction)):
                if iterations_limit is None:
                    raise ValueError("Iterations limit must be given if stagnation limit is a fraction.")
                stagnation_limit = int(stagnation_limit * iterations_limit)
            if iterations_limit is not None:
                stagnation_limit = min(iterations_limit, stagnation_limit)
        
        ## Check inertia and personal coefficient.
        if not (0.0 <= inertia <= 1.0):
            raise ValueError(f"Inertia must be between 0.0 and 1.0. Got; {inertia}")
        if not (0.0 <= personal_coef <= 1.0):
            raise ValueError(f"Personal coefficient must be between 0.0 and 1.0. Got; {personal_coef}")
        if not (0.0 <= global_coef <= 1.0):
            raise ValueError(f"Global coefficient must be between 0.0 and 1.0. Got; {global_coef}")
        ## TODO: Check initial coefs.
        
        ## If the decay rate is not given or None, then the decay constant is one.
        if inertia_decay is None:
            inertia_decay = 1.0
        if coef_decay is None:
            coef_decay = 1.0
        
        ## If the final inertia or personal coefficient is not given, then it is zero.
        if final_inertia is None:
            final_inertia = 0.0
        
        ## If the decay start is less than 0, then raise an error.
        if inertia_decay_start < 0:
            raise ValueError(f"Inertia decay start must be at least 0. Got; {inertia_decay_start}")
        if coef_decay_start < 0:
            raise ValueError(f"Coefficient decay start must be at least 0. Got; {coef_decay_start}")
        
        ## If the decay end is not given, then set it to the iterations limit.
        if inertia_decay_end is None:
            inertia_decay_end = iterations_limit
        if coef_decay_end is None:
            coef_decay_end = iterations_limit
        
        ## If the decay end is less than or equal to the decay start, then raise an error.
        if inertia_decay_end <= inertia_decay_start:
            raise ValueError(f"Inertia decay end must be greater than or equal to the inertia decay start. Got; {inertia_decay_end}")
        if coef_decay_end <= coef_decay_start:
            raise ValueError(f"Coefficient decay end must be greater than or equal to the coefficient decay start. Got; {coef_decay_end}")
        
        ## Get the decay functions for the inertia and coefficients.
        if inertia_decay_type is not None:
            inertia_decay_function = get_decay_function(inertia_decay_type, inertia, final_inertia,
                                                        inertia_decay_start, inertia_decay_end, inertia_decay)
        if coef_decay_type is not None:
            personal_coef_decay_function = get_decay_function(coef_decay_type, personal_coef, final_personal_coef,
                                                              coef_decay_start, coef_decay_end, coef_decay)
            global_coef_decay_function = get_decay_function(coef_decay_type, global_coef, final_global_coef,
                                                            coef_decay_start, coef_decay_end, coef_decay)
            if use_neighbourhood:
                neighbourhood_coef_decay_function = get_decay_function(coef_decay_type, neighbour_coef, final_neighbour_coef,
                                                                       coef_decay_start, coef_decay_end, coef_decay)
        
        ## Random number generator
        rng = np.random.default_rng()
        
        ## Array of dimensions: (lower_bound, upper_bound, max_velocity)
        dimensions = np.array([dataclasses.astuple(dimension) for dimension in self.__dimensions])
        
        ## Create positions and velocities for each particle:
        ##   - Arrays have row for each particle and column for each dimension.
        total_dimensions: int = dimensions.shape[0]
        match init_strategy:
            case "random":
                position_vectors = rng.uniform(dimensions[:,0], dimensions[:,1], (total_particles, total_dimensions))
            case "linspace":
                particles_across_dimensions: int = math.floor(total_particles ** (1.0 / total_dimensions))
                deterministic_total_particles = int(particles_across_dimensions ** total_dimensions)
                linspace_across_dimensions = [np.linspace(dimension[0], dimension[1], particles_across_dimensions) for dimension in dimensions]
                position_vectors = np.dstack(np.meshgrid(*linspace_across_dimensions)).reshape(-1, total_dimensions)
                if deterministic_total_particles < total_particles:
                    position_vectors = np.concatenate((position_vectors, rng.uniform(dimensions[:,0], dimensions[:,1], (total_particles - deterministic_total_particles, total_dimensions))))
        velocity_vectors = rng.uniform(-dimensions[:,2], dimensions[:,2], (total_particles, total_dimensions))
        
        ## Evaluate fitness of each particle
        particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors)
        
        ## Personal best fitness and position for each particle (initialise to current fitness)
        best_particles_fitness = particles_fitness.copy()
        best_position_vectors = position_vectors.copy()
        
        ## Global best fitness and position over all particles
        if self.__maximise:
            global_best_index = np.argmax(particles_fitness)
        else: global_best_index = np.argmin(particles_fitness)
        global_best_fitness = particles_fitness[global_best_index]
        global_best_position = position_vectors[global_best_index]
        
        ## K-dimensional tree representing the particle "history" cloud;
        ##  - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
        ##  - https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms
        if use_neighbourhood:
            ## Keep a record of the best position and fitness for each particle for evaluating the neighbourhood.
            neighbour_best_particles_fitness = best_particles_fitness.copy()
            neighbour_best_position_vectors = best_position_vectors.copy()
            
            if neighbourhood_update < 1:
                raise ValueError(f"Neighbourhood update must be at least 1. Got; {neighbourhood_update}")
            if neighbourhood_size < 1:
                raise ValueError(f"Neighbourhood size must be at least 1. Got; {neighbourhood_size}")
            
            neighbour_cloud: skn.KDTree | skn.BallTree
            match neighbourhood_method:
                case "kd_tree":
                    neighbour_cloud = skn.KDTree(position_vectors)
                case "ball_tree":
                    neighbour_cloud = skn.BallTree(position_vectors)
                case _:
                    raise ValueError(f"Invalid neighbourhood method. Got; {neighbourhood_method}")
        
        ## Decision tree regressor representing an approximation of the fitness function.
        if use_fitness_approximation:
            ## Particle position and fitness clouds are a history of the particles' exploration of the search space,
            ## which is used to build a regression model that approximates the fitness function over the search space.
            position_cloud = position_vectors.copy()
            fitness_cloud = particles_fitness.copy()
            
            if fitness_approximation_update <= 1:
                raise ValueError(f"Fitness approximation update rate must be at least 2. Got; {fitness_approximation_update}")
            if fitness_approximation_start < 0:
                raise ValueError(f"Fitness approximation start must be at least 0. Got; {fitness_approximation_start}")
            
            fitness_approximation_model: skt.DecisionTreeRegressor | sks.SVR
            match fitness_approximation_method:
                case "dtr":
                    fitness_approximation_model = skt.DecisionTreeRegressor()
                case "svr":
                    fitness_approximation_model = sks.SVR()
                case _:
                    raise ValueError(f"Invalid fitness approximation method. Got; {fitness_approximation_method}")
        
        ## Loop variables
        iterations: int = 0
        stagnated_iterations: int = 0
        stagnated_at: int = 0
        
        ## Stop conditions
        iteration_limit_reached: bool = False
        stagnation_limit_reached: bool = False
        fitness_limit_reached: bool = False
        
        ## Data structure for storing history of best fitness and position.
        dataholder = DataHolder(["iteration",
                                 "global_best_fitness", "global_best_position",
                                 "personal_coef", "global_coef"],
                                converters={"personal_coef" : float,
                                            "global_coef" : float})
        
        if use_tqdm:
            progress_bar = ResourceProgressBar(total=iterations_limit, desc="Iterations")
        
        ## Iterate until some stop condition is reached.
        while not ((iterations_limit is not None
                    and (iteration_limit_reached := iterations >= iterations_limit))
                   or (stagnation_limit is not None
                       and (stagnation_limit_reached := stagnated_iterations >= stagnation_limit))
                   or (fitness_limit is not None
                       and (fitness_limit_reached := global_best_fitness >= fitness_limit))):
            
            if iterations != 0:
                if inertia_decay_type is not None:
                    inertia = inertia_decay_function(iterations)
                if coef_decay_type is not None:
                    personal_coef = personal_coef_decay_function(iterations)
                    global_coef = global_coef_decay_function(iterations)
                    if use_neighbourhood:
                        neighbourhood_coef = neighbourhood_coef_decay_function(iterations)
            
            ## For each particle, generate two random numbers for each dimension;
            ##      - one for the personal best, and one for the global best.
            update_vectors = rng.random((total_particles, 3 if use_neighbourhood else 2, total_dimensions))
            
            if use_neighbourhood:
                neighbours_indices = neighbour_cloud.query(position_vectors, k=neighbourhood_size, return_distance=False)
                neighbours_best_position = neighbour_best_position_vectors[np.argmax(neighbour_best_particles_fitness[np.ravel(neighbours_indices)].reshape(total_particles, neighbourhood_size), axis=1)]
            
            ## Velocity is updated based on inertia and random contribution of displacement from;
            ##      - personal best position,
            ##      - global best position.
            personal_displacements = best_position_vectors - position_vectors
            global_displacements = global_best_position - position_vectors
            velocity_vectors = (inertia * velocity_vectors
                                + (personal_coef * update_vectors[:,0] * personal_displacements)
                                + (global_coef * update_vectors[:,1] * global_displacements))
            if use_neighbourhood:
                neighbour_displacements = neighbours_best_position - position_vectors
                velocity_vectors += (neighbourhood_coef * update_vectors[:,2] * neighbour_displacements)
            velocity_vectors = np.maximum(np.minimum(velocity_vectors, dimensions[:, 2]), -dimensions[:, 2])
            
            ## Position is updated based on previous position and velocity:
            ##  - take the maximum of;
            ##      - the lower bound of each dimension and the minimum of;
            ##          - the upper bound of each dimension and the new position (old position plus velocity).
            position_vectors = np.maximum(np.minimum(position_vectors + velocity_vectors, dimensions[:, 1]), dimensions[:, 0])
            
            ## Evaluate fitness of each particle and update personal best.
            if use_fitness_approximation:
                if iterations % fitness_approximation_update == 0 or iterations < fitness_approximation_start:
                    particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors) ## TODO: Parallelise!
                    position_cloud = np.concatenate((position_cloud, position_vectors), axis=0)
                    fitness_cloud = np.concatenate((fitness_cloud, particles_fitness), axis=0)
                    if iterations >= fitness_approximation_start:
                        fitness_approximation_model.fit(position_cloud, fitness_cloud)
                else:
                    particles_fitness = fitness_approximation_model.predict(position_vectors)
            else:
                particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors)
            
            ## Find the personal best fitness and position for each particle.
            if self.__maximise:
                particles_better_than_best = particles_fitness >= best_particles_fitness
            else: particles_better_than_best = particles_fitness < best_particles_fitness
            best_particles_fitness[particles_better_than_best] = particles_fitness[particles_better_than_best]
            best_position_vectors[particles_better_than_best] = position_vectors[particles_better_than_best]
            
            ## Update record current particle positions, and their personal best fitness and position, if using neighbourhood.
            if use_neighbourhood and iterations % neighbourhood_update == 0:
                match neighbourhood_method:
                    case "kd_tree":
                        neighbour_cloud = skn.KDTree(position_vectors)
                    case "ball_tree":
                        neighbour_cloud = skn.BallTree(position_vectors)
                neighbour_best_particles_fitness = best_particles_fitness.copy()
                neighbour_best_position_vectors = best_position_vectors.copy()
            
            ## Find the global best fitness and position over all particles.
            if self.__maximise:
                current_best_index = np.argmax(particles_fitness)
            else: current_best_index = np.argmin(particles_fitness)
            current_best_fitness = particles_fitness[current_best_index]
            if ((self.__maximise and current_best_fitness >= global_best_fitness)
                or (not self.__maximise and current_best_fitness < global_best_fitness)):
                global_best_fitness = current_best_fitness
                global_best_position = position_vectors[current_best_index]
                stagnated_iterations = 0
                stagnated_at = iterations
            else: stagnated_iterations += 1
            iterations += 1
            
            if use_tqdm:
                progress_bar.update(data={"Stagnated" : stagnated_iterations})
            
            dataholder.add_row([iterations,
                                global_best_fitness,
                                global_best_position,
                                personal_coef,
                                global_coef])
        
        return (ParticleSwarmSolution(global_best_position,
                                      global_best_fitness,
                                      position_vectors,
                                      particles_fitness,
                                      iterations,
                                      stagnated_at,
                                      iteration_limit_reached,
                                      stagnation_limit_reached,
                                      fitness_limit_reached),
                dataholder.to_dataframe())
