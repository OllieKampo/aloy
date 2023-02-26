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
import os
from fractions import Fraction
from numbers import Real
from typing import Any, Callable, Literal, Sequence

from dask.distributed import Client, LocalCluster
import dask.array as daskarray

import numpy as np
import numpy.typing as npt
import sklearn.neighbors as skn
import sklearn.svm as sks
import sklearn.tree as skt
from scipy import spatial as scipy_spatial
from sklearn.metrics import mean_absolute_error

from auxiliary.progressbars import ResourceProgressBar
from datahandling.makedata import DataHolder
from optimization.decayfunctions import DecayFunctionType, get_decay_function

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
    `best_position : npt.NDArray[np.float64]` - The best position of the swarm of shape `(n_dimensions,)`.
    
    `best_fitness : np.float64` - The best fitness of the swarm.
    
    `positions : npt.NDArray[np.float64]` - The positions of the particles in the swarm at the end of the algorithm of shape `(n_particles, n_dimensions)`.
    
    `fitnesses : npt.NDArray[np.float64]` - The fitnesses of the particles in the swarm at the end of the algorithm of shape `(n_particles,)`.
    
    `total_iterations : int` - The total number of iterations that the algorithm ran for.
    
    `stagnated_at_iteration : int` - The iteration at which the stagnation limit was reached.
    
    `iterations_limit_reached : bool` - Whether the iterations limit was reached.
    
    `stagnation_limit_reached : bool` - Whether the stagnation limit was reached.
    
    `fitness_limit_reached : bool` - Whether the fitness limit was reached.
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
    
    ## Approximation error
    approximation_error: float = 0.0
    
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
    
    There are three coefficients used in particle velocity updates; the personal and global coefficients.
    
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
            total_particles: int = 100,
            init_strategy: Literal["random", "linspace"] = "random",

            # replace_clustered: bool = True,
            # merge_clustered: bool = True,
            # replace_clustered_for: int | float = 0.1,
            # replace_clustered_rate: int | float = 10,
            # replace_clustered_quantity: int | float = 0.1,
            
            # replace_low_fitness: bool = True,
            # replace_low_fitness_for: int | float = 0.1,
            # replace_low_fitness_rate: int | float = 10,
            # replace_low_fitness_quantity: int | float = 0.1,

            # replace_stagnated: bool = True,
            # replace_stagnated_for: int | float = 0.1,
            # replace_stagnated_rate: int | float = 10,
            # replace_stagnated_quantity: int | float = 0.1,

            # replace_random: bool = True,
            # replace_random_rate: int | float = 10,
            # replace_random_quantity: int | float = 0.1,
            
            # replace_strategy: Literal["random-space", "random-partner", "random-cast"] = "random",

            ## Stop criteria
            iterations_limit: int | None = 1000,
            stagnation_limit: int | float | Fraction | None = 0.25,
            fitness_limit: Real | None = None,
            bounce: bool = True,
            
            ## Particle inertia
            inertia: float = 1.0,
            final_inertia: float = 0.25,

            ## Particle inertia decay
            inertia_decay_type: DecayFunctionType = "lin",
            inertia_decay_start: int = 0,
            inertia_decay_end: int | None = None,
            inertia_decay_rate: float = 1.0,
            
            ## Particle direction of movement coefficients
            personal_coef: float = 1.0,
            personal_coef_final: float = 0.0,
            global_coef: float = 0.0,
            global_coef_final: float = 1.0,
            neighbour_coef: float = 0.25,
            neighbour_coef_final: float = 0.75,

            ## Particle direction of movement coefficients decay
            coef_decay_type: DecayFunctionType = "lin",
            coef_decay_start: int = 0,
            coef_decay_end: int | None = None,
            coef_decay_rate: float = 1.0,

            personal_coef_decay_type: DecayFunctionType | None = None,
            personal_coef_decay_start: int | None = None,
            personal_coef_decay_end: int | None = None,
            personal_coef_decay_rate: float | None = None,

            global_coef_decay_type: DecayFunctionType | None = None,
            global_coef_decay_start: int | None = None,
            global_coef_decay_end: int | None = None,
            global_coef_decay_rate: float | None = None,

            neighbour_coef_decay_type: DecayFunctionType | None = None,
            neighbour_coef_decay_start: int | None = None,
            neighbour_coef_decay_end: int | None = None,
            neighbour_coef_decay_rate: float | None = None,
            
            ## Neighbourhood options.
            use_neighbourhood: bool = False,
            neighbour_update: int = 10,
            neighbour_size: int = 5,
            neighbour_method: Literal["kd_tree", "ball_tree", "knr"] = "kd_tree",
            neighbour_method_params: dict[str, Any] = {"leaf_size" : 20, "metric" : "minkowski"},
            
            ## Fitness approximation options.
            use_fitness_approximation: bool = False,
            fitness_approximation_update: int = 10,
            fitness_approximation_start: int = 0,
            fitness_approximation_method: Literal["dtr", "knr", "svr"] = "knr",
            fitness_approximation_model_params: dict[str, Any] = {},
            fitness_approximation_threshold: float | None = None,
            
            ## Parallelisation options.
            parallelise: bool = False,
            threads: int | None = None,
            
            ## Miscellaneous options.
            gather_stats: bool = False,
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
        
        inertia_decay_function, personal_coef_decay_function, \
              global_coef_decay_function, neighbour_coef_decay_function \
                  = self.get_decay_functions(**locals())
        
        ## Random number generator
        rng = np.random.default_rng()
        
        ## Array of dimensions: (lower_bound, upper_bound, max_velocity)
        dimensions = np.array([dataclasses.astuple(dimension) for dimension in self.__dimensions])
        
        ## Create positions and velocities for each particle;
        ##   - Arrays have row for each particle and column for each dimension.
        total_dimensions: int = dimensions.shape[0]
        position_vectors, velocity_vectors = self._initialise_particles(total_particles, init_strategy, rng, dimensions, total_dimensions)
        
        ## Evaluate fitness of particles;
        particles_fitness = self._calculate_fitness(total_particles, parallelise, threads, total_dimensions, position_vectors)
        
        ## Personal best fitness and position for each particle (initialise to current fitness)
        best_particles_fitness: np.ndarray = particles_fitness.copy()
        best_position_vectors: np.ndarray = position_vectors.copy()
        
        ## Global best fitness and position over all particles
        if self.__maximise:
            global_best_index = np.argmax(particles_fitness)
        else: global_best_index = np.argmin(particles_fitness)
        global_best_fitness = particles_fitness[global_best_index]
        global_best_position = position_vectors[global_best_index]

        ## Create a model for finding the neighbours of each particle.
        neighbour_model: skn.BallTree | skn.KDTree | skn.KNeighborsRegressor | None = None
        if use_neighbourhood:
            if neighbour_update < 1:
                raise ValueError(f"Neighbourhood update must be at least 1. Got; {neighbour_update}.")
            if neighbour_size < 1:
                raise ValueError(f"Neighbourhood size must be at least 1. Got; {neighbour_size}.")
            
            ## Keep a record of the best position and fitness for each particle for evaluating the neighbourhood.
            neighbour_best_particles_fitness = best_particles_fitness.copy()
            neighbour_best_position_vectors = best_position_vectors.copy()
            
            ## Create a model for finding the neighbours of each particle.
            neighbour_model = self._create_neighbour_model(neighbour_update, neighbour_method, neighbour_method_params,
                                                           fitness_approximation_update, fitness_approximation_method, position_vectors)
        
        ## Create a model for approximating the fitness function.
        if use_fitness_approximation:
            if fitness_approximation_update < 2:
                raise ValueError(f"Fitness approximation update rate must be at least 2. Got; {fitness_approximation_update}.")
            if fitness_approximation_start < 0:
                raise ValueError(f"Fitness approximation start must be at least 0. Got; {fitness_approximation_start}.")
            
            ## Particle position and fitness clouds are a history of the particles' exploration of the search space.
            ## These are used to build a regression model that approximates the fitness function over the search space.
            position_cloud = position_vectors.copy()
            fitness_cloud = particles_fitness.copy()

            ## Create a model for approximating the fitness function.
            fitness_approximation_model = self._create_fitness_approximation_model(fitness_approximation_method, fitness_approximation_model_params,
                                                                                   neighbour_size, neighbour_method, neighbour_method_params)
            if use_neighbourhood and fitness_approximation_method == "knr" and neighbour_method == "knr":
                neighbour_model = fitness_approximation_model
                neighbour_model.fit(position_cloud, fitness_cloud)
        
        ## Variable for storing the running fitness approximation error.
        approximation_error: float = 0.0
        
        if parallelise and threads is None:
            threads = os.cpu_count()
        
        ## Loop variables
        iterations: int = 0
        stagnated_iterations: int = 0
        stagnated_at: int = 0
        
        ## Stop conditions
        iteration_limit_reached: bool = False
        stagnation_limit_reached: bool = False
        fitness_limit_reached: bool = False
        
        ## Data structure for storing history of best fitness and position.
        if gather_stats:
            dataholder = self.create_data_holder()
        
        if use_tqdm:
            progress_bar = ResourceProgressBar(total=iterations_limit, desc="Iterations")
        
        ## Iterate until some stop condition is reached.
        while not ((iterations_limit is not None
                    and (iteration_limit_reached := iterations >= iterations_limit))
                   or (stagnation_limit is not None
                       and (stagnation_limit_reached := stagnated_iterations >= stagnation_limit))
                   or (fitness_limit is not None
                       and (fitness_limit_reached := ((self.__maximise and global_best_fitness >= fitness_limit
                                                       or global_best_fitness <= fitness_limit))))):
            
            ## Update inertia and coefficients.
            if iterations != 0:
                if inertia_decay_function is not None:
                    inertia = inertia_decay_function(iterations)
                if personal_coef_decay_function is not None:
                    personal_coef = personal_coef_decay_function(iterations)
                if global_coef_decay_function is not None:
                    global_coef = global_coef_decay_function(iterations)
                if use_neighbourhood and neighbour_coef_decay_function is not None:
                    neighbour_coef = neighbour_coef_decay_function(iterations)
            
            ## For each particle, generate two random numbers for each dimension;
            ##      - one for the personal best, and one for the global best.
            update_vectors = rng.random((total_particles, 3 if use_neighbourhood else 2, total_dimensions))
            
            if use_neighbourhood:
                if neighbour_method == "knr":
                    neighbours_indices = neighbour_model.kneighbors(position_vectors, return_distance=False)
                    neighbours_best_position = position_cloud[np.argmax(fitness_cloud[np.ravel(neighbours_indices)].reshape(total_particles, neighbour_size), axis=1)]
                else:
                    neighbours_indices = neighbour_model.query(position_vectors, k=neighbour_size, return_distance=False)
                    neighbours_best_position = neighbour_best_position_vectors[np.argmax(neighbour_best_particles_fitness[np.ravel(neighbours_indices)].reshape(total_particles, neighbour_size), axis=1)]
            
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
                velocity_vectors += (neighbour_coef * update_vectors[:,2] * neighbour_displacements)
            velocity_vectors = np.maximum(np.minimum(velocity_vectors, dimensions[:, 2]), -dimensions[:, 2])
            
            ## Position is updated based on previous position and velocity:
            ##  - take the maximum of;
            ##      - the lower bound of each dimension and the minimum of;
            ##          - the upper bound of each dimension and the new position (old position plus velocity).
            position_vectors += velocity_vectors
            if bounce:
                less_than = position_vectors < dimensions[:,0]
                print(less_than)
                if np.any(less_than):
                    position_vectors[less_than] = (dimensions[:,0] * 2.0) - position_vectors[less_than]
                greater_than = position_vectors > dimensions[:,1]
                print(greater_than)
                if np.any(greater_than):
                    position_vectors[greater_than] = (dimensions[:,1] * 2.0) - position_vectors[greater_than]
            else: position_vectors = np.maximum(np.minimum(position_vectors, dimensions[:, 1]), dimensions[:, 0])
            
            ## Evaluate fitness of each particle.
            if not use_fitness_approximation:
                particles_fitness = self._calculate_fitness(total_particles, parallelise, threads, total_dimensions, position_vectors)

            else:
                if iterations < fitness_approximation_start or (iterations - fitness_approximation_start) % fitness_approximation_update == 0:
                    particles_fitness = self._calculate_fitness(total_particles, parallelise, threads, total_dimensions, position_vectors)
                    
                    ## Compare fitness approximation model prediction (with model from before update) with actual fitnesses of current particle positions
                    ## to find prediction error, which is used to calculate a fitness approximation error score.
                    if iterations >= (fitness_approximation_start + fitness_approximation_update):
                        predicted_particle_fitness = fitness_approximation_model.predict(position_vectors)
                        approximation_updates = (iterations - fitness_approximation_start) / fitness_approximation_update
                        approximation_error = ((approximation_error * (approximation_updates - 1)) \
                            + mean_absolute_error(particles_fitness, predicted_particle_fitness)) / approximation_updates
                    
                    ## Update the fitness approximation model with the current particle positions and fitnesses.
                    if iterations >= fitness_approximation_start or fitness_approximation_method == "knr" and neighbour_method == "knr" and iterations % neighbour_update == 0:
                        position_cloud = np.concatenate((position_cloud, position_vectors), axis=0)
                        fitness_cloud = np.concatenate((fitness_cloud, particles_fitness), axis=0)
                        if iterations >= fitness_approximation_start:
                            fitness_approximation_model.fit(position_cloud, fitness_cloud)
                    
                else:
                    ## Fitness approximation is only done if the particle is currently "close" to N existing points in the position cloud.
                    ## Use kdtree to find the nearest neighbours of each particle, if the cdist of the particle to the nearest neighbours is less than a threshold, use the fitness approximation.
                    if fitness_approximation_method == "knr" and fitness_approximation_threshold is not None:
                        distances, indices = fitness_approximation_model.kneighbors(position_vectors, return_distance=True)
                        use_approximation = ~np.any(distances > (fitness_approximation_threshold * (dimensions[:,1] - dimensions[:,0])), axis=1)
                        particles_fitness[use_approximation] = fitness_approximation_model.predict(position_vectors[use_approximation])
                        particles_fitness[~use_approximation] = self._calculate_fitness(total_particles, parallelise, threads, total_dimensions, position_vectors[~use_approximation])
                    else: particles_fitness = fitness_approximation_model.predict(position_vectors)
            
            ## Find the personal best fitness and position for each particle.
            if self.__maximise:
                particles_better_than_best = particles_fitness >= best_particles_fitness
            else: particles_better_than_best = particles_fitness < best_particles_fitness
            best_particles_fitness[particles_better_than_best] = particles_fitness[particles_better_than_best]
            best_position_vectors[particles_better_than_best] = position_vectors[particles_better_than_best]
            
            ## Update record current particle positions, and their personal best fitness and position, if using neighbourhood.
            if use_neighbourhood and neighbour_method != "knr" and iterations % neighbour_update == 0:
                match neighbour_method:
                    case "kd_tree":
                        neighbour_model = skn.KDTree(position_vectors, **neighbour_method_params)
                    case "ball_tree":
                        neighbour_model = skn.BallTree(position_vectors, **neighbour_method_params)
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
            
            if gather_stats:
                ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
                particles_distance_from_global_best = scipy_spatial.distance.cdist(position_vectors, global_best_position[np.newaxis,:])
                dataholder.add_row([iterations,
                                    stagnated_iterations,
                                    total_particles,
                                    global_best_fitness,
                                    global_best_position,
                                    best_particles_fitness.mean(),
                                    best_particles_fitness.std(),
                                    particles_fitness.mean(),
                                    particles_fitness.std(),
                                    position_vectors.mean(),
                                    position_vectors.std(),
                                    # (position_vectors.std(axis=0) / (dimensions[:,1] - dimensions[:,0])).mean(),
                                    particles_distance_from_global_best.mean(),
                                    particles_distance_from_global_best.std(),
                                    velocity_vectors.mean(),
                                    velocity_vectors.std(),
                                    # np.exp(np.abs(velocity_vectors) - dimensions[:,2]).mean(),
                                    inertia,
                                    personal_coef,
                                    global_coef,
                                    neighbour_coef])
        
        solution = ParticleSwarmSolution(## Solution attributes
                                         global_best_position,
                                         global_best_fitness,
                                         position_vectors,
                                         particles_fitness,
                                         ## Stop conditions
                                         iterations,
                                         stagnated_at,
                                         iteration_limit_reached,
                                         stagnation_limit_reached,
                                         fitness_limit_reached,
                                         ## Performance scores
                                         approximation_error)
        if gather_stats:
            return (solution, dataholder.to_dataframe())
        else: return solution

    def create_data_holder(self):
        dataholder = DataHolder(["iteration",
                                 "stagnated",
                                 "total_particles",
                                 "global_best_fitness",
                                 "global_best_position",
                                 "mean_best_particles_fitness",
                                 "std_best_particles_fitness",
                                 "mean_particles_fitness",
                                 "std_particles_fitness",
                                 "mean_particles_position",
                                 "std_particles_position",
                                 "mean_distance_from_global_best",
                                 "std_distance_from_global_best",
                                 "mean_velocity",
                                 "std_velocity",
                                 "inertia",
                                 "personal_coef",
                                 "global_coef",
                                 "neighbour_coef"],
                                 converters={"personal_coef" : float,
                                             "global_coef" : float,
                                             "neighbour_coef" : float})
        return dataholder
    
    @staticmethod
    def get_decay_functions(self, *,
                            iterations_limit,
                            inertia,
                            final_inertia,
                            inertia_decay_type,
                            inertia_decay_start,
                            inertia_decay_end,
                            inertia_decay_rate,
                            personal_coef,
                            personal_coef_final,
                            global_coef,
                            global_coef_final,
                            neighbour_coef,
                            neighbour_coef_final,
                            coef_decay_type,
                            coef_decay_start,
                            coef_decay_end,
                            coef_decay_rate,
                            personal_coef_decay_type,
                            personal_coef_decay_start,
                            personal_coef_decay_end,
                            personal_coef_decay_rate,
                            global_coef_decay_type,
                            global_coef_decay_start,
                            global_coef_decay_end,
                            global_coef_decay_rate,
                            neighbour_coef_decay_type,
                            neighbour_coef_decay_start,
                            neighbour_coef_decay_end,
                            neighbour_coef_decay_rate,
                            use_neighbourhood,
                            **kwargs) -> tuple[Callable[[int], float], ...]:
        """Get the decay functions for the inertia, personal, global and neighbourhood coefficients."""

        if inertia_decay_end is None:
            inertia_decay_end = iterations_limit
        if coef_decay_end is None:
            coef_decay_end = iterations_limit
        
        def _get(_var, _default):
            if _var is None:
                return _default
            return _var
        
        personal_coef_decay_type = _get(personal_coef_decay_type, coef_decay_type)
        global_coef_decay_type = _get(global_coef_decay_type, coef_decay_type)
        neighbour_coef_decay_type = _get(neighbour_coef_decay_type, coef_decay_type)

        personal_coef_decay_start = _get(personal_coef_decay_start, coef_decay_start)
        global_coef_decay_start = _get(global_coef_decay_start, coef_decay_start)
        neighbour_coef_decay_start = _get(neighbour_coef_decay_start, coef_decay_start)

        personal_coef_decay_end = _get(personal_coef_decay_end, coef_decay_end)
        global_coef_decay_end = _get(global_coef_decay_end, coef_decay_end)
        neighbour_coef_decay_end = _get(neighbour_coef_decay_end, coef_decay_end)

        personal_coef_decay_rate = _get(personal_coef_decay_rate, coef_decay_rate)
        global_coef_decay_rate = _get(global_coef_decay_rate, coef_decay_rate)
        neighbour_coef_decay_rate = _get(neighbour_coef_decay_rate, coef_decay_rate)
        
        get_df = get_decay_function
        inertia_decay_func = None
        personal_coef_decay_func = None
        global_coef_decay_func = None
        neighbour_coef_decay_func = None

        if inertia_decay_type is not None:
            inertia_decay_func = get_df(inertia_decay_type, inertia, final_inertia,
                                        inertia_decay_start, inertia_decay_end, inertia_decay_rate)
        if coef_decay_type is not None:
            personal_coef_decay_func = get_df(personal_coef_decay_type, personal_coef, personal_coef_final,
                                              personal_coef_decay_start, personal_coef_decay_end, personal_coef_decay_rate)
            global_coef_decay_func = get_df(global_coef_decay_type, global_coef, global_coef_final,
                                            global_coef_decay_start, global_coef_decay_end, global_coef_decay_rate)
            if use_neighbourhood:
                neighbour_coef_decay_func = get_df(neighbour_coef_decay_type, neighbour_coef, neighbour_coef_final,
                                                   neighbour_coef_decay_start, neighbour_coef_decay_end, neighbour_coef_decay_rate)
                                                                   
        return inertia_decay_func, personal_coef_decay_func, global_coef_decay_func, neighbour_coef_decay_func

    def _create_fitness_approximation_model(self,
                                            fitness_approximation_method: str,
                                            fitness_approximation_model_params: dict,
                                            neighbour_size: int,
                                            neighbour_method: str,
                                            neighbour_model_params: dict
                                            ) -> skt.DecisionTreeRegressor | skn.KNeighborsRegressor | sks.SVR:
        """Create a fitness approximation model."""
        neighbour_model: skn.KNeighborsRegressor | None = None
        if fitness_approximation_method == "dtr":
            ## https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            fitness_approximation_model = skt.DecisionTreeRegressor(**fitness_approximation_model_params)
        elif fitness_approximation_method == "knr":
            ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
            if neighbour_method == "knr":
                fitness_approximation_model = skn.KNeighborsRegressor(n_neighbors=neighbour_size, **neighbour_model_params, **fitness_approximation_model_params)
            else: fitness_approximation_model = skn.KNeighborsRegressor(**fitness_approximation_model_params)
        elif fitness_approximation_method == "svr":
            ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
            fitness_approximation_model = sks.SVR(**fitness_approximation_model_params)
        else:
            raise ValueError(f"Invalid fitness approximation method. Got; {fitness_approximation_method}.")
        return fitness_approximation_model

    def _create_neighbour_model(self,
                                neighbour_update: int,
                                neighbour_model: str,
                                neighbour_model_params: dict,
                                fitness_approximation_update: int,
                                fitness_approximation_method: str,
                                position_vectors: np.ndarray
                                ) -> skn.KDTree | skn.BallTree:
        """Create a model for finding the neighbours of particles."""
        if neighbour_model == "kd_tree":
            ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
            return skn.KDTree(position_vectors, **neighbour_model_params)
        elif neighbour_model == "ball_tree":
            ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
            return skn.BallTree(position_vectors, **neighbour_model_params)
        elif neighbour_model == "knr":
            if fitness_approximation_method != "knr":
                raise ValueError(f"Neighbourhood method 'knr' requires fitness approximation method 'knr'. Got; {fitness_approximation_method}.")
            if neighbour_update != fitness_approximation_update:
                raise ValueError(f"Neighbourhood method 'knr' requires neighbour update rate (got; {neighbour_update}) \
                                 to be equal to fitness approximation update (got; {fitness_approximation_update}).")
        else:
            raise ValueError(f"Invalid neighbourhood method. Got; {neighbour_model}.")

    def _initialise_particles(self,
                              total_particles: int,
                              init_strategy: Literal["random", "linspace"],
                              rng: np.random.Generator,
                              dimensions: np.ndarray,
                              total_dimensions: int
                              ) -> tuple[np.ndarray, np.ndarray]:
        """Initialise the position and velocity vectors for the particles."""
        ## Random position initialisation strategy, randomly place particles over the search space with uniform probability distribution.
        if init_strategy == "random":
                position_vectors = rng.uniform(dimensions[:,0], dimensions[:,1], (total_particles, total_dimensions))
        
        ## Linearly spaced position initialisation strategy, deterministically place particles even spaced over the search space.
        elif init_strategy == "linspace":
            ## The number of particles to place along each dimension is the root to the total number of dimensions of the total number of particles.
            particles_across_dimensions: int = math.floor(total_particles ** (1.0 / total_dimensions))
            
            ## Linearly space the particles across each dimension and then mesh them together to get the position vectors.
            linspace_across_dimensions = np.linspace(dimensions[:,0], dimensions[:,1], particles_across_dimensions, axis=-1)
            position_vectors = np.dstack(np.meshgrid(*linspace_across_dimensions)).reshape(-1, total_dimensions)
            
            ## If the number of particles is not a perfect square, add some random particles to fill the rest of the space.
            deterministic_total_particles = int(particles_across_dimensions ** total_dimensions)
            if deterministic_total_particles < total_particles:
                position_vectors = np.concatenate((position_vectors, rng.uniform(dimensions[:,0], dimensions[:,1], (total_particles - deterministic_total_particles, total_dimensions))))
        
        ## Initialise velocity vectors to be random values between -max_velocity and max_velocity with uniform probability distribution.
        velocity_vectors = rng.uniform(-dimensions[:,2], dimensions[:,2], (total_particles, total_dimensions))
        
        return position_vectors, velocity_vectors
    
    def _calculate_fitness(self,
                           total_particles: int,
                           parallelise: bool,
                           threads: int,
                           total_dimensions: int,
                           position_vectors: np.ndarray
                           ) -> np.ndarray:
        """Calculate the fitness of all particles using numpy or dask."""
        if parallelise:
            dask_array = daskarray.from_array(position_vectors, chunks=(total_particles // threads, total_dimensions))
            dask_array = dask_array.map_blocks(lambda block: np.apply_along_axis(self.__fitness_function, 1, block),
                                               chunks=(total_particles // threads,), drop_axis=[1], dtype=np.float64)
            particles_fitness = np.array(dask_array.compute())
        else:
            particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors)
        return particles_fitness

import math
import numpy as np
import matplotlib.pyplot as plt

from control.pid import PIDController
from control.systems import InvertedPendulumSystem
from control.controlutils import simulate_control

if __name__ == "__main__":
    
    cluster = LocalCluster(n_workers=20, processes=True, threads_per_worker=2)
    client = Client(cluster)
    
    pend_system = InvertedPendulumSystem()
    controller = PIDController(0.0, 0.0, 0.0, initial_error=pend_system.get_control_input() - pend_system.get_setpoint())

    ticks: int = 100
    delta_time: float = 0.1

    def control_evaluator(vec):
        controller.reset()
        controller.set_gains(*vec)
        pend_system.reset()
        return simulate_control(pend_system, controller, ticks, delta_time)

    psystem = ParticleSwarmSystem([Dimension(0.0, 25.0, 2.5),
                                Dimension(0.0, 5.0, 0.5),
                                Dimension(0.0, 10.0, 1.0)],
                                control_evaluator,
                                maximise=False)

    result, data = psystem.run(total_particles=10000,
                            init_strategy="linspace",
                            iterations_limit=1000,
                            stagnation_limit=1.0,
                            inertia=1.0,
                            final_inertia=0.25,
                            inertia_decay_rate=1.0,
                            inertia_decay_type="lin",
                            personal_coef=2.0,
                            global_coef_final=1.5,
                            use_neighbourhood=False,
                            coef_decay_rate=1.0,
                            coef_decay_type="sin",
                            bounce=False,
                            use_fitness_approximation=False,
                            fitness_approximation_method="dtr",
                            parallelise=True,
                            threads=40,
                            gather_stats=True,
                            use_tqdm=True)
    
    print(f"Best position :: {result.best_position}")

    fig, (fitness_axes, position_axes, velocity_axes, coefficient_axes) = plt.subplots(1, 4)
    fitness_axes.plot(data["iteration"], data["global_best_fitness"], label="Global best fitness")
    # fitness_axes.plot(data["iteration"], data["mean_best_particles_fitness"], label="Mean best particles fitness")
    # fitness_axes.plot(data["iteration"], data["std_particles_fitness"], label="Std current particles fitness")
    position_axes.plot(data["iteration"], data["mean_distance_from_global_best"], "b", label="Mean distance from global best")
    position_axes.plot(data["iteration"], data["mean_distance_from_global_best"] + data["std_distance_from_global_best"], "r--", label="Std distance from global best")
    position_axes.plot(data["iteration"], data["mean_distance_from_global_best"] - data["std_distance_from_global_best"], "r--", label="Std distance from global best")
    velocity_axes.plot(data["iteration"], data["mean_velocity"], "b", label="Mean velocity")
    velocity_axes.plot(data["iteration"], data["mean_velocity"] + data["std_velocity"], "r--", label="Std velocity")
    velocity_axes.plot(data["iteration"], data["mean_velocity"] - data["std_velocity"], "r--", label="Std velocity")
    coefficient_axes.plot(data["iteration"], data["personal_coef"], label="personal")
    coefficient_axes.plot(data["iteration"], data["global_coef"], label="global")
    coefficient_axes.plot(data["iteration"], data["neighbour_coef"], label="global")
    coefficient_axes.legend()
    fig.set_size_inches(18.5, 10.5)
    fig.subplots_adjust(hspace=2.0)
    plt.show()
