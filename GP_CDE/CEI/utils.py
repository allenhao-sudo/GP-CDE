###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import numpy as np


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def round_vars(x, int_var, lb, ub):
    """Round integer variables to closest integer in the domain.

    :param x: Set of points, of size npts x dim
    :type x: numpy.array
    :param int_var: Set of indices of integer variables
    :type int_var: numpy.array
    :param lb: Lower bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper bounds, of size 1 x dim
    :type ub: numpy.array
    :return: The set of points with the integer variables
        rounded to the closest integer in the domain
    :rtype: numpy.array
    """

    if len(int_var) > 0:
        # Round the original ranged integer variables
        x[:, int_var] = np.round(x[:, int_var])
        # Make sure we don't violate the bound constraints
        for i in int_var:
            ind = np.where(x[:, i] < lb[i])
            x[ind, i] = lb[i]
            ind = np.where(x[:, i] > ub[i])
            x[ind, i] = ub[i]
    return x

def from_unit_box(x, lb, ub):
    """Maps a set of points from the unit box to the original domain

    :param x: Points to be mapped from the unit box, of size npts x dim
    :type x: numpy.array
    :param lb: Lower bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper bounds, of size 1 x dim
    :type ub: numpy.array
    :return: Points mapped to the original domain
    :rtype: numpy.array
    """
    return lb + (ub - lb) * x


class GeneticAlgorithm:
    """Genetic algorithm.

    Implementation of the real-valued Genetic algorithm. The mutations are
    normally distributed perturbations, the selection mechanism is a tournament
    selection, and the crossover oepration is the standard linear combination
    taken at a randomly generated cutting point.

    The total number of evaluations are popsize x ngen

    :param function: Function that can be used to evaluate the entire
        population. It needs to take an input of size pop_size x dim and
        return a numpy.array of size pop_size x 1
    :type function: Object
    :param dim: Number of dimensions
    :type dim: int
    :param lb: Lower variable bounds, of length dim
    :type lb: numpy.array
    :param ub: Lower variable bounds, of length dim
    :type ub: numpy.array
    :param int_var: List of indices with the integer valued variables
        (e.g., [0, 1, 5])
    :type int_var: list
    :param pop_size: Population size
    :type pop_size: int
    :param num_gen: Number of generations
    :type num_gen: int
    :param start: Method for generating the initial population
    :type start: string

    :ivar nvariables: Number of variables (dimensions)
    :ivar nindividuals: population size
    :ivar lower_boundary: lower bounds for the optimization problem
    :ivar upper_boundary: upper bounds for the optimization problem
    :ivar integer_variables: List of variables that are integer valued
    :ivar start: Method for generating the initial population
    :ivar sigma: Perturbation radius. Each pertubation is N(0, sigma)
    :ivar p_mutation: Mutation probability (1/dim)
    :ivar tournament_size: Size of the tournament (5)
    :ivar p_cross: Cross-over probability (0.9)
    :ivar ngenerations: Number of generations
    :ivar function: Object that can be used to evaluate the objective function
    """

    def __init__(self, function, dim, lb, ub, int_var=None, pop_size=100, num_gen=100, start="SLHD"):

        self.nvariables = dim
        self.nindividuals = pop_size + (pop_size % 2)  # Make sure this is even
        self.lower_boundary = np.array(lb)
        self.upper_boundary = np.array(ub)
        self.integer_variables = []
        if int_var is not None:
            self.integer_variables = np.array(int_var)
        self.start = start
        self.sigma = 0.2
        self.p_mutation = 1.0 / dim
        self.tournament_size = 5
        self.p_cross = 0.9
        self.ngenerations = num_gen
        self.function = function

    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """
        #  Initialize population
        if isinstance(self.start, np.ndarray):
            if self.start.shape[0] != self.nindividuals or self.start.shape[1] != self.nvariables:
                raise ValueError("Initial population has incorrect size")
            if any(np.min(self.start, axis=0) >= self.lower_boundary) or any(
                np.max(self.start, axis=0) <= self.upper_boundary
            ):
                raise ValueError("Initial population is outside the domain")
            population = self.start
        elif self.start == "SLHD":
            from experimental_design import SymmetricLatinHypercube

            exp_des = SymmetricLatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            from experimental_design import LatinHypercube

            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            population = self.lower_boundary + np.random.rand(self.nindividuals, self.nvariables) * (
                self.upper_boundary - self.lower_boundary
            )
        else:
            raise ValueError("Unknown argument for initial population")

        new_population = []
        #  Round positions
        if len(self.integer_variables) > 0:
            new_population = np.copy(population)
            population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
            for i in self.integer_variables:
                ind = np.where(population[:, i] < self.lower_boundary[i])
                population[ind, i] += 1
                ind = np.where(population[:, i] > self.upper_boundary[i])
                population[ind, i] -= 1

        #  Evaluate all individuals
        function_values = self.function(population)
        if len(function_values.shape) == 2:
            function_values = np.squeeze(np.asarray(function_values))

        # Save the best individual
        ind = np.argmin(function_values)
        best_individual = np.copy(population[ind, :])
        best_value = function_values[ind]

        if len(self.integer_variables) > 0:
            population = new_population

        # Main loop
        for _ in range(self.ngenerations):
            # Do tournament selection to select the parents
            competitors = np.random.randint(0, self.nindividuals, (self.nindividuals, self.tournament_size))
            ind = np.argmin(function_values[competitors], axis=1)
            winner_indices = np.zeros(self.nindividuals, dtype=int)
            for i in range(self.tournament_size):  # This loop is short
                winner_indices[np.where(ind == i)] = competitors[np.where(ind == i), i]

            parent1 = population[winner_indices[0 : self.nindividuals // 2], :]
            parent2 = population[winner_indices[self.nindividuals // 2 : self.nindividuals], :]

            # Averaging Crossover
            cross = np.where(np.random.rand(self.nindividuals // 2) < self.p_cross)[0]
            nn = len(cross)  # Number of crossovers
            alpha = np.random.rand(nn, 1)

            # Create the new chromosomes
            parent1_new = np.multiply(alpha, parent1[cross, :]) + np.multiply(1 - alpha, parent2[cross, :])
            parent2_new = np.multiply(alpha, parent2[cross, :]) + np.multiply(1 - alpha, parent1[cross, :])
            parent1[cross, :] = parent1_new
            parent2[cross, :] = parent2_new
            population = np.concatenate((parent1, parent2))

            # Apply mutation
            scale_factors = self.sigma * (self.upper_boundary - self.lower_boundary)  # Scale
            perturbation = np.random.randn(self.nindividuals, self.nvariables)  # Generate perturbations
            perturbation = np.multiply(perturbation, scale_factors)  # Scale accordingly
            perturbation = np.multiply(
                perturbation, (np.random.rand(self.nindividuals, self.nvariables) < self.p_mutation)
            )

            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(self.lower_boundary, (1, self.nvariables)), population)
            population = np.minimum(np.reshape(self.upper_boundary, (1, self.nvariables)), population)

            # Round chromosomes
            new_population = []
            if len(self.integer_variables) > 0:
                new_population = np.copy(population)
                population = round_vars(population, self.integer_variables, self.lower_boundary, self.upper_boundary)

            # Keep the best individual
            population[0, :] = best_individual

            #  Evaluate all individuals
            function_values = self.function(population)
            if len(function_values.shape) == 2:
                function_values = np.squeeze(np.asarray(function_values))

            # Save the best individual
            ind = np.argmin(function_values)
            best_individual = np.copy(population[ind, :])
            best_value = function_values[ind]

            # Use the positions that are not rounded
            if len(self.integer_variables) > 0:
                population = new_population

        return best_individual, best_value