from heuristic_methods import GeneticAlgorithm
from experimental_design import LatinHypercube, SymmetricLatinHypercube
import numpy as np
import time


class constrained_GeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, function, cons_vio, dim, xlow, xup, intvar=None, popsize=100, ngen=100, start="SLHD",
                 projfun=None):
        super().__init__(function, dim, xlow, xup, intvar=intvar, popsize=popsize, ngen=ngen, start=start,
                         projfun=projfun)
        self.cons_vio = cons_vio
        self.sigma_ini = 0.2
        self.his = np.zeros([int(self.ngenerations * self.nindividuals), self.nvariables + 2])
        self.his_best_individual = np.zeros([int(self.ngenerations), 2])
        self.constrained_if = True

    def Eval_Pop(self, population):

        #  Evaluate all individuals
        function_values = self.function(population)
        if len(function_values.shape) == 2:
            function_values = np.squeeze(np.asarray(function_values))

        if self.constrained_if == False:
            # Save the best individual
            ind = np.argmin(function_values)
            best_individual = np.copy(population[ind, :])
            best_value = function_values[ind]
            self.his[:self.nindividuals, :self.nvariables] = np.copy(population)
            self.his[:self.nindividuals, self.nvariables] = np.copy(function_values)
            return function_values, best_individual, best_value
        else:
            cons_viosum = self.cons_vio(population)
            if len(cons_viosum.shape) == 2:
                cons_viosum = np.squeeze(np.asarray(cons_viosum))
            # sort the pop based on the cons_viosum
            min_id = np.where(cons_viosum == cons_viosum.min())

            if len(min_id[0]) > 1:
                f_best_id = min_id[0][0]
                for k in range(len(min_id[0])):
                    if function_values[f_best_id] > function_values[min_id[0][k]]:
                        f_best_id = min_id[0][k]
                best_individual = np.copy(population[f_best_id, :])
                best_value = np.copy(function_values[f_best_id])
                best_vio = np.copy(cons_viosum[f_best_id])
            else:
                best_individual = np.copy(population[min_id[0], :])
                best_value = np.copy(function_values[min_id[0]])
                best_vio = np.copy(cons_viosum[min_id[0]])
            return function_values, cons_viosum, best_individual, best_value, best_vio

    def tournament_selection(self, competitors, cons_viosum, function_values):
        ind = np.zeros([self.nindividuals])
        for m in range(self.nindividuals):
            ind[m] = int(0)
            for k in range(competitors.shape[1]):

                if cons_viosum[competitors[m, k]] < cons_viosum[competitors[m, int(ind[m])]]:
                    ind[m] = k
                elif cons_viosum[competitors[m, k]] == cons_viosum[competitors[m, int(ind[m])]]:
                    if function_values[competitors[m, k]] < function_values[competitors[m, int(ind[m])]]:
                        ind[m] = k

        return ind

    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """
        t = []
        #  Initialize population
        if isinstance(self.start, np.ndarray):
            # if initial sampling size doesn't match the number of individuals and variable dimension, print error
            if self.start.shape[0] != self.nindividuals or self.start.shape[1] != self.nvariables:
                raise ValueError("Unknown method for generating the initial population")
            # if initial positions are outside the domain, print error
            # np.min(A, axis = 0) returns the minimum in all rows
            if (not all(np.min(self.start, axis=0) >= self.lower_boundary)) or \
                    (not all(np.max(self.start, axis=0) <= self.upper_boundary)):
                raise ValueError("Initial population is outside the domain")
            population = self.start
        elif self.start == "SLHD":
            exp_des = SymmetricLatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                         (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                         (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            # population = self.lower_boundary + np.random.rand(self.nindividuals, self.nvariables) * \
            #              (self.upper_boundary - self.lower_boundary)
            population = np.asarray(self.lower_boundary + np.random.random([self.nindividuals, self.nvariables]) *
                                 (self.upper_boundary - self.lower_boundary))
        else:
            raise ValueError("Unknown argument for initial population")
        # print(population)
        # print(population.shape)
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
        function_values, cons_viosum, best_individual, best_value, best_vio = self.Eval_Pop(population)
        self.his[:self.nindividuals, 0:self.nvariables] = np.copy(population)
        self.his[:self.nindividuals, self.nvariables] = np.copy(function_values)
        self.his[:self.nindividuals, self.nvariables + 1] = np.copy(cons_viosum)
        self.his_best_individual[0, 0] = np.copy(best_value)
        self.his_best_individual[0, 1] = np.copy(best_vio)
        if len(self.integer_variables) > 0:
            population = new_population

        # Main loop

        for ngen in range(self.ngenerations - 1):
            self.sigma = (1 - (ngen) / self.ngenerations) * self.sigma_ini
            # Do tournament selection to select the parents
            competitors = np.random.randint(0, self.nindividuals, (self.nindividuals, self.tournament_size))
            # find the best index
            # ind = np.argmin(cons_viosum[competitors], axis=1)
            ind = self.tournament_selection(competitors, cons_viosum, function_values)
            winner_indices = np.zeros(self.nindividuals, dtype=int)
            for i in range(self.tournament_size):  # This loop is short
                winner_indices[np.where(ind == i)] = competitors[np.where(ind == i), i]

            parent1 = population[winner_indices[0:self.nindividuals // 2], :]
            parent2 = population[winner_indices[self.nindividuals // 2:self.nindividuals], :]

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
            scale_factors = self.sigma * (self.upper_boundary - self.lower_boundary)  # Account for dimensions ranges
            perturbation = np.random.randn(self.nindividuals, self.nvariables)  # Generate perturbations
            perturbation = np.multiply(perturbation, scale_factors)  # Scale accordingly
            perturbation = np.multiply(perturbation, (np.random.rand(self.nindividuals,
                                                                     self.nvariables) < self.p_mutation))

            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(self.lower_boundary, (1, self.nvariables)), population)
            population = np.minimum(np.reshape(self.upper_boundary, (1, self.nvariables)), population)

            # Map to feasible region if method exists
            if self.projfun is not None:
                for i in range(self.nindividuals):
                    population[i, :] = self.projfun(population[i, :])

            # Round chromosomes
            new_population = []
            if len(self.integer_variables) > 0:
                new_population = np.copy(population)
                population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
                for i in self.integer_variables:
                    ind = np.where(population[:, i] < self.lower_boundary[i])
                    population[ind, i] += 1
                    ind = np.where(population[:, i] > self.upper_boundary[i])
                    population[ind, i] -= 1

            # Keep the best individual

            function_values, cons_viosum, best_individual, best_value, best_vio = self.Eval_Pop(population)
            self.his_best_individual[ngen+1, 0] = np.copy(best_value)
            self.his_best_individual[ngen+1, 1] = np.copy(best_vio)
            self.his[(ngen + 1) * self.nindividuals:(ngen + 2) * self.nindividuals, :self.nvariables] = np.copy(
                population)
            self.his[(ngen + 1) * self.nindividuals:(ngen + 2) * self.nindividuals, self.nvariables] = np.copy(
                function_values)
            if self.constrained_if == True:
                self.his[(ngen + 1) * self.nindividuals:(ngen + 2) * self.nindividuals, self.nvariables + 1] = np.copy(
                    cons_viosum)
            # Use the positions that are not rounded
            if len(self.integer_variables) > 0:
                population = new_population
        return best_individual, best_value, best_vio, t, self.his_best_individual.flatten('F'), self.his



