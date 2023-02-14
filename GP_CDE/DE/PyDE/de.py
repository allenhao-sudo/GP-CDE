"""
Implements the differential evolution optimization method by Storn & Price
(Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)

.. moduleauthor:: Hannu Parviainen <hpparvi@gmail.com>
"""

from numba import njit
from numpy import asarray, zeros, zeros_like, tile, array, argmin, mod
from numpy.random import random, randint, rand, seed as rseed, uniform
import numpy as np
import time


class SymmetricLatinHypercube(object):
    """Symmetric Latin Hypercube experimental design

    :param dim: Number of dimensions
    :type dim: int
    :param npts: Number of desired sampling points
    :type npts: int

    :ivar dim: Number of dimensions
    :ivar npts: Number of desired sampling points
    """

    def __init__(self, dim, npts):
        self.dim = dim
        self.npts = npts

    def _slhd(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
        :rtype: numpy.array
        """

        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.npts, self.dim])
        points[:, 0] = np.arange(1, self.npts+1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.npts//2

        # special manipulation if odd number of rows
        if self.npts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.npts-i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.npts):
            points[i, :] = self.npts + 1 - points[self.npts - 1 - i, :]

        return points/self.npts

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
            that is of full rank
        :rtype: numpy.array
        :raises ValueError: Unable to find an SLHD of rank at least dim + 1
        """

        rank_pmat = 0
        pmat = np.ones((self.npts, self.dim + 1))
        xsample = None
        max_tries = 100
        counter = 0
        while rank_pmat != self.dim + 1:
            xsample = self._slhd()
            pmat[:, 1:] = xsample
            rank_pmat = np.linalg.matrix_rank(pmat)
            counter += 1
            if counter == max_tries:
                raise ValueError("Unable to find a SLHD of rank at least dim + 1, is npts too smal?")
        return xsample


def wrap(v, vmin, vmax):
    w = vmax - vmin
    return vmin + mod(asarray(v) - vmin, w)


@njit
def evolve_vector(i, pop, f, c):
    npop, ndim = pop.shape

    # --- Vector selection ---
    v1, v2, v3 = i, i, i
    while v1 == i:
        v1 = randint(npop)
    while (v2 == i) or (v2 == v1):
        v2 = randint(npop)
    while (v3 == i) or (v3 == v2) or (v3 == v1):
        v3 = randint(npop)

    # --- Mutation ---
    v = pop[v1] + f * (pop[v2] - pop[v3])

    # --- Cross over ---
    jf = randint(ndim)
    co = rand(ndim)
    for j in range(ndim):
        if co[j] > c and j != jf:
            v[j] = pop[i, j]
    return v


@njit("float64[:,:](float64[:,:], float64[:,:], float64, float64)")
def evolve_population(pop, pop2, f, c):
    npop, ndim = pop.shape

    for i in range(npop):

        # --- Vector selection ---
        v1, v2, v3 = i, i, i
        while v1 == i:
            v1 = randint(npop)
        while (v2 == i) or (v2 == v1):
            v2 = randint(npop)
        while (v3 == i) or (v3 == v2) or (v3 == v1):
            v3 = randint(npop)

        # --- Mutation ---
        v = pop[v1] + f * (pop[v2] - pop[v3])

        # --- Cross over ---
        co = rand(ndim)
        for j in range(ndim):
            if co[j] <= c:
                pop2[i, j] = v[j]
            else:
                pop2[i, j] = pop[i, j]

        # --- Forced crossing ---
        j = randint(ndim)
        pop2[i, j] = v[j]

    return pop2


class DiffEvol(object):
    """
    Implements the differential evolution optimization method by Storn & Price
    (Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)

    :param fun:
       the function to be minimized

    :param bounds:
        parameter bounds as [npar,2] array

    :param npop:
        the size of the population (5*D - 10*D)

    :param  f: (optional)
        the difference amplification factor. Values of 0.5-0.8 are good in most cases.

    :param c: (optional)
        The cross-over probability. Use 0.9 to test for fast convergence, and smaller
        values (~0.1) for a more elaborate search.

    :param seed: (optional)
        Random seed

    :param maximize: (optional)
        Switch setting whether to maximize or minimize the function. Defaults to minimization.
    """

    def __init__(self, fun, cons, bounds, npop, ngen, f=None, c=None, seed=None, maximize=True, constrained=False,
                 vectorize=False, cbounds=(0.25, 1),
                 fbounds=(0.25, 0.75), pool=None, min_ptp=1e-2, args=[], kwargs={}):
        if seed is not None:
            rseed(seed)

        self.minfun = _function_wrapper(fun, args, kwargs)
        self.consvio = _function_wrapper(cons, args, kwargs)
        self.bounds = asarray(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.n_pop = npop
        self.n_par = self.bounds.shape[0]
        self.bl = tile(self.bounds[:, 0], [npop, 1])
        self.bw = tile(self.bounds[:, 1] - self.bounds[:, 0], [npop, 1])
        # print(self.bl)
        # print(self.bw)
        # print(self.lower_bounds,'\n',self.upper_bounds)
        self.m = -1 if maximize else 1
        self.pool = pool
        self.args = args
        self.ngen = ngen
        self._t = []
        if self.pool is not None:
            self.map = self.pool.map
        else:
            self.map = map

        self.periodic = []
        self.min_ptp = min_ptp

        self.cmin = cbounds[0]
        self.cmax = cbounds[1]
        self.cbounds = cbounds
        self.fbounds = fbounds

        self.seed = seed
        self.f = f
        self.c = c

        # self._population = asarray(self.bl + random([self.n_pop, self.n_par]) * self.bw)
        exp_des = SymmetricLatinHypercube(self.n_par, self.n_pop)
        self._population = self.lower_bounds + exp_des.generate_points() * \
                     (self.upper_bounds - self.lower_bounds)
        self._fitness = zeros(npop)
        self._consvio = zeros(npop)
        self._minidx = None

        self._trial_pop = zeros_like(self._population)
        self._trial_fit = zeros_like(self._fitness)
        self._trial_vio = zeros_like(self._consvio)

        self._his = zeros([int(self.n_pop * self.ngen), self.n_par + 2])
        self._his_best_individual = zeros([int(self.ngen), 2])
        if vectorize == True and constrained == False:
            self._eval = self._eval_vfun
        elif vectorize == False and constrained == False:
            self._eval = self._eval_sfun
        elif vectorize == False and constrained == True:
            self._eval = self._eval_sfun_constrained
        else:
            pass

    @property
    def population(self):
        """The parameter vector population"""
        return self._population

    @property
    def minimum_value(self):
        """The best-fit value of the optimized function"""
        return self._fitness[self._minidx]

    @property
    def minimum_vio(self):
        """he best-vio value of the optimized function"""
        return self._consvio[self._minidx]

    @property
    def minimum_location(self):
        """The best-fit solution"""
        return self._population[self._minidx, :]

    @property
    def minimum_index(self):
        """Index of the best-fit solution"""
        return self._minidx

    @property
    def his(self):
        """Index of the best-fit solution"""
        return self._his

    @property
    def time(self):
        """return the time cost for x times of function evaluation"""
        return self._t

    @property
    def his_best_individuals(self):
        """return the time cost for x times of function evaluation"""
        return self._his_best_individual

    def optimize(self, ngen):
        """Run the optimizer for ``ngen`` generations"""
        for res in self(ngen):
            pass
        return res

    def __call__(self, ngen=1):
        return self._eval(ngen)

    def _eval_sfun_constrained(self, ngen):
        """Run DE for a constrained function that takes a single pv as an input and retuns a single value."""
        popc, fitc, cons_vioc = self._population, self._fitness, self._consvio
        popt, fitt, cons_viot = self._trial_pop, self._trial_fit, self._trial_vio
        t0 = time.time()
        # print(t0)
        for ipop in range(self.n_pop):
            fitc[ipop] = self.m * self.minfun(np.reshape(popc[ipop, :], (-1, self.n_par)))
            cons_vioc[ipop] = self.consvio(np.reshape(popc[ipop, :], (-1, self.n_par)))
        # save to his
        self._his[:self.n_pop, 0:self.n_par] = np.copy(popc)
        self._his[:self.n_pop, self.n_par] = np.copy(fitc)
        self._his[:self.n_pop, self.n_par + 1] = np.copy(cons_vioc)
        self._minidx = 0
        self._his_best_individual[0, 0] = np.copy(fitc[self._minidx])
        self._his_best_individual[0, 1] = np.copy(cons_vioc[self._minidx])
        for igen in range(ngen - 1):

            f = self.f or uniform(*self.fbounds)
            c = self.c or uniform(*self.cbounds)

            popt = evolve_population(popc, popt, f, c)
            popt = np.maximum(np.reshape(self.lower_bounds, (1, self.n_par)), popt)
            popt = np.minimum(np.reshape(self.upper_bounds, (1, self.n_par)), popt)
            for ipop in range(self.n_pop):
                fitt[ipop] = self.m * self.minfun(np.reshape(popt[ipop, :], (-1, self.n_par)))
                cons_viot[ipop] = self.consvio(np.reshape(popt[ipop, :], (-1, self.n_par)))

                # update popc
                if cons_viot[ipop] < cons_vioc[ipop]:
                    fitc[ipop] = fitt[ipop]
                    popc[ipop, :] = popt[ipop, :]
                    cons_vioc[ipop] = cons_viot[ipop]
                elif cons_viot[ipop] == cons_vioc[ipop]:
                    if fitt[ipop] <= fitc[ipop]:
                        fitc[ipop] = fitt[ipop]
                        popc[ipop, :] = popt[ipop, :]
                        cons_vioc[ipop] = cons_viot[ipop]
                else:
                    pass

                # update current best
                if cons_viot[ipop] < cons_vioc[self._minidx]:
                    self._minidx = ipop
                elif cons_viot[ipop] == cons_vioc[self._minidx]:
                    if fitt[ipop] <= fitc[self._minidx]:
                        self._minidx = ipop
                else:
                    pass

            # save to his
            # print('igen =', str(igen))
            # print(self._his)
            self._his[(igen + 1) * self.n_pop:(igen + 2) * self.n_pop, :self.n_par] = np.copy(popt)
            self._his[(igen + 1) * self.n_pop:(igen + 2) * self.n_pop, self.n_par] = np.copy(fitt)
            self._his[(igen + 1) * self.n_pop:(igen + 2) * self.n_pop, self.n_par + 1] = np.copy(cons_viot)
            self._his_best_individual[igen+1, 0] = np.copy(fitc[self._minidx])
            self._his_best_individual[igen+1, 1] = np.copy(cons_vioc[self._minidx])
            #
            # if fitc.ptp() < self.min_ptp:
            #     break

            yield popc[self._minidx, :], fitc[self._minidx]

    def _eval_sfun(self, ngen=1):
        """Run DE for a function that takes a single pv as an input and retuns a single value."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        for ipop in range(self.n_pop):
            fitc[ipop] = self.m * self.minfun(popc[ipop, :])

        for igen in range(ngen):
            f = self.f or uniform(*self.fbounds)
            c = self.c or uniform(*self.cbounds)

            popt = evolve_population(popc, popt, f, c)
            fitt[:] = self.m * array(list(self.map(self.minfun, popt)))

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            if fitc.ptp() < self.min_ptp:
                break

            yield popc[self._minidx, :], fitc[self._minidx]

    def _eval_vfun(self, ngen=1):
        """Run DE for a function that takes the whole population as an input and retuns a value for each pv."""
        popc, fitc = self._population, self._fitness
        popt, fitt = self._trial_pop, self._trial_fit

        fitc[:] = self.m * self.minfun(self._population)

        for igen in range(ngen):
            f = self.f or uniform(*self.fbounds)
            c = self.c or uniform(*self.cbounds)

            popt = evolve_population(popc, popt, f, c)
            fitt[:] = self.m * self.minfun(popt)

            msk = fitt < fitc
            popc[msk, :] = popt[msk, :]
            fitc[msk] = fitt[msk]

            self._minidx = argmin(fitc)
            if fitc.ptp() < self.min_ptp:
                break

            yield popc[self._minidx, :], fitc[self._minidx]


class _function_wrapper(object):
    def __init__(self, f, args=[], kwargs={}):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
