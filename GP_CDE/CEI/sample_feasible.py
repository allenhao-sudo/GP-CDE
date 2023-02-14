import numpy as np
from utils import GeneticAlgorithm as GA
from scipy.stats import norm
import scipy.spatial as scpspatial
import time
from utils import from_unit_cube, to_unit_cube
import gpytorch
import torch
from copy import deepcopy


def sample_feasible(problem, GP_obj, mu_obj, sigma_obj, GP_Con, mu_con, sigma_con, best, x, num_q=1, rho=0):
    # use the CEI criterion when feasible solution exists
    new_x = np.zeros([num_q, problem.dim])
    n_g = len(GP_Con)
    for i in range(num_q):
        def objctive(p):
            n = p.shape[0]
            mu_g, sigma_g = model_predict(deepcopy(p), GP_Con, mu_con, sigma_con, problem)

            pof = norm.cdf(-mu_g / sigma_g)  # probability of feasibility

            mu_y, sigma_y = model_predict(deepcopy(p), GP_obj, mu_obj, sigma_obj, problem)
            gamma = np.maximum((best - mu_y), 0) / sigma_y
            beta = gamma * norm.cdf(gamma) + norm.pdf(gamma)
            ei = (sigma_y * beta).reshape(-1, 1)  # expected improvement

            POF = np.prod(pof, 1, keepdims=True)
            EIC = POF * ei  # expected improvement for constrained
            # dists = scpspatial.distance.cdist(p, x)
            # dmerit = np.amin(dists, axis=1, keepdims=True)
            # EIC[dmerit < rho] = -1e5
            return -EIC

        ga = GA(
            function=objctive,
            dim=problem.dim,
            lb=problem.lb,
            ub=problem.ub,
            pop_size=max([2 * problem.dim, 100]),
            num_gen=100,
        )
        # t1 = time.time()
        x_best, f_min = ga.optimize()
        # print("EIC optmization time cost:", time.time() - t1)
        new_x[i, :] = x_best
        return new_x


def model_predict(X_cand, gp, mu, sigma, problem):
    X_cand = to_unit_cube(X_cand, problem.lb, problem.ub)

    # Figure out what device we are running on
    if len(X_cand) < 300:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = torch.device("cuda"), torch.float64
    # We may have to move the GP to a new device
    n_g = len(gp)
    n_cand = X_cand.shape[0]
    mu_cand = np.zeros([n_cand, n_g])
    sigma_cand = np.zeros([n_cand, n_g])
    for i in range(n_g):
        gp[i] = gp[i].to(dtype=dtype, device=device)

    with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000):
        X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
        for k in range(n_g):
            mu_cand[:, k] = gp[k](X_cand_torch).mean.cpu().detach().numpy()
            sigma_cand[:, k] = gp[k](X_cand_torch).variance.cpu().detach().numpy()
    # De-standardize the sampled values
    mu_cand = mu + sigma * mu_cand
    sigma_cand = sigma * sigma * sigma_cand
    # y_cand = np.maximum(y_cand, 0)
    # sum_g = np.sum(y_cand, axis=1)
    return mu_cand, sigma_cand