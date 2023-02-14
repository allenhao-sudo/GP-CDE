import os
import numpy as np
from ..CEC2010 import CEC2010
from ..CEC2017 import CEC2017
from experimental_design import SymmetricLatinHypercube
from sample_feasible import sample_feasible
from sample_infeasible import sample_infeasible
import time
from feasibility_rule import feasibility_rule
import torch
from gp import train_gp
from utils import to_unit_cube
from copy import deepcopy
import gpytorch
import matlab.engine


def CEI(run_index, func_id, D, maxeval):
    dirname = os.path.dirname(__file__)
    filename = dirname + '/results/CEC2010/Function_' + str(func_id + 1) + '/dim_' + str(D) + '/Maxeval_' + str(
        maxeval) + '/'
    eng = matlab.engine.start_matlab()
    CEC_path = os.path.dirname(dirname)
    eng.cd(CEC_path + '/CEC2010/', nargout=0)
    problem = CEC2010.CEC2010(int(func_id), D, eng)
    dim, lb, ub = problem.dim, problem.lb, problem.ub  # get basic info of problem
    l = np.min(ub - lb)
    num_initial = 2 * (dim + 1)  # num for initial LHS design
    num_q = 1  # num of generate points based on GP at each iteration
    num_eval = 0
    max_eval = maxeval  # max num of real function eval
    n_g = problem.ng  # num of constraints
    y = np.empty([0, 2])
    x = np.empty([0, dim])
    g = np.empty([0, n_g])
    his_best_y = np.zeros([0, 2])
    initial_y = np.zeros([num_initial, 2])
    initial_g = np.zeros([num_initial, n_g])

    t1 = time.time()
    train_time = 0
    # initial design with LHS
    exp_des = SymmetricLatinHypercube(dim, num_initial)
    inital_x = lb + exp_des.generate_points() * (ub - lb)
    for i in range(num_initial):
        initial_y[i, 0] = problem.obj(inital_x[i, :])
        initial_g[i, :] = problem.expensive_con(inital_x[i, :])
    initial_y[:, 1] = np.sum(np.maximum(initial_g, 0), axis=1, keepdims=False)
    y = np.vstack((y, initial_y))
    g = np.vstack((g, initial_g))
    x = np.vstack((x, inital_x))
    num_eval += num_initial
    better = feasibility_rule(initial_y)
    his_best_y = np.vstack((his_best_y, better))
    best = his_best_y[-1, :].reshape(-1, 2)

    # main loop
    while num_eval < max_eval:

        # Warp inputs
        X = to_unit_cube(deepcopy(x), lb, ub)

        # Standardize values
        copy_g = deepcopy(g)
        copy_y = deepcopy(y[:, 0])

        # build global gp for expensive constraints
        tic = time.time()
        GP_con, mu_con, sigma_con = model_fit(X, copy_g, n_training_steps=25)

        # build global gp for objective
        GP_obj, mu_obj, sigma_obj = model_fit(X, copy_y, n_training_steps=25)
        train_time = train_time + (time.time() - tic)
        # print("GP fit time cost:", time.time() - t2)

        fea_num = np.sum(g <= 1e-6, 1)
        fea_index = np.where(fea_num == n_g)[0]
        rho = 1 / (5 * max_eval * num_q) * np.sqrt(dim) * (max_eval - num_eval) / max_eval * l
        if np.sum(fea_index) == 0:
            print('***process-{}***: evaluation:{} , no feasible solution'.format(run_index, num_eval))
            sample_x = sample_infeasible(problem, GP_con, mu_con, sigma_con, x, num_q, rho)
        else:
            best_new = his_best_y[-1, :].reshape(-1, 2)
            if best_new[0, 1] < best[0, 1]:
                print('***process-{}***:  evaluation:{} , new best solution:{}'.format(run_index, num_eval,
                                                                                       best_new[0, 0]))
                best = best_new
            elif best_new[0, 1] == best[0, 1] and best_new[0, 0] < best[0, 0]:
                print('***process-{}***:  evaluation:{} , new best solution:{}'.format(run_index, num_eval,
                                                                                       best_new[0, 0]))
                best = best_new
            else:
                print('***process-{}***:  evaluation:{} , new best solution:{}'.format(run_index, num_eval,
                                                                                       best_new[0, 0]))
            sample_x = sample_feasible(problem, GP_obj, mu_obj, sigma_obj,
                                       GP_con, mu_con, sigma_con, best[0, 0], x, num_q, rho)

        n = sample_x.shape[0]
        sample_y = np.zeros([n, 2])
        sample_g = np.zeros([n, n_g])
        for i in range(n):
            sample_y[i, 0] = problem.obj(sample_x[i, :])
            sample_g[i, :] = problem.expensive_con(sample_x[i, :])
        sample_y[:, 1] = np.sum(np.maximum(sample_g, 0), axis=1, keepdims=False)
        y = np.vstack((y, sample_y))
        x = np.vstack((x, sample_x))
        g = np.vstack((g, sample_g))
        num_eval += n
        better = feasibility_rule(sample_y, best=best)
        his_best_y = np.vstack((his_best_y, better))
    # save the result

    time_cost = time.time() - t1
    fea_num = np.sum(g <= 1e-6, 1)
    fea_index = np.where(fea_num == n_g)[0]
    best_x = np.empty([0, dim])
    best_y = None
    if np.sum(fea_index) == 0:
        print('***process-{}***:  evaluation:{} , no feasible solution'.format(run_index, num_eval))
    else:
        ymin = np.argmin(y[fea_index, 0])
        ymin_id = fea_index[ymin]
        best_x = x[ymin_id, :]
        best_y = y[ymin_id, 0]
        print('***process-{}***:  evaluation:{} , best solution:{}'.format(run_index, num_eval, best[0, 0]))
    result = {'his_x': x.flatten(), 'his_y': y[:, 0], 'his_g': g.flatten(), 'time': time_cost,
              "best_x": best_x, 'best_y': best_y, 'his_best_y': his_best_y[:, 0],
              'his_best_g': his_best_y[:, 1], 'train_time': train_time}
    if not os.path.isdir(filename):
        os.makedirs(filename)
    np.save(filename + str(run_index) + ".npy", result)


def model_fit(X, fX, n_training_steps):
    # Standardize function values.
    mu, sigma = np.median(fX, axis=0), np.std(fX, axis=0)

    sigma = np.where(sigma < 1e-6, 1, sigma)

    fX = (deepcopy(fX) - mu) / sigma
    if len(fX.shape) == 1:
        fX = fX.reshape(-1, 1)
    # Figure out what device we are running on
    if len(X) < 1024:
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = torch.device("cuda"), torch.float64

    # We use CG + Lanczos for training if we have enough data
    n_g = fX.shape[1]
    gp = [0] * n_g
    with gpytorch.settings.max_cholesky_size(2000):
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        for i in range(n_g):
            y_torch = torch.tensor(fX[:, i]).to(device=device, dtype=dtype)
            gp[i] = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=True, num_steps=n_training_steps, hypers={}
            )
    del X_torch, y_torch
    return gp, mu, sigma


if __name__ == "__main__":
    CEC2010_set = np.array([0, 6, 7, 12])
    CEC2017_set = np.array([0, 1, 3, 4, 12, 19])
    CEI(run_index=0, func_id=CEC2010_set[0], D=10, maxeval=500)
