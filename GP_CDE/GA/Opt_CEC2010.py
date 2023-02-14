from constrained_GA import constrained_GeneticAlgorithm
from ..CEC2010 import CEC2010
import matlab.engine
import numpy as np
import os
import time


def Opt_CEC2010(run_index, para, func_id, D, maxeval):
    dirname = os.path.dirname(__file__)
    path = os.path.dirname(dirname)
    eng = matlab.engine.start_matlab()
    eng.cd(path + '/CEC2010/', nargout=0)
    problem = CEC2010.CEC2010(int(func_id), D, eng)
    ga = constrained_GeneticAlgorithm(problem.obj, problem.expensive_con, D, problem.bounds[:, 0], problem.bounds[:, 1],
                                      popsize=para[0], ngen=para[1], start="SLHD")
    tic = time.time()
    x_best, f_best, v_best, time1, his_best_individual, his = ga.optimize()
    time_cost = time.time() - tic
    fea_index = np.where(his[:, D + 1] <= 1e-6)[0]
    best_x = np.empty([0, D])
    best_y = None
    if np.sum(fea_index) == 0:
        print('**process--{}**: no feasible solution'.format(run_index))
        print("time_cost:", time_cost)
    else:
        minrows = np.argmin(his[fea_index, D])
        minrows = fea_index[minrows]
        best_x = his[minrows, :D]
        best_y = his[minrows, D]
        print('**process--{}**:  best solution:{}'.format(run_index, his[minrows, :].flatten()))
        print("time_cost:", time_cost)
    # save the result
    his_best = np.zeros([maxeval, D + 2])
    his_best[0, :] = his[0, :]
    for id in range(1, maxeval):
        if his[id, D + 1] < his_best[id - 1, D + 1]:
            his_best[id, :] = his[id, :]
        elif his[id, D + 1] == his_best[id - 1, D + 1]:
            if his[id, D] < his_best[id - 1, D]:
                his_best[id, :] = his[id, :]
            else:
                his_best[id, :] = his_best[id - 1, :]
        else:
            his_best[id, :] = his_best[id - 1, :]
    result = {'his_x': his[:, :D].flatten(), 'his_y': his[:, D].flatten(),
              'his_g': his[:, D + 1].flatten(), 'time': time_cost,
              "best_x": best_x, 'best_y': best_y,
              "his_best_y": his_best[:, D], "his_best_g": his_best[:, D + 1]}
    dirname = os.path.dirname(__file__)
    filename = dirname + '/results/CEC2010/Function_' + str(func_id + 1) + '/dim_' + str(D) + '/Maxeval_' + str(
        maxeval) + '/'
    if not os.path.isdir(filename):
        os.makedirs(filename)
    np.save(filename + str(run_index + 20) + ".npy", result)


if __name__ == "__main__":
    func_set = np.array([0, 6, 7, 12])
    Opt_CEC2010(run_index=1, para=[25, 20], func_id=func_set[0], D=10, maxeval=500)
