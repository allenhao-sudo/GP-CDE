from PyDE import DiffEvol
import numpy as np
from constrained_problems import *
from MyDB import MyDB
from PyDE import DiffEvol


if __name__ == "__main__":

    data = constrained_problem_6()
    ga = constrained_GeneticAlgorithm(data.objfunction, data.cons_vio, data.dim, data.xlow, data.xup, popsize=20,
                          ngen=50, start="SLHD")



    de = DiffEvol(data.objfunction, data.cons_vio, rbf, [[-5, 6], [-5, 6]], npop)


    db = MyDB(db_name='constrained_GA_data')
    for run_id in range(3):
        x_best, f_best, v_best, his = ga.optimize()

        # Print the best solution found
        print("\nBest function value: {0}".format(f_best))
        print("\nBest vio_sum value: {0}".format(v_best))
        print("Best solution: {0}".format(x_best))





        a = {'best_x': x_best.flatten().tolist(), 'best_f': f_best.flatten().tolist(), 'best_v': v_best.flatten().tolist(), 'history': his.flatten().tolist()}
        db.save('constrained_GA_data', 'run_'+str(run_id+1), a,)

    db.print_all_db()