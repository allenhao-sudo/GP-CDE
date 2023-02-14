import numpy as np


def feasibility_rule(sample, best=None):
    n = sample.shape[0]
    better = np.zeros([n, 2])
    if best is None:
        better[0, :] = sample[0, :]
    else:
        if sample[0, 1] < best[0, 1]:
            better[0, :] = sample[0, :]
        elif sample[0, 1] == better[0, 1] and sample[0, 0] < best[0, 0]:
            better[0, :] = sample[0, :]
        else:
            better[0, :] = best
    if n > 1:
        for id in range(1, n):
            if sample[id, 1] < better[id - 1, 1]:
                better[id, :] = sample[id, :]
            elif sample[id, 1] == better[id - 1, 1] and sample[id, 0] < better[id - 1, 0]:
                better[id, :] = sample[id, :]
            else:
                better[id, :] = better[id - 1, :]
        return better
    else:
        return better
