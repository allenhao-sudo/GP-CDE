import numpy as np
import scipy.optimize as op
from PSSmechV3 import PSSmechV3


# Init the 6PSS object
pss = PSSmechV3()
qL = 0.2235  # Lower bound of the slider displacement
qU = qL + 0.6  # Upper bound of the slider displacement
sU = np.deg2rad(40)  # maximum rotation angle of the spherical joint
rL = 0.45  # Lower boundary of the distribution radius of the upper spherical joint
rU = 0.60  # upper boundary of the distribution radius of the upper spherical joint
thetaL = np.deg2rad(16)  # Lower boundary of upper spherical joint distribution angle
thetaU = np.deg2rad(17)  # Upper boundary of upper spherical joint distribution angle
rbaseU = 1.8  # Upper boundary of the base radius
qvelU = 3  # Maximum velocity of the slider（m/s）
qaccU = 5 * 9.81  # Maximum acceleration of the slider（m/s^2）
ha = 0.3
lb = 0.17358
hb = 0.21988
cz = 1.4

ra_bounds = [rL * np.cos(np.pi / 3 - thetaL / 2),
             rU * np.cos(np.pi / 3 - thetaU / 2)]
da_bounds = [2 * rL * np.sin(np.pi / 3 - thetaU / 2),
             2 * rU * np.sin(np.pi / 3 - thetaL / 2)]
la_bounds = [0.1, 0.15]
rb_bounds = [1, 1.5]
db_bounds = [0.15, 0.16]
tb_bounds = [np.deg2rad(30), np.deg2rad(46)]
q0_bounds = [qL, qU]
a_bounds = [0.002, 0.20]
bounds = [ra_bounds, da_bounds, la_bounds, rb_bounds,
          db_bounds, tb_bounds, q0_bounds, a_bounds]
bounds = np.asarray(bounds)


def coneFinder(xa, ya, za):
    """
    Envelope cone solver, given trajectory,
    returns the spherical coordinate angle
    and half-cone angle of the envelope cone
    """

    def Fobj(x):
        return x[2]

    def con(x):
        res = []
        for i in range(len(xa)):
            vab = np.array([xa[i], ya[i], za[i]])
            p = np.array([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0])
                          * np.sin(x[1]), np.cos(x[1])])
            res.append(x[2] - np.arccos(np.dot(vab, p) /
                                        (np.linalg.norm(vab) * np.linalg.norm(p))))
        return np.array(res)

    ineq_cons = {'type': 'ineq',
                 'fun': con}
    bounds = [[-np.pi, np.pi], [-np.pi, np.pi], [0, np.pi]]
    x0 = np.array([0.01, 0.01, 0.01])
    res = op.minimize(Fobj, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': False},
                      bounds=bounds)
    return res.x


def obj(p):
    """
    Objective function: the maximum side length of the 6D coupled cube
    """
    p = p.flatten()
    return -p[7]


def con(p):
    """
    Various constraints
    """
    # ——————————————————————————————————————————————————————————————————
    # Updating the parameters
    # ——————————————————————————————————————————————————————————————————
    p = p.flatten()
    pss.set_param(ra=p[0], da=p[1], la=p[2], ha=ha, rb=p[3],
                  db=p[4], lb=lb, hb=hb, tb=p[5], q0=p[6], cz=cz)
    a = p[7]  # side length of the 6D coupled cube
    # ——————————————————————————————————————————————————————————————————
    # Simulated trajectories
    # ——————————————————————————————————————————————————————————————————
    t1, T1, pos1, vel1, acc1 = pss.sinTest_ALL(n=100)
    # 3D coupled space
    pos2 = []
    A = 0.13
    for i in [-A, A]:
        for j in [-A, A]:
            for k in [-A, A]:
                pos2.append([i, j, k, 0, 0, 0])
    # 6D coupled space
    B = np.deg2rad(15)
    for i in [-a, a]:
        for j in [-a, a]:
            for k in [-a, a]:
                for al in [-B, B]:
                    for be in [-B, B]:
                        for ga in [-B, B]:
                            pos2.append([i, j, k, al, be, ga])
    pos2 = np.array(pos2).T
    # Appending all above motions
    pos3 = np.hstack((pos1, pos2))
    # ——————————————————————————————————————————————————————————————————
    # Constraints violation calculation
    # ——————————————————————————————————————————————————————————————————
    qvel = pss.qvel(X=pos1, dX=vel1)
    qacc = pss.qacc(X=pos1, dX=vel1, ddX=acc1)
    qvel = np.abs(qvel.flatten())
    qacc = np.abs(qacc.flatten())
    con_qvelU = qvelU - qvel.max()  # 1
    con_qaccU = qaccU - qacc.max()  # 2
    qpos2 = pss.IK(x=pos2[0], y=pos2[1], z=pos2[2],
                   yaw=pos2[3], pitch=pos2[4], roll=pos2[5])
    qpos2 = qpos2.flatten()
    con_qpos2L = qpos2.min() - qL  # 3
    con_qpos2U = qU - qpos2.max()  # 4
    p_Mab, p_Oba = pss.rodRelativeTrajectory(
        x=pos3[0], y=pos3[1], z=pos3[2], yaw=pos3[3], pitch=pos3[4], roll=pos3[5])
    cone = np.zeros((2, 6)).tolist()
    for i in range(0, 6):
        cone[0][i] = (coneFinder(p_Mab[0, i], p_Mab[1, i], p_Mab[2, i]))[2]
        cone[1][i] = (coneFinder(p_Oba[0, i], p_Oba[1, i], p_Oba[2, i]))[2]
    con_coneU = sU - np.array(cone).flatten().max()  # 5

    return np.array(
        [con_qvelU, con_qaccU, con_qpos2L, con_qpos2U, con_coneU])


class SixDOF_Problem():
    def __init__(self):
        self.dim = 8
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]
        self.bounds = bounds
        self.nexpensive = 5
        self.ncheap = 5

    def expensive_con(self, x):
        vio = -con(x)
        for i in range(len(vio)):
            if np.isnan(vio[i]):
                vio[i] = 1.0
        return vio

    def cheap_con(self, x):
        ra = x[0]
        da = x[1]
        r = np.sqrt(ra ** 2 + (da / 2) ** 2)
        theta = 2 * np.pi / 3 - 2 * np.arctan2(da / 2, ra)
        con_rL = r - rL  # 6
        con_rU = rU - r  # 7
        con_thetaL = theta - thetaL  # 8
        con_thetaU = thetaU - theta  # 9
        rb = x[3]
        con_rbaseU = rbaseU - np.sqrt((rb + 0.22472) ** 2 + 1.0703 ** 2)  # 10
        cheap_con = np.array([con_rL, con_rU, con_thetaL, con_thetaU, con_rbaseU])
        return -cheap_con

    def obj(self, x):
        if len(x.shape) > 1:
            return -x[:, 7]
        else:
            return -x[7]


if __name__ == "__main__":
    problem = SixDOF_Problem()
    p = np.array([0.27704771, 0.70920979, 0.1, 1.22250421,
                  0.15000007, 0.73920619, 0.45477733, 0.01116788])
    obj = problem.obj(p)
    expensive_con = problem.expensive_con(p)
    cheap_con = problem.cheap_con(p)
    print('Objective:', obj, '\n',
          "expensive_con:", expensive_con, '\n',
          'cheap_con:', cheap_con)
