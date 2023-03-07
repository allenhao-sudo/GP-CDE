import numpy as np
from PSSfuncV3 import solve_q, solve_dq, solve_ddq, solve_pOa, solve_pOb, solve_jacobian, solve_vmab, solve_voba
import scipy.optimize as op


class PSSmechV3(object):
    """
    The PPSmech class is used to calculate the kinematics
    and dynamics of 6PPS parallel manipulator.
    """
    def __init__(self, ra=0.298151, da=0.752346, la=0.15, ha=0.3, rb=1.063493, db=0.188, lb=0.1736, hb=0.216,
                 tb=np.deg2rad(45), q0=0.477606, cz=1.4, lr=0.81966059):
        self._ra = ra
        self._da = da
        self._la = la
        self._ha = ha
        self._rb = rb
        self._db = db
        self._lb = lb
        self._hb = hb
        self._tb = tb
        self._q0 = q0
        self._cz = cz
        self._lr = lr

    # region Getter

    def get_param(self):
        print('ra = ', self._ra)
        print('da = ', self._da)
        print('la = ', self._la)
        print('ha = ', self._ha)
        print('rb = ', self._rb)
        print('db = ', self._db)
        print('lb = ', self._lb)
        print('hb = ', self._hb)
        print('tb = ', np.rad2deg(self._tb))
        print('q0 = ', self._q0)
        print('cz = ', self._cz)
        print('lr = ', self._lr)
        return np.array(
            [self._ra, self._da, self._la, self._ha, self._rb, self._db, self._lb, self._hb, self._tb, self._q0,
             self._cz, self._lr])

    def get_ra(self):
        return self._ra

    def get_da(self):
        return self._da

    def get_la(self):
        return self._la

    def get_ha(self):
        return self._ha

    def get_rb(self):
        return self._rb

    def get_db(self):
        return self._db

    def get_lb(self):
        return self._lb

    def get_hb(self):
        return self._hb

    def get_tb(self):
        return self._tb

    def get_q0(self):
        return self._q0

    def get_cz(self):
        return self._cz

    def get_lr(self):
        return self._lr

    # endregion

    # region Setter

    def set_param(self, ra, da, la, ha, rb, db, lb, hb, tb, q0, cz):
        self._ra = ra
        self._da = da
        self._la = la
        self._ha = ha
        self._rb = rb
        self._db = db
        self._lb = lb
        self._hb = hb
        self._tb = tb
        self._q0 = q0
        self._cz = cz
        self._lr = np.sqrt((-da / 2 + db / 2) ** 2 + (lb * np.sin(tb) - q0 * np.cos(tb) - ra + rb) ** 2 + (
                -cz + hb + la + lb * np.cos(tb) + q0 * np.sin(tb)) ** 2)

    # endregion

    def IK(self, x, y, z, yaw, pitch, roll):
        """
        Inverse kinematics calculation function
        """
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        lr = self._lr
        q = solve_q(x, y, z, yaw, pitch, roll, ra, da, la, ha, rb, db, lb, hb, tb, cz, lr)
        return q

    def qvel(self, X, dX):
        """
        Velocity calculation
        """
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        lr = self._lr
        dq = solve_dq(X, dX, ra, da, la, ha, rb, db, lb, hb, tb, cz, lr)
        return dq

    def qacc(self, X, dX, ddX):
        """
        Acceleration calculation
        """
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        lr = self._lr
        ddq = solve_ddq(X, dX, ddX, ra, da, la, ha, rb, db, lb, hb, tb, cz, lr)
        return ddq

    def jacobian(self, x, y, z, yaw, pitch, roll):
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        lr = self._lr
        J = solve_jacobian(x, y, z, yaw, pitch, roll, ra, da, la, ha, rb, db, lb, hb, tb, cz, lr)
        return J

    def jacobianL(self, x, y, z, yaw, pitch, roll, q):
        return np.eye(6, 6)

    def KL(self, diameter, thickness, length, E):
        A = 0.5 * np.pi * (diameter ** 2 - (diameter - 2 * thickness) ** 2)
        K = E * A / length
        K = np.diag(K)
        return K

    def KxL(self, x, y, z, yaw, pitch, roll, KL):
        q = self.IK(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
        J = self.jacobianL(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, q=q)
        Kx = np.matmul(J.T, KL)
        Kx = np.matmul(Kx, J)
        return Kx

    def jointTrajectory(self, x, y, z, yaw, pitch, roll):
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        p_Oa = solve_pOa(x, y, z, yaw, pitch, roll, ra, da, la, ha)
        q = self.IK(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
        p_Ob = solve_pOb(q, ha, rb, db, lb, hb, tb, cz)
        return p_Oa, p_Ob, q

    def rodRelativePosition(self, x, y, z, yaw, pitch, roll):
        ra = self._ra
        da = self._da
        la = self._la
        ha = self._ha
        rb = self._rb
        db = self._db
        lb = self._lb
        hb = self._hb
        tb = self._tb
        cz = self._cz
        q = self.IK(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
        vmab = solve_vmab(x, y, z, yaw, pitch, roll, q, ra, da, la, ha, rb, db, lb, hb, tb, cz)
        voba = solve_voba(x, y, z, yaw, pitch, roll, q, ra, da, la, ha, rb, db, lb, hb, tb, cz)
        return vmab, voba

    def rodRelativeTrajectory(self, x, y, z, yaw, pitch, roll):
        """
        Calculating the envelope cone of trajectories of the pull rod
        """
        p_Oa, p_Ob, _ = self.jointTrajectory(
            x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
        p_ab = p_Ob - p_Oa
        p_ba = -p_ab
        p_ab = np.vstack((p_ab, np.ones_like([p_ab[0]])))
        p_Mab = np.ones_like(p_ab)
        T_MO = np.array([[np.cos(pitch) * np.cos(yaw),
                          np.sin(pitch) * np.sin(roll) * np.cos(yaw) +
                          np.sin(yaw) * np.cos(roll),
                          -np.sin(pitch) * np.cos(roll) * np.cos(yaw) +
                          np.sin(roll) * np.sin(yaw),
                          -x * np.cos(pitch) * np.cos(yaw) + y * (
                                  -np.sin(pitch) * np.sin(roll) * np.cos(yaw) - np.sin(yaw) * np.cos(roll)) + z * (
                                  np.sin(pitch) * np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(yaw))],
                         [-np.sin(yaw) * np.cos(pitch),
                          -np.sin(pitch) * np.sin(roll) * np.sin(yaw) +
                          np.cos(roll) * np.cos(yaw),
                          np.sin(pitch) * np.sin(yaw) * np.cos(roll) +
                          np.sin(roll) * np.cos(yaw),
                          x * np.sin(yaw) * np.cos(pitch) + y * (
                                  np.sin(pitch) * np.sin(roll) * np.sin(yaw) - np.cos(roll) * np.cos(yaw)) + z * (
                                  -np.sin(pitch) * np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw))],
                         [np.sin(pitch), -np.sin(roll) * np.cos(pitch), np.cos(pitch) * np.cos(roll),
                          -x * np.sin(pitch) + y * np.sin(roll) * np.cos(pitch) - z * np.cos(pitch) * np.cos(roll)],
                         [0 + 0 * x, 0 + 0 * x, 0 + 0 * x, 1 + 0 * x]])
        for i in range(len(p_ab[0, 0, :])):
            p_Mab[:, :, i] = np.matmul(T_MO[:, :, i], p_ab[:, :, i])
        p_Mab = p_Mab[0:3]
        return p_Mab, p_ba

    def coneFinder(self, xa, ya, za):
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
        x0 = np.array([0.05, 0.05, 0.05])
        res = op.minimize(Fobj, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': False},
                          bounds=bounds)
        return res.x

    def nominalJointCone(self, a):
        t1, T1, pos1, vel1, acc1 = self.sinTest_ALL(n=100)
        pos2 = []
        A = 0.13
        for i in [-A, A]:
            for j in [-A, A]:
                for k in [-A, A]:
                    pos2.append([i, j, k, 0, 0, 0])
        B = np.deg2rad(15)
        for i in [-a, a]:
            for j in [-a, a]:
                for k in [-a, a]:
                    for al in [-B, B]:
                        for be in [-B, B]:
                            for ga in [-B, B]:
                                pos2.append([i, j, k, al, be, ga])
        pos2 = np.array(pos2).T
        t2 = t1[-1] + 0.1 * np.arange(1, len(pos2[0]) + 1)
        pos3 = np.hstack((pos1, pos2))
        t3 = np.append(t1, t2)
        vmab, voba = self.rodRelativePosition(
            x=pos3[0], y=pos3[1], z=pos3[2], yaw=pos3[3], pitch=pos3[4], roll=pos3[5])
        cone = np.zeros((2, 6)).tolist()
        for i in range(0, 6):
            cone[0][i] = self.coneFinder(vmab[0, i], vmab[1, i], vmab[2, i])
            cone[1][i] = self.coneFinder(voba[0, i], voba[1, i], voba[2, i])
        return np.array(cone)

    def workspace(self, qmin, qmax, thres_rod, thres_slider):
        return 0

    def rangeCheck(self, q, qmin, qmax):
        q_ = np.abs(q.flatten())
        return q_.min() >= qmin and q_.max() <= qmax

    def interfereMonitor(self, x, y, z, yaw, pitch, roll, thres_rod=0.08, thres_slider=0.1633):
        p_Oa, p_Ob, q = self.jointTrajectory(
            x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
        d = []
        interfere = []
        for i in [0, 2, 4]:
            v1 = (p_Ob[:, i] - p_Oa[:, i]).T
            v2 = (p_Ob[:, i + 1] - p_Oa[:, i + 1]).T
            vab = (p_Ob[:, i] - p_Oa[:, i + 1]).T
            n = np.cross(v1, v2)
            temp = np.array([np.abs(np.dot(vab[i], n[i]) / np.linalg.norm(n[i]))
                             for i in range(len(n))])
            d.append(temp)
            interfere.append(temp > thres_rod)
        return np.array(d), np.array(interfere), q

    def sinTest_x(self, T=2.0, dt=0.01, f=2, A=0.13, C=0.0):
        t = np.arange(0.0, T, dt)
        x = A * np.sin(2 * np.pi * f * t) + C
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        yaw = np.zeros_like(x)
        pitch = np.zeros_like(x)
        roll = np.zeros_like(x)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_y(self, T=2.0, dt=0.01, f=2, A=0.13, C=0.0):
        t = np.arange(0.0, T, dt)
        y = A * np.sin(2 * np.pi * f * t) + C
        x = np.zeros_like(y)
        z = np.zeros_like(y)
        yaw = np.zeros_like(y)
        pitch = np.zeros_like(y)
        roll = np.zeros_like(y)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_z(self, T=2.0, dt=0.01, f=3.47, A=0.13, C=0.0):
        t = np.arange(0.0, T, dt)
        z = A * np.sin(2 * np.pi * f * t) + C
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        yaw = np.zeros_like(z)
        pitch = np.zeros_like(z)
        roll = np.zeros_like(z)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_yaw(self, T=2.0, dt=0.01, f=3.18, A=np.deg2rad(15), C=0.0):
        t = np.arange(0.0, T, dt)
        yaw = A * np.sin(2 * np.pi * f * t) + C
        x = np.zeros_like(yaw)
        y = np.zeros_like(yaw)
        z = np.zeros_like(yaw)
        pitch = np.zeros_like(yaw)
        roll = np.zeros_like(yaw)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_pitch(self, T=2.0, dt=0.01, f=3.18, A=np.deg2rad(15), C=0.0):
        t = np.arange(0.0, T, dt)
        pitch = A * np.sin(2 * np.pi * f * t) + C
        x = np.zeros_like(pitch)
        y = np.zeros_like(pitch)
        z = np.zeros_like(pitch)
        yaw = np.zeros_like(pitch)
        roll = np.zeros_like(pitch)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_roll(self, T=2.0, dt=0.01, f=3.18, A=np.deg2rad(15), C=0.0):
        t = np.arange(0.0, T, dt)
        roll = A * np.sin(2 * np.pi * f * t) + C
        x = np.zeros_like(roll)
        y = np.zeros_like(roll)
        z = np.zeros_like(roll)
        yaw = np.zeros_like(roll)
        pitch = np.zeros_like(roll)
        pos = np.array([x, y, z, yaw, pitch, roll])
        return t, pos

    def sinTest_ALL(self, f=np.array([2, 2, 3.47, 3.18, 2, 3.18, 2, 3.18, 2]), A=np.array(
        [0.13, 0.13, 0.13, np.pi * 5.0 / 180.0, np.pi * 15.0 / 180.0, np.pi * 5.0 / 180.0, np.pi * 15.0 / 180.0,
         np.pi * 5.0 / 180.0, np.pi * 15.0 / 180.0]), C=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), n=100):
        f = f
        A = A
        C = C
        T = 1 / f
        t = np.linspace(0.0, T[0], n)
        pos = np.zeros((9, n))
        vel = np.zeros((9, n))
        acc = np.zeros((9, n))
        pos[0] = A[0] * np.sin(2 * np.pi * f[0] * t) + C[0]
        vel[0] = A[0] * np.cos(2 * np.pi * f[0] * t) * 2 * np.pi * f[0]
        acc[0] = -A[0] * np.sin(2 * np.pi * f[0] * t) * 2 * np.pi * f[0] * 2 * np.pi * f[0]
        for i in range(1, 9):
            t_ = np.linspace(0.0, T[i], n)
            pos_ = np.zeros((9, n))
            vel_ = np.zeros((9, n))
            acc_ = np.zeros((9, n))
            pos_[i] = A[i] * np.sin(2 * np.pi * f[i] * t_) + C[i]
            vel_[i] = A[i] * np.cos(2 * np.pi * f[i] * t_) * 2 * np.pi * f[i]
            acc_[i] = -A[i] * np.sin(2 * np.pi * f[i] * t_) * 2 * np.pi * f[i] * 2 * np.pi * f[i]
            t = np.concatenate((t, t_[1:] + T[:i].sum()))
            pos = np.concatenate((pos, pos_[:, 1:]), 1)
            vel = np.concatenate((vel, vel_[:, 1:]), 1)
            acc = np.concatenate((acc, acc_[:, 1:]), 1)
        for i in [3, 5, 7]:
            pos[i] = pos[i] + pos[i + 1]
            vel[i] = vel[i] + vel[i + 1]
            acc[i] = acc[i] + acc[i + 1]
        pos[4] = pos[5]
        pos[5] = pos[7]
        pos = pos[:6, :]
        vel[4] = vel[5]
        vel[5] = vel[7]
        vel = vel[:6, :]
        acc[4] = acc[5]
        acc[5] = acc[7]
        acc = acc[:6, :]
        for i in range(1, len(T)):
            T[i] = T[i] + T[i - 1]
        return t, T, pos, vel, acc

    # endregion


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    pss = PSSmechV3()
    pss.set_param(ra=0.298151, da=0.752346, la=0.15, ha=0.3, rb=1.063493, db=0.188, lb=0.1736, hb=0.216,
                  tb=np.deg2rad(45), q0=0.509, cz=1.4)
    print(pss.get_param())
    t, T, pos, vel, acc = pss.sinTest_ALL()
    dp = pd.DataFrame(pos.T, index=t, columns=['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
    dp.plot()
    dv = pd.DataFrame(vel.T, index=t, columns=['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
    dv.plot()
    da = pd.DataFrame(acc.T, index=t, columns=['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
    da.plot()
    plt.show()
