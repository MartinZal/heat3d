import numpy as np
import copy
import scipy.io as sio




def calcDN(h, e, cp, mi, k, u,v):
    Dh = 4 * h * e / (2 * h + 2 * e)
    Pr = cp * mi / k
    Re = u * Dh / v
    f = (0.79 * np.log(Re) - 1.64)**(-2)

    #Nu1 = 0.023 * Re ** (4 / 5) * Pr ** (0.4)
    Nu2 = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** (1 / 2) * (Pr ** (2 / 3) - 1))

    #alpha1 = Nu1*k/Dh
    alpha2 = Nu2*k/Dh

    return alpha2


def neighFun(a, b, Nz, Nzv):
    neigh = np.zeros([Nz, Nzv])
    for i in range(Nz):
        for j in range(Nzv):
            # b je cele v a
            if (a[1, i] >= b[1, j] and a[0, i] <= b[0, j]):
                neigh[i, j] = b[1, j] - b[0, j]
            # naopak , a cele v b
            elif (a[1, i] <= b[1, j] and a[0, i] >= b[0, j]):
                neigh[i, j] = a[1, i] - a[0, i]
            # a je vys jak b
            elif (a[1, i] >= b[1, j] and a[0, i] <= b[1, j]):
                neigh[i, j] = b[1, j] - a[0, i]
                # b je vys jak a
            elif (a[1, i] <= b[1, j] and a[1, i] >= b[0, j]):
                neigh[i, j] = a[1, i] - b[0, j]
    return neigh


def moveFun(bIn, dzv, dzp, Nzv, d):

    b2 = copy.copy(bIn)
    nout = 0
    for i in range(0, Nzv - 1):
        b2[1, i] = b2[1, i] + dzp
        b2[0, i + 1] = b2[0, i + 1] + dzp
        if (b2[0, i] >= d):
            nout = nout + 1
    b1 = np.zeros([2, Nzv])
    # tvl zkusit pres roll taky, bude o dost lepsi
    for i in range(0, Nzv - 1 - nout):
        b1[1, i + nout] = b2[1, i]
        b1[0, i + 1 + nout] = b2[0, i + 1]

    for i in range(0, nout):
        b1[0, i + 1] = b2[0, 1] + (-nout + i) * dzv
        b1[1, i] = b2[1, 0] + (-nout + i) * dzv

    b1[0, 0] = 0
    b1[-1, -1] = d
    return b1, nout


def moveFun1(bIn, dzv, dzp, Nzv, d):

    b2 = copy.copy(bIn)
    nout = 0
    for i in range(0, Nzv - 1):
        b2[1, i] = b2[1, i] + dzp
        b2[0, i + 1] = b2[0, i + 1] + dzp
        if (b2[0, i] >= d):
            nout = nout + 1
            # nezere to nout
    b1 = copy.copy(b2)
    b2 = np.roll(b1, nout, axis=1)

    for i in range(0, nout):
        b2[0, i + 1] = b1[0, 1] + (-nout + i) * dzv
        b2[1, i] = b1[1, 0] + (-nout + i) * dzv

    b2[0, 0] = 0
    b2[-1, -1] = d
    return b2, nout

def heatIn(tmax,dt,Ny,Nz,T,Nzv,Tvz,dz,alpha,neighField):
    #krelit teplo in v case
    heatSum = np.zeros([200])
    for p in np.arange(0, int(tmax / dt), 1):
        for i in range(0, Ny):
            for j in range(0, Nz):
                heatSum[p] = heatSum[p] + ql(i, j, p, T, Nzv, Tvz, dz, alpha, neighField) + qp(i, j, p, T, Nzv, Tvz, dz, alpha, neighField)
    return heatSum

def heatPoint(tmax,dt,Ny,Nz,T,Nzv,Tvz,dz,alpha,neighField):

    heatSum = np.zeros([Ny,Nz])
    for i in range(0, Ny):
        for j in range(0, Nz):
            for p in np.arange(0, int(tmax / dt), 1):
                heatSum[i,j] = heatSum[i,j] + ql(i, j, p, T, Nzv, Tvz, dz, alpha, neighField) + qp(i, j, p, T, Nzv, Tvz, dz, alpha, neighField)
    return heatSum

def heatInAll(tmax,dt,Ny,Nz,T,Nzv,Tvz,dz,dy,alpha,neighField,h,d):
    heatSum = 0
    for p in np.arange(0, int(tmax / dt), 1):
        for i in range(0, Ny):
            for j in range(0, Nz):
                heatSum = heatSum + (alpha / dz * np.dot(neighField[j, :], (Tvz[i, 0:Nzv, p, 0] - T[0, i, j, p]))+alpha / dz * np.dot(neighField[j, :], (Tvz[i, 0:Nzv, p, 1] - T[-1, i, j, p])))*yout(i,dy,Ny) + 1000*0.91*0.95*h*d
    return heatSum

def qprl(n, q, p, T, Tvz, dy, alpha, neighField):
    return alpha * dy * np.dot(neighField[:, q], (T[0, n, :, p] - Tvz[n, q, p, 0]))


def qprp(n, q, p, T, Tvz, dy, alpha, neighField):
    return alpha * dy * np.dot(neighField[:, q], (T[-1, n, :, p] - Tvz[n, q, p, 1]))


def sout(p):
    s = 1
    if p>1:
        s = 0
    return s

def yout(n,dy,Ny):
    if (n == 0) or (n == Ny-1):
        y1 = dy/2
    else:
        y1 = dy
    return y1



def ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField):
    return alpha / dz * np.dot(neighField[o, :], (Tvz[n, 0:Nzv, p, 0] - T[0, n, o, p])) + 1000*0.91*0.95


def qp(n, o, p, T, Nzv, Tvz, dz, alpha, neighField):
    return alpha / dz * np.dot(neighField[o, :], (Tvz[n, 0:Nzv, p, 1] - T[-1, n, o, p]))





def exportData(Tvz, Tsol):
    sio.savemat('dataTvz.mat', mdict={'Tvz': Tvz})
    sio.savemat('dataT.mat', mdict={'T': Tsol})



def heat(l, h, d, a, b, qh, qd, qf, qb, qloss, rho, rhov, lambd, alpha, c, cv, T0p, Nx, Ny, Nz, Nzv, dzv,
                     dzp, dt, tmax):


    dx = l / (Nx - 1)
    dy = h / (Ny - 1)
    dz = d / (Nz - 1)

    T = np.zeros([Nx, Ny, Nz, int(tmax / dt) + 1])
    Tvz = np.zeros([Ny, 300, int(tmax / dt) + 2, 2])

    T[:, :, :, 0].fill(T0p)
    Tvz[:, :, :, :].fill(50)

    bIt = copy.copy(b)
    neighField = neighFun(a, b, Nz, Nzv)

    for p in np.arange(0, int(tmax / dt), 1):

        # vzduch leva rohy

        Tvz[0, 0, p + 1, 0] = Tvz[0, 0, p, 0] + (2 * dt) / (dx * dy * neighField[0, 0] * cv * rhov) * \
        (qprl(0, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, 0, p + 1, 0] = Tvz[-1, 0, p, 0] + (2 * dt) / (dx * dy * neighField[0, 0] * cv * rhov) * \
        (qprl(-1, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[0, Nzv - 1, p + 1, 0] = Tvz[0, Nzv - 1, p, 0] + (2 * dt) / (dx * dy * neighField[-1, -1] * cv * rhov) \
        * (qprl(0, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, Nzv - 1, p + 1, 0] = Tvz[-1, Nzv - 1, p, 0] + (2 * dt) / (dx * dy * neighField[-1, -1] * cv * rhov) * \
        (qprl(-1, Nzv - 1, p, T, Tvz, dy, alpha,neighField) + qloss)

        # vzduch prava rohy

        Tvz[0, 0, p + 1, 1] = Tvz[0, 0, p, 1] + (2 * dt) / (dx * dy * neighField[0, 0] * cv * rhov) * \
        (qprp(0, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, 0, p + 1, 1] = Tvz[-1, 0, p, 1] + (2 * dt) / (dx * dy * neighField[0, 0] * cv * rhov) * \
        (qprp(-1, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[0, Nzv - 1, p + 1, 1] = Tvz[0, Nzv - 1, p, 1] + (2 * dt) / (dx * dy * neighField[-1, -1] * cv * rhov) \
        *(qprp(0, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        Tvz[-1, Nzv - 1, p + 1, 1] = Tvz[-1, Nzv - 1, p, 1] + (2 * dt) / (dx * dy * neighField[-1, -1] * cv * rhov) * \
        (qprp(-1, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

        T[0, 0, 0, p + 1] = T[0, 0, 0, p] + (2 * dt) / (c(T[0, 0, 0, p]) * rho) * \
        (ql(0, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[0, 0, 0, p]) * rho) * \
        ((T[1, 0, 0, p] - T[0, 0, 0, p]) / dx ** 2 + (T[0, 1, 0, p] - T[0, 0, 0, p]) / dy ** 2 +
        (T[0, 0, 1, p] - T[0, 0, 0, p]) / dz ** 2)

        T[-1, 0, 0, p + 1] = T[-1, 0, 0, p] + (2 * dt) / (c(T[-1, 0, 0, p]) * rho) * (
        qp(0, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qf / dz) + (c(T[-1, 0, 0, p]) * rho) * \
        ((T[-2, 0, 0, p] - T[-1, 0, 0, p]) / dx ** 2 + (T[-1, 1, 0, p] - T[-1, 0, 0, p]) / dy ** 2 +
        (T[-1, 0, 1, p] - T[-1, 0, 0, p]) / dz ** 2)

        T[0, -1, 0, p + 1] = T[0, -1, 0, p] + (2 * dt) / (c(T[0, -1, 0, p]) * rho) * \
        (ql(-1, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[0, -1, 0, p]) * rho) * ((T[1, -1, 0, p] - T[0, -1, 0, p]) / dx ** 2 +
        (T[0, -2, 0, p] - T[0, -1, 0, p]) / dy ** 2 + (T[0, -1, 1, p] - T[0, -1, 0, p]) / dz ** 2)

        T[-1, -1, 0, p + 1] = T[-1, -1, 0, p] + (2 * dt) / (c(T[-1, -1, 0, p]) * rho) * (
        qp(-1, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qf / dz) + \
        (2 * lambd * dt) / (c(T[-1, -1, 0, p]) * rho) * ((T[-2, -1, 0, p] - T[-1, -1, 0, p]) / dx ** 2 +
        (T[-1, -2, 0, p] - T[-1, -1, 0, p]) / dy ** 2 + (T[-1, -1, 1, p] - T[-1, -1, 0, p]) / dz ** 2)

        T[0, 0, -1, p + 1] = T[0, 0, -1, p] + (2 * dt) / (c(T[0, 0, -1, p]) * rho) * \
        (ql(0, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[0, 0, -1, p]) * rho) * ((T[1, 0, -1, p] - T[0, 0, -1, p]) / dx ** 2 +
        (T[0, 1, -1, p] - T[0, 0, -1, p]) / dy ** 2 + (T[0, 0, -2, p] - T[0, 0, -1, p]) / dz ** 2)

        T[-1, 0, -1, p + 1] = T[-1, 0, -1, p] + (2 * dt) / (c(T[-1, 0, -1, p]) * rho) * (
        qp(0, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[-1, 0, -1, p]) * rho) * ((T[-2, 0, -1, p] - T[-1, 0, -1, p]) / dx ** 2 +
        (T[-1, 1, -1, p] - T[-1, 0, -1, p]) / dy ** 2 + (T[-1, 0, -2, p] - T[-1, 0, -1, p]) / dz ** 2)

        T[0, -1, -1, p + 1] = T[0, -1, -1, p] + (2 * dt) / (c(T[0, -1, -1, p]) * rho) * \
        (ql(-1, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[0, -1, -1, p]) * rho) * ((T[1, -1, -1, p] - T[0, -1, -1, p]) / dx ** 2 +
        (T[0, -2, -1, p] - T[0, -1, -1, p]) / dy ** 2 + (T[0, -1, -2, p] - T[0, -1, -1, p]) / dz ** 2)

        T[-1, -1, -1, p + 1] = T[-1, -1, -1, p] + (2 * dt) / (c(T[-1, -1, -1, p]) * rho) * (
        qp(-1, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy + qb / dz) + \
        (2 * lambd * dt) / (c(T[-1, -1, -1, p]) * rho) * (
        (T[-2, -1, -1, p] - T[-1, -1, -1, p]) / dx ** 2 +
        (T[-1, -2, -1, p] - T[-1, -1, -1, p]) / dy ** 2 + (
        T[-1, -1, -2, p] - T[-1, -1, -1, p]) / dz ** 2)

        for m in range(1, Nx - 1):

            T[m, 0, 0, p + 1] = T[m, 0, 0, p] + (2 * dt) / (c(T[m, 0, 0, p]) * rho) * (qh / dy + qf / dz) + \
            (2 * lambd * dt) / (c(T[m, 0, 0, p]) * rho) * (
            (T[m - 1, 0, 0, p] + T[m + 1, 0, 0, p] - 2 * T[m, 0, 0, p]) / (
            dx ** 2 * 2) + (T[m, 1, 0, p] - T[m, 0, 0, p]) / dy ** 2 + (
            T[m, 0, 1, p] - T[m, 0, 0, p]) / dz ** 2)

            T[m, -1, 0, p + 1] = T[m, -1, 0, p] + (2 * dt) / (c(T[m, -1, 0, p]) * rho) * (qd / dy + qf / dz) + \
            (2 * lambd * dt) / (c(T[m, -1, 0, p]) * rho) * (
            (T[m - 1, -1, 0, p] + T[m + 1, -1, 0, p] - 2 * T[m, -1, 0, p]) / (
            dx ** 2 * 2) + (T[m, -2, 0, p] - T[m, -1, 0, p]) / dy ** 2 + (
            T[m, -1, 1, p] - T[m, -1, 0, p]) / dz ** 2)

            T[m, 0, -1, p + 1] = T[m, 0, -1, p] + (2 * dt) / (c(T[m, 0, -1, p]) * rho) * (qh / dy + qb / dz) + \
            (2 * lambd * dt) / (c(T[m, 0, -1, p]) * rho) * ((T[m - 1, 0, -1, p] + T[m + 1, 0, -1, p] -
            2 *T[m, 0, -1, p]) / (dx ** 2 * 2) + (T[m, 1, -1, p] - T[m, 0, -1, p]) / dy ** 2 +
            (T[m, 0, -2, p] - T[m, 0, -1, p]) / dz ** 2)

            T[m, -1, -1, p + 1] = T[m, -1, -1, p] + (2 * dt) / (c(T[m, -1, -1, p]) * rho) * (qd / dy + qb / dz) + \
            (2 * lambd * dt) / (c(T[m, -1, -1, p]) * rho) * ((T[m - 1, -1, -1, p] +
            T[m + 1, -1, -1, p] - 2 *T[m, -1, -1, p]) / (dx ** 2 * 2) +
            (T[m, -2, -1, p] - T[m, -1, -1, p]) / dy ** 2 + (T[m, -1, -2, p] -T[m, -1, -1, p]) / dz ** 2)

            for n in range(1, Ny - 1):

                T[0, n, 0, p + 1] = T[0, n, 0, p] + (2 * dt) / (c(T[0, n, 0, p]) * rho) * \
                                                    (ql(n, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qf / dz) + \
                                    (2 * lambd * dt) / (c(T[0, n, 0, p]) * rho) * (
                                        (T[0, n + 1, 0, p] + T[0, n - 1, 0, p] - 2 * T[0, n, 0, p]) / (
                                            dy ** 2 * 2) + (T[1, n, 0, p] - T[0, n, 0, p]) / dx ** 2 + (
                                            T[0, n, 1, p] - T[0, n, 0, p]) / dz ** 2)

                T[-1, n, 0, p + 1] = T[-1, n, 0, p] + (2 * dt) / (c(T[-1, n, 0, p]) * rho) * (
                qp(n, 0, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qf / dz) + \
                                     (2 * lambd * dt) / (c(T[-1, n, 0, p]) * rho) * (
                                         (T[-1, n + 1, 0, p] + T[-1, n - 1, 0, p] - 2 * T[-1, n, 0, p]) / (
                                             dy ** 2 * 2) + (T[-2, n, 0, p] - T[-1, n, 0, p]) / dx ** 2 + (
                                             T[-1, n, 1, p] - T[-1, n, 0, p]) / dz ** 2)

                T[0, n, -1, p + 1] = T[0, n, -1, p] + (2 * dt) / (c(T[0, n, -1, p]) * rho) * (
                ql(n, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qb / dz) + (2 * lambd * dt) / (
                c(T[0, n, -1, p]) * rho) * ((T[0, n + 1, -1, p] + T[0, n - 1, -1, p] - 2 * T[0, n, -1, p]) / (
                dy ** 2 * 2) + (T[1, n, -1, p] - T[0, n, -1, p]) / dx ** 2 + (
                                            T[0, n, -2, p] - T[0, n, -1, p]) / dz ** 2)

                T[-1, n, -1, p + 1] = T[-1, n, -1, p] + (2 * dt) / (c(T[-1, n, -1, p]) * rho) * (
                qp(n, -1, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qb / dz) + (2 * lambd * dt) / (
                c(T[-1, n, -1, p]) * rho) * ((T[-1, n + 1, -1, p] + T[-1, n - 1, -1, p] - 2 * T[-1, n, -1, p]) / (
                dy ** 2 * 2) + (T[-2, n, -1, p] - T[-1, n, -1, p]) / dx ** 2 + (
                                             T[-1, n, -2, p] - T[-1, n, -1, p]) / dz ** 2)
                T[m, n, 0, p + 1] = T[m, n, 0, p] + (2 * qf * dt) / (rho * c(T[m, n, 0, p]) * dz) + (2 * lambd * dt) / (
                    c(T[m, n, 0, p]) * rho) * ((T[m + 1, n, 0, p] + T[m - 1, n, 0, p] - 2 * T[m, n, 0, p]) / (
                    dx ** 2 * 2) + (T[m, n + 1, 0, p] + T[m, n - 1, 0, p] - 2 * T[m, n, 0, p]) / (
                                                   dy ** 2 * 2) + (
                                                   T[m, n, 1, p] - T[m, n, 0, p]) / dz ** 2)

                T[m, n, -1, p + 1] = T[m, n, -1, p] + (2 * qb * dt) / (rho * c(T[m, n, -1, p]) * dz) + \
                (c(T[m, n, -1, p]) * rho) * ((T[m + 1, n, -1, p] + T[m - 1, n, -1, p] -
                2 * T[m, n, -1, p]) / (dx ** 2 * 2) + (T[m, n + 1, -1, p] + T[m, n - 1, -1, p] -
                2 *T[m, n, -1, p]) / (dy ** 2 * 2) + (T[m, n, -2, p] - T[m, n, -1, p]) / dz ** 2)
                for o in range(1, Nz - 1):
                    T[m, 0, o, p + 1] = T[m, 0, o, p] + (2 * qh * dt) / (rho * c(T[m, 0, o, p]) * dy) + (
                                                                                                            2 * lambd * dt) / (
                                                                                                        c(T[
                                                                                                              m, 0, o, p]) * rho) * (
                                                                                                            (T[
                                                                                                                 m + 1, 0, o, p] +
                                                                                                             T[
                                                                                                                 m - 1, 0, o, p] - 2 *
                                                                                                             T[
                                                                                                                 m, 0, o, p]) / (
                                                                                                                dx ** 2 * 2) + (
                                                                                                            T[
                                                                                                                m, 0, o + 1, p] +
                                                                                                            T[
                                                                                                                m, 0, o - 1, p] - 2 *
                                                                                                            T[
                                                                                                                m, 0, o, p]) / (
                                                                                                            dz ** 2 * 2) + (
                                                                                                                T[
                                                                                                                    m, 1, o, p] -
                                                                                                                T[
                                                                                                                    m, 0, o, p]) / dy ** 2)

                    T[m, -1, o, p + 1] = T[m, -1, o, p] + (2 * qd * dt) / (rho * c(T[m, -1, o, p]) * dy) + (
                                                                                                               2 * lambd * dt) / (
                                                                                                           c(T[
                                                                                                                 m, -1, o, p]) * rho) * (
                                                                                                               (T[
                                                                                                                    m + 1, -1, o, p] +
                                                                                                                T[
                                                                                                                    m - 1, -1, o, p] - 2 *
                                                                                                                T[
                                                                                                                    m, -1, o, p]) / (
                                                                                                                   dx ** 2 * 2) + (
                                                                                                                   T[
                                                                                                                       m, -1, o + 1, p] +
                                                                                                                   T[
                                                                                                                       m, -1, o - 1, p] - 2 *
                                                                                                                   T[
                                                                                                                       m, -1, o, p]) / (
                                                                                                               dz ** 2 * 2) + (
                                                                                                                   T[
                                                                                                                       m, -2, o, p] -
                                                                                                                   T[
                                                                                                                       m, -1, o, p]) / dy ** 2)

                    T[0, n, o, p + 1] = T[0, n, o, p] + (
                                                            2 * ql(n, o, p, T, Nzv, Tvz, dz, alpha,
                                                                   neighField) * dt) / (
                                                            rho * c(T[0, n, o, p]) * dx) + (2 * lambd * dt) / (
                        c(T[0, n, o, p]) * rho) * (
                                                                                               (T[0, n + 1, o, p] + T[
                                                                                                   0, n - 1, o, p] - 2 *
                                                                                                T[0, n, o, p]) / (
                                                                                                   dy ** 2 * 2) + (
                                                                                               T[0, n, o + 1, p] + T[
                                                                                                   0, n, o - 1, p] - 2 *
                                                                                               T[
                                                                                                   0, n, o, p]) / (
                                                                                               dz ** 2 * 2) + (
                                                                                                   T[1, n, o, p] - T[
                                                                                                       0, n, o, p]) / dx ** 2)

                    T[-1, n, o, p + 1] = T[-1, n, o, p] + (2 * qp(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) * dt) / (
                    rho * c(T[-1, n, o, p]) * dx) + (
                                                        2 * lambd * dt) / (c(T[-1, n, o, p]) * rho) * (
                                                        (T[-1, n + 1, o, p] + T[-1, n - 1, o, p] - 2 * T[
                                                            -1, n, o, p]) / (
                                                            dy ** 2 * 2) + (
                                                            T[-1, n, o + 1, p] + T[-1, n, o - 1, p] - 2 * T[
                                                                -1, n, o, p]) / (dz ** 2 * 2) + (
                                                            T[-2, n, o, p] - T[-1, n, o, p]) / dx ** 2)

                    T[0, 0, o, p + 1] = T[0, 0, o, p] + (2 * dt) / (c(T[0, 0, o, p]) * rho) * (
                        ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy) + (2 * lambd * dt) / (
                        c(T[0, 0, o, p]) * rho) * (
                                                                                              (T[0, 0, o + 1, p] + T[
                                                                                                  0, 0, o - 1, p] - 2 *
                                                                                               T[0, 0, o, p]) / (
                                                                                                  dz ** 2 * 2) + (
                                                                                              T[1, 0, o, p] - T[
                                                                                                  0, 0, o, p]) / dx ** 2 + (
                                                                                                  T[0, 1, o, p] - T[
                                                                                                      0, 0, o, p]) / dy ** 2)

                    T[0, -1, o, p + 1] = T[0, -1, o, p] + (2 * dt) / (c(T[0, -1, o, p]) * rho) * (
                        ql(n, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy) + (2 * lambd * dt) / (
                        c(T[0, -1, o, p]) * rho) * (
                                                                                              (T[0, -1, o + 1, p] + T[
                                                                                                  0, -1, o - 1, p] - 2 *
                                                                                               T[0, -1, o, p]) / (
                                                                                                  dz ** 2 * 2) + (
                                                                                              T[1, -1, o, p] - T[
                                                                                                  0, -1, o, p]) / dx ** 2 + (
                                                                                                  T[0, -2, o, p] - T[
                                                                                                      0, -1, o, p]) / dy ** 2)

                    T[-1, 0, o, p + 1] = T[-1, 0, o, p] + (2 * dt) / (c(T[-1, 0, o, p]) * rho) * (
                    qp(0, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qh / dy) + (
                                                                                          2 * lambd * dt) / (
                                                                                      c(T[-1, 0, o, p]) * rho) * (
                                                                                          (T[-1, 0, o + 1, p] + T[
                                                                                              -1, 0, o - 1, p] - 2 * T[
                                                                                               -1, 0, o, p]) / (
                                                                                              dz ** 2 * 2) + (
                                                                                          T[-2, 0, o, p] - T[
                                                                                              -1, 0, o, p]) / dx ** 2 + (
                                                                                              T[-1, 1, o, p] - T[
                                                                                                  -1, 0, o, p]) / dy ** 2)

                    T[-1, -1, o, p + 1] = T[-1, -1, o, p] + (2 * dt) / (c(T[-1, -1, o, p]) * rho) * (
                        qp(-1, o, p, T, Nzv, Tvz, dz, alpha, neighField) / dx + qd / dy) + (2 * lambd * dt) / (
                    c(T[-1, -1, o, p]) * rho) * \
                                                                                           ((T[-1, -1, o + 1, p] + T[
                                                                                               -1, -1, o - 1, p] - 2 *
                                                                                             T[-1, -1, o, p]) /
                                                                                            (dz ** 2 * 2) + (
                                                                                            T[-2, -1, o, p] - T[
                                                                                                -1, -1, o, p]) / dx ** 2 + (
                                                                                                T[-1, -2, o, p] - T[
                                                                                                    -1, -1, o, p]) / dy ** 2)

                    T[m, n, o, p + 1] = T[m, n, o, p] + (lambd * dt) / (rho * dx ** 2 * c(T[m, n, o, p])) * (
                        T[m + 1, n, o, p] - 2 * T[m, n, o, p] + T[m - 1, n, o, p]) + (lambd * dt) / (
                        rho * dy ** 2 * c(T[m, n, o, p])) * (
                                                                                         T[m, n + 1, o, p] - 2 * T[
                                                                                             m, n, o, p] + T[
                                                                                             m, n - 1, o, p]) + (
                                                                                                                    lambd * dt) / (
                                                                                                                rho * dz ** 2 * c(
                                                                                                                    T[
                                                                                                                        m, n, o, p])) * (
                                                                                                                    T[
                                                                                                                        m, n, o + 1, p] - 2 *
                                                                                                                    T[
                                                                                                                        m, n, o, p] +
                                                                                                                    T[
                                                                                                                        m, n, o - 1, p])

        for n in range(1, Ny - 1):

            Tvz[n, 0, p + 1, 0] = Tvz[n, 0, p, 0] + dt / (dx * dy * neighField[0, 0] * cv * rhov) * (
                qprl(n, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, Nzv - 1, p + 1, 0] = Tvz[n, Nzv - 1, p, 0] + dt / (dx * dy * neighField[-1, -1] * cv * rhov) * (
                qprl(n, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, 0, p + 1, 1] = Tvz[n, 0, p, 1] + dt / (dx * dy * neighField[0, 0] * cv * rhov) * (
                qprp(n, 0, p, T, Tvz, dy, alpha, neighField) + qloss)

            Tvz[n, Nzv - 1, p + 1, 1] = Tvz[n, Nzv - 1, p, 1] + dt / (dx * dy * neighField[-1, -1] * cv * rhov) * (
                qprp(n, Nzv - 1, p, T, Tvz, dy, alpha, neighField) + qloss)

            for q in range(1, Nzv - 1):
                Tvz[n, q, p + 1, 0] = Tvz[n, q, p, 0] + dt / (dx * dy * dzv * cv * rhov) * (
                    qprl(n, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[0, q, p + 1, 0] = Tvz[0, q, p, 0] + (2 * dt) / (dx * dy * dzv * cv * rhov) * (
                    qprl(0, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[-1, q, p + 1, 0] = Tvz[-1, q, p, 0] + (2 * dt) / (dx * dy * dzv * cv * rhov) * (
                    qprl(-1, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[n, q, p + 1, 1] = Tvz[n, q, p, 1] + dt / (dx * dy * dzv * cv * rhov) * (
                    qprp(n, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[0, q, p + 1, 1] = Tvz[0, q, p, 1] + (2 * dt) / (dx * dy * dzv * cv * rhov) * (
                    qprp(0, q, p, T, Tvz, dy, alpha, neighField) + qloss)

                Tvz[-1, q, p + 1, 1] = Tvz[-1, q, p, 1] + (2 * dt) / (dx * dy * dzv * cv * rhov) * (
                    qprp(-1, q, p, T, Tvz, dy, alpha, neighField) + qloss)


        bN, nout = moveFun1(bIt, dzv, dzp, Nzv, d)
        bIt = bN
        Tvz[:, :, p + 1] = np.roll(Tvz[:, :, p + 1], nout, axis=1)
        Tvz[:, :, p + 2] = np.roll(Tvz[:, :, p + 1], nout, axis=1)
        print(p*dt)

    heatP = heatPoint(tmax, dt, Ny, Nz, T, Nzv, Tvz, dz, alpha, neighField)


    sio.savemat('heatp.mat', mdict={'hp': heatP})

    return T,Tvz

