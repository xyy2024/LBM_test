#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

try:
    from .lbm_border import Circulation, Half_Way_Mirror, New_Bounce_Back
    from .lbm_d2q9 import D2Q9_ELBM, D2Q9_LBM_BGK, D2Q9_Mixed
    from .lbm_core import LBMCore
    from .lbm_d2 import D2
except ImportError:
    from lbm_border import Circulation, Half_Way_Mirror, New_Bounce_Back
    from lbm_d2q9 import D2Q9_ELBM, D2Q9_LBM_BGK, D2Q9_Mixed
    from lbm_core import LBMCore
    from lbm_d2 import D2

class elbmc(D2Q9_ELBM, Circulation): pass

class elbmhwm(D2Q9_ELBM, Half_Way_Mirror): pass

class elbmbb(D2Q9_ELBM, New_Bounce_Back): pass

class lbmc(D2Q9_LBM_BGK, Circulation): pass

class lbmhwm(D2Q9_LBM_BGK, Half_Way_Mirror): pass

class lbmbb(D2Q9_LBM_BGK, New_Bounce_Back): pass

class mixhwm(D2Q9_Mixed, Half_Way_Mirror): pass

class mixc(D2Q9_Mixed, Circulation): pass

class mixbb(D2Q9_Mixed, New_Bounce_Back): pass

def _test(solver: D2, dt):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    solver.add_flow(fig_ax=(fig, ax1))
    t1 = int(1/16/dt)
    solver.iter(times = t1, showlog = False, nohistory = True)
    solver.add_flow(fig_ax=(fig, ax2))
    t2 = t1*3
    solver.iter(times = t2, showlog = False, nohistory = True)
    solver.add_flow(fig_ax=(fig, ax3))

    plt.show()

def Taylor_Green_test(cls, n=64):
    m = n
    k = 2*math.pi/n
    ki= math.pi/n
    dt = 1/m**2
    solver = cls.initial(
        np.ones((n,n)), #密度为1
        np.array([[[
            -math.cos(k*x+ki)*math.sin(k*y+ki),
            math.sin(k*x+ki)*math.cos(k*y+ki)
        ] for y in range(n)] for x in range(n)]),
        dx=1/n, dt=dt, max_deltaH=1e-13, nu=1e-5)
    
    _test(solver, dt)

if __name__ == "__main__":
    Taylor_Green_test(lbmhwm)