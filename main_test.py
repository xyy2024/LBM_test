#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

try:
    from .lbm_border import Circulation, Half_Way_Mirror
    from .lbm_d2q9 import D2Q9_ELBM, D2Q9_LBM_BGK
except ImportError:
    from lbm_border import Circulation, Half_Way_Mirror
    from lbm_d2q9 import D2Q9_ELBM, D2Q9_LBM_BGK

class elbmc(D2Q9_ELBM, Circulation): pass

class elbmhwm(D2Q9_ELBM, Half_Way_Mirror): pass

class lbmc(D2Q9_LBM_BGK, Circulation): pass

class lbmhwm(D2Q9_LBM_BGK, Circulation): pass

if __name__ == "__main__":
    '''
    n = 256
    solver = d2q9_elbm.initial(
        np.ones((n,n)),
        np.array([[[
            0.004*math.tanh(80*((y-n//4)/n if y < n//2 else (n-n//4-y)/n)),
            0.002*math.sin(math.pi*(x/n+0.5))
        ] for y in range(n)] for x in range(n)]), 
        dx=1/n, 
        dt=1/256, 
        max_deltaH=1e-13, 
        nu=1e-5)
    solver.iter(256) #'''
    #print(solver.getmomentum())

    #'''
    n = 64
    m = n
    t = 64
    k = 2*math.pi/n
    solver = elbmc.initial(
        np.ones((n,n)),
        np.array([[[
            -math.cos(k*x)*math.sin(k*y),
            math.sin(k*x)*math.cos(k*y)
        ] for y in range(n)] for x in range(n)]),
        dx=1/n, dt=1/m**2, max_deltaH=1e-13, nu=1e-5)
    solver.iter(times = t, showlog = True, nohistory = True) #'''

    solver.show_flow()
    solver.show_density()