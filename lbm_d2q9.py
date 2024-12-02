#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

try:
    from .lbm_border import Circulation, Half_Way_Mirror
    from .lbm_d2 import D2_BGK, D2_ELBM, D2_Mixed
except ImportError:
    from lbm_border import Circulation, Half_Way_Mirror
    from lbm_d2 import D2_BGK, D2_ELBM, D2_Mixed


__all__ = ["D2Q9_ELBM", "D2Q9_LBM_BGK", "D2Q9_Mixed"]

class D2Q9:
    '''D2Q9 方形格点'''
    w = np.array([4/9]+[1/9]*4+[1/36]*4)
    c = np.array([[0,1,0,-1,0,1,-1,-1,1],
                  [0,0,1,0,-1,1,1,-1,-1]], dtype=float)
    
    dimension = 2

    __init__ = D2_ELBM.__init__

class D2Q9_ELBM(D2Q9, D2_ELBM): pass

class D2Q9_LBM_BGK(D2Q9, D2_BGK): pass

class D2Q9_Mixed(D2Q9, D2_Mixed): pass