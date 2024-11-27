#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numbers import Number

try:
    from .lbm_core import LBMCore, Constant
except ImportError:
    from lbm_core import LBMCore, Constant

__all__ = ["D2_BGK", "D2_ELBM"]

class D2(LBMCore):
    def __init__(self, f:np.ndarray, dx:Number, dt:Number, nu:Number, **kwargs):
        LBMCore.__init__(self, f=f, dx=dx, dt=dt)
        self.nu = nu

    @classmethod
    def initial(cls, density:np.ndarray, momentum:np.ndarray, dx:Number, dt:Number, **kwargs):
        cs_square_negative = 3*dt**2/dx**2
        cs_square_negative_square = cs_square_negative**2
        div_2density = 0.5/density
        momentum_square_thing = (momentum*momentum).sum(2)*density*cs_square_negative/2
        feq = np.array([[
            [
                cls.w[i]
                *(
                    density[x,y]
                    +(dot_product:=np.dot(cls.c[:,i],momentum[x,y,:]))*cs_square_negative
                    +dot_product**2*cs_square_negative_square*div_2density[x,y]
                    -momentum_square_thing[x,y]
                )
            for i in range(cls.w.shape[0])]
            for y in range(density.shape[1])] for x in range(density.shape[0])])
        return cls(f=feq, dx=dx, dt=dt, **kwargs)

class D2_BGK(D2):
    def getrelax(self) -> np.ndarray:
        return Constant(self.nu)

    def getfeq(self) -> np.ndarray:
        #求出密度和动量，为求feq做准备
        density = self.getdensity()
        momentum = self.getmomentum()

        #求feq
        cs_square_negative = 3*self.dt**2/self.dx**2
        cs_square_negative_square = cs_square_negative**2
        div_2density = 0.5/density
        momentum_square_thing = (momentum*momentum).sum(2)*density*cs_square_negative/2
        feq = np.array([[
            [
                self.w[i]
                *(
                    density[x,y]
                    +(dot_product:=np.dot(self.c[:,i],momentum[x,y,:]))*cs_square_negative
                    +dot_product**2*cs_square_negative_square*div_2density[x,y]
                    -momentum_square_thing[x,y]
                )
            for i in range(self.w.shape[0])]
            for y in range(density.shape[1])] for x in range(density.shape[0])])
        return feq

def H(grid): return (grid[grid > 0]*np.log(grid[grid > 0])).sum(0)

class D2_ELBM(D2_BGK):
    def __init__(self, f:np.ndarray, dx:Number, dt:Number, nu:Number, max_deltaH:Number=1e-12, **kwargs):
        LBMCore.__init__(self, f=f, dx=dx, dt=dt)
        self.nu = nu
        self.max_deltaH = max_deltaH

    @property
    def beta(self): return self.dx/(6*self.nu*self.dt*self.dt+self.dx)

    def getrelax(self) -> np.ndarray:
        f   = self.f
        feq = self.getfeq()

        def getalpha(x,y):
            index = f[x,y,:]>feq[x,y,:]
            if index.any():
                alpha = (f[x,y,:][index]/(f[x,y,:][index]-feq[x,y,:][index])).min(0)
                f[x,y],feq[x,y]
                if alpha > 2: alpha = 2
                
                hf = H(f[x,y,:]); hfeq = H(feq[x,y,:])
                hfalp = H(f[x,y,:]+alpha*(feq[x,y,:]-f[x,y,:]))
                while 0 < hf-hfalp < hf*self.max_deltaH:
                    alpha = alpha - (hfalp-hf)/(hfalp-hfeq)*(1-alpha)
                    hfalp = H(f[x,y,:]+alpha*(feq[x,y,:]-f[x,y,:]))
                return alpha
            else:
                return 0
        
        beta = self.beta
        alphabeta = np.array([[
            getalpha(x,y)*beta
            for y in range(self.shape[1])] for x in range(self.shape[0])])
        return alphabeta

