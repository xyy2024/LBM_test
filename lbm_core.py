#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

from typing import Self, Callable
from numbers import Number

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec as gs

__all__ = ["LATEST", "LBMCore", "Border_BC", "Collision_BC", "Constant"]

LATEST = -1

class LBMCore:
    '''LBM核心
    在此基础上，需要定义：

    w np.ndarray 速度权重序列
    示例： np.array([w_0, w_1, ...])

    c np.ndarray 速度序列
    示例： np.array([[x_0, x_1, ...],
                     [y_0, y_1, ...],
                     [z_0, z_1, ...]])
    该序列各项只代表方向，因此在一般情况下均为整数。
    
    move Callable[[Self], None] 移动方法
    这一项与边界条件有关。

    getfeq Callable[[Self], np.ndarray] 平衡状态
    如果自定义了 collision 方法，可以不设置此项。

    getrelax Callable[[Self], np.ndarray] 松弛系数
    如果自定义了 collision 方法，可以不设置此项。
    '''
    w:np.ndarray
    c:np.ndarray

    getfeq:Callable[[Self], np.ndarray]
    getrelax:Callable[[Self], np.ndarray]
    move:Callable[[Self, np.ndarray], np.ndarray]

    def __init__(self, f:np.ndarray, dx:Number, dt:Number, **kwargs):
        self.history = [f]
        self.shape = f.shape[:-1]
        try:
            self.dimension
        except AttributeError:
            self.dimension = len(self.shape)
        self.dx = dx
        self.dt = dt

    @property
    def f(self) -> np.ndarray:
        '''获得当前计算出的系统状态'''
        return self.history[-1]
    
    @property
    def cs(self) -> Number:
        '''声速'''
        return self.dx/self.dt/math.sqrt(3)

    def getdensity(self, time=LATEST) -> np.ndarray:
        '''获得系统各格点的密度'''
        return self.history[time].sum(len(self.f.shape)-1)

    def getmomentum(self, time=LATEST) -> np.ndarray:
        '''获得系统各格点的动量'''
        if self.dimension == 1:
            return np.array([
                [(self.history[time][x,:]*self.c[0,:].sum(0))]
                for x in range(self.shape[0])])
        if self.dimension == 2:
            return np.array([[
                [(self.history[time][x,y,:]*self.c[0,:]).sum(0),
                 (self.history[time][x,y,:]*self.c[1,:]).sum(0)]
                for y in range(self.shape[1])]
                for x in range(self.shape[0])])
        if self.dimension == 3:
            return np.array([[[
                [(self.history[time][x,y,z,:]*self.c[0,:]).sum(0),
                 (self.history[time][x,y,z,:]*self.c[1,:]).sum(0),
                 (self.history[time][x,y,z,:]*self.c[2,:]).sum(0),]
                for z in range(self.shape[2])]
                for y in range(self.shape[1])]
                for x in range(self.shape[0])])

    def iter(self, times:int=1, showlog:bool=True, nohistory:bool=True) -> None:
        '''默认的迭代步骤'''
        if nohistory:
            self.history = [self.f]
        for time in range(times):
            if showlog: print(f"正在进行第{time+1}次迭代...",end="")

            newf = self.collision()
            newf = self.move(newf)

            if nohistory: self.history[-1] = newf
            else:         self.history.append(newf)
            if showlog: print("完成")

    def show_density(self, time=LATEST) -> None:
        '''显示密度图'''
        if self.dimension == 2:
            density = self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            y = np.linspace(0,self.dx*(self.shape[1]-1),num=self.shape[1])
            y,x = np.meshgrid(y,x)
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            surf = ax.plot_surface(x, y, density, cmap=cm.coolwarm)
            ax.set(xlabel="x",ylabel="y")
            fig.colorbar(surf)
            plt.show()
            return None
        if self.dimension == 1:
            density = self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            fig, ax = plt.subplots()
            line = ax.plot(x, density)
            ax.set(xlabel="x",ylabel="density")
            plt.show()
            return None
        if self.dimension == 3:
            raise ValueError("三维密度图尚未支持")

    def show_flow(self, time=LATEST) -> None:
        '''显示流体流向'''
        if self.dimension == 2:
            density = self.getdensity(time)
            momentum= self.getmomentum(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            y = np.linspace(0,self.dx*(self.shape[1]-1),num=self.shape[1])
            x,y = np.meshgrid(x,y)
            u = momentum[:,:,0]/density
            v = momentum[:,:,1]/density
            fig, ax = plt.subplots()
            s = np.sqrt(u**2+v**2)
            lines = ax.streamplot(x,y,u,v,color=s,cmap="cool")
            ax.set(xlabel="x",ylabel="y")
            fig.colorbar(lines.lines)
            plt.show()
            return None
        if self.dimension == 1:
            velocity = self.getmomentum(time)[:,0]/self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            fig, ax = plt.subplots()
            line = ax.plot(x, velocity)
            ax.set(xlabel="x",ylabel="velocity")
            plt.show()
            return None
        if self.dimension == 3:
            raise ValueError("三维流图尚未支持")

    def add_density(self, fig_ax=None, time=LATEST) -> None:
        '''增加密度图'''
        if self.dimension == 2:
            if fig_ax == None:
                fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            else:
                fig, ax = fig_ax
            density = self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            y = np.linspace(0,self.dx*(self.shape[1]-1),num=self.shape[1])
            y,x = np.meshgrid(y,x)
            fig, ax = fig_ax
            surf = ax.plot_surface(x, y, density, cmap=cm.coolwarm)
            ax.set(xlabel="x",ylabel="y")
            fig.colorbar(surf)
            return None
        if self.dimension == 1:
            if fig_ax == None:
                fig, ax = plt.subplots()
            else:
                fig, ax = fig_ax
            density = self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            line = ax.plot(x, density)
            ax.set(xlabel="x",ylabel="density")
            return None
        if self.dimension == 3:
            raise ValueError("三维密度图尚未支持")

    def add_flow(self, fig_ax=None, time=LATEST) -> None:
        '''显示流体流向'''
        if fig_ax == None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if self.dimension == 2:
            density = self.getdensity(time)
            momentum= self.getmomentum(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            y = np.linspace(0,self.dx*(self.shape[1]-1),num=self.shape[1])
            x,y = np.meshgrid(x,y)
            u = momentum[:,:,0]/density
            v = momentum[:,:,1]/density
            s = np.sqrt(u**2+v**2)
            lines = ax.streamplot(x,y,u,v,color=s,cmap="cool")
            ax.set(xlabel="x",ylabel="y")
            fig.colorbar(lines.lines)
            return None
        if self.dimension == 1:
            velocity = self.getmomentum(time)[:,0]/self.getdensity(time)
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            line = ax.plot(x, velocity)
            ax.set(xlabel="x",ylabel="velocity")
            return None
        if self.dimension == 3:
            raise ValueError("三维流图尚未支持")

    def collision(self) -> np.ndarray:
        '''不同格点有不同松弛参数的碰撞'''
        f   = self.f
        feq = self.getfeq()
        step= feq - f
        rlx = self.getrelax()
        
        for x in range(self.shape[0]):
            if self.dimension > 1:
                for y in range(self.shape[1]):
                    if self.dimension > 2:
                        for z in range(self.shape[2]): step[x,y,z,:] *= rlx[x,y,z]
                    else: step[x,y,:] *= rlx[x,y]
            else: step[x,:] *= rlx[x]
        return f + step

class Border_BC:
    '''LBM 基类。用于 VS Code 识别自动补全。
    边界条件的基类'''
    c:np.ndarray
    dimension:int
    def move(self, newf:np.ndarray) -> np.ndarray: pass

class Collision_BC:
    '''LBM 基类。用于 VS Code 识别自动补全。
    碰撞条件的基类'''
    def getfeq(self) -> np.ndarray: pass

    def getrelax(self) -> np.ndarray: pass

    def collision(self) -> np.ndarray: pass

class Constant:
    '''当松弛量和位置无关时，可以使用这个'''
    def __init__(self, value):
        self.value = value
    def __getitem__(self, index):
        return self.value
    def __getattr__(self, index):
        return self.value
    