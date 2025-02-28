#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

#D2Q9 速度范围
E_RANGE = [0, 1, -1]
E = np.array([[[Ex, Ey] for Ey in E_RANGE] for Ex in E_RANGE])

#D2Q9 权重项
corner, edge, center = 1/36, 1/9, 4/9
W = np.array([[
    (edge if (Wtype:=abs(Ex)+abs(Ey)) == 1 else 
    (corner if Wtype == 2 else center))
    for Ey in E_RANGE]
    for Ex in E_RANGE])
del corner, edge, center

#D2Q9 平衡分布
#该方法的速度被归一化了（即 c=1 ）
def feqx0(density, momentum_devidebyC) -> np.ndarray:
    '''返回对单一采样点的 Maxwell Boltzmann 分布'''
    div_2density = 0.5/density
    momentum_square = (momentum_devidebyC*momentum_devidebyC).sum()
    dot_product3 = (momentum_devidebyC[0]*E[:,:,0]+momentum_devidebyC[1]*E[:,:,1])*3

    feq = W*(density + dot_product3*(1 + dot_product3*div_2density) - momentum_square*div_2density*3)
    return feq

class D2Q9Space:
    dimension = 2
    def __init__(self, density:np.ndarray, momentum:np.ndarray, dx:float, dt:float, nu:float, **kwargs):
        '''初始化 D2Q9 LBM 解算器'''
        self.base = [0, 0]
        self.dx = dx                  #空间步长
        self.dt = dt                  #时间步长
        self.nu = nu                  #流体黏性
        self.shape = density.shape    #空间形状
        self.c = dx/dt                #模型粒子速度
        self.cs = self.c/math.sqrt(3) #模型声速
        self.iter_count = 0           #迭代次数
        
        #从流体黏性 nu 计算数值弛豫时间 tau
        # tau = dt / tau_0
        # tau_0 = 实际弛豫时间

        #计算过程使用了公式
        # nu = c_s**2 * (tau - 0.5) * dt
        # nu = dx**2 / dt / 3 * (tau - 0.5)
        # 3*nu / dx / c = tau - 0.5
        self.tau = 3*self.nu/self.dx/self.c+0.5

        #松弛参数是 tau 的倒数
        self.reciprocal_tau = 1/self.tau

        #将密度和动量输入到模型中，进行初始化
        self.init_grids(density, momentum)

        #初始化迁移步骤参数
        self.grid_setting = [[
            None
            for j in range(self.shape[1])]
            for i in range(self.shape[0])]
        self.init_border(**kwargs)
    
    def init_grids(self, density:np.ndarray, momentum:np.ndarray):
        '''用平衡分布作为初始状态'''
        self.f = np.array([[
            feqx(solver = self, density = density[i, j], momentum = momentum[i, j, :])
                for j in range(self.shape[1])]
                for i in range(self.shape[0])])
        self.feq = self.f
    
    def init_border(self, **kwargs):
        #默认使用循环边值条件
        #格点设置的格式参见 iter_once 中判断特殊格点的方式
        #以及 iter_for_grid_with_special_setting 中解包设置的方式

        #对于迁移步骤而言，f(x[i], y[j], t+dt) 在迁移前对应的点是 f(x[i-ex], y[j-ey], t)
        imax = self.shape[0]-1
        jmax = self.shape[1]-1
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i == 0 or i == imax or j == 0 or j == jmax:
                    self.grid_setting[i][j] = [[
                        [(imax if i-ex < 0 else (0 if i-ex > imax else i-ex)),
                         (jmax if j-ey < 0 else (0 if j-ey > jmax else j-ey))]
                    for ey in E_RANGE]
                    for ex in E_RANGE]

    def getdensity(self) -> np.ndarray:
        '''获得各个采样点的密度信息
        返回：一个二维的 np.ndarray'''
        return np.array([[
            self.f[i,j,:,:].sum()
            for j in range(self.shape[1])] for i in range(self.shape[0])])
    
    @property
    def density(self): return self.getdensity()

    def getmomentum_devidebyC(self) -> np.ndarray:
        '''获得各个采样点的动量信息
        返回：一个三维的 np.ndarray'''
        return np.array([[
            [(self.f[i,j,:,:]*E[:,:,0]).sum(),
             (self.f[i,j,:,:]*E[:,:,1]).sum()]
            for j in range(self.shape[1])] for i in range(self.shape[0])])

    def getmomentum(self) -> np.ndarray:
        '''获得各个采样点的动量信息
        返回：一个三维的 np.ndarray'''
        return self.getmomentum_devidebyC()*self.c
    
    @property
    def momentum(self): return self.getmomentum()

    def getspeed(self) -> np.ndarray:
        '''获得各个采样点的速度信息
        返回：一个三维的 np.ndarray'''
        density = self.density
        momentum = self.momentum
        return np.array([[
            [momentum[i,j,0]/density[i,j],
             momentum[i,j,1]/density[i,j]]
            for j in range(self.shape[1])] for i in range(self.shape[0])])

    @property
    def speed(self): return self.getspeed()

    @property
    def time(self): return self.dt*self.iter_count

    def iter_once(self):
        density = self.density
        momentum_devidebyC = self.getmomentum_devidebyC()
        self.feq = np.array([[
            feqx0(density = density[i, j], momentum_devidebyC = momentum_devidebyC[i, j, :])
                for j in range(self.shape[1])]
                for i in range(self.shape[0])])
        nextf = np.zeros(self.f.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                #并非边界点
                if self.grid_setting[i][j] is None:
                    for ex in E_RANGE:
                        for ey in E_RANGE:
                            #对于迁移步骤而言，f(x[i], y[j], t+dt) 在迁移前对应的点是 f(x[i-ex], y[j-ey], t)
                            nextf[i,j,ex,ey] += (
                                (1-self.reciprocal_tau)*self.f[i-ex, j-ey, ex, ey]
                                +  self.reciprocal_tau *self.feq[i-ex, j-ey, ex, ey]
                                )
                #边界点
                else:
                    for ex in E_RANGE:
                        for ey in E_RANGE:
                            #边界点中，向边界外的速度
                            #不依赖于边界条件，可以直接求出
                            if self.grid_setting[i][j][ex][ey] is None:
                                nextf[i,j,ex,ey] += (
                                (1-self.reciprocal_tau)*self.f[i-ex, j-ey, ex, ey]
                                +  self.reciprocal_tau *self.feq[i-ex, j-ey, ex, ey]
                                )
                            #边界点中，向边界内的速度
                            #依赖于边界条件，读取特殊设置
                            else:
                                self.iter_for_grid_with_special_setting(i, j, ex, ey, nextf)
        self.f = nextf
        self.iter_count += 1

    def iter_for_grid_with_special_setting(self, i, j, ex, ey, nextf):
        from_index = self.grid_setting[i][j][ex][ey]
        nextf[i,j,ex,ey] += (
            (1-self.reciprocal_tau)*self.f[from_index[0], from_index[1], ex, ey]
            +  self.reciprocal_tau *self.feq[from_index[0], from_index[1], ex, ey]
        )

    def iter(self, times:int=1):
        for _ in range(times):
            self.iter_once()

    def show_density(self) -> None:
        '''显示密度图'''
        if self.dimension == 2:
            density = self.getdensity()
            x = np.arange(self.shape[0], dtype=float)*self.dx
            y = np.arange(self.shape[1], dtype=float)*self.dx
            y,x = np.meshgrid(y,x)
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            surf = ax.plot_surface(x, y, density, cmap=cm.coolwarm)
            ax.set(xlabel="x",ylabel="y")
            fig.colorbar(surf)
            plt.show()
            return None
        if self.dimension == 1:
            density = self.getdensity()
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            fig, ax = plt.subplots()
            line = ax.plot(x, density)
            ax.set(xlabel="x",ylabel="density")
            plt.show()
            return None
        if self.dimension == 3:
            raise ValueError("三维密度图尚未支持")

    def show_flow(self) -> None:
        '''显示流体流向'''
        if self.dimension == 2:
            density = self.getdensity()
            momentum= self.getmomentum()
            x = np.arange(self.shape[0], dtype=float)*self.dx
            y = np.arange(self.shape[1], dtype=float)*self.dx
            x,y = np.meshgrid(x,y)
            u = momentum[:,:,0]/density
            v = momentum[:,:,1]/density
            fig, ax = plt.subplots()
            s = np.sqrt(u**2+v**2)
            lines = ax.streamplot(x,y,u.T,v.T,color=s.T,cmap="cool")
            ax.set(xlabel=f"time = {self.time}",ylabel="")
            fig.colorbar(lines.lines)
            plt.show()
            return None
        if self.dimension == 1:
            velocity = self.getmomentum()[:,0]/self.getdensity()
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            fig, ax = plt.subplots()
            line = ax.plot(x, velocity)
            ax.set(xlabel="x",ylabel="velocity")
            plt.show()
            return None
        if self.dimension == 3:
            raise ValueError("三维流图尚未支持")

    def add_density(self, fig_ax=None) -> None:
        '''增加密度图'''
        if self.dimension == 2:
            if fig_ax == None:
                fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            else:
                fig, ax = fig_ax
            density = self.getdensity()
            x = np.arange(self.shape[0], dtype=float)*self.dx
            y = np.arange(self.shape[1], dtype=float)*self.dx
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
            density = self.getdensity()
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            line = ax.plot(x, density)
            ax.set(xlabel="x",ylabel="density")
            return None
        if self.dimension == 3:
            raise ValueError("三维密度图尚未支持")

    def add_flow(self, fig_ax=None) -> None:
        '''显示流体流向'''
        if fig_ax == None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if self.dimension == 2:
            density = self.getdensity()
            momentum= self.getmomentum()
            x = np.arange(self.shape[0], dtype=float)*self.dx
            y = np.arange(self.shape[1], dtype=float)*self.dx
            x,y = np.meshgrid(x,y)
            u = momentum[:,:,0]/density
            v = momentum[:,:,1]/density
            s = np.sqrt(u**2+v**2)
            lines = ax.streamplot(x,y,u.T,v.T,color=s.T,cmap="cool")
            ax.set(xlabel=f"time = {self.time}",ylabel="")
            fig.colorbar(lines.lines)
            return None
        if self.dimension == 1:
            velocity = self.getmomentum()[:,0]/self.getdensity()
            x = np.linspace(0,self.dx*(self.shape[0]-1),num=self.shape[0])
            line = ax.plot(x, velocity)
            ax.set(xlabel="x",ylabel="velocity")
            return None
        if self.dimension == 3:
            raise ValueError("三维流图尚未支持")

#平衡分布
#速度未被归一化
def feqx(solver:D2Q9Space, density, momentum) -> np.ndarray:
    '''返回对单一采样点的 Maxwell Boltzmann 分布'''
    return feqx0(density, momentum/solver.c)

from typing import Callable

OUTOFBOUND_VALUE = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0]])

class D2Q9Space_SingleNode_IrregularDomain(D2Q9Space):
    def __init__(self, density:np.ndarray, momentum:np.ndarray, dx:float, dt:float, nu:float,
        p:Callable, u:Callable, **kwargs):
        '''p(x)<0 为求解区域, u 为求边界速度的函数'''
        self.border_position_func = p
        self.border_value_func = u
        self.not_ready_to_show = True
        super().__init__(density, momentum, dx, dt, nu, **kwargs)

    def init_border(self, **kwargs):
        # 需要计算的区域为 p(x,y) < 0

        p = self.border_position_func
        u = self.border_value_func
        #top_left = True #调试用
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                #界外格点
                if p(i*self.dx, j*self.dx) > 0:                    
                    self.grid_setting[i][j] = False
                    self.f[i,j,:,:] *= 0
                    #该格点所有数值都无意义。但为了不引发ZeroDivisionError，密度设为1。
                    self.f[i,j,0,0] = 1
                #边界内格点，需要计算
                else:
                    #默认格点不在边界上
                    at_border = False
                    for ex in E_RANGE:
                        for ey in E_RANGE:
                            #无需通过零速度来计算边界
                            if ex == 0 and ey == 0: continue
                            #对于迁移步骤而言，f(x[i], y[j], t+dt) 在迁移前对应的点是 f(x[i-ex], y[j-ey], t)
                            #如果迁移前对应的点在边界外，则该点为边界格点
                            elif p((i-ex)*self.dx, (j-ey)*self.dx) > 0:
                                if not at_border:
                                    #在初次判定出边界格点时，按照边界格点的默认设置进行设置
                                    at_border = True
                                    self.grid_setting[i][j] = [[
                                        None
                                        for e_y in E_RANGE]
                                        for e_x in E_RANGE]
                                self.init_for_grid_need_special_setting(i, j, ex, ey, p, u)

    def iter_once(self):
        time = self.time
        density = self.density
        momentum = self.momentum
        self.feq = np.array([[
            OUTOFBOUND_VALUE if self.grid_setting[i][j] is False else feqx(solver = self, density = density[i, j], momentum = momentum[i, j, :])
                for j in range(self.shape[1])]
                for i in range(self.shape[0])])
        nextf = np.zeros(self.f.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                #一般点
                if self.grid_setting[i][j] is None:
                    for ex in E_RANGE:
                        for ey in E_RANGE:
                            #对于迁移步骤而言，f(x[i], y[j], t+dt) 在迁移前对应的点是 f(x[i-ex], y[j-ey], t)
                            nextf[i,j,ex,ey] += (
                                (1-self.reciprocal_tau)*self.f[i-ex, j-ey, ex, ey]
                                +  self.reciprocal_tau *self.feq[i-ex, j-ey, ex, ey]
                                )
                #界外点
                elif self.grid_setting[i][j] is False:
                    nextf[i,j,:,:] = OUTOFBOUND_VALUE
                #边界点
                else:
                    for ex in E_RANGE:
                        for ey in E_RANGE:
                            #边界点中，向边界外的速度
                            #不依赖于边界条件，可以直接求出
                            if self.grid_setting[i][j][ex][ey] is None:
                                nextf[i,j,ex,ey] += (
                                    (1-self.reciprocal_tau)*self.f[i-ex, j-ey, ex, ey]
                                    +  self.reciprocal_tau *self.feq[i-ex, j-ey, ex, ey]
                                    )
                            else:                                                             #依赖边界条件的
                                self.iter_for_grid_with_special_setting(i, j, ex, ey, time, density, nextf)
        self.f = nextf
        self.iter_count += 1

    def init_for_grid_need_special_setting(self, i, j, ex, ey, p, u):
        #边界格点需要特别设置

        #在 (x[i-ex], y[j-ey]) 和 (x[i], y[j]) 之间存在边界
        #边界位置位于 (x[i-gamma*ex], y[j-gamma*ey])
        #边界满足    p(x[i-gamma*ex], y[j-gamma*ey])=0
        gamma_func = lambda gamma: p((i-gamma*ex)*self.dx,(j-gamma*ey)*self.dx)
        gamma = dichotomy(gamma_func)
        del gamma_func
    
        #计算各项系数
        a4 = 0
        a1 = 1/(1 + 2*gamma)
        a5 = 2*a1
        a3 = gamma*a5*self.reciprocal_tau
        a2 = gamma*a5-a3
    
        #把结果写入设置，格式为：
        # a[0]: 计算边界速度时使用的函数
        #       函数参数为(边界横坐标，边界纵坐标，时间，self)，返回边界速度(ux, uy)元组
        # a[1] ~ a[5]: SingleNode 那篇论文所给出的系数
        # a[6]: 边界横坐标
        # a[7]: 边界纵坐标
        self.grid_setting[i][j][ex][ey] = [
            u, a1, a2, a3, a4, a5,
            (i-gamma*ex)*self.dx,
            (j-gamma*ey)*self.dx
            ]

    def iter_for_grid_with_special_setting(self, i, j, ex, ey, time, density, nextf):
        #对具有特别设置的格点进行操作
        a = self.grid_setting[i][j][ex][ey]
        ux, uy = a[0](a[6], a[7], time, self)
        # nextf 为 t+dt 时间下的离散分布函数
        nextf[i,j,ex,ey] = (
              a[1]*self.f[i,j,-ex,-ey]    #f_-i
            + a[2]*self.f[i,j, ex, ey]    #f_i
            + a[3]*self.feq[i,j, ex, ey]  #feq_i
            + a[4]*self.feq[i,j,-ex,-ey]  #feq_-i
            + a[5]*W[ex, ey]*density[i,j]*3/self.c
                *(ex*ux+ey*uy)
        )

    #为了在流的图像上显示不规则边界，设置以下方法
    def init_graph_border(self):
        self.contour_x = np.linspace(0, self.dx*self.shape[0], 1001)
        self.contour_y = np.linspace(0, self.dx*self.shape[1], 1001)
        self.contour_x, self.contour_y = np.meshgrid(self.contour_x, self.contour_y)
        self.contour_z = self.border_value_func(self.contour_x, self.contour_y)
        self.not_ready_to_show = False

    def show_flow(self) -> None:
        if self.not_ready_to_show: self.init_graph_border()
        '''显示流体流向'''
        density = self.getdensity()
        momentum= self.getmomentum()
        x = np.arange(self.shape[0], dtype=float)*self.dx
        y = np.arange(self.shape[1], dtype=float)*self.dx
        x,y = np.meshgrid(x,y)
        u = momentum[:,:,0]/density
        v = momentum[:,:,1]/density
        fig, ax = plt.subplots()
        s = np.sqrt(u**2+v**2)
        lines = ax.streamplot(x,y,u.T,v.T,color=s.T,cmap="cool")
        ax.set(xlabel=f"time = {self.time}",ylabel="")
        fig.colorbar(lines.lines)
        ax.contour(self.contour_x, self.contour_y, self.contour_z, levels=[0], colors="black")
        plt.show()
        return None

    def add_flow(self, fig_ax=None) -> None:
        '''显示流体流向'''
        if fig_ax == None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if self.not_ready_to_show: self.init_graph_border()
        density = self.getdensity()
        momentum= self.getmomentum()
        x = np.arange(self.shape[0], dtype=float)*self.dx
        y = np.arange(self.shape[1], dtype=float)*self.dx
        x,y = np.meshgrid(x,y)
        u = momentum[:,:,0]/density
        v = momentum[:,:,1]/density
        s = np.sqrt(u**2+v**2)
        lines = ax.streamplot(x,y,u.T,v.T,color=s.T,cmap="cool")
        ax.set(xlabel=f"time = {self.time}",ylabel="")
        fig.colorbar(lines.lines)
        ax.contour(self.contour_x, self.contour_y, self.contour_z, levels=[0], colors="black")
        return None

def dichotomy(f, xmin=0, xmax=1):
    '''二分法 / dichotomy'''
    if f(xmin) == 0: return xmin
    if f(xmax) == 0: return xmax
    for iter_time in range(1000):
        newx = (xmin+xmax)/2
        if (fnewx:=f(newx)) == 0: return newx
        elif fnewx < 0:           xmin = newx
        else:                     xmax = newx
    return (xmin+xmax)/2