#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

try:
    from .lbm_core import Border_BC
except ImportError:
    from lbm_core import Border_BC

__all__ = ["Circulation", "Half_Way_Mirror"]

class Circulation(Border_BC):
    '''周期格式'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        if self.dimension == 1:
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                if x < 0:
                    for _ in range(0,x,-1):
                        temp = newf[0,i]
                        newf[:-1,i] = newf[1:,i]
                        newf[-1,i] = temp
                else:
                    for _ in range(0,x,1):
                        temp = newf[-1,i]
                        newf[1:,i] = newf[:-1,i]
                        newf[0,i] = temp
            return newf
        if self.dimension == 2:
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                if x < 0:
                    for _ in range(0,x,-1):
                        temp = newf[0,:,i].copy()
                        newf[:-1,:,i] = newf[1:,:,i]
                        newf[-1,:,i] = temp
                else:
                    for _ in range(0,x,1):
                        temp = newf[-1,:,i].copy()
                        newf[1:,:,i] = newf[:-1,:,i]
                        newf[0,:,i] = temp
                if y < 0:
                    for _ in range(0,y,-1):
                        temp = newf[:,0,i].copy()
                        newf[:,:-1,i] = newf[:,1:,i]
                        newf[:,-1,i] = temp
                else:
                    for _ in range(0,y,1):
                        temp = newf[:,-1,i].copy()
                        newf[:,1:,i] = newf[:,:-1,i]
                        newf[:,0,i] = temp
            return newf
        if self.dimension == 3:
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                z = round(self.c[2,i])
                if x < 0:
                    for _ in range(0,x,-1):
                        temp = newf[0,:,:,i].copy()
                        newf[:-1,:,:,i] = newf[1:,:,:,i]
                        newf[-1,:,:,i] = temp
                else:
                    for _ in range(0,x,1):
                        temp = newf[-1,:,:,i].copy()
                        newf[1:,:,:,i] = newf[:-1,:,:,i]
                        newf[0,:,:,i] = temp
                if y < 0:
                    for _ in range(0,y,-1):
                        temp = newf[:,0,:,i].copy()
                        newf[:,:-1,:,i] = newf[:,1:,:,i]
                        newf[:,-1,:,i] = temp
                else:
                    for _ in range(0,y,1):
                        temp = newf[:,-1,:,i].copy()
                        newf[:,1:,:,i] = newf[:,:-1,:,i]
                        newf[:,0,:,i] = temp
                if z < 0:
                    for _ in range(0,z,-1):
                        temp = newf[:,:,0,i].copy()
                        newf[:,:,:-1,i] = newf[:,:,1:,i]
                        newf[:,:,-1,i] = temp
                else:
                    for _ in range(0,z,1):
                        temp = newf[:,:,-1,i].copy()
                        newf[:,:,1:,i] = newf[:,:,:-1,i]
                        newf[:,:,0,i] = temp
            return newf
        raise ValueError(f"维度数据异常：{self.dimension}")

class Bounce_Back(Border_BC):
    '''经典反弹格式（一阶精度）

    不在边界执行碰撞
    边界格点向外的速度方向完全反向
    移动'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        raise SyntaxError("尚未完成") #TODO

class New_Bounce_Back(Border_BC):
    '''修正反弹格式（二阶精度）
    
    在边界执行碰撞
    边界格点向外的速度方向完全反向
    移动'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        raise SyntaxError("尚未完成") #TODO

class Half_Way(Border_BC):
    '''修正反弹格式（二阶精度）
    
    在边界执行碰撞
    向内移动
    边界格点向外的速度方向完全反向
    向外不移动，相当于把边界设置在距离边界格点 dx/2 的位置
    '''
    def move(self, newf:np.ndarray) -> np.ndarray:
        raise SyntaxError("尚未完成") #TODO

class Mirror(Border_BC):
    '''镜面反弹格式
    适用于无摩擦损失的光滑表面
    
    不在边界执行碰撞
    边界格点向外的速度方向镜面反向
    移动'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        raise SyntaxError("尚未完成") #TODO

class New_Mirror(Border_BC):
    '''修正镜面反弹格式

    在边界执行碰撞
    边界格点向外的速度方向镜面反向
    移动'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        raise SyntaxError("尚未完成") #TODO

class Half_Way_Mirror(Border_BC):
    '''Half-Way 镜面反射格式'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        maxmove = round(self.c.max())+1
        if self.dimension == 1:
            movef = np.zeros(
                (newf.shape[0]+2*maxmove, 
                 newf.shape[1]))
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                movef[maxmove+x:-maxmove+x,i] = newf[:,i]
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == -x:
                        movef[maxmove: 2*maxmove,j] += movef[maxmove-1::-1,i]
                        movef[-2*maxmove:-maxmove,j] += movef[-1:-maxmove-1:-1,i]
                        break
            newf[:,:] = movef[maxmove:-maxmove,:]
            return newf
        if self.dimension == 2:
            movef = np.zeros(
                (newf.shape[0]+2*maxmove, 
                 newf.shape[1]+2*maxmove, 
                 newf.shape[2]))
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                movef[maxmove+x:-maxmove+x, maxmove+y:-maxmove+y, i] = newf[:,:,i]
            for i in range(self.c.shape[1]):
                x = self.c[0,i]
                y = self.c[1,i]
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == -x and self.c[1,j] == y:
                        movef[maxmove: 2*maxmove,:,i] += movef[maxmove-1::-1,:,j]
                        movef[-maxmove-1:-2*maxmove-1:-1,:,i] += movef[-maxmove::1,:,j]
                        break
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == x and self.c[1,j] == -y:
                        movef[:,maxmove: 2*maxmove,i] += movef[:,maxmove-1::-1,j]
                        movef[:,-maxmove-1:-2*maxmove-1:-1,i] += movef[:,-maxmove::1,j]
                        break
            newf[:,:,:] = movef[maxmove:-maxmove,maxmove:-maxmove,:]
            return newf
        if self.dimension == 3:
            movef = np.zeros(
                (newf.shape[0]+2*maxmove, 
                 newf.shape[1]+2*maxmove, 
                 newf.shape[2]+2*maxmove, 
                 newf.shape[3]))
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                z = round(self.c[2,i])
                movef[maxmove+x:-maxmove+x,
                      maxmove+y:-maxmove+y,
                      maxmove+z:-maxmove+z,
                      i] = newf[:,:,i]
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == -x and self.c[1,j] == y and self.c[2,j] == z:
                        movef[maxmove: 2*maxmove,:,:,j] += movef[maxmove-1::-1,:,:,i]
                        movef[-2*maxmove:-maxmove,:,:,j] += movef[-1:-maxmove-1:-1,:,:,i]
                        break
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == x and self.c[1,j] == -y and self.c[2,j] == z:
                        movef[:,maxmove: 2*maxmove,:,j] += movef[:,maxmove-1::-1,:,i]
                        movef[:,-2*maxmove:-maxmove,:,j] += movef[:,-1:-maxmove-1:-1,:,i]
                        break
                for j in range(self.c.shape[1]):
                    if self.c[0,j] == x and self.c[1,j] == y and self.c[2,j] == -z:
                        movef[:,:,maxmove: 2*maxmove,j] += movef[:,:,maxmove-1::-1,i]
                        movef[:,:,-2*maxmove:-maxmove,j] += movef[:,:,-1:-maxmove-1:-1,i]
                        break
            newf[:,:,:] = movef[maxmove:-maxmove,maxmove:-maxmove,:]
            return newf
        raise ValueError(f"维度数据异常：{self.dimension}")
