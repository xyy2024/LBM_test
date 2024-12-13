#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

try:
    from .lbm_core import Border_BC
except ImportError:
    from lbm_core import Border_BC

__all__ = ["Circulation", "Half_Way_Mirror", "New_Bounce_Back"]

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
        #手动对 newf 中每一个数据进行处理
        #并把处理结果存储到 result 中。
        result = np.zeros(newf.shape)
        if self.dimension == 1:
            #按速度进行处理
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                
                #找出相反的速度
                for j in range(self.c.shape[1]):
                    if round(self.c[0,j]) == -x:
                        break
                
                #如果速度 > 0，左侧的格点可以直接使用，右侧的格点会产生碰撞
                if x > 0:
                    result[x:,i] += newf[:-x,i]
                    for xb in range(-x, 0, 1):
                        result[-x-1-xb, j] += newf[-xb,i]
                elif x < 0:
                    result[:x,i] += newf[-x:,i]
                    for xb in range(0, x-1, 1):
                        result[x-xb, j] += newf[xb, i]
                else:
                    result[:,i] += newf[:,i]
            return result
        raise ValueError(f"维度数据异常：{self.dimension}")

class New_Bounce_Back(Border_BC):
    '''修正反弹格式（二阶精度）
    
    在边界执行碰撞
    边界格点向外的速度方向完全反向
    移动'''
    def move(self, newf:np.ndarray) -> np.ndarray:
        #手动对 newf 中每一个数据进行处理
        #并把处理结果存储到 result 中。
        result = np.zeros(newf.shape)
        if self.dimension == 1:
            #按速度进行处理
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])

                #如果速度为零，所有格点都能直接使用
                if x == 0:
                    result[:,i] += newf[:,i]
                    continue

                #找出相反的速度
                for j in range(self.c.shape[1]):
                    if round(self.c[0,j]) == -x:
                        break

                #如果速度 > 0，左侧的格点可以直接使用
                # 右侧的格点会与边界碰撞
                if x > 0:
                    result[x:,i] += newf[:-x,i]
                    #手动处理碰撞
                    # xb -> result
                    # -1 -> 向右0步，向左x步 -> -x-1
                    # -2 -> 向右1步，向左x-1步 -> -x
                    # -3 -> 向右2步，向左x-2步 -> -x+1
                    # result = -x-2-xb
                    for xb in range(-x, 0, 1):
                        result[-x-2-xb, j] += newf[xb, i]
                
                #类似处理
                elif x < 0:
                    result[:x,i] += newf[-x:,i]
                    #手动处理碰撞
                    # xb -> result
                    # 0 -> 向左0步，向右-x步 -> -x
                    # 1 -> 向左1步，向右-x-1步 -> -x-1
                    # 2 -> 向左2步，向右-x-2步 -> -x-2
                    # result = -x-xb
                    for xb in range(0, x-1, 1):
                        result[-x-xb, j] += newf[xb, i]
            return result
        if self.dimension == 2:
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                if x == 0 and y == 0:
                    result[:,:,i] += newf[:,:,i]
                    continue
                for j in range(self.c.shape[1]):
                    if round(self.c[0,j]) == -x and round(self.c[1,j]) == -y:
                        break
                
                #分开处理两个分量
                temp = np.zeros(newf.shape[:-1]+(2,))

                if x == 0:
                    temp[:, :, 0] += newf[:, :, i]
                elif x > 0:
                    temp[x:, :, 0] += newf[:-x, :, i]
                    for xb in range(-x, 0, 1):
                        temp[-x-2-xb, :, 1] += newf[xb, :, i]
                elif x < 0:
                    temp[:x, :, 0] += newf[-x:, :,i]
                    for xb in range(0, -x-1, 1):
                        temp[-x-xb, :, 1] += newf[xb, :, i]

                if y == 0:
                    result[:,:,i] += temp[:,:,0]
                    result[:,:,j] += temp[:,:,1]
                if y > 0:
                    result[:, y:, i] += temp[:, :-y, 0]
                    result[:, y:, j] += temp[:, :-y, 1]
                    for yb in range(-y, 0, 1):
                        #TODO 对于两个方向速度分量均>1的，可能碰撞两次
                        result[:, -y-2-yb, j] += temp[:, -yb, 0]
                        result[:, -y-2-yb, j] += temp[:, -yb, 1]
                elif y < 0:
                    result[:, :y, i] += temp[:, -y:, 0]
                    result[:, :y, j] += temp[:, -y:, 1]
                    for yb in range(-y, 0, 1):
                        result[:, -y-yb, j] += temp[:, -yb, 0]
                        result[:, -y-yb, j] += temp[:, -yb, 1]
            return result
        if self.dimension == 3: raise ValueError("3维尚未支持")
        raise ValueError(f"维度数据异常：{self.dimension}")

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
            #自由移动
            for i in range(self.c.shape[1]):
                x = round(self.c[0,i])
                y = round(self.c[1,i])
                movef[maxmove+x:-maxmove+x, maxmove+y:-maxmove+y, i] = newf[:,:,i]
            #对折
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
