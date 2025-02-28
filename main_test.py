#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as poly

_k = 2
_b = -0.5*math.pi

def u0(x, y):          #速度初值
    return (
        -math.cos(_k*x+_b)*math.sin(_k*y+_b),
         math.sin(_k*x+_b)*math.cos(_k*y+_b))

_kt = -2*_k**2

def u(x, y, time, nu): #解析解的速度
    that_time_thing = math.exp(_kt*nu*time)
    ux0, uy0 = u0(x, y)
    return ux0*that_time_thing, uy0*that_time_thing

def u_fromsolver(x, y, time, solver):
    return u(x, y, time, solver.nu)

def Taylor_Green_compare_Irregular_Domain(solver):
    x = np.arange(solver.shape[0], dtype=float)*solver.dx
    y = np.arange(solver.shape[1], dtype=float)*solver.dx
    x,y = np.meshgrid(x,y)

    solution_speed = solver.speed
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    s1 = np.sqrt(solution_speed[:,:,0]**2+solution_speed[:,:,1]**2)
    lines1 = ax1.streamplot(x,y,solution_speed[:,:,0].T,solution_speed[:,:,1].T,color=s1.T,cmap="cool")
    ax1.set(xlabel=f"Numerical Solution: dx={solver.dx}",ylabel=f"time = {solver.time}")
    fig.colorbar(lines1.lines)
    ax1.contour(contour_x, contour_y, contour_z, levels=[0], colors="black")

    dx = solver.dx

    scale = math.exp(_kt*solver.nu*solver.time)
    #scale = math.exp(-solver.time*solver.nu*8*math.pi**2)
    precise_solution = np.array([[
        (u0(i*dx,j*dx) if (P(i*dx, j*dx) <= 0) else (0, 0))
        for j in range(solver.shape[1])] for i in range(solver.shape[0])])*scale
    s2 = np.sqrt(precise_solution[:,:,0]**2+precise_solution[:,:,1]**2)
    lines2 = ax2.streamplot(x,y,precise_solution[:,:,0].T,precise_solution[:,:,1].T,color=s2.T,cmap="cool")
    ax2.set(xlabel=f"Precise Solution",ylabel="")
    fig.colorbar(lines2.lines)
    ax2.contour(contour_x, contour_y, contour_z, levels=[0], colors="black")

    error = solution_speed - precise_solution

    s3 = np.sqrt(error[:,:,0]**2+error[:,:,1]**2)
    max_err = (np.abs(s3)).max()
    lines3 = ax3.streamplot(x,y,error[:,:,0].T,error[:,:,1].T,color=s3.T,cmap="cool")
    ax3.set(xlabel=f"Error: Max={max_err}",ylabel="")
    fig.colorbar(lines3.lines)
    ax3.contour(contour_x, contour_y, contour_z, levels=[0], colors="black")

    plt.show()
    return max_err

#空间集合为 P(x,y) <= 0
def P(x, y):
    return 122 + x*(-81 + x*(49 + x*(-12 + x))) + y*(-137 + y*(84 + y*(-22 + y*2)))

contour_x = np.linspace(0, 6, 1001)
contour_y = np.linspace(0, 6, 1001)
contour_x, contour_y = np.meshgrid(contour_x, contour_y)
contour_z = P(contour_x, contour_y)

def Taylor_Green_init_Irregular_Domain(cls, dx=1/32, nu=1/6):
    dt = dx**2
    c_ = (dt/dx)**2
    #x_i = (i+0.5)*dx
    #y_j = (j+0.5)*dx
    n = round(6/dx) + 1   #空间为[0,6]*[0,6]集合的子集
    speed = np.array([[
        u0(i*dx, j*dx)
        for j in range(n)] for i in range(n)])
    density = np.array([[
        1 - 3/4*c_*(math.cos(2*_k*i*dx) + math.cos(2*_k*j*dx))
        for j in range(n)] for i in range(n)])
    for x in range(n):
        for y in range(n):
            speed[x,y,:]*=density[x,y] #Change speed to solution_speed
    solver = cls(
        density = density,
        momentum = speed,
        dx=dx, dt=dt, nu=nu, p=P, u=u_fromsolver)
    return solver

def Taylor_Green_ranktest_Irregular_Domain(cls, nu=1/6):
    dxarray = [1/4, 1/6, 1/8, 1/12, 1/16]
    error = []

    for dx in dxarray:
        solver = Taylor_Green_init_Irregular_Domain(cls, dx=dx, nu=nu)
        #solver.iter(1)
        solver.iter(round(1/dx)**2//4)
        error.append(Taylor_Green_compare_Irregular_Domain(solver))

    dxarray= np.array(dxarray, dtype=float)
    error = np.array(error)

    x = np.log(dxarray)
    y = np.log(error)

    fig, ax = plt.subplots()
    dots = ax.scatter(x, y)

    p2 = poly.fit(x, y, deg=(0, 1))
    line2 = ax.plot(x, p2(x), label=f"{p2}")

    print("Rank:", (rank:=p2(1)-p2(0)))
    plt.show()
    return rank

if __name__ == "__main__":
    pass
