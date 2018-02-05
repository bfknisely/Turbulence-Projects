# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29, 2018

@author: Brian Knisely

The purpose of this code is to use finite difference to solve an elliptic
partial differential equation.

The PDE is: ∂u/∂x - ∂^2(u)/∂y^2 = y

Subject to the following boundary conditions (BCs):
    u(x, 0) = 0
    u(x, 1) = 1
    u(0, y) = y

The Crank-Nicolson Algorithm is to be used to march in the x-direction, using
a uniform grid and central differencing scheme that is fourth-order in y.
"""

# %% Import packages for arrays and plotting
import numpy as np
import matplotlib.pyplot as plt

# %% Initialize grid

Nx = 21  # number of nodes in x-direction
Ny = 21  # number of nodes in y-direction

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)

u = np.zeros([Ny, Nx])

xx, yy = np.meshgrid(x, y, sparse=True)
z = xx + yy

plt.contourf(x,y,z)
