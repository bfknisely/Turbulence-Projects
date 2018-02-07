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

# %% Initialize uniform grid

Nx = 21  # number of nodes in x-direction
Ny = 11  # number of nodes in y-direction

# Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
x = np.linspace(0, 1, Nx)

# Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
y = np.linspace(0, 1, Ny)

# Calculate spacings Δx and Δy; these are constant with the uniform grid
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initialize U-array: 2D array of zeros of dimension [Nx columns] by [Ny rows]
u = np.zeros([Ny, Nx])

# %% Apply boundary conditions to U matrix

# Set u(x, 0) to be 0 from Dirichlet boundary condition
u[0, :] = 0  # Set 0th row, all columns to be 0

# Set u(x, 1) to be 1 from Dirichlet boundary condition
u[-1, :] = 1  # Set last row, all columns to be 1

# Set u(0, y) to be y from Dirichlet boundary condition
u[:, 0] = y  # Set all rows, 0th column to be y


# %% Set up matrix for internal nodes using Crank-Nicolson algorithmS

# Initialize matrix A for equation [[A]]*[u] = [b]
A = np.zeros([Ny - 2, Ny - 2])
# Y-dimension reduced by two because u(x, 0) and u(x, 1) are known already

# Use 2nd-Order-Accurate scheme at j = Ny - 1 and j = 1
# at j = 1
A[0, 0] = 1
A[0, 1] = 1

# at j = Ny - 1
A[-1, -1] = -1
A[-1, -2] = -2


print(A)

# Use 4th-Order-Accurate scheme for j = 2 to j = Ny - 2


# %% Use LU Decomposition matrix solver (Thomas algorithm)


# %% Display results spatially

plt.figure(figsize=(6, 4))
plt.contourf(x, y, u, cmap='plasma')
cbar = plt.colorbar()
xlab = 'x'
ylab = 'y'
fs = 17  # Define font size for figures
fn = 'Calibri'  # Define font for figures
figFileName = 'fig1.png'
plt.xlabel(xlab, fontsize=fs, fontname=fn, fontweight='bold')
plt.ylabel(ylab + '     ', fontsize=fs, rotation=0, fontname=fn,
           fontweight='bold')
plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
cbar.ax.set_ylabel('    u', rotation=0, fontname=fn, fontsize=fs,
                   weight='bold')
cbar.ax.set_yticklabels(
        [round(cbar.get_ticks()[n], 2) for n in range(len(cbar.get_ticks()))],
        fontsize=fs-2, fontname=fn, weight='bold')
plt.close()
