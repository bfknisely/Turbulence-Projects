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
from numpy.linalg import inv
import matplotlib.pyplot as plt

# %% Initialize uniform grid

Nx = 21  # number of nodes in x-direction
Ny = 21  # number of nodes in y-direction

# Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
x = np.linspace(0, 1, Nx)

# Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
y = np.linspace(0, 1, Ny)

# Calculate spacings Δx and Δy; these are constant with the uniform grid
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initialize U-array: 2D array of zeros of dimension [Nx columns] by [Ny rows]
U = np.zeros([Ny, Nx])
# capital U is a 2D array of velocities at all spatial locations

# %% Apply boundary conditions to U matrix

# Set u(x, 0) to be 0 from Dirichlet boundary condition
U[0, :] = 0  # Set 0th row, all columns to be 0

# Set u(x, 1) to be 1 from Dirichlet boundary condition
U[-1, :] = 1  # Set last row, all columns to be 1

# Set u(0, y) to be y from Dirichlet boundary condition
U[:, 0] = y  # Set all rows, 0th column to be y



for i in range(len(x)-1):

    # %% Set up matrix and RHS "b" matrix (knowns) for Crank-Nicolson algorithm
    # Initialize matrix A for equation [[A]]*[u] = [b]
    A = np.zeros([Ny - 2, Ny - 2])
    b = np.zeros([Ny - 2, 1])
    # Y-dimension reduced by two because u(x, 0) and u(x, 1) are known already

    # # # Use 2nd-Order-Accurate scheme for first interior nodes

    # at j = 1 (near bottom border of domain)
    A[0, 0] = 1/dx + 1/dy**2  # Assign value in matrix location [1, 1]
    A[0, 1] = -1/(2*dy**2)  # Assign value in matrix location [1, 2]

    # Assign right-hand side of equation (known values) for 0th element in b
    b[0] = (y[1] + 1/(2*dy**2)*U[0, i] + (-1/dy**2 + 1/dx)*U[1, i]
            + 1/(2*dy**2)*U[2, i] + 1/(2*dy**2)*U[0, i+1])

    # at j = Ny-2 (near top border of domain)
    A[-1, -1] = 1/dx + 1/dy**2  # Assign value to last diagonal element
    A[-1, -2] = -1/(2*dy**2)  # Assign value to left of last diagonal element

    # Assign right-hand side of equation (known values) for last element in b
    b[-1] = (y[-2] + 1/(2*dy**2)*U[-3, i] + (-1/dy**2 + 1/dx)*U[-2, i]
             + 1/(2*dy**2)*U[-1, i] + 1/(2*dy**2)*U[-1, i+1])

    # # # Use 4th-Order-Accurate scheme for j = 2 to j = Ny-3

    # j = 2
    A[1, 0] = -2/(3*dy**2)
    A[1, 1] = 5/(4*dy**2) + 1/dx
    A[1, 2] = -2/(3*dy**2)
    A[1, 3] = 1/(24*dy**2)

    b[1] = (y[2] + -1/(24*dy**2)*U[0, i] + 2/(3*dy**2)*U[1, i]
            + (1/dx-5/(4*dy**2))*U[2, i] + 2/(3*dy**2)*U[3, i]
            + (-1/(24*dy**2))*U[4, i] + (-1/(24*dy**2))*U[0, i+1])

    A[-2, -1] = -2/(3*dy**2)
    A[-2, -2] = 5/(4*dy**2) + 1/dx
    A[-2, -3] = -2/(3*dy**2)
    A[-2, -4] = 1/(24*dy**2)

    b[-2] = (y[-3] + -1/(24*dy**2)*U[-5, i] + 2/(3*dy**2)*U[-4, i]
             + (1/dx-5/(4*dy**2))*U[-3, i] + 2/(3*dy**2)*U[-2, i]
             + (-1/(24*dy**2))*U[-1, i] + (-1/(24*dy**2))*U[-1, i+1])

    for j in range(2, Ny-4):
        A[j, j-2] = 1/(24*dy**2)
        A[j, j-1] = -2/(3*dy**2)
        A[j, j] = 5/(4*dy**2) + 1/dx
        A[j, j+1] = -2/(3*dy**2)
        A[j, j+2] = 1/(24*dy**2)
        b[j] = (y[j+1] + -1/(24*dy**2)*U[j-1, i] + 2/(3*dy**2)*U[j, i]
                + (1/dx-5/(4*dy**2))*U[j+1, i] + 2/(3*dy**2)*U[j+2, i]
                + (-1/(24*dy**2))*U[j+3, i])

#    print(A[0:5, 0:5])
#    print(A[-5:, -5:])
#    print(b)

    # %% Use LU Decomposition matrix solver (Thomas algorithm)

    # skip this for now, use built-in matrix solver to test
    U[1:-1, i+1] = (inv(A)@b).transpose()

# %% Display results spatially

# Create contour plot of U vs x and y
plt.figure(figsize=(6, 4))
plt.contourf(x, y, U, cmap='plasma')
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

# Look at solution at various x-locations
