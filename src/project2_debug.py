# -*- coding: utf-8 -*-
"""
This code is used to debug the main function in project2.py

Created on Thu Mar 15 19:06:50 2018

@author: Brian

The resulting nondimensional equations are:
    du/dx + dv/dy = 0
    u du/dx + v du/dy = 1/RD d/dy(nu du/dy)

The boundary conditions (BCs) in nondimensional form are:
    u(x, 0) = 0  (no-slip condition)
    u(x, yMax) = 1  (velocity is equal to freestream at top edge of domain)
    u(0, y <= 1) = sin(pi*y/2)  (starting profile)
    u(0, y > 1) = 1  (starting profile)
    v(x, 0) = 0  (impermeable wall condition)
    v(x, yMax) = 0  (freestream is purely in x-direction, no y-component)
"""

# Import packages for arrays, plotting, timing, and file I/O
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import sin, pi


def pentaLU(A, b):  # Define LU Decomposition function to solve A*x = b for x
    """
    Function to solve A*x = b for a given a pentadiagonal 2D array A and right-
    hand side 1D array, b. Dims of A must be at least 5 x 5.
    """

    Ny = np.shape(A)[0] + 2  # Extract dims of input matrix A
    yVec = np.zeros([1, Ny - 2])  # initialize y vector for LU decomposition
    xVec = np.zeros([1, Ny - 2])  # initialize x vector for LU decomposition

    lo = np.eye(np.shape(A)[0])  # initialize lower array with identity
    up = np.zeros([np.shape(A)[0], np.shape(A)[0]])  # initialize upper

    # Assign values at beginning of matrix
    # 0th row
    up[0, 0:3] = A[0, 0:3]
    # 1st row
    lo[1, 0] = A[1, 0]/up[0, 0]
    up[1, 3] = A[1, 3]
    up[1, 2] = A[1, 2] - lo[1, 0]*up[0, 2]
    up[1, 1] = A[1, 1] - lo[1, 0]*up[0, 1]

    # Assign values in middle of matrix
    for n in range(2, np.shape(A)[0]-2):
        up[n, n+2] = A[n, n+2]
        lo[n, n-2] = A[n, n-2]/up[n-2, n-2]
        lo[n, n-1] = (A[n, n-1] - lo[n, n-2]*up[n-2, n-1])/up[n-1, n-1]
        up[n, n+1] = A[n, n+1] - lo[n, n-1]*up[n-1, n+1]
        up[n, n] = A[n, n] - lo[n, n-2]*up[n-2, n] - lo[n, n-1]*up[n-1, n]

    # Assign values at end of matrix
    # Second-to-last row
    lo[-2, -4] = A[-2, -4]/up[-4, -4]
    lo[-2, -3] = (A[-2, -3] - lo[-2, -4]*up[-4, -3])/up[-3, -3]
    up[-2, -1] = A[-2, -1] - lo[-2, -3]*up[-3, -1]
    up[-2, -2] = A[-2, -2] - lo[-2, -4]*up[-4, -2] - lo[-2, -3]*up[-3, -2]
    # Last row
    lo[-1, -3] = A[-1, -3]/up[-3, -3]
    lo[-1, -2] = (A[-1, -2] - lo[-1, -3]*up[-3, -2])/up[-2, -2]
    up[-1, -1] = A[-1, -1] - lo[-1, -3]*up[-3, -1] - lo[-1, -2]*up[-2, -1]
    # LU Decomposition is complete at this point

    # Now solve for y vector in lo*yVec = b by forward substitution
    yVec[0, 0] = b[0]  # Calculate 0th element of y vector
    yVec[0, 1] = b[1] - lo[1, 0]*yVec[0, 0]  # Compute 1st element of y vector
    for n in range(2, np.shape(yVec)[1]):
        # Compute nth value of y vector
        yVec[0, n] = b[n] - (lo[n, n-2]*yVec[0, n-2] + lo[n, n-1]*yVec[0, n-1])

    # Solve for x vector in up*xVec = yVec
    xVec[0, -1] = 1/up[-1, -1] * yVec[0, -1]  # Calculate last element of xVec
    # Calculate second-to-last element of x vector
    xVec[0, -2] = 1/up[-2, -2] * (yVec[0, -2] - up[-2, -1]*xVec[0, -1])
    for n in np.arange(np.shape(xVec)[1]-3, -1, -1):
        # Step backwards from end to beginning to compute x vector
        xVec[0, n] = 1/up[n, n] * (yVec[0, n] - (up[n, n+1]*xVec[0, n+1] +
                                                 up[n, n+2]*xVec[0, n+2]))
    # Output value of x vector from function
    return xVec


# def main(Nx, Ny, method):  # Define main function to set up grid and A matrix
#                and march in x-direction using Crank-Nicolson algorithm
#       Inputs:  Nx = number of nodes in x-direction
#                Ny = number of nodes in y-direction
#                method = "lu" or "inv" for matrix inversion
Nx = 41
Ny = 41
method = 'lu'

yMax = 5

# Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
x = np.linspace(0, 1, Nx)

# Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
y = np.linspace(0, yMax, Ny)

# Calculate spacings Δx and Δy; these are constant with the uniform grid
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initialize u- and v-arrays: 2D arrays of zeros
# of dimension [Nx columns] by [Ny rows]
u = np.zeros([Ny, Nx])
v = np.zeros([Ny, Nx])
# u is a 2D array of nondimensional x-velocities at all spatial locations
# u is a 2D array of nondimensional y-velocities at all spatial locations

# Apply boundary conditions to u matrix

# Set u(x, 0) to be 0 from no-slip boundary condition
u[0, :] = 0  # Set 0th row, all columns to be 0

# Set u(x, 1) to be 1 from Dirichlet boundary condition
u[-1, :] = 1  # Set last row, all columns to be 1

# Set u(0, y) to starting profile
for n in range(len(y)):
    # u(0, y <= 1) = sin(pi*y/2)
    if y[n] <= 1:
        u[n, 0] = sin(pi*y[n]/2)
    # u(0, y > 1) = 1
    elif y[n] > 1:
        u[n, 0] = 1

# Apply boundary conditions to v matrix

# Set v(x, 0) to be 0 from impermeable wall condition
v[0, :] = 0

# Set v(x, yMax) to be 0 due to upper freestream condition
v[-1, :] = 0


for i in range(len(x)-1):

    # Set up matrix and RHS "b" matrix (knowns)
    # for Crank-Nicolson algorithm

    # Initialize matrix A for equation [[A]]*[u] = [b]
    A = np.zeros([Ny - 2, Ny - 2])
    b = np.zeros([Ny - 2, 1])
    # Y-dimension reduced by two because u(x, 0) and u(x, 1)
    # are known already

    # # # Use 2nd-Order-Accurate scheme for first interior nodes

    # at j = 1 (near bottom border of domain)
    A[0, 0] = 1/dx + 1/dy**2  # Assign value in matrix location [1, 1]
    A[0, 1] = -1/(2*dy**2)  # Assign value in matrix location [1, 2]

    # Assign right-hand side of equation (known values) for 0th value in b
    b[0] = (y[1] + 1/(2*dy**2)*u[0, i] + (-1/dy**2 + 1/dx)*u[1, i]
            + 1/(2*dy**2)*u[2, i] + 1/(2*dy**2)*u[0, i+1])

    # at j = Ny-2 (near top border of domain)
    A[-1, -1] = 1/dx + 1/dy**2  # Assign value to last diagonal element
    A[-1, -2] = -1/(2*dy**2)  # Assign value to left of last diag element

    # Assign right-hand side of equation (known values) for last value in b
    b[-1] = (y[-2] + 1/(2*dy**2)*u[-3, i] + (-1/dy**2 + 1/dx)*u[-2, i]
             + 1/(2*dy**2)*u[-1, i] + 1/(2*dy**2)*u[-1, i+1])

    # # # Use 4th-Order-Accurate scheme for j = 2 to j = Ny-3
    # Store coefficients for second internal node
    A[1, 0] = -2/(3*dy**2)
    A[1, 1] = 5/(4*dy**2) + 1/dx
    A[1, 2] = -2/(3*dy**2)
    A[1, 3] = 1/(24*dy**2)
    b[1] = (y[2] + -1/(24*dy**2)*u[0, i] + 2/(3*dy**2)*u[1, i]
            + (1/dx-5/(4*dy**2))*u[2, i] + 2/(3*dy**2)*u[3, i]
            + (-1/(24*dy**2))*u[4, i] + (-1/(24*dy**2))*u[0, i+1])

    # Store coefficients for second-from-last internal node
    A[-2, -1] = -2/(3*dy**2)
    A[-2, -2] = 5/(4*dy**2) + 1/dx
    A[-2, -3] = -2/(3*dy**2)
    A[-2, -4] = 1/(24*dy**2)
    b[-2] = (y[-3] + -1/(24*dy**2)*u[-5, i] + 2/(3*dy**2)*u[-4, i]
             + (1/dx-5/(4*dy**2))*u[-3, i] + 2/(3*dy**2)*u[-2, i]
             + (-1/(24*dy**2))*u[-1, i] + (-1/(24*dy**2))*u[-1, i+1])

    # Loop over internal nodes to compute and store coefficients
    for j in range(2, Ny-4):
        A[j, j-2] = 1/(24*dy**2)
        A[j, j-1] = -2/(3*dy**2)
        A[j, j] = 5/(4*dy**2) + 1/dx
        A[j, j+1] = -2/(3*dy**2)
        A[j, j+2] = 1/(24*dy**2)
        b[j] = (y[j+1] + -1/(24*dy**2)*u[j-1, i] + 2/(3*dy**2)*u[j, i]
                + (1/dx-5/(4*dy**2))*u[j+1, i] + 2/(3*dy**2)*u[j+2, i]
                + (-1/(24*dy**2))*u[j+3, i])

    if method == 'lu':  # if input was for LU decomposition
        u[1:-1, i+1] = pentaLU(A, b)  # call the pentaLU solver
    if method == 'inv':  # if input was for built-in inv (for testing)
        u[1:-1, i+1] = (inv(A)@b).transpose()  # solve by inverting matrix

# output is the u-matrix
# return u

# Plot results
plt.figure(figsize=(6, 4))
plt.contourf(x, y, u, cmap='plasma', levels=np.linspace(0., np.amax(u), 11))
cbar = plt.colorbar()
plt.title('u')

plt.figure(figsize=(6, 4))
plt.contourf(x, y, v, cmap='plasma', levels=np.linspace(0., 1, 11))
cbar = plt.colorbar()
plt.title('v')
