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
from math import sin, pi, sinh, cosh, asinh


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
#                s = stretching factor
Nx = 41
Ny = 151
method = 'lu'
s = 5

# Define given dimensional quantities
nuInfDim = 1.5e-6  # Dimensional freestream viscosity in m^2/s
uInfDim = 40  # Dimensional freestream velocity in m/s
LDim = 0.5  # Length of plate in m
deltaDim = 0.005  # Initial BL thickness in m

# Calculate derived nondimensional quantity
RD = uInfDim*deltaDim**2/(LDim*nuInfDim)
nu = 1  # Assume viscosity in BL is equal to freestream viscosity

# Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
x = np.linspace(0, 1, Nx)

# Make linear-spaced 1D array of eta-values from 0 to 1 with Ny elements
eta = np.linspace(0, 1, Ny)

# Define bounds of computational domain
xMax = 40
yMax = 10


# %% Compute values of y based on eta and ymax
y = [yMax*sinh(s*et)/sinh(s) for et in eta]

# Evaluate values of fPrime and fDoublePrime
fPrime = [s*yMax*cosh(s*et)/sinh(s) for et in eta]
fDoublePrime = [s**2*yMax*sinh(s*et)/sinh(s) for et in eta]
# Determine etaY and etaYY
etaY = [1/fP for fP in fPrime]
etaYY = [-fDoublePrime[n]/(fPrime[n]**3) for n in range(len(fPrime))]

# Calculate spacings dx and de; these are constant with the uniform x-eta grid
dx = x[1] - x[0]
de = eta[1] - eta[0]

# Initialize u- and v-arrays: 2D arrays of zeros
# of dimension [Nx columns] by [Ny rows]
u = np.zeros([Ny, Nx])
v = np.zeros([Ny, Nx])
# u is a 2D array of nondimensional x-velocities at all spatial locations
# v is a 2D array of nondimensional y-velocities at all spatial locations

# %% Apply boundary conditions to u matrix

# Set u(x, 0) to be 0 from no-slip boundary condition
u[0, :] = 0  # Set 0th row, all columns to be 0

# Set u(x, 1) to be 1 from Dirichlet boundary condition
u[-1, :] = 1  # Set last row, all columns to be 1

# Set u(0, eta) to starting profile
etaY1 = asinh(1*sinh(s)/yMax)/s  # Determine eta value corresponding to y = 1
for n in range(len(y)):
    # u(0, y <= 1) = sin(pi*y/2)
    if eta[n] <= etaY1:
        u[n, 0] = sin(pi*yMax*sinh(s*eta[n])/sinh(s)/2)
    # u(0, y > 1) = 1
    elif y[n] > etaY1:
        u[n, 0] = 1

# %% Apply boundary conditions to v matrix
# Set v(x, eta=0) to be 0 from impermeable wall condition
v[0, :] = 0

# %% Loop over each x-step
for i in range(len(x)-1):

    # Set up matrix and RHS "b" matrix (knowns)
    # for Crank-Nicolson algorithm

    # Initialize matrix A for equation [[A]]*[u] = [b]
    A = np.zeros([Ny - 2, Ny - 2])
    b = np.zeros([Ny - 2, 1])
    # Y-dimension reduced by two because u(x, 0) and u(x, 1)
    # are known already

    # %% Use 2nd-Order-Accurate scheme for first interior nodes

    # at j = 2 (near bottom border of domain)
    # Calculate values of A(eta) and B(eta) at j = 2
    Ae = v[1, i]*etaY[1] - nu*etaY[1]*etaYY[1]/RD
    Be = -nu*etaY[1]**2/RD

    # Populate coefficient matrix
    A[0, 0] = u[1, i]/dx - Be/de**2  # Assign value in matrix location [1, 1]
    A[0, 1] = Ae/(4*de) + Be/(2*de**2)  # Assign value in matrix location
    b[0] = (u[0, i]*(Ae/(4*de) - Be/(2*de**2)) + u[1, i]*Be/de**2
            + u[1, i]**2/dx + u[2, i]*(-Ae/(4*de) - Be/(2*de**2))
            + u[0, i+1]*(Ae/(4*de) - Be/(2*de**2)))

    # %% at j = Ny-1 (near top border of domain)
    # Calculate values of A(eta) and B(eta) at j = Ny-1
    Ae = v[-2, i]*etaY[-2] - nu*etaY[-2]*etaYY[-2]/RD
    Be = -nu*etaY[-2]**2/RD

    # Populate coefficient matrix
    A[-1, -1] = u[-2, i]/dx - Be/de**2  # Assign value to last diagonal element
    A[-1, -2] = -Ae/(4*de) + Be/(2*de**2)  # Assign value to left of last diag
    b[-1] = (u[-3, i]*(Ae/(4*de) - Be/(2*de**2)) + u[-2, i]*Be/de**2
             + u[-2, i]**2/dx + u[-1, i]*(-Ae/(4*de) - Be/(2*de**2))
             + u[-1, i+1]*(-Ae/(4*de) - Be/(2*de**2)))

    # %% Use 4th-Order-Accurate scheme for j = 3 to j = Ny-2

    # Second internal node (j = 3)
    # Calculate values of A(eta) and B(eta) at j
    Ae = v[2, i]*etaY[2] - nu*etaY[2]*etaYY[2]/RD
    Be = -nu*etaY[2]**2/RD

    # Store coefficients and value for RHS vector b
    A[1, 0] = -Ae/(3*de) + 2*Be/(3*de**2)
    A[1, 1] = -5*Be/(4*de**2) + u[2, i]/dx
    A[1, 2] = Ae/(3*de) + 2*Be/(3*de**2)
    A[1, 3] = -Ae/(24*de) - Be/(24*de**2)
    b[1] = (u[0, i]*(-Ae/(24*de) + Be/(24*de**2))
            + u[1, i]*(Ae/(3*de) - 2*Be/(3*de**2))
            + u[2, i]*(5*Be/(4*de**2) + u[2, i]/dx)
            + u[3, i]*(-Ae/(3*de) - 2*Be/(3*de**2))
            + u[4, i]*(Ae/(24*de) + Be/(24*de**2))
            + u[0, i+1]*(-Ae/(24*de) + Be/(24*de**2)))

    # %% Loop over internal nodes to compute and store coefficients
    for j in range(2, Ny-4):
        # Calculate values of A(eta) and B(eta) at j
        Ae = v[j, i]*etaY[j] - nu*etaY[j]*etaYY[j]/RD
        Be = -nu*etaY[j]**2/RD

        A[j, j-2] = Ae/(24*de) - Be/(24*de**2)
        A[j, j-1] = -Ae/(3*de) + 2*Be/(3*de**2)
        A[j, j] = -5*Be/(4*de**2) + u[j, i]/dx
        A[j, j+1] = Ae/(3*de) + 2*Be/(3*de**2)
        A[j, j+2] = -Ae/(24*de) - Be/(24*de**2)
        b[j] = (u[j-2, i]*(-Ae/(24*de) + Be/(24*de**2))
                + u[j-1, i]*(Ae/(3*de) - 2*Be/(3*de**2))
                + u[j, i]*(5*Be/(4*de**2) + u[j, i]/dx)
                + u[j+1, i]*(-Ae/(3*de) - 2*Be/(3*de**2))
                + u[j+2, i]*(Ae/(24*de) + Be/(24*de**2)))

    # %% Second-to-last internal node (j = Ny-2)
    # Calculate values of A(eta) and B(eta) at j
    Ae = v[-3, i]*etaY[-3] - nu*etaY[-3]*etaYY[-3]/RD
    Be = -nu*etaY[-3]**2/RD

    # Store coefficients and value for RHS vector b
    A[-2, -4] = Ae/(24*de) - Be/(24*de**2)
    A[-2, -3] = -Ae/(3*de) + 2*Be/(3*de**2)
    A[-2, -2] = -5*Be/(4*de**2) + u[-3, i]/dx
    A[-2, -1] = Ae/(3*de) + 2*Be/(3*de**2)
    b[-2] = (u[-5, i]*(-Ae/(24*de) + Be/(24*de**2))
             + u[-4, i]*(Ae/(3*de) - 2*Be/(3*de**2))
             + u[-3, i]*(5*Be/(4*de**2) + u[-3, i]/dx)
             + u[-2, i]*(-Ae/(3*de) - 2*Be/(3*de**2))
             + u[-1, i]*(Ae/(24*de) + Be/(24*de**2))
             + u[-1, i+1]*(Ae/(24*de) + Be/(24*de**2)))

    # Perform matrix inversion to solve for u
    if method == 'lu':  # if input was for LU decomposition
        u[1:-1, i+1] = pentaLU(A, b)  # call the pentaLU solver

    if method == 'inv':  # if input was for built-in inv (for testing)
        u[1:-1, i+1] = (inv(A)@b).transpose()  # solve by inverting matrix

    # %% Now that u at x+1 has been solved for, use continuity to solve for v

    # Initialize matrix A and vector b for equation [[A]]*[v] = [b]
    A = np.zeros([Ny - 1, Ny - 1])
    b = np.zeros([Ny - 1, 1])

    for j in range(1, Ny):

        if i == 0:  # if on first iteration of x-step loop, use first-order x

            # Use third order FDS in eta, 2nd order Crank Nicolson in x
            # Use biased 3rd order scheme for node adjacent to bottom boundary
            if j == 1:
                A[j-1, j-1] = -etaY[j]/(2*de)
                A[j-1, j] = etaY[j]/(de)
                A[j-1, j+1] = -etaY[j]/(6*de)
                b[j-1] = (u[j, i] - u[j, i+1])/dx

            # One up from bottom boundary - use 4th order scheme about j - 1/2
            elif j == 2:
                A[j-1, j-2] = -9*etaY[2]/(8*de)
                A[j-1, j-1] = 9*etaY[2]/(8*de)
                A[j-1, j] = -etaY[2]/(24*de)
                b[j-1] = ((u[j, i+1] - u[j, i] + u[j-1, i+1] - u[j-1, i])
                          / (-2*dx))

            # At top boundary use 3rd order scheme about Ny - 1/2
            elif j == Ny-1:
                A[j-1, j-3] = 1/2
                A[j-1, j-2] = -2
                A[j-1, j-1] = 3/2
                b[j-1] = 0

            # Loop over internal nodes - use 4th order scheme about j - 1/2
            else:
                A[j-1, j-3] = etaY[j+1]/(24*de)
                A[j-1, j-2] = -9*etaY[j+1]/(8*de)
                A[j-1, j-1] = 9*etaY[j+1]/(8*de)
                A[j-1, j] = -etaY[j+1]/(24*de)
                b[j-1] = ((u[j, i+1] - u[j, i] + u[j-1, i+1] - u[j-1, i])
                          / (-2*dx))
        else:
            # Use third order FDS in eta, 2nd order Crank Nicolson in x
            # Use biased 3rd order scheme for node adjacent to bottom boundary
            if j == 1:
                A[j-1, j-1] = -etaY[j]/(2*de)
                A[j-1, j] = etaY[j]/(de)
                A[j-1, j+1] = -etaY[j]/(6*de)
                b[j-1] = -1/dx*(3/2*u[j, i+1] - 2*u[j, i] + 1/2*u[j, i-1])

            # One up from bottom boundary - use 4th order scheme about j - 1/2
            elif j == 2:
                A[j-1, j-2] = -9*etaY[2]/(8*de)
                A[j-1, j-1] = 9*etaY[2]/(8*de)
                A[j-1, j] = -etaY[2]/(24*de)
                b[j-1] = -1/dx*(3/2*u[j, i+1] - 2*u[j, i] + 1/2*u[j, i-1]
                                + 3/2*u[j-1, i+1]-2*u[j-1, i]+1/2*u[j-1, i-1])

            # At top boundary use 3rd order scheme about Ny - 1/2
            elif j == Ny-1:
                A[j-1, j-3] = 1/2
                A[j-1, j-2] = -2
                A[j-1, j-1] = 3/2
                b[j-1] = 0

            # Loop over internal nodes - use 4th order scheme about j - 1/2
            else:
                A[j-1, j-3] = etaY[j+1]/(24*de)
                A[j-1, j-2] = -9*etaY[j+1]/(8*de)
                A[j-1, j-1] = 9*etaY[j+1]/(8*de)
                A[j-1, j] = -etaY[j+1]/(24*de)
                b[j-1] = -1/dx*(3/2*u[j, i+1] - 2*u[j, i] + 1/2*u[j, i-1]
                                + 3/2*u[j-1, i+1]-2*u[j-1, i]+1/2*u[j-1, i-1])

#    # Use third order FDS in eta, 2nd order Crank Nicolson in x
#    # Use biased 3rd order scheme for node adjacent to bottom boundary
#    A[0, 0] = -etaY[1]/(4*de)
#    A[0, 1] = etaY[1]/(2*de)
#    A[0, 2] = -etaY[1]/(12*de)
#    b[0] = ((u[j, i] - u[j, i+1])/dx
#            - etaY[1]/12*(-v[3, i]+6*v[2, i]-3*v[1, i])/de)
#
#    # One up from bottom boundary - now use 4th order scheme about j - 1/2
#    A[1, 0] = -9*etaY[2]/(16*de)
#    A[1, 1] = 9*etaY[2]/(16*de)
#    A[1, 2] = -etaY[2]/(48*de)
#    b[1] = ((u[2, i]-u[2, i+1])/dx
#            - (etaY[2]/48)*(27*v[1, i]-27*v[2, i]+v[3, i])/de)
#
#    # Loop over internal nodes - still used 4th order scheme about j - 1/2
#    for j in range(2, Ny-2):
#        A[j, j-2] = etaY[j+1]/(48*de)
#        A[j, j-1] = -9*etaY[j+1]/(16*de)
#        A[j, j] = 9*etaY[j+1]/(16*de)
#        A[j, j+1] = -etaY[j+1]/(48*de)
#        b[j] = ((u[j+1, i]-u[j+1, i+1])/dx - (etaY[j+1]/48)
#                * (-v[j-2, i]+27*v[j-1, i]-27*v[j, i]+v[j+1, i])/de)
#
#    # Finally at top boundary use 3rd order scheme about Ny - 1/2
#    A[-1, -3] = 1
#    A[-1, -2] = -4
#    A[-1, -1] = 3
#    b[-1] = 0

    # Perform matrix inversion to solve for v
#    if method == 'lu':  # if input was for LU decomposition
#        v[1:, i+1] = pentaLU(A, b)  # call the pentaLU solver
#    if method == 'inv':  # if input was for built-in inv (for testing)
#        v[1:, i+1] = (inv(A)@b).transpose()  # solve by inverting matrix

# output is the u-matrix
# return u

# Plot A matrix
#    c = np.linspace(1, np.shape(A)[0]-1, np.shape(A)[0])
#    plt.contourf(c, c, A, cmap='plasma',
#                 levels=np.linspace(0., np.amax(u), 11))
#    plt.colorbar()
#
#    np.round(A[0:4, 0:4], 4)
#    np.round(A[-4:, -4:], 5)

# %% Plot results
# plt.figure(figsize=(6, 4))
# plt.contourf(x, eta, u, cmap='plasma',
#              levels=np.linspace(0., np.amax(u), 11))
# cbar = plt.colorbar()
# cbar.ax.set_ylabel('    u', rotation=0,
#                   weight='bold')
# plt.xlabel('x')
# plt.ylabel('eta', rotation=0)

nSteps = -1
plt.figure(figsize=(6, 4))
plt.contourf(x[0:nSteps], y, u[:, 0:nSteps], cmap='plasma')
cbar = plt.colorbar()
cbar.ax.set_ylabel('    u', rotation=0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)


plt.figure(figsize=(6, 4))
plt.contourf(x, y, v, cmap='plasma', levels=np.linspace(0., 1, 11))
cbar = plt.colorbar()
cbar.ax.set_ylabel('    v', rotation=0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

# eta = np.linspace(0., 1., len(y))
# s = 3
# yy = np.array([yMax*sinh(s*x)/sinh(s) for x in eta])
# plt.plot(eta, np.zeros(len(eta)), 'o', yy, np.zeros(len(yy)), 'x')
#
# eta = np.array([asinh(yi/yMax*sinh(s))/s for yi in y])
# plt.plot(eta, y, '.')

plt.figure(figsize=(6, 4))
plt.plot(u[:, 0], eta, 'r', u[:, 20], eta, 'g', u[:, -1], eta, 'b')
plt.legend(['X=0', 'X=20', 'X=40'])
plt.xlabel('u')
plt.ylabel('eta')


def blasius(Nx, Ny):  # Define function to calculate Blasius BL solution
    print('Blasius')
