# -*- coding: utf-8 -*-
"""
@author: Brian Knisely

AERSP/ME 525: Turbulence and Applications to CFD: RANS
Computer Project Number 2

The purpose of this code is to use finite difference to solve the laminar
boundary layer equations for a 2D flat plate and compare to the Blasius
solution.

The PDEs with dimensional variables (denoted with ~) are:
    du~/dx~ + dv~/dy~ = 0
    u~ du~/dx~ + v~ du~/dy~ = d/dy~(nu~ du~/dy~)

Note: all derivatives are partial derivatives

The equations are nondimensionalized with:
    x = x~/L~
    y = y~/delta~
    u = u~/Uinf~
    v = v~L~/(Uinf~delta~)
    nu = nu~/nu_inf~
    RD = L~nu_inf~/(Uinf~(delta~)^2)

The resulting nondimensional equations are:
    du/dx + dv/dy = 0
    u du/dx + v du/dy = 1/RD d/dy(nu du/dy)

The boundary conditions (BCs) in nondimensional form are:
    u(x, 0) = 0  (no-slip condition)
    u(x, yMax) = 1  (velocity is equal to freestream at top edge of domain)
    u(0, y <= 1) = sin(pi*y/2)  (starting profile)
    u(0, y > 1) = 1  (starting profile)
    v(x, 0) = 0  (impermeable wall condition)

The Crank-Nicolson Algorithm is to be used to march in the x-direction, using
a uniform grid and central differencing scheme that is fourth-order in y. An LU
decompsition algorithm is used to solve the pentadiagonal matrix for u-values
at each x-step. After computing the u-values, the continuity equation is solved
for the v-values at that step. A fourth-order noncentered scheme is used to
generate the coefficient matrix for v to avoid singularities when inverting the
matrix. The solution is compared to the Blasius solution for a flat plate
boundary layer. The effect of stretching factor is investigated.
"""

# Import packages for arrays, plotting, timing, and file I/O
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import sin, pi, sinh, cosh, asinh, sqrt
import csv
import os


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


# %% Main function
def main(Nx, Ny, method, s):  # Define main function to set up grid and matrix
    #                           and march in x-direction using Crank-Nicolson
    #                           algorithm

    #    Inputs:  Nx = number of nodes in x-direction
    #             Ny = number of nodes in y-direction
    #             method = "lu" or "inv" for matrix inversion
    #             s = stretching factor

    # Inputs for testing
    # Nx = 41
    # Ny = 151
    # method = 'lu'
    # s = 5

    # Define bounds of computational domain
    xMax = 20  # Dimensional distance in m
    yMax = 10  # Scaled by BL height

    # Define given dimensional quantities
    nuInfDim = 1.5e-6  # Dimensional freestream viscosity in m^2/s
    uInfDim = 40  # Dimensional freestream velocity in m/s
    LDim = 0.5  # Length of plate in m
    deltaDim = 0.005  # Initial BL thickness in m

    # Calculate derived nondimensional quantity
    RD = uInfDim*deltaDim**2/(LDim*nuInfDim)

    # Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
    x = np.linspace(0, xMax, Nx)

    # Make linear-spaced 1D array of eta-values from 0 to 1 with Ny elements
    eta = np.linspace(0, 1, Ny)

    # %% Compute values of y based on eta and ymax
    y = [yMax*sinh(s*et)/sinh(s) for et in eta]

    # Evaluate values of fPrime and fDoublePrime
    fPrime = [s*yMax*cosh(s*et)/sinh(s) for et in eta]
    fDoublePrime = [s**2*yMax*sinh(s*et)/sinh(s) for et in eta]
    # Determine etaY and etaYY
    etaY = [1/fP for fP in fPrime]
    etaYY = [-fDoublePrime[n]/(fPrime[n]**3) for n in range(len(fPrime))]

    # Calculate spacings dx and de; constant with the uniform x-eta grid
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
    etaY1 = asinh(1*sinh(s)/yMax)/s  # Determine eta value corresponding to y=1
    for n in range(len(y)):
        # u(0, y <= 1) = sin(pi*y/2)
        if eta[n] <= etaY1:
            u[n, 0] = sin(pi*y[n]/2)
        # u(0, y > 1) = 1
        elif y[n] > etaY1:
            u[n, 0] = 1

    # %% Apply boundary condition to v matrix
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
        Ae = v[1, i]*etaY[1] - etaY[1]*etaYY[1]/RD
        Be = -etaY[1]**2/RD

        # Populate coefficient matrix
        A[0, 0] = u[1, i]/dx - Be/de**2  # Assign value in location [1, 1]
        A[0, 1] = Ae/(4*de) + Be/(2*de**2)  # Assign value in matrix location
        b[0] = (u[0, i]*(Ae/(4*de) - Be/(2*de**2)) + u[1, i]*Be/de**2
                + u[1, i]**2/dx + u[2, i]*(-Ae/(4*de) - Be/(2*de**2))
                + u[0, i+1]*(Ae/(4*de) - Be/(2*de**2)))

        # %% at j = Ny-1 (near top border of domain)
        # Calculate values of A(eta) and B(eta) at j = Ny-1
        Ae = v[-2, i]*etaY[-2] - etaY[-2]*etaYY[-2]/RD
        Be = -etaY[-2]**2/RD

        # Populate coefficient matrix
        A[-1, -1] = u[-2, i]/dx - Be/de**2  # Assign value to last diag element
        A[-1, -2] = -Ae/(4*de) + Be/(2*de**2)  # Assign value to left of diag
        b[-1] = (u[-3, i]*(Ae/(4*de) - Be/(2*de**2)) + u[-2, i]*Be/de**2
                 + u[-2, i]**2/dx + u[-1, i]*(-Ae/(4*de) - Be/(2*de**2))
                 + u[-1, i+1]*(-Ae/(4*de) - Be/(2*de**2)))

        # %% Use 4th-Order-Accurate scheme for j = 3 to j = Ny-2

        # Second internal node (j = 3)
        # Calculate values of A(eta) and B(eta) at j
        Ae = v[2, i]*etaY[2] - etaY[2]*etaYY[2]/RD
        Be = -etaY[2]**2/RD

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
            Ae = v[j, i]*etaY[j] - etaY[j]*etaYY[j]/RD
            Be = -etaY[j]**2/RD

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
        Ae = v[-3, i]*etaY[-3] - etaY[-3]*etaYY[-3]/RD
        Be = -etaY[-3]**2/RD

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

        # %% u at x+1 has been solved for, now use continuity to solve for v

        # Initialize matrix A and vector b for equation [[A]]*[v] = [b]
        A = np.zeros([Ny - 1, Ny - 1])
        b = np.zeros([Ny - 1, 1])

        # Use third order FDS in eta, 2nd order Crank Nicolson in x
        # Use biased 3rd order scheme for node adjacent to bottom boundary
        A[0, 0] = -etaY[1]/(4*de)
        A[0, 1] = etaY[1]/(2*de)
        A[0, 2] = -etaY[1]/(12*de)
        b[0] = ((u[j, i] - u[j, i+1])/dx
                - etaY[1]/12*(-v[3, i]+6*v[2, i]-3*v[1, i])/de)

        # One up from bottom boundary - now use 4th order scheme about j - 1/2
        A[1, 0] = -9*etaY[2]/(16*de)
        A[1, 1] = 9*etaY[2]/(16*de)
        A[1, 2] = -etaY[2]/(48*de)
        b[1] = ((u[2, i]-u[2, i+1])/dx
                - (etaY[2]/48)*(27*v[1, i]-27*v[2, i]+v[3, i])/de)

        # Loop over internal nodes - still used 4th order scheme about j - 1/2
        for j in range(2, Ny-2):
            A[j, j-2] = etaY[j+1]/(48*de)
            A[j, j-1] = -9*etaY[j+1]/(16*de)
            A[j, j] = 9*etaY[j+1]/(16*de)
            A[j, j+1] = -etaY[j+1]/(48*de)
            b[j] = ((u[j+1, i]-u[j+1, i+1])/dx - (etaY[j+1]/48)
                    * (-v[j-2, i]+27*v[j-1, i]-27*v[j, i]+v[j+1, i])/de)

        # Finally at top boundary use 3rd order scheme about Ny - 1/2
        A[-1, -3] = 1
        A[-1, -2] = -4
        A[-1, -1] = 3
        b[-1] = 0

    # Perform matrix inversion to solve for v
    if method == 'lu':  # if input was for LU decomposition
        v[1:, i+1] = pentaLU(A, b)  # call the pentaLU solver
    if method == 'inv':  # if input was for built-in inv (for testing)
        v[1:, i+1] = (inv(A)@b).transpose()  # solve by inverting matrix

    # output is the u-matrix
    return u, y


# %% Plot results
# Define function to plot versus Blasius
def plotsVsBlasius(u, y):

    # Yes! It is possible to read text files in Python!

    # Read CSV containing Blasius solution and store variables
    with open('readfiles//Blasius.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = [row[0].split(',') for row in reader]
        etaB = [eval(val) for val in data[0]]  # Store similarity variable list
        g = [eval(val) for val in data[2]]  # Store normalized velocity list

    xLoc = [0, 5, 10, 15, 20]  # Dimensional x-locations (m)
    xMax = 20  # Dimensional distance in m
    # Define given dimensional quantities
    nuInf = 1.5e-6  # Dimensional freestream viscosity in m^2/s
    uInf = 40  # Dimensional freestream velocity in m/s
    deltaDim = 0.005  # Initial BL thickness in m
    Nx = len(u[0, :])  # number of grid points in x-direction

    delta99 = []  # initialize to store calculated 99% BL thickness values
    xInds = []  # initialize to store x-indices corresponding to xLocs above
    setbacks = []  # initialize array to store estimates for x~
    # Loop through each x-location and solve for 99% BL thickness and estimate
    # x~ based on each BL thickness
    mrk = 0  # marker to count how many while loops
    for xi in xLoc:
        xInd = round(xi/xMax*(Nx-1))  # find index corresponding to that xLoc
        xInds.append(xInd)
        yInd = 0  # store index corresponding to BL height
        while True:
            if u[yInd, xInd] > 0.99:  # if 99% velocity satisfied
                delta99.append(deltaDim*y[yInd])  # Store BL height (m)
                setbacks.append(delta99[-1]**2*uInf/nuInf
                                / 4.91**2-xLoc[mrk])
                mrk += 1
                break  # stop while loop
            yInd += 1

    # Dimensionalize velocity from Blasius solution
    uB = [gi*uInf for gi in g]

    # Dimensionalize y-coordinates for numerical solution
    yDim = [deltaDim*yi for yi in y]

    # Initialize figure with 5 subplots/axes
    axs = ['ax' + str(n) for n in range(1, 6)]
    fig, axs = plt.subplots(figsize=(8, 6), ncols=5,
                            sharey='row')

    # Loop through all desired positions and plot numerical result vs Blasius
    for ii in range(5):
        xCorr = xLoc[ii] + 0.005**2 * uInf / nuInf / 4.91**2
        yB = [sqrt(xCorr)*etaBi/sqrt(uInf/nuInf) for etaBi in etaB]
        ax = axs[ii]
        ax.plot(uInf*u[:, xInds[ii]], yDim, 'r', uB, yB, 'k*')
        ax.set_xlabel('u [m/s]', fontweight='bold')
        ax.set_title('x = {0:0.0f} L'.format(xLoc[ii]/0.5))
        if ii == 0:
            ax.set_ylabel('y [m]        ', fontweight='bold', rotation=0)

    plt.legend(['FDM', 'Blasius'])  # add legend
    plt.ylim(ymin=0, ymax=0.01)  # axes limits
    plt.savefig(os.getcwd()+"\\figures\\profiles.png", dpi=320,
                bbox_inches='tight')  # save figure to file
    plt.close()  # close figure

    # Estimate initial position using BL thickness at beginning
    xInitial = 0.005**2*uInf/nuInf/4.91**2
    # Print result to console
    print('Estimated initial value of x~ is {0:0.2f} m'.format(xInitial))
    # Output value of xInitial
    return xInitial


# %% Calculate displacement thickness and momentum thickness
def thicc(u, y):  # function to compute displacement thickness, momentum
    #               thickness, and shape factor
    nuInf = 1.5e-6  # Dimensional freestream viscosity in m^2/s
    uInf = 40  # Dimensional freestream velocity in m/s
    deltaDim = 0.005  # Initial BL thickness in m
    xInitial = 0.005**2*uInf/nuInf/4.91**2
    Ny = len(y)  # Number of grid points in y-direction
    deltaStar = 0  # initialize displacement thickness
    thetaStar = 0  # initialize momentum thickness
    for i in range(1, Ny):
        uUi = ((1 - u[i, -1]) + (1 - u[i-1, -1]))/2  # find velocity deficit
        dy = (y[i] - y[i-1])*deltaDim  # find dimensional y-height difference
        deltaStar += (uUi)*dy  # add ith contribution
        thetaStar += uUi*(1 - uUi)*dy  # add ith contribution

    # Calculate Blasius estimates from solution
    deltaStarBlasius = 1.702*sqrt(nuInf*(20+xInitial)/uInf)
    thetaStarBlasius = 0.664*sqrt(nuInf*(20+xInitial)/uInf)

    # Print results to console
    print('Displacement thickness at x = 40 L is {0:0.6f} m'.format(deltaStar))
    print('Estimated Blasius displacement thickness at x = 40 L is {0:0.6f} m'
          .format(deltaStarBlasius))
    print('Momentum thickness at x = 40 L is {0:0.6f} m'.format(thetaStar))
    print('Estimated Blasius momentum thickness at x = 40 L is {0:0.6f} m'
          .format(thetaStarBlasius))


# %% Define function to show effect of stretching on profile
def stretchEffect(Nx, Ny, stretch):
    uInf = 40  # freestream velocity
    ls = ['r.', 'gx', 'b*', 'kd', 'mp', 'yh']  # line styles
    # uuu = np.zeros([Ny, Nx, len(stretch)])  # initialize array to compare u
    axs = ['ax' + str(n) for n in range(1, 1+len(stretch))]  # name axes
    # set up subplots on same figure
    fig, axs = plt.subplots(figsize=(8, 6), ncols=len(stretch), sharey='row')
    # Loop for each stretching factor
    for s in range(len(stretch)):
        u, y = main(Nx, Ny, 'lu', stretch[s])  # get u and y from main
        ax = axs[s]  # set axis number
        ax.plot(u*uInf, y, ls[s])  # plot y vs u
        ax.set_xlabel('u [m/s]', fontweight='bold')  # label x axis
        ax.set_title('s = {0:0.0f}'.format(stretch[s]))  # label title
        if s == 0:  # set y-label for the leftmost subplot
                ax.set_ylabel('y [m]        ', fontweight='bold', rotation=0)
    #    plt.savefig(os.getcwd()+"\\figures\\stretching.png", dpi=320,
    #                bbox_inches='tight')  # save figure to file


# %% Run functions in order
Nx = 41  # number of nodes in x-direction
Ny = 151  # number of nodes in y-direction
stretching = 5  # stretching factor
u, y = main(Nx, Ny, 'lu', stretching)

# Plot results compared to Blasius solution and calculate initial x~ position
plotsVsBlasius(u, y)

# Calculate thicknesses at end and compare to Blasius
thicc(u, y)

# %% Run function to see effect of stretching
stretch = [1, 5, 10]
stretchEffect(Nx, Ny, stretch)
