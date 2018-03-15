"""
Mon, Feb 19, 2018

@author: Brian F. Knisely

AERSP/ME 525: Turbulence and Applications to CFD: RANS
Computer Project Number 1

The purpose of this code is to use finite difference to solve a parabolic
partial differential equation.

The PDE is: ∂u/∂x - ∂^2(u)/∂y^2 = y

Subject to the following boundary conditions (BCs):
    u(x, 0) = 0
    u(x, 1) = 1
    u(0, y) = y

The Crank-Nicolson Algorithm is to be used to march in the x-direction, using
a uniform grid and central differencing scheme that is fourth-order in y. An LU
decompsition algorithm is used to solve the pentadiagonal matrix for u-values
at each x-step.

Developed in Spyder 3.2.6 for Windows
"""

# Import packages for arrays, plotting, and timing
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import os

def analytic(Nx, Ny):  # Define function to output analytic solution
    """
    This function solves the PDE ∂u/∂x - ∂^2(u)/∂y^2 = y analytically using
    the closed-form Fourier sine series solution with N = 100 summation terms.
    """
    # Import functions as necessary
    from math import sin, exp, pi

    # Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
    x = np.linspace(0, 1, Nx)

    # Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
    y = np.linspace(0, 1, Ny)

    # Initialize solution as 2D array of zeros
    # of dimension [Nx columns] by [Ny rows]
    u = np.zeros([Ny, Nx])

    nTerms = 100  # choose number of terms for finite sum in solution

    # Loop to solve for u at each (x, y) location
    for i in range(len(x)):  # loop over all x values
        for j in range(len(y)):  # loop over all y values
            v = 0  # set v function to zero initially
            for n in range(1, nTerms+1):  # loop over all eigvalues from 0 to N
                # calculate the Bn for the nth eigenvalue
                Bn = 2*(-1)**n / (n*pi)**3
                # add terms to v one at a time
                v = v + Bn * sin(n*pi*y[j]) * exp(-(n*pi)**2*x[i])
            U = -y[j]**3/6 + 7*y[j]/6  # solution to U(y)
            # Store result in the jth row, ith column
            u[j, i] = U + v  # Add solutions of U and v together to get u(x, y)
    return u


def pentaLU(A, b):  # Define LU Decomposition function to solve A*x = b for x
    """
    Function to solve A*x = b for a given a pentadiagonal 2D array A and right-
    hand side 1D array, b. Shape of A must be at least 5 x 5.
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


def main(Nx, Ny, method):  # Define main function to set up grid and A matrix
    #                and march in x-direction using Crank-Nicolson algorithm
    #       Inputs:  Nx = number of nodes in x-direction
    #                Ny = number of nodes in y-direction
    #                method = "lu" or "inv" for matrix inversion

    # Initialize uniform grid
    # Nx = 21  # number of nodes in x-direction
    # Ny = 21  # number of nodes in y-direction

    # Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
    x = np.linspace(0, 1, Nx)

    # Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
    y = np.linspace(0, 1, Ny)

    # Calculate spacings dx and dy; these are constant with the uniform grid
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize u-array: 2D array of zeros
    # of dimension [Nx columns] by [Ny rows]
    u = np.zeros([Ny, Nx])
    # u is a 2D array of velocities at all spatial locations

    # Apply boundary conditions to u matrix

    # Set u(x, 0) to be 0 from Dirichlet boundary condition
    u[0, :] = 0  # Set 0th row, all columns to be 0

    # Set u(x, 1) to be 1 from Dirichlet boundary condition
    u[-1, :] = 1  # Set last row, all columns to be 1

    # Set u(0, y) to be y from Dirichlet boundary condition
    u[:, 0] = y  # Set all rows, 0th column to be y

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
    return u
    # End of main function


def makePlots(u):  # Display results spatially

    # Form x and y arrays
    Ny, Nx = np.shape(u)
    # Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
    x = np.linspace(0, 1, Nx)
    # Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
    y = np.linspace(0, 1, Ny)

    # Create contour plot of u vs x and y
    plt.figure(figsize=(6, 4))
    plt.contourf(x, y, u, cmap='plasma', levels=np.linspace(0., 1., 11))
    cbar = plt.colorbar()
    fs = 17  # Define font size for figures
    fn = 'Calibri'  # Define font for figures
    plt.xlabel('x', fontsize=fs, fontname=fn, fontweight='bold')
    plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
               fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    cbar.ax.set_ylabel('    u', rotation=0, fontname=fn, fontsize=fs,
                       weight='bold')
    cbar.ax.set_yticklabels([round(cbar.get_ticks()[n], 2)
                            for n in range(len(cbar.get_ticks()))],
                            fontsize=fs-2, fontname=fn, weight='bold')
    plt.savefig(os.getcwd()+"\\figures\\contour"+str(Nx)+'_'+str(Ny)+".png",
                dpi=320, bbox_inches='tight')  # save figure
    plt.close()  # close figure

    # Look at solution at selected x-locations
    xLocs = [0.05, 0.1, 0.2, 0.5, 1.0]
    lines = ['y-', 'r-.', 'm--', 'b:', 'k-']  # define line styles
    # Loop for each x-location to make plots at each location comparing exact
    # solution with analytical solution
    legStr = []  # initialize value to store strings for legend
    plt.figure()  # create new figure
    for n in range(len(xLocs)):
        col = np.argmin(abs(x-xLocs[n]))  # extract value closest to given xLoc
        plt.plot(u[:, col], y, lines[n])
        legStr.append('x = {}'.format(xLocs[n]))  # append value to leg string
    plt.legend(legStr, fontsize=fs-2)
    plt.xlabel('u', fontsize=fs, fontname=fn, fontweight='bold')
    plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
               fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.savefig(os.getcwd()+"\\figures\\curves"+str(Nx)+'_'+str(Ny)+".png",
                dpi=320, bbox_inches='tight')  # save figure
    plt.close()  # close figure
    # End of makePlots function


def errorPlots(u, u_an):  # function to plot error in u vs u_an
    # Form x and y arrays
    Ny, Nx = np.shape(u)
    # Make linear-spaced 1D array of x-values from 0 to 1 with Nx elements
    x = np.linspace(0, 1, Nx)
    # Make linear-spaced 1D array of y-values from 0 to 1 with Ny elements
    y = np.linspace(0, 1, Ny)
    # Create contour plot of error (u-u_an) vs x and y
    plt.figure(figsize=(6, 4))
    cmax = round(np.amax(abs(u-u_an)), 3)  # maximum absolute contour value
    lev = np.linspace(0, cmax, 11)
    plt.contourf(x, y, abs(u-u_an), cmap='plasma',
                 levels=lev)
    cbar = plt.colorbar()  # make colorbar
    fs = 17  # Define font size for figures
    fn = 'Calibri'  # Define font for figures
    # Label axes and set tick styles
    plt.xlabel('x', fontsize=fs, fontname=fn, fontweight='bold')
    plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
               fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    cbar.ax.set_ylabel('                   |${u-u_{an}}$|', rotation=0,
                       fontname=fn, fontsize=fs, weight='bold')
    cbar.ax.set_yticklabels([round(cbar.get_ticks()[n], 4)
                            for n in range(len(cbar.get_ticks()))],
                            fontsize=fs-2, fontname=fn, weight='bold')
    plt.savefig(os.getcwd()+"\\figures\\err_contour"+str(Nx)+'_'+str(Ny)+".png",
                dpi=320, bbox_inches='tight')  # save figure
    plt.close()  # close figure

    # Look at error in solution at selected x-locations
    xLocs = [0.05, 0.1, 0.2, 0.5, 1.0]
    lines = ['y-', 'r-.', 'm--', 'b:', 'k-']
    # Loop for each x-location to make plots at each location comparing exact
    # solution with analytical solution
    legStr = []
    plt.figure()  # create new figure
    for n in range(len(xLocs)):
        col = np.argmin(abs(x-xLocs[n]))  # extract value closest to given xLoc
        plt.plot(abs(u[:, col]-u_an[:, col]), y, lines[n])
        legStr.append('x = {}'.format(xLocs[n]))
    plt.legend(legStr, fontsize=fs-2)
    plt.xlabel('|${u-u_{an}}$|', fontsize=fs, fontname=fn, fontweight='bold')
    plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
               fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.savefig(os.getcwd()+"\\figures\\err_curves"+str(Nx)+'_'+str(Ny)+".png",
                dpi=320, bbox_inches='tight')  # save figure
    plt.close()  # close figure
    # End of errorPlots function


def plotDxDyEffects():  # function to plot effects of dx and dy spacings

    # Data collected on variation of dx values (with dy constant at 0.05)
    dxes = [0.05, 0.04, 0.0333333, 0.025, 0.02, 0.0166667, 0.0125, 0.01, 0.005,
            0.0025, 0.00166667, 0.001]  # dx values themselves
    dxerrs = [0.00276807, 0.00222227, 0.00183598, 0.00132455, 0.00101785,
              0.000861093, 0.000647488, 0.000507671, 0.00019631, 3.41267e-05,
              1.97199e-05, 3.37169e-05]  # maximum error in u
    dxt = [0.3222, 0.343764, 0.406265, 0.406255, 0.499992, 0.499991, 0.609383,
           0.671899, 0.843771, 1.70315, 2.55095, 4.21325]  # computation time
    # Create plot to show effects of dx on error and computation time
    plt.figure()
    plt.loglog(dxes, dxerrs, 'ko', dxes, dxt, 'bx')  # plot on log-log axes
    fs = 17  # Define font size for figures
    fn = 'Calibri'  # Define font for figures
    # Label axes and set tick styles
    plt.xlabel('$\Delta$ x', fontsize=fs, fontname=fn, fontweight='bold')
    # plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
    #            fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.legend(['Absolute error in u', 'Computation time [s]'])
    plt.savefig(os.getcwd()+"\\figures\\dxEffect.png", dpi=320,
                bbox_inches='tight')
    plt.close()  # close figure

    # Data collected on variation of dy values (with dx constant at 0.05)
    # dy values themselves
    dyes = [0.05, 0.04, 0.0333333, 0.025, 0.02, 0.0166667, 0.0125, 0.01, 0.005,
            0.0025, 0.00166667, 0.001]
    # maximum error in u
    dyerrs = [0.00276807, 0.00278632, 0.00278221, 0.00278486, 0.00279274,
              0.00279906, 0.002803, 0.00280316, 0.00280319, 0.00280319,
              0.0028033, 0.00280335]
    # computation time
    dyt = [0.0937653, 0.109376, 0.140636, 0.171877, 0.218763, 0.265623,
           0.34375, 0.425119, 0.859382, 1.73438, 2.61386, 4.33241]
    # Create plot to show effects of dy on error and computation time
    plt.figure()
    plt.loglog(dyes, dyerrs, 'ko', dyes, dyt, 'bx')  # plot on log-log axes
    # Label axes and set tick styles
    plt.xlabel('$\Delta$ y', fontsize=fs, fontname=fn, fontweight='bold')
    # plt.ylabel('y' + '     ', fontsize=fs, rotation=0, fontname=fn,
    #            fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.legend(['Absolute error in u', 'Computation time [s]'])
    plt.savefig(os.getcwd()+"\\figures\\dyEffect.png", dpi=320,
                bbox_inches='tight')
    plt.close()  # close figure

    # Data collected on varying both dx and dy together
    dxdyes = [0.05, 0.04, 0.0333333, 0.025, 0.02, 0.0166667, 0.0125, 0.01,
              0.005, 0.0025, 0.00166667, 0.001]  # dx and dy values themselves
    dxdyerrs = [0.00276807, 0.00218433, 0.00185296, 0.00139747, 0.00110844,
                0.000932838, 0.00069616, 0.000560447, 0.000280287, 0.000140149,
                9.34256e-05, 5.60506e-05]
    dxdyt = [0.0937514, 0.140621, 0.187512, 0.343739, 0.536466, 0.75001,
             1.34378, 2.09376, 8.223, 32.7938, 77.9015, 207.268]
    # Create plot to show effects of dy on error and computation time
    plt.figure()
    # plot on log-log axes
    plt.loglog(dxdyes, dxdyerrs, 'ko', dxdyes, dxdyt, 'bx')
    # Label axes and set tick styles
    plt.xlabel('$\Delta$ x = $\Delta$ y', fontsize=fs, fontname=fn,
               fontweight='bold')
    plt.xticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.yticks(fontsize=fs-2, fontname=fn, fontweight='bold')
    plt.legend(['Absolute error in u', 'Computation time [s]'])
    plt.savefig(os.getcwd()+"\\figures\\dxdyEffect.png", dpi=320,
                bbox_inches='tight')
    plt.close()
    # End of plotDxDyEffects function


# Run functions in order
t0 = time.time()  # begin timer
Nx = 21  # number of nodes in x-direction
Ny = 21  # number of nodes in y-direction

# execute main numerical solver to calculate u, use LU solver
u = main(Nx, Ny, 'lu')

# execute analytic solver to calculate u_analytic
u_an = analytic(Nx, Ny)

# run function to make plots of results for u
makePlots(u)

# run function to make plots of errors compared to analytic solution
errorPlots(u, u_an)

# calculate difference in time from current to when code started (elapsed time)
elapsed = time.time() - t0

# run function to make plots showing the effects of dx and dy spacing
plotDxDyEffects()

print('dx = {0:0.6}'.format(1/(Nx-1)))  # print current dx spacing
print('dy = {0:0.6}'.format(1/(Ny-1)))  # print current dy spacing
print('Max error in u is {0:0.6}'.format(np.amax(abs(u-u_an))))  # print error
print('Elapsed time is {0:0.4}'.format(elapsed) + ' s.')  # print elapsed time
