#===============================================================================
# TITLE: Hertz vector current loop simulation
# AUTHOR: Daniel Scarbrough
# EMAIL: dscarbro@mines.edu
# DATE: 5/1/2019
#
# MADE FOR: Colorado School of Mines
#           PHGN 507 Electromagnetic Theory
#           Final Project - Hertz Vectors
#
# DESCRIPTION: This script is used to simulate the electric field around a time-
# varying current traveling uniformly in a thin wire on the z-axis. The full
# 3D grid is simulated for future 3D plotting. Stream plots are produced, but 
# the contour plots are a better representation of what is happening. The 
# simulation is run through one period of current oscillation and saved as a 
# gif for each of the 3 planes.
#
# REQUIRED PACKAGES AND SOURCE:
# - numpy          (pip)
# - matplotlib     (pip)
# - imagemagick    (apt-get on Ubuntu)
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#===============================================================================

# Returns one of the first three Legendre polynomials
def legendre(x, l):
    if l == 0:
        return 1
    elif l == 1:
        return x
    elif l == 2:
        return (1/2)*(3*x**2 - 1)

#-------------------------------------------------------------------------------

# Custom form of the arctan function with two inputs (y/x) as in normal arctan
# definition. This function is unique in that for this script, it sets the 
# negative half of the coordinate grid to be flipped. This allows proper theta
# values, as not doing so results in low angles when they should be near pi
# due to the negative input to the arctan function.
def n2arctan(x, y):
    theta = np.arctan(y / x)
    theta[0:49, :, :] = (theta[0:49, :, :] - np.pi)
    return abs(theta)

#-------------------------------------------------------------------------------

# This custom arctan function is just a simple shortcut for calculating phi 
# based on x, y, and z in the meshgrid
def n3arctan(x, y, z):
    arg = np.sqrt(x**2 + y**2) / z
    return abs(np.arctan(arg))

#-------------------------------------------------------------------------------

# Calculates all of the r-components of the electric field in the meshgrid
def calcEr(X, Y, Z, I, halfL, theta, omega, t, dz):
    Er = np.zeros(shape=(len(X), len(Y), len(Z)))
    eps = 1
    mu = 1
    r = np.sqrt(X**2 + Y**2 + Z**2)
    l = 2 * halfL * dz
    c = 3 * 10**8   

    for indl in range(3): # sum for Legendre polynomials
        for lenStep in range(-halfL, halfL):
            # sums over each discrete piece of the wire

            if lenStep == 0:
                continue # don't do anything at the origin

            rp = lenStep * dz # Length to current wire piece

            # retarded time
            rt = t - (np.abs(np.sqrt(X**2 + Y**2 + Z**2) - rp)/c)
            
            # account for negative side of wire
            if rp > 0:
                Er +=I * l * np.cos(omega * rt) / (4 * np.pi) * abs(rp**indl) \
                    / r**(l+1) * np.cos(theta) * (-(indl + 1) * (indl + 2) / \
                        (r**2 * eps * omega) - mu * omega) * \
                            legendre(np.cos(theta), indl)
            else:
                Er +=I * l * np.cos(omega * rt) / (4 * np.pi) * abs(rp**indl) \
                    / r**(l+1) * np.cos(theta) * (-(indl + 1) * (indl + 2) / \
                        (r**2 * eps * omega) - mu * omega) * \
                            legendre(np.cos(np.pi - theta), indl)

    return Er

#-------------------------------------------------------------------------------

# nearly identical to the calcEr function
def calcEth(X, Y, Z, I, halfL, theta, omega, t, dz):
    Eth = np.zeros(shape=(len(X), len(Y), len(Z)))
    eps = 1
    mu = 1
    r = np.sqrt(X**2 + Y**2 + Z**2)
    l = 2 * halfL * dz
    c = 3 * 10**8   

    for indl in range(3):
        for lenStep in range(-halfL, halfL):
            if lenStep == 0:
                continue
            rp = lenStep * dz
            rt = t - (np.abs(np.sqrt(X**2 + Y**2 + Z**2) - rp)/c)
            if rp > 0:
                Eth +=I * l * np.cos(omega * rt) / (4 * np.pi) * abs(rp**indl) \
                    / r**(l+1) * np.sin(theta) * (-(indl + 1) / \
                        (r**2 * eps * omega) + mu * omega) * \
                            legendre(np.cos(theta), indl)
            else:
                Eth +=I * l * np.cos(omega * rt) / (4 * np.pi) * abs(rp**indl) \
                    / r**(l+1) * np.sin(theta) * (-(indl + 1) / \
                        (r**2 * eps * omega) + mu * omega) * \
                            legendre(np.cos(np.pi - theta), indl)

    return Eth

#-------------------------------------------------------------------------------

# Main function which defines system parameters, generates the grids, calls
# appropriate calculations, evolves system through time, and generates
# animations
def main():
    # simulation grid dimensions
    Nx = 100
    Ny = 100
    Nz = 100

    # max dimension of simulation grid i.e. x is [-dim, dim]
    dim = 0.2

    # generates one dimensional grids based on dim and number of points
    xGrid = np.linspace(-dim, dim, Nx)
    yGrid = np.linspace(-dim, dim, Ny)
    zGrid = np.linspace(-dim, dim, Nz)

    # actual spacing of zGrid
    dz = zGrid[1] - zGrid[0]

    # meshgrid to ease calculations at every point
    X, Y, Z = np.meshgrid(xGrid, yGrid, zGrid)

    # calculate the theta and phi values at each coordinate so that 
    # spherical calculations can be done
    theta = n3arctan(X, Y, Z)
    phi = n2arctan(X, Y)

    # initialize figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # set wire parameters
    I = 100
    halfL = 10 # half the wire length (centers at zero)
    omega = 2 * np.pi
    
    # set the time parameters for simulation. End time of 1 ensures smooth 
    # gif looping
    t = 0
    tEnd = 1
    dt = 0.02

    # Calculate each component over the entire grid at the initial time
    # - note: in spherical components
    Er = calcEr(X, Y, Z, I, halfL, theta, omega, t, dz)
    Eth = calcEth(X, Y, Z, I, halfL, theta, omega, t, dz)

    # convert spherical components to cartesian
    Ex = np.sin(theta) * np.cos(phi) * Er + np.cos(theta) * np.cos(phi) * Eth
    Ey = np.sin(theta) * np.sin(phi) * Er + np.cos(theta) * np.sin(phi) * Eth
    Ez = np.cos(theta) * Er - np.sin(theta) * Eth

    # Initialize an array to store the values for each time
    ExArr = [Ex]
    EyArr = [Ey]
    EzArr = [Ez]

    # simulate through time
    for i in range(1, int(tEnd / dt)):
        t = i * dt

        Er = calcEr(X, Y, Z, I, halfL, theta, omega, t, dz)
        Eth = calcEth(X, Y, Z, I, halfL, theta, omega, t, dz)

        Ex = np.sin(theta) * np.cos(phi) * Er + np.cos(theta) * np.cos(phi) * Eth
        Ey = np.sin(theta) * np.sin(phi) * Er + np.cos(theta) * np.sin(phi) * Eth
        Ez = np.cos(theta) * Er - np.sin(theta) * Eth

        # append values at current time to array
        ExArr.append(Ex)
        EyArr.append(Ey)
        EzArr.append(Ez)


    # The following functions handle animation in matplotlib
    #---------------------------------------------------------------------------
    def animateStreamEXZ(i):
        ax.clear()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$z$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        color = 2 * np.log(np.hypot(ExArr[i][:,int(Ny/2),:], EzArr[i][:,int(Nz/2),:]))
        stream = ax.streamplot(xGrid, zGrid, ExArr[i][:,int(Nx/2),:], EzArr[i][:,int(Nz/2),:], \
            color=color, linewidth=1, cmap=plt.cm.inferno, density=1, \
            arrowstyle='->', arrowsize=1.5)
        return stream

    def animateStreamEXY(i):
        ax.clear()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        color = 2 * np.log(np.hypot(ExArr[i][:,int(Nx/2),:], EyArr[i][:,int(Ny/2),:]))
        stream = ax.streamplot(xGrid, yGrid, ExArr[i][:,int(Nx/2),:], EyArr[i][:,int(Ny/2),:], \
            color=color, linewidth=1, cmap=plt.cm.inferno, density=1, \
            arrowstyle='->', arrowsize=1.5)
        return stream

    def animateStreamEYZ(i):
        ax.clear()
        ax.set_xlabel('$y$')
        ax.set_ylabel('$z$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        color = 2 * np.log(np.hypot(EyArr[i][:,int(Ny/2),:], EzArr[i][:,int(Nz/2),:]))
        stream = ax.streamplot(yGrid, zGrid, EyArr[i][:,int(Ny/2),:], EzArr[i][:,int(Nz/2),:], \
            color=color, linewidth=1, cmap=plt.cm.inferno, density=1, \
            arrowstyle='->', arrowsize=1.5)
        return stream

    def animateContourEXZ(i):
        ax.clear()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$z$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        contour = ax.contourf(xGrid, zGrid, EyArr[i][:,int(Ny/2),:])
        return contour

    def animateContourEXY(i):
        ax.clear()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        contour = ax.contourf(xGrid, yGrid, EzArr[i][:,int(Nz/2),:])
        return contour

    def animateContourEYZ(i):
        ax.clear()
        ax.set_xlabel('$y$')
        ax.set_ylabel('$z$')
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_aspect('equal')
        contour = ax.contourf(yGrid, zGrid, ExArr[i][:,int(Nx/2),:])
        return contour

    #---------------------------------------------------------------------------

    # animate and save each gif
    anim = animation.FuncAnimation(fig, animateStreamEXZ, frames=int(tEnd / dt))
    anim.save('exzStream.gif', writer="imagemagick", fps=10)

    anim = animation.FuncAnimation(fig, animateStreamEXY, frames=int(tEnd / dt))
    anim.save('exyStream.gif', writer="imagemagick", fps=10)

    anim = animation.FuncAnimation(fig, animateStreamEYZ, frames=int(tEnd / dt))
    anim.save('eyzStream.gif', writer="imagemagick", fps=10)

    anim = animation.FuncAnimation(fig, animateContourEXZ, frames=int(tEnd / dt))
    anim.save('exzContour.gif', writer="imagemagick", fps=10)

    anim = animation.FuncAnimation(fig, animateContourEXY, frames=int(tEnd / dt))
    anim.save('exyContour.gif', writer="imagemagick", fps=10)

    anim = animation.FuncAnimation(fig, animateContourEYZ, frames=int(tEnd / dt))
    anim.save('eyzContour.gif', writer="imagemagick", fps=10)


main()

#===============================================================================