#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:16:29 2022

A 3D plot script for spherical coordinates
"""

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Kaitlin Doublestein'
__email__ = 'kdoubles@umich.edu'


#coordinates = {'cartesian': ('x','y','z'),
#               'spherical': ('r','phi','theta')
               


def cartesian(r,phi,theta):
    """Converts spherical coordinates to cartesian coordinates and place 
    them in a dictionary.
    
    Agrs:
        r(float):
            The radial value.
        phi(float):
            The azimuthal angle, which can run from 0 to 180 degrees.
        theta(float):
            The radial angle, which can run from 0 to 360 degrees.
        
    Returns:
        cartesian_coords (dictionary): the converted cartesian coordinates
        from the spherical coordinate inputs.
        
    Example:
        from numpy as np
        cartesian(4,np.pi,(3/4)*np.pi)
            
    """
    x =  r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    cartesian_coords = {'x': x,
                     'y': y,
                     'z': z}
    
    return cartesian_coords

fig = plt.figure()
axes = fig.gca(projection='3d')
r_coord = np.linspace(0,1)
phi_coord = np.linspace(0,2*np.pi)
theta_coord = np.linspace(0,2*np.pi)
coords = cartesian(r_coord,phi_coord,theta_coord)
axes.plot(coords['x'],coords['y'],coords['z'])


if __name__ == '__main__':
    print("(r=0,phi=0,theta=0) = ", cartesian(0,0,0))
    print("(r=1,phi=pi,theta=0)", cartesian(1,np.pi,0))