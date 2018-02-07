# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:33:24 2018

@author: Brian

The purpose of this code is to visualize the u-array when solving for one
column of u-values at some xi location.
"""

Ny = 11  # Number of elements in y-direction

uList = []  # Initialize list
for n in range(1, Ny-1):
    s = 'u_i+1,{}'.format(n)
    uList.append(s)

print(uList)
