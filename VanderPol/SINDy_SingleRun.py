#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:50:56 2020

@author: kadikadi
"""

# =============================================================================
# The following code will swipe the effect of noise level on SINDy
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from utils_NSS_SINDy import *
import time
from datetime import datetime
import os

#%%
NoiseLevel=6.5

# Set a pin to generate new noise every run
pin=2

# SoftStart?
softstart=1

#%% Define some parameters to swipe the different noise level

# Define the simulation parameters
p0=np.array([0.5])

# Define the initial conditions
x0=np.array([-2.0,-1.0])

# Define the time points
T=10.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(VanderPol(np.transpose(x), 0, p0))

libOrder=3
        
# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the sparsity parameter you would like to use
lam=0.02

# Define the parameters for the SINDy
N_iter=25
disp=0
NormalizeLib=0

# Define the true parameters
Xi_base=np.array([[0,-1],
                  [1,0.5],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,0],
                  [0,-0.5],
                  [0,0],
                  [0,0]])

#%%

np.random.seed(pin)

# Generate the noise
NoiseMag=[np.std(x[:,i])*NoiseLevel*0.01 for i in range(stateVar)]
Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

# Add the noise and get the noisy data
xn=x+Noise

if softstart==1:
    # Soft Start
    NoiseEs,xes=approximate_noise(np.transpose(xn), 3)
    NoiseEs=np.transpose(NoiseEs)
    xes=np.transpose(xes)
else:
    # Hard Start
    xes=xn

# Prepare the derivative and library
dxes=CalDerivative(xes,dt,1)
Theta=Lib(xes,libOrder)

Xi=SINDy(Theta,dxes,lam,N_iter,disp,NormalizeLib)


print(Xi)


print(np.linalg.norm(Xi_base-Xi)/np.linalg.norm(Xi_base))

    
#%% Plot the true signal and denoised signal

plt.figure()
plt.plot(x[:,0])
plt.plot(xes[:,0])






