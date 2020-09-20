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
NoiseLevel=40

# Set a pin to generate new noise every run
pin=2

# SoftStart?
softstart=0

#%% Simulate
# Define the random seed for the noise generation
np.random.seed(0)

# Define the parameters, the first value determines the number of states while the second one defines the forcing
p0=np.array([6,8])

# Define the initial conditions
x0=p0[1]*np.ones(p0[0])
x0[0]=1

x0_test=p0[1]*np.ones(p0[0])
x0_test[0]=2

# Define the time points
T=25.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz96,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz96(np.transpose(x), 0, p0))

x_test=odeint(Lorenz96,x0_test,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx_test=np.transpose(Lorenz96(np.transpose(x_test), 0, p0))

libOrder=3
Theta_test=Lib(x_test,libOrder)
        
# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the sparsity parameter you would like to use
lam=0.08

# Define the parameters for the SINDy
N_iter=15
disp=0
NormalizeLib=0

# Define the true parameters
Theta_base=Lib(x,libOrder)
Xi_base=SINDy(Theta_base,dx,0.2,N_iter,disp,NormalizeLib)
Xi_base[Xi_base!=0]=-1
Xi_base[0,:]=8
Xi_base[9,1]=1
Xi_base[17,0]=1
Xi_base[15,2]=1
Xi_base[20,3]=1
Xi_base[24,4]=1
Xi_base[11,5]=1
print(Xi_base)

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

print(np.linalg.norm((Xi!=0).astype(int)-(Xi_base!=0).astype(int))==0)


