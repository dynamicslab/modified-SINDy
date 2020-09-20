# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:51:06 2020

@author: kahdi
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

#%% SoftStart?
softstart=1

#%% Define the parameters to swipe

# Noise level to use
#NoiseLevelArray=np.linspace(2,50,25)
NoiseLevelArray=np.linspace(2,10,17)
NoiseNum=len(NoiseLevelArray)

# Sparsity parameters to use
lamArray=np.linspace(0.01,0.5,90)
lamNum=len(lamArray)

# Set a pin to generate new noise every run
pinArray=np.linspace(0,4,5)
pinNum=len(pinArray)

#%% Define some parameters to swipe the different noise level

# Define the simulation parameters
p0=np.array([0.5])

# Define the initial conditions
x0=np.array([-2.0,-1.0])
x0_test=np.array([-3.0,2.0])

# Define the time points
T=10.0
T_test=5.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(VanderPol(np.transpose(x), 0, p0))

x_test=odeint(VanderPol,x0_test,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx_test=np.transpose(VanderPol(np.transpose(x_test), 0, p0))

libOrder=3
Theta_test=Lib(x_test,libOrder)
        
# Get the data size info
stateVar,dataLen=np.transpose(x).shape

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

# Define the parameters for the SINDy
N_iter=15
disp=0
NormalizeLib=0

#%%
SuccessOrNot=np.zeros((NoiseNum,pinNum))
ParameterError=np.zeros((NoiseNum,pinNum))
Xi_Matrix=np.zeros((NoiseNum,pinNum,np.size(Xi_base,0),np.size(Xi_base,1)))

for i in range(NoiseNum):
    print("Using noise level ",NoiseLevelArray[i],"\n")
    # Define the Noise Mag
    NoiseMag=[np.std(x[:,ij])*NoiseLevelArray[i]*0.01 for ij in range(stateVar)]
    
    for j in range(pinNum):
        print("\t Using using random seed ",pinArray[j],"\n")
        # Define noise seed
        np.random.seed(int(pinArray[j]))
        
        # Define noise
        Noise=np.hstack([NoiseMag[ij]*np.random.randn(dataLen,1) for ij in range(stateVar)])
        
        xn=x+Noise
        
        # Process noise if needed
        if softstart==1:
            # Soft Start
            NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
            NoiseEs=np.transpose(NoiseEs)
            xes=np.transpose(xes)
        else:
            # Hard Start
            xes=xn
        
        # Prepare the derivative and library
        dxes=CalDerivative(xes,dt,1)
        Theta=Lib(xes,libOrder)
        
        # Set dummy variables to store the variable
        Xi=[]
        Success_dum=np.zeros((lamNum,1))
        Evec_dum=np.zeros((lamNum,1))
        
        # Swipe the sparsity parameters
        for k in range(lamNum):
            print("\t\t Setting lambda as ",lamArray[k],"\n")
            Xi0=SINDy(Theta,dxes,lamArray[k],N_iter,disp,NormalizeLib)
            Xi.append(Xi0)
            
            # Calculate the testing error
            Evec_dum[k]=np.linalg.norm(dx_test-Theta_test@Xi0,'fro')**2/np.linalg.norm(dx_test,'fro')**2
        
            # Store whether it successed ot not
            if np.linalg.norm((Xi0!=0).astype(int)-(Xi_base!=0).astype(int))==0:
                Success_dum[k]=1
            
        if sum(Success_dum)>0:
            SuccessOrNot[i,j]=1
            Xi0=Xi[np.argmax(Success_dum)]
            Xi_Matrix[i,j,:,:]=Xi0
            ParameterError[i,j]=np.linalg.norm(Xi_base-Xi0)/np.linalg.norm(Xi_base)
        else:
            Xi0=Xi[np.argmin(Evec_dum)]
            Xi_Matrix[i,j,:,:]=Xi0
            ParameterError[i,j]=np.linalg.norm(Xi_base-Xi0)/np.linalg.norm(Xi_base)
        

    


