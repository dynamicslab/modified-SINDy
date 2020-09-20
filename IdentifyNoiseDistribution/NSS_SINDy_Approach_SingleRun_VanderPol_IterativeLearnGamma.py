# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:57:42 2020

@author: kahdi
"""

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_SINDy import *
import time
from datetime import datetime
import seaborn as sns
from fitter import Fitter

#%% Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    
    return None

#%% Define how many percent of noise you need
NoisePercentage=20

#%% Simulate
NoiseType='gamma'

# Define the random seed for the noise generation
np.random.seed(0)

# Define the parameters
p0=np.array([0.5])

# Define the initial conditions
x0=np.array([-2,1])

# Define the time points
T=10.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(VanderPol(np.transpose(x), 0, p0))

# Get the size info
stateVar,dataLen=np.transpose(x).shape

# Generate the noise
NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]

Noise=np.hstack([NoiseMag[i]*np.random.gamma(1,1,(dataLen,1)) for i in range(stateVar)])

# Add the noise and get the noisy data
xn=x+Noise
  
# Test the SINDy
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=3
lam=0.15

# SoftStart?
softstart=1

#%% Now plot the result of Van der Pol

lw=5

plt.figure()
pp1=plt.plot(x[:,0],x[:,1],linewidth=lw,color='k',linestyle='-')
pp1=plt.plot(xn[:,0],xn[:,1],linewidth=1.0,color='red',linestyle='--')
plt.ylabel('x')
plt.grid(False)
plt.axis('off')

#%% Define a neural network
# Check the GPU status
CheckGPU()

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

#%% Define the data
# Define the prediction step
q=2

# Get the weight for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

#%% Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)

#%% Finally start training!
Nloop=4
N_train=5000

# Set a list to store the noise value
NoiseLoopID=[]
NoiseList=[]
NoiseIDList_SingleRun=[]
TrainTimeList_SingleRun=np.zeros((Nloop,1))
Enoise_error_List_SingleRun=np.zeros((Nloop,1))
Evector_field_error_list_SingleRun=np.zeros((Nloop,1))
Epre_error_list_SingleRun=np.zeros((Nloop,1))
x_sim_list_SingleRun=[]
Xi_List_SingleRun=[]
xn_list=[]

# Define the mean of the noise
NoiseMean=np.array([0,0])

for jj in range(3):
    print("Iteration",str(jj+1))
    if jj==0:
        xn=xn-NoiseMean
    else:
        NoiseMean=np.mean(NoiseID,axis=0)
        xn=xn-NoiseMean
        
    # Store it
    xn_list.append(xn)
    
    # Get the middel part of the measurement data (it will be define as constant)
    Y=tf.constant(xn,dtype=dataType)
    Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)
    
    # Ge the forward and backward measurement data (it is a constant that wouldn't change)
    Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
    Ypre_F=tf.constant(Ypre_F,dtype=dataType)
    Ypre_B=tf.constant(Ypre_B,dtype=dataType)
    
    if softstart==1:
        # Soft Start
        NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
        NoiseEs=np.transpose(NoiseEs)
        xes=np.transpose(xes)
    else:
        # Hard Start
        NoiseEs=np.zeros((xn.shape[0],xn.shape[1]))
        xes=xn-NoiseEs
            
    dxes=CalDerivative(xes,dt,1)
    
    # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
    NoiseVar=tf.Variable(NoiseEs,dtype=tf.dtypes.float32)
    
    if jj==0:
        # Get the initial guess of the SINDy parameters
        Theta=Lib(xes,libOrder)
    
        Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
    else:
        if softstart==1:
            Xi0=Xi.numpy()
        else:
            # Get the initial guess of the SINDy parameters
            Theta=Lib(xes,libOrder)
        
            Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
            #Xi0=Xi.numpy()
            
    print(Xi0)
    
    # Define the initial guess of the selection parameters
    Xi=tf.Variable(Xi0,dtype=dataType)
    
    # Set the initial active matrix
    Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)
    
    for k in range(Nloop):
        print("Runing the loop ",str(k+1))
        # Denoise the signal
        NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
        
        print("\t Current loop takes ",totalTime)
        # After the first iteration, minus the noise identified from the noisy measurement data
        xes=xn-NoiseID
        xes=xes[q+1:-q-1,:]
        dxes=CalDerivative(xes,dt,1)
             
        print("Current Xi result")
        print(Xi)
        
        # Do SINDy on the denoised data
        Theta=Lib(xes,libOrder)
        Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
        
        # Do SINDy on the denoised data
        index_min=abs(Xi.numpy())>lam
        Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
        Xi_num=Xi.numpy()
        Xi_num=Xi_num*Xi_act_dum
        index_min=Xi_act_dum.astype(bool)
        
        # Regress
        for r in range(xes.shape[1]):
            Xi_num[index_min[:,r],r]=solve_minnonzero(Theta[:,index_min[:,r]],dxes[:,r])
        
        # Print the new initial start point
        print("New Xi result")
        print(Xi_num)
        
        # Determine which term should we focus on to optimize next
        Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
        Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
        
        # Calculate the performance
        Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
        
        # Print the performance
        print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
        print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
        print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")
        
        NoiseIDList_SingleRun.append(NoiseID)
        x_sim_list_SingleRun.append(x_sim)
        TrainTimeList_SingleRun[k]=totalTime
        Enoise_error_List_SingleRun[k]=Enoise_error
        Evector_field_error_list_SingleRun[k]=Evector_field_error
        Epre_error_list_SingleRun[k]=Epre_error
        Xi_List_SingleRun.append(Xi.numpy())
    
    # Store the noise
    NoiseLoopID.append(NoiseID)
    
    
#%% Plot the distribution of noise
colorNoise='black'
colorNoiseID='green'

lw3=30
lw4=30

Alpha1=0.9
Alpha2=0.4

Bins=int(180/1)

plt.figure(figsize=(30,15))
ChangeFontSize(100)
plt.grid()

Iter=2

sns.distplot(xn_list[Iter][:,0]-x[:,0], hist=False, kde=True,
             bins=Bins, color = colorNoise, 
             hist_kws={'edgecolor':'black','alpha':Alpha1},
             kde_kws={'shade': False,'linewidth': lw4,'color':colorNoise,'linestyle':'-'},norm_hist=False)

sns.distplot(NoiseLoopID[Iter][:,0], hist=False, kde=True,
             bins=Bins, color = colorNoiseID, 
             hist_kws={'edgecolor':'black','alpha':Alpha2},
             kde_kws={'shade': False,'linewidth': lw3,'color':colorNoiseID,'linestyle':':'},norm_hist=False)

#%%% Plot the noisy data

color_trueState='k'
color_noisyState='red'

lw1=20
lw2=5

plt.figure(figsize=(30,25))

# Cahnge the Font size
ChangeFontSize(100)

Iter=2

plt.plot(x[q+1:-q-1,0],x[q+1:-q-1,1],linewidth=lw1,color=color_trueState,linestyle='-')
plt.plot(xn_list[Iter][q+1:-q-1,0],xn_list[Iter][q+1:-q-1,1],linewidth=lw2,color=color_noisyState,linestyle='--')
plt.grid(True)
plt.axis('on')

#%%
Iter=2
plt.figure()
pp1=plt.plot(x[q+1:-q-1,0],x[q+1:-q-1,1],linewidth=lw,color='k',linestyle='-')
pp1=plt.plot(xn_list[Iter][q+1:-q-1,0]-NoiseLoopID[Iter][q+1:-q-1,0],xn_list[Iter][q+1:-q-1,1]-NoiseLoopID[Iter][q+1:-q-1,1],linewidth=lw,color='orange',linestyle='--')
plt.ylabel('x')
plt.grid(False)
plt.axis('off')

#%%
Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)

print("The error between the true noise and estimated noise is:",Enoise_error)
print("The error between the true vector field and estimated vector field is:",Evector_field_error)
print("The error between the true trajector and simulted trajectory is:",Epre_error)


#%%
# Cahnge the Font size
ChangeFontSize(10)
plt.figure()
plt.plot(Enoise_error_List_SingleRun)
plt.ylabel('NoiseID Error')
plt.xlabel('Loop Index')

plt.figure()
plt.plot(Evector_field_error_list_SingleRun)
plt.ylabel('Derivative Error')
plt.xlabel('Loop Index')

plt.figure()
plt.plot(Epre_error_list_SingleRun)
plt.ylabel('Simulation Error')
plt.xlabel('Loop Index')

