# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:10:39 2020

@author: kahdi
"""
# =============================================================================
# The following code will swipe the effect of noise level on using SINDy to perform noise signal speration 
# We set T=25, dt=0.01, q=10, x0=[-5.0,5.0,25.0], ro=0.9.
# The SINDy library has 9 nonlinear features. We also perform the loop multiple times to record the performance of SINDy.
# =============================================================================

#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS_Swipe import *
import time
import tensorflow_probability as tfp
from datetime import datetime
import os
import importlib
#%% Create a path and folder to save the result
FolderName="Result_NSS_SwipeNoiseLevel_HighSampling\\"
FilePath=os.getcwd()
SavePath=FilePath+'\\'+FolderName
# Create the folder
try:
    os.mkdir(SavePath)
    print("The file folder does not exist, will create a new one....\n")
except:
    print("The folder already exist, will store the new result in current folder...\n")

#%% Define some parameters to swipe the different noise level
# Define how mant times you would like to run each noise level
N_run=10

# Define how many iterations you allow when training your model
N_train=5000

# Define the number of SINDy loop
Nloop=8

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
#x0=np.array([-5.0,5.0,25.0])
x0=np.array([5.0,5.0,25.0])

# Define the time points
T=25.0
dt=0.001

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

# Define the SINDy parameters
# Test the SINDy
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=2
lam=0.12

# Check the GPU status
CheckGPU()

# Define the prediction step
#q=10
q=3

# Define the optimizer to that will updates the Neural Network weights
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
        
# Get the weights for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Define the noise percent you would like to swipe
#NoisePercentageToSwipe=[25,30,35,40,45,50]
NoisePercentageToSwipe=[0,2,4,6,8,10,12,14,16,18,20]
NoiseNum=len(NoisePercentageToSwipe)

# Set a pin to generate new noise every run
pin=0

# Set a list to store the noise value
NoiseList=[]
NoiseEsList=[]
NoiseIDList=[]
TrainTimeList=np.zeros((N_run,NoiseNum,Nloop))
Enoise_error_List=np.zeros((N_run,NoiseNum,Nloop))
Evector_field_error_list=np.zeros((N_run,NoiseNum,Nloop))
Epre_error_list=np.zeros((N_run,NoiseNum,Nloop))
x_sim_list=[]
Xi_List=[]
Xi0_List=[]

# Softstart?
Softstart=0

#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    print("This is the run:",str(i+1),"\n")
    
    for j in range(NoiseNum):
        # Set up the lambda
        if j==0:
            lam=0.05
        elif j<=5:
            lam=0.1
        else:
            lam=0.15
        
        # Recompute computational graph by calling tf.function one more time
        LibGPU,RK45_F_SINDy,RK45_B_SINDy,SliceNoise,Prediction_SINDy,CalDerivativeMatrix,WeightMSE,OneStepLoss_NSS_SINDy,Train_NSS_SINDy,NSS_SINDy,ID_Accuracy_SINDy=ReloadFunction_SINDy()
        
        print("\t Setting the noise percentage as:",NoisePercentageToSwipe[j],"%\n")
        # First, let's set the noise for this run
        # Define the random seed for the noise generation
        pin=pin+1
        np.random.seed(pin)
        
        # Generate the noise
        NoiseMag=[np.std(x[:,i])*NoisePercentageToSwipe[j]*0.01 for i in range(stateVar)]
        Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
        
        # Store the generated noise 
        NoiseList.append(Noise)
        
        # Add the noise and get the noisy data
        xn=x+Noise
        
        print("\t Getting initial guess...")
        
        # Get the derivative of the noisy data, we directly take the derivative here without smoothing
        if Softstart==1:
            # Do the smoothing
            NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
            NoiseEs=np.transpose(NoiseEs)
            xes=np.transpose(xes)
            NoiseEsList.append(NoiseEs)
            
            # Get derivative and library
            dxes=CalDerivative(xes,dt,1)
            Theta=Lib(xes,libOrder)
            
            # Get initial guess
            Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
            
            # Set up optimization parameter
            NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
        else:
            dxn=CalDerivative(xn,dt,1)
            Theta=Lib(xn,libOrder)
            Xi0=SINDy(Theta,dxn,lam,N_SINDy_Iter,disp,NormalizeLib)
            NoiseVar=tf.Variable(tf.random.normal((dataLen,stateVar), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None))
            
        # Store initial guess
        Xi0_List.append(Xi0)
        
        print("\t The initial guess of the parameters are:")
        print(Xi0)
        
        # Define the initial guess of the selection parameters
        Xi=tf.Variable(Xi0,dtype=dataType)
        
        # Set the initial active matrix (all active)
        Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)
        
        print("\t Setting up the parameters...\n")
        # Get the middel part of the measurement data (it will be define as constant)
        Y=tf.constant(xn,dtype=dataType)
        Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)
        
        # Ge the forward and backward measurement data (it is a constant that wouldn't change)
        Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
        Ypre_F=tf.constant(Ypre_F,dtype=dataType)
        Ypre_B=tf.constant(Ypre_B,dtype=dataType)
        
        # Satrt training!
        print("\t Start training...\n\n")
        
        for k in range(Nloop):
            print("Runing the loop ",str(k+1))
            # Denoise the signal
            NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
            
            print("\t Current loop takes ",totalTime)
            # After the first iteration, minus the noise identified from the noisy measurement data
            xes=xn-NoiseID
            dxes=CalDerivative(xes,dt,1)
            
            Theta=Lib(xes,libOrder)
            
            print("Current Xi result")
            print(Xi)
            
            # Do SINDy on the denoised data
            index_min=abs(Xi.numpy())>lam
            Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
            Xi_num=Xi.numpy()
            Xi_num=Xi_num*Xi_act_dum
            index_min=Xi_act_dum.astype(bool)
            
            # Regress
            for r in range(3):
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
            
            # Store the result of identified noise and training time
            NoiseIDList.append(NoiseID)
            x_sim_list.append(x_sim)
            TrainTimeList[i,j,k]=totalTime
            Enoise_error_List[i,j,k]=Enoise_error
            Evector_field_error_list[i,j,k]=Evector_field_error
            Epre_error_list[i,j,k]=Epre_error
            Xi_List.append(Xi.numpy())
            
#%%
print("\n\n\n\n Training finished! Please save the file using the Spyder variable explorer!")

#%%

ind1=7

plt.figure()
plt.subplot(3,1,1)
plt.plot(Epre_error_list[0,:,ind1])
plt.title("Epre")
plt.tight_layout
plt.subplot(3,1,2)
plt.plot(Evector_field_error_list[0,:,ind1])
plt.title("Ef")
plt.tight_layout
plt.subplot(3,1,3)
plt.plot(Enoise_error_List[0,:,ind1])
plt.title("En")
plt.tight_layout

#%%
ind2=7
plt.figure()
plt.subplot(3,1,1)
plt.plot(Epre_error_list[0,ind2,:])
#plt.yscale("log")
plt.title("Epre")

plt.subplot(3,1,2)
plt.plot(Evector_field_error_list[0,ind2,:])
#plt.yscale("log")
plt.title("Ef")


plt.subplot(3,1,3)
plt.plot(Enoise_error_List[0,ind2,:])
plt.yscale("log")
plt.title("En")
plt.tight_layout




