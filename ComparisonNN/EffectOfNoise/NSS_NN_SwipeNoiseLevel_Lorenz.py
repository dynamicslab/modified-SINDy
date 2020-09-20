# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:41:55 2020

@author: kahdi
"""
# =============================================================================
# The following code will swipe the effect of noise level on using neural network to perform noise signal speration 
# We set T=25, dt=0.01, q=10, x0=[-5.0,5.0,25.0], ro=0.9, beta=10**(-8), gamma=10**(-5)
# NN has 3 hidden layers with 64 neurons each layler. 
# Those parameters are obtianed from: https://arxiv.org/abs/1808.02578
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
FolderName="Result_NewIC_SwipeNoiseLevel\\"
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
N_train=30000

# Define the simulation parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
#x0=np.array([-5.0,5.0,25.0])
x0=np.array([5.0,5.0,25.0])

# Define the time points
T=25.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

# Get the data size info
stateVar,dataLen=np.transpose(x).shape

# Define the data type
dataType=tf.dtypes.float32

dt=tf.constant(dt)

# Define the neuron size
ly1=64

# Define the l2 norm penalty for the neural network weights
beta=10**(-8)

# Define the noise penalty
gamma=10**(-5)

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
NoisePercentageToSwipe=[0,2,4,6,8,10,12,14,16,18,20]
NoiseNum=len(NoisePercentageToSwipe)

# Set a pin to generate new noise every run
pin=0

# Set a list to store the noise value
NoiseList=[]
NoiseIDList=[]
NoiseEsList=[]
TrainTimeList=np.zeros((N_run,NoiseNum))
Enoise_error_List=np.zeros((N_run,NoiseNum))
Evector_field_error_list=np.zeros((N_run,NoiseNum))
Epre_error_list=np.zeros((N_run,NoiseNum))
x_sim_list=[]

# Softstart?
Softstart=1

#%%
# =============================================================================
# Start the noise level swip from here! Good luck!
# =============================================================================
for i in range(N_run):
    print("This is the run:",str(i+1),"\n")
    for j in range(NoiseNum):
        # Recompute computational graph by calling tf.function one more time
        RK45_F,RK45_B,SliceNoise,Prediction,WeightMSE,OneStepLoss_NSS_NN,Train_NSS_NN,ID_Accuracy_NN=ReloadFunction()
        
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
        
        # Get the estimate of the noise
        NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
        NoiseEs=np.transpose(NoiseEs)
        xes=np.transpose(xes)
        NoiseEsList.append(NoiseEs)
        
        print("\t Generating the new model to train on...")
        # Refine the neural network to initialize the training process
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(ly1, activation=tf.nn.relu, input_shape=(stateVar,),kernel_regularizer=tf.keras.regularizers.l2(l=beta)),  # input shape required
            tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
            tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
            tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
            tf.keras.layers.Dense(stateVar,kernel_regularizer=tf.keras.regularizers.l2(l=beta))
            ])
        
        
        print("\t Setting up the parameters...\n")
        # Get the middel part of the measurement data (it will be define as constant)
        Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)

        # Ge the forward and backward measurement data (it is a constant that wouldn't change)
        Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
        Ypre_F=tf.constant(Ypre_F,dtype=dataType)
        Ypre_B=tf.constant(Ypre_B,dtype=dataType)
        
        # Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
        if Softstart==1:
            NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
        else:
            NoiseVar=tf.Variable(tf.random.normal((dataLen,stateVar), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None))
        
        # Satrt training!
        print("\t Start training...\n\n")
         
        NoiseID,totalTime=Train_NSS_NN(Y0,Ypre_F,Ypre_B,NoiseVar,model,weights,dt,q,stateVar,dataLen,gamma,optimizer,N_train)

        print("\t The total training time on GPU is",totalTime,"sec...\n")
        
        Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_NN(x,dx,Noise,NoiseID,model,dataLen,dt)
        
        print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
        print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
        print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")

        
        # Save the trained model
        print("\t\t Current training loop finished, saving the trained model...\n")
        SavedModelName=SavePath+'Run_'+str(i+1)+'_NoiseLevel_'+str(NoisePercentageToSwipe[j])
        model.save(SavedModelName)
        
        # Store the result of identified noise and training time
        NoiseIDList.append(NoiseID)
        x_sim_list.append(x_sim)
        TrainTimeList[i,j]=totalTime
        Enoise_error_List[i,j]=Enoise_error
        Evector_field_error_list[i,j]=Evector_field_error
        Epre_error_list[i,j]=Epre_error
        
#%%
print("\n\n\n\n Training finished! Please save the file using the Spyder variable explorer!")

        



    


