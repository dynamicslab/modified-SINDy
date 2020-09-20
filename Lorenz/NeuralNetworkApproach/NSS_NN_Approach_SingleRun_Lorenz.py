# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:54:43 2020

@author: kahdi
"""
#%% Import packages
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
import matplotlib.pyplot as plt
from utils_NSS import *
import time
import tensorflow_probability as tfp
from datetime import datetime
#from keras.models import Sequential
#from keras.layers import Dense 

#%% Define how many percent of noise you need
NoisePercentage=0

#%% Simulate
# Define the random seed for the noise generation
np.random.seed(0)

# Define the parameters
p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

# Define the initial conditions
x0=np.array([5.0,5.0,25.0])

# Define the time points
T=25.0
dt=0.01

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

# Get the size info
stateVar,dataLen=np.transpose(x).shape

# Generate the noise
NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

# Add the noise and get the noisy data
xn=x+Noise

# SoftStart?
softstart=1

#%% Now plot the result of Lorenz

plt.figure()

plt.subplot(3,1,1)
pp1=plt.plot(t,x[:,0],linewidth=0.5,color='k')
pp1=plt.scatter(t,xn[:,0],s=0.5,color='b')
plt.ylabel('x')
plt.grid(True)

plt.subplot(3,1,2)
pp2=plt.plot(t,x[:,1],linewidth=0.5,color='k')
pp1=plt.scatter(t,xn[:,1],s=0.5,color='b')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3,1,3)
pp3=plt.plot(t,x[:,2],linewidth=0.5,color='k')
pp1=plt.scatter(t,xn[:,2],s=0.5,color='b')
plt.ylabel('z')
plt.xlabel('t')
plt.tight_layout()
plt.grid(True)

plt.figure()
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0], x[:,1], x[:,2], 'gray')
pp4.plot3D(xn[:,0], xn[:,1], xn[:,2], linestyle='-.',color='red')
#%% Define a neural network
# Check the GPU status
CheckGPU()

# Define the data type
dataType=tf.dtypes.float32

dt=tf.constant(dt)

# Define the neuron size
ly1=64

# Define the l2 norm penalty for the neural network weights
beta=10**(-8)

# Define the noise penalty
gamma=10**(-5)

# Define a neural network
model = tf.keras.Sequential([
  tf.keras.layers.Dense(ly1, activation=tf.nn.relu, input_shape=(stateVar,),kernel_regularizer=tf.keras.regularizers.l2(l=beta)),  # input shape required
  tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
  tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
  tf.keras.layers.Dense(ly1, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
  tf.keras.layers.Dense(stateVar,kernel_regularizer=tf.keras.regularizers.l2(l=beta))
])


#%% Define the data
# Define the prediction step
q=3

# Get the middel part of the measurement data (it will be define as constant)
Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)

# Ge the forward and backward measurement data (it is a constant that wouldn't change)
Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
Ypre_F=tf.constant(Ypre_F,dtype=dataType)
Ypre_B=tf.constant(Ypre_B,dtype=dataType)

# Get the weight for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

# Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
if softstart==1:
    # Soft Start
    NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
    NoiseEs=np.transpose(NoiseEs)
    xes=np.transpose(xes)
    NoiseVar=tf.Variable(NoiseEs, dtype=tf.dtypes.float32)
else:
    # Hard Start
    NoiseEs=np.random.randn(xn.shape[0],xn.shape[1])
    xes=xn-NoiseEs
    NoiseVar=tf.Variable(tf.random.normal((dataLen,stateVar), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None))

#%% Define the optimizer

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-07)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

N_train=30000

#%% Finally start training!
NoiseID,totoalTime=Train_NSS_NN(Y0,Ypre_F,Ypre_B,NoiseVar,model,weights,dt,q,stateVar,dataLen,gamma,optimizer,N_train)
print("The total training time on GPU is",totoalTime)

#%% Now plot the noise signal speration result

# First plot the noise: true v.s. identified
StartIndex=200 # Choose how many noise data point you would like to plot
EndIndex=500

plt.figure()
plt.subplot(3,1,1)
pp1=plt.plot(t[StartIndex:EndIndex],Noise[StartIndex:EndIndex,0],linewidth=1.5,color='b')
pp1=plt.plot(t[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,0],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nx')
plt.xlabel('t')
plt.legend(['Noise Truth:x', 'Noise Estimate:x'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,2)
pp1=plt.plot(t[StartIndex:EndIndex],Noise[StartIndex:EndIndex,1],linewidth=1.5,color='b')
pp1=plt.plot(t[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,1],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Ny')
plt.xlabel('t')
plt.legend(['Noise Truth:y', 'Noise Estimate:y'],loc='upper right')
plt.grid(True)

plt.subplot(3,1,3)
pp1=plt.plot(t[StartIndex:EndIndex],Noise[StartIndex:EndIndex,2],linewidth=1.5,color='b')
pp1=plt.plot(t[StartIndex:EndIndex],NoiseID[StartIndex:EndIndex,2],linewidth=1.5,color='k',linestyle='--')
plt.ylabel('Nz')
plt.xlabel('t')
plt.legend(['Noise Truth:z', 'Noise Estimate:z'],loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.figure()
pp4=plt.axes(projection='3d')
pp4.plot3D(x[:,0], x[:,1], x[:,2], color='black',linewidth=1)
pp4.plot3D((xn-NoiseID)[:,0], (xn-NoiseID)[:,1], (xn-NoiseID)[:,2], color='red',linestyle='--',linewidth=1)
pp4.plot3D((xn)[:,0], (xn)[:,1], (xn)[:,2], color='blue',linestyle='-.',linewidth=1)

Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_NN(x,dx,Noise,NoiseID,model,dataLen,dt)

print("The error between the true noise and estimated noise is:",Enoise_error)
print("The error between the true vector field and estimated vector field is:",Evector_field_error)
print("The error between the true trajector and simulted trajectory is:",Epre_error)

#%%
fontSize=60
plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontSize) 

preLen=500
plt.figure(figsize=(30,25))
pp5=plt.axes(projection='3d')
pp5.plot3D(x[1:preLen+1,0], x[1:preLen+1,1], x[1:preLen+1,2], color='black',linewidth=8)
pp5.plot3D(x_sim[0:preLen,0], x_sim[0:preLen,1], x_sim[0:preLen,2], color='m',linestyle='--',linewidth=8)
pp5.plot([x[0,0]], [x[0,1]], [x[0,2]], markerfacecolor='black', markeredgecolor='orange', marker='*', markeredgewidth=5,markersize=60, alpha=1)
pp5.view_init(30, -30)
pp5.grid(True)
pp5.axis('on')
pp5.set_facecolor("white")

Epre_short=np.linalg.norm(x[1:preLen+1]-x_sim[0:preLen],'fro')**2/np.linalg.norm(x[1:preLen+1],'fro')**2

print("The short term prediction error is:",Epre_short)

plt.savefig("NN_Lorez_Pre_LongTrain.pdf")

#%%
fontSize=10
plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontSize) 
n_bins=5
plt.figure()
plt.subplot(3,1,1)
plt.grid(True)
pp6=plt.hist(Noise[:,0], color = 'blue', alpha=0.9,edgecolor = 'black',bins = int(180/n_bins),density="true")
pp6=plt.hist(NoiseID[:,0], color = 'orange', alpha=0.75,edgecolor = 'black',bins = int(180/n_bins),density="true")
#pp6=plt.plot(np.linspace(-3, 3, 1000),0.5*Gaussian(np.linspace(-3, 3, 1000), 0, 1),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Density')
plt.xlabel('Noise:x')
axes = plt.gca()
#axes.set_xlim([-3,3])
#axes.set_ylim([0,0.6])

plt.subplot(3,1,2)
plt.grid(True)
pp6=plt.hist(Noise[:,1], color = 'blue', alpha=0.9,edgecolor = 'black',bins = int(180/n_bins),density="true")
pp6=plt.hist(NoiseID[:,1], color = 'orange', alpha=0.75,edgecolor = 'black',bins = int(180/n_bins),density="true")
#pp6=plt.plot(np.linspace(-3, 3, 1000),0.5*Gaussian(np.linspace(-3, 3, 1000), 0, 1),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Density')
plt.xlabel('Noise:y')
axes = plt.gca()
#axes.set_xlim([-3,3])
#axes.set_ylim([0,0.6])

plt.subplot(3,1,3)
plt.grid(True)
pp6=plt.hist(Noise[:,2], color = 'blue', alpha=0.9,edgecolor = 'black',bins = int(180/n_bins),density="true")
pp6=plt.hist(NoiseID[:,2], color = 'orange', alpha=0.75,edgecolor = 'black',bins = int(180/n_bins),density="true")
#pp6=plt.plot(np.linspace(-3, 3, 1000),0.5*Gaussian(np.linspace(-3, 3, 1000), 0, 1),color ='black',alpha=0.9,linewidth=2.5)
plt.ylabel('Density')
plt.xlabel('Noise:z')
axes = plt.gca()
#axes.set_xlim([-3,3])
#axes.set_ylim([0,0.6])
plt.tight_layout()

    


