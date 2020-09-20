# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:24:02 2020

@author: kahdi
"""
#%% Import packages
import numpy as np
import tensorflow as tf
from scipy.linalg import qr
import time

#%% Define functions here
# =============================================================================
# Define the ODE of Van der Pol
# =============================================================================
def  VanderPol(u,t,p):
    du1=u[1]
    du2=p[0]*(1-u[0]**2)*u[1]-u[0]
    
    du=np.array([du1,du2])
    
    #p0=0.5
    
    return du

# =============================================================================
# Define the ODE for cubic oscalator
# =============================================================================
def CubicOsc(u,t,p):
    du1=p[0]*u[0]**3+p[1]*u[1]**3
    du2=p[2]*u[0]**3+p[3]*u[1]**3
    
    du=np.array([du1,du2])
    
    #p0=[-0.1,2,-2,-0.1]
    
    return du 

# =============================================================================
# Define the ODE for the Lorenz system
# =============================================================================
def Lorenz(u, t, p):
   du1 = p[0]*u[0]+p[1]*u[1]
   du2 = p[2]*u[0]+p[3]*u[0]*u[2]+p[4]*u[1]
   du3 = p[5]*u[0]*u[1]+p[6]*u[2]
   
   du=np.array([du1,du2,du3])
   
   #p0=[-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3]
   
   return du

# =============================================================================
# Define the function for calculating the derivative
# =============================================================================
def CalDerivative(x,dx,d):
    # First we get the information of the data length. The x should be a n x m vector.
    Dev=np.zeros(x.shape)
    n,m=x.shape
    
    # Define the coeficient for different orders of derivative
    if d==1:
        p1=1/12 
        p2=-2/3 
        p3=0 
        p4=2/3 
        p5=-1/12
    elif d==2:
        p1=-1/12
        p2=4/3
        p3=-5/2
        p4=4/3
        p5=-1/12
    elif d==3:
        p1=-1/2
        p2=1
        p3=0
        p4=-1
        p5=1/2
    
    
    # Calculate the derivative of the middel point
    for i in range(2,n-2):
        Dev[i,:]=(p1*x[i-2,:]+p2*x[i-1,:]+p3*x[i,:]+p4*x[i+1,:]+p5*x[i+2,:])
        if d==1:
            Dev[i,:]=Dev[i,:]/dx
        elif d==2:
            Dev[i,:]=Dev[i,:]/dx^2
        elif d==3:
            Dev[i,:]=Dev[i,:]/dx^3

    # Ge the derivative of first two points using forward difference
    if d==1:
        q1=-3/2
        q2=2
        q3=-1/2
        q4=0
        q5=0
    elif d==2:
        q1=2
        q2=-5
        q3=4
        q4=-1
        q5=0
    elif d==3:
        q1=-5/2
        q2=9
        q3=-12
        q4=7
        q5=-3/2
    
    for i in range(2):
        Dev[i,:]=(q1*x[i,:]+q2*x[i+1,:]+q3*x[i+2,:]+q4*x[i+3,:]+q5*x[i+4,:])
        if d==1:
            Dev[i,:]=Dev[i,:]/dx;
        elif d==2:
            Dev[i,:]=Dev[i,:]/dx^2;
        elif d==3:
            Dev[i,:]=Dev[i,:]/dx^3;

    
    # Get the derivative of last two points using backward difference
    if d==1:
        m1=3/2,
        m2=-2,
        m3=1/2,
        m4=0,
        m5=0
    elif d==2:
        m1=2
        m2=-5
        m3=4
        m4=-1
        m5=0
    elif d==3:
        m1=5/2
        m2=-9
        m3=12
        m4=-7
        m5=3/2
    
    for i in range(n-2,n):
        Dev[i,:]=(m1*x[i,:]+m2*x[i-1,:]+m3*x[i-2,:]+m4*x[i-3,:]+m5*x[i-4,:])
        if d==1:
            Dev[i,:]=Dev[i,:]/dx;
        elif d==2:
            Dev[i,:]=Dev[i,:]/dx^2;
        elif d==3:
            Dev[i,:]=Dev[i,:]/dx^3;

    return Dev

# =============================================================================
# Check whether the GPU is available
# =============================================================================
def CheckGPU():
    if tf.test.is_gpu_available():
        print("\n\n\n\n\n")
        print("The GPU is available")
        print("\n\n\n\n\n")
    else:
        print("\n\n\n\n\n")
        print("The GPU is not available")
        print("\n\n\n\n\n")
    return None
 
# =============================================================================
# Define the RK45 for the forward prediction, for general function
# =============================================================================
@tf.function
def RK45_F(xin,f,dt):
    K1=f(xin)*dt
        
    K2=f(tf.math.add(xin,tf.constant(0.5)*K1))*dt
        
    K3=f(tf.math.add(xin,tf.constant(0.5)*K2))*dt                          
        
    K4=f(tf.math.add(xin,K3))*dt

    return tf.math.add_n([xin,tf.constant(1/6)*K1,tf.constant(1/3)*K2,tf.constant(1/3)*K3,tf.constant(1/6)*K4])

# =============================================================================
# Define the RK45 for the backward prediction, for general function
# =============================================================================
@tf.function
def RK45_B(xin,f,dt):
    K1=-f(xin)*dt
        
    K2=-f(tf.math.add(xin,tf.constant(0.5)*K1))*dt
        
    K3=-f(tf.math.add(xin,tf.constant(0.5)*K2))*dt                          
        
    K4=-f(tf.math.add(xin,K3))*dt

    return tf.math.add_n([xin,tf.constant(1/6)*K1,tf.constant(1/3)*K2,tf.constant(1/3)*K3,tf.constant(1/6)*K4])

# =============================================================================
# Define a function that will generate multiple initial condtion for the forward and backward simulation.
# The input tensor should be n x m where the n is the number of states and m is the time horizon.
# =============================================================================
def GetInitialCondition(X,q,n):
    if q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        X0=X[q:n-q,:]

    return X0

# =============================================================================
# Define a function that will slice the data provided
# =============================================================================
def SliceData(Y,q,dataLen):
    if q==0:
        Ypre_F=Y
        Ypre_B=Y
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else: 
        Ypre_F=[]
        Ypre_B=[]
        for j in range(1,q+1):
            if j==1:
                Ypre_F=Y[q+j:dataLen-q+j,:]
                Ypre_B=Y[q-j:dataLen-q-j,:]
            else:
                Ypre_F=np.append(Ypre_F,Y[q+j:dataLen-q+j,:],axis=1)
                Ypre_B=np.append(Ypre_B,Y[q-j:dataLen-q-j,:],axis=1)      

    return Ypre_F,Ypre_B

# =============================================================================
# Define a function that will slice the noise variable into future and previous state
# =============================================================================
@tf.function
def SliceNoise(NoiseVar,q,dataLen,stateVar):
    if q==0:
        NoiseVar_F=Noise
        NoiseVar_B=Noise
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else: 
        NoiseVar_F=tf.slice(NoiseVar,[q+1,0],[dataLen-2*q,stateVar])
        NoiseVar_B=tf.slice(NoiseVar,[q-1,0],[dataLen-2*q,stateVar])

        for i in range(1,q):
            NoiseVar_F=tf.concat([NoiseVar_F,tf.slice(NoiseVar,[q+1+i,0],[dataLen-2*q,stateVar])],axis=1)
            NoiseVar_B=tf.concat([NoiseVar_B,tf.slice(NoiseVar,[q-1-i,0],[dataLen-2*q,stateVar])],axis=1)
            
    return NoiseVar_F,NoiseVar_B

# =============================================================================
# Define a function will calculate the prediction result given initial condition matrix and prediction step.
# =============================================================================
@tf.function
def Prediction(X0,f,dt,q,stateVar,dataLen):
    if q==0:
        Xpre_F=X0
        Xpre_B=X0
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:    
        Xpre_F=RK45_F(X0,f,dt)
        Xpre_B=RK45_B(X0,f,dt)
        
        for i in range(q-1):
            Xpre_F=tf.concat([Xpre_F,RK45_F(tf.slice(Xpre_F,[0,stateVar*i],[dataLen-2*q,stateVar]),f,dt)],axis=1)
            Xpre_B=tf.concat([Xpre_B,RK45_B(tf.slice(Xpre_B,[0,stateVar*i],[dataLen-2*q,stateVar]),f,dt)],axis=1)
        
    return Xpre_F,Xpre_B

# =============================================================================
# Define a function that calculate the decay factor
# =============================================================================
def DecayFactor(ro,stateVar,q):
    if q==0:
        weights=1
    elif q<0:
        raise Exception("The prediction step must be equals or greater than zero")
    else:
        weights=[]
        for j in range(q):
            for i in range(stateVar):
                weights=np.append(weights,ro**(j))
    
    return weights

# =============================================================================
# Calculate the derivative of the measurement. The first two and lst two point will be discarded.
# =============================================================================
@tf.function
def CalDerivativeMatrix(Y,dataLen,stateVar,dt):
    p1=tf.constant(1/12)
    p2=tf.constant(-2/3)
    p3=tf.constant(0.0)
    p4=tf.constant(2/3)
    p5=tf.constant(-1/12)
    
    Dev=tf.math.add_n([tf.math.multiply(p1,tf.slice(Y,[0,0],[dataLen-4,stateVar])),\
                   tf.math.multiply(p2,tf.slice(Y,[1,0],[dataLen-4,stateVar])),\
                   tf.math.multiply(p3,tf.slice(Y,[2,0],[dataLen-4,stateVar])),\
                   tf.math.multiply(p4,tf.slice(Y,[3,0],[dataLen-4,stateVar])),\
                   tf.math.multiply(p5,tf.slice(Y,[4,0],[dataLen-4,stateVar]))])/dt
    
    return Dev

# =============================================================================
# Define a function that calculate the weighted mean suqare error
# =============================================================================
@tf.function
def WeightMSE(Yt_F,Xp_F,Yt_B,Xp_B,NoiseVar_F,NoiseVar_B,weights):
    # Calculate the prediction of noisy data
    Yp_F=Xp_F+NoiseVar_F
    Yp_B=Xp_B+NoiseVar_B
    
    # Calculate the loss
    Jwmse_F=tf.reduce_mean(tf.math.multiply(tf.math.squared_difference(Yt_F,Yp_F),weights))
    Jwmse_B=tf.reduce_mean(tf.math.multiply(tf.math.squared_difference(Yt_B,Yp_B),weights))
    
    Jwmse=tf.math.add(Jwmse_F,Jwmse_B)
    
    return Jwmse

# =============================================================================
# Define the one step loss function for the noise signal speration: NN approach
# =============================================================================
@tf.function
def OneStepLoss_NSS_NN(Y0,Ypre_F,Ypre_B,NoiseVar,model,weights,dt,q,stateVar,dataLen,gamma,optimizer):
    with tf.GradientTape() as g:
        # First get the initial condition guess based on the measurement data and noise variable
        X0=tf.math.subtract(Y0,tf.slice(NoiseVar,[q,0],[dataLen-2*q,stateVar]))
        
        # Next, simulate the system forward and backward
        Xpre_F,Xpre_B=Prediction(X0,model,dt,q,stateVar,dataLen)
        
        # Similarly get the forward and backward noise
        NoiseVar_F,NoiseVar_B=SliceNoise(NoiseVar,q,dataLen,stateVar)
        
        # Next calculate the weighted loss
        Jw=WeightMSE(Ypre_F,Xpre_F,Ypre_B,Xpre_B,NoiseVar_F,NoiseVar_B,weights)
        
        # Next calculate the noise penalty
        Jn=gamma*tf.nn.l2_loss(NoiseVar)
        
        # Finally, add all the noise together
        J=tf.math.add(Jw,Jn)
        
    # Calculate the gradient with respect to the variables
    optVal=model.trainable_variables
    optVal.append(NoiseVar)
    gard=g.gradient(J,optVal)
    optimizer.apply_gradients(zip(gard,optVal))
        
    return J

# =============================================================================
# Now define the training function for the noise signal speration: NN approach
# =============================================================================
def Train_NSS_NN(Y0,Ypre_F,Ypre_B,NoiseVar,model,weights,dt,q,stateVar,dataLen,gamma,optimizer,N_train):
    start=time.time() 
    for i in range(N_train):
        # Calculate the cost and updte the gradient
        J=OneStepLoss_NSS_NN(Y0,Ypre_F,Ypre_B,NoiseVar,model,weights,dt,q,stateVar,dataLen,gamma,optimizer)
        
        if i%1000==0:
            tf.print(J)
    
    totoalTime=time.time()-start
    
    return NoiseVar.numpy(),totoalTime

# =============================================================================
# Define a function that calculates the noise signal speration accuracy
# =============================================================================
def ID_Accuracy_NN(x,dx,Noise,NoiseID,model,dataLen,dt):
    Enoise_error=np.linalg.norm(Noise-NoiseID,'fro')**2/dataLen
    Evector_field_error=np.linalg.norm(dx-model(tf.constant(x,dtype='float32')),'fro')**2/np.linalg.norm(dx,'fro')**2
    xpre=[]
    xpre=RK45_F(tf.constant([x[0,:]],dtype="float32"),model,dt).numpy()
    
    try:
        for i in range(1,dataLen-1):
            dummy=RK45_F(tf.constant([xpre[-1]],dtype="float32"),model,dt).numpy()
            xpre=np.append(xpre,[dummy[0]],axis=0)
        
        Epre_error=np.linalg.norm(x[1:]-xpre,'fro')**2/np.linalg.norm(x,'fro')**2
    except:
        print("The simulation blows up...Current Neural Network is not stable...")
        Epre_error=float('nan')
    
    
    return Enoise_error,Evector_field_error,Epre_error,xpre

# =============================================================================
# Define a function that will generate gasiian 
# =============================================================================
def Gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# =============================================================================
# This code is used for approximate the noise that we added into the signal.
# Reference: https://github.com/snagcliffs/RKNN
# =============================================================================
def approximate_noise(Y, lam):
	n,m = Y.shape

	D = np.zeros((m,m))
	D[0,:4] = [2,-5,4,-1]
	D[m-1,m-4:] = [-1,4,-5,2]

	for i in range(1,m-1):
	    D[i,i] = -2
	    D[i,i+1] = 1
	    D[i,i-1] = 1
	    
	D = D.dot(D)

	X_smooth = np.vstack([np.linalg.solve(np.eye(m) + lam*D.T.dot(D), Y[j,:].reshape(m,1)).reshape(1,m) for j in range(n)])

	N_hat = Y-X_smooth

	return N_hat, X_smooth



