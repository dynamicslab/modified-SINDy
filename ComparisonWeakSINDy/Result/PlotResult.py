#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:53:17 2020

@author: kadikadi
"""

# =============================================================================
# This file will plot the merged result of swiping the noise level
# =============================================================================
#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#%% Load result file

# Define the plot parameters
def ChangeFontSize(fontSize):
    plt.rc('xtick', labelsize=fontSize)    
    plt.rc('ytick', labelsize=fontSize) 
    
    return None

def CustomizeViolin(data,line_color,edge_color,lw1):
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)

    inds = np.arange(0,len(medians))

    plt.plot(medians,linewidth=lw1,color=line_color,alpha=0.6,zorder=0)
    
    return None

# Define the color
color_WS=[0.5,0.3,0.1]
color_NSS_SINDy='c'
edge_color="black"

# Define the transparancy
transAlpha=0.95

# Define the line width
lw1=8
lw2=2
lw3=3.5

width_violin=0.25

# Define the lable
labelNSS_SINDy="NSS-SINDy"

# Define the marker properties
MarkerSize_Violin=6
MarkerEdgeWidth=3
MarkerSize=200

NoisePercentageToSwipe=NoisePercentageToSwipe[0,:]
#%% Plot the parameter error

quartile1a, median1, quartile1b = np.percentile(Ep_WS_01, [25, 50, 75], axis=0)
quartile2a, median2, quartile2b = np.percentile(Ep_WS_001, [25, 50, 75], axis=0)
quartile3a, median3, quartile3b = np.percentile(Epm_Final01, [25, 50, 75], axis=0)
quartile4a, median4, quartile4b = np.percentile(Epm_Final001, [25, 50, 75], axis=0)

plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(36)

# Plot the line
plt.plot(median1,linewidth=lw1,color=color_WS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(median3,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the violin plot
sns.violinplot(data=Ep_WS_01,cut=0,inner="box",width=width_violin,scale="width",color=color_WS,linewidth=lw2,saturation=transAlpha,zorder=0)
sns.violinplot(data=Epm_Final01,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0)

# Plot the scatter
plt.scatter(range(NoiseNum),median1,s=MarkerSize,marker='o',color=color_WS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),median3,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),NoisePercentageToSwipe)

plt.yscale('log')
plt.yticks([1e-1,1e-3,1e-5])
plt.grid()
plt.savefig("Plots/Epm01.pdf")


plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(36)

# Plot the line
plt.plot(median2,linewidth=lw1,color=color_WS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(median4,linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')


# Plot the violin plot
sns.violinplot(data=Ep_WS_001,cut=0,inner="box",width=width_violin,scale="width",color=color_WS,linewidth=lw2,saturation=transAlpha,zorder=0)
sns.violinplot(data=Epm_Final001,cut=0,inner="box",width=width_violin,scale="width",color=color_NSS_SINDy,linewidth=lw2,saturation=transAlpha,zorder=0)

# Plot the scatter
plt.scatter(range(NoiseNum),median2,s=MarkerSize,marker='o',color=color_WS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),median4,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),NoisePercentageToSwipe)

plt.yscale('log')
plt.yticks([1e0,1e-3,1e-6,1e-9,1e-12])
plt.grid()
plt.savefig("Plots/Epm001.pdf")


#%% Plot success rate

plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(36)


# Plot the line
plt.plot(np.mean(SuccessOrNot_WS_01,axis=0)*100,linewidth=lw1,color=color_WS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(SuccessOrNot_NSS_SINDy01[0,:],linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the scatter
plt.scatter(range(NoiseNum),np.mean(SuccessOrNot_WS_01,axis=0)*100,s=MarkerSize,marker='o',color=color_WS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),SuccessOrNot_NSS_SINDy01,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),NoisePercentageToSwipe)

#plt.yscale('log')
plt.yticks([0,25,50,75,100])
plt.ylim([0,105])
plt.grid()
plt.savefig("Plots/SuccessRate01.pdf")


plt.figure(figsize=(14,5))
# Cahnge the Font size
ChangeFontSize(36)

# Plot the line
plt.plot(np.mean(SuccessOrNot_WS_001,axis=0)*100,linewidth=lw1,color=color_WS,alpha=0.6,zorder=0,linestyle='solid')
plt.plot(SuccessOrNot_NSS_SINDy001[0,:],linewidth=lw1,color=color_NSS_SINDy,alpha=0.6,zorder=0,linestyle='solid')

# Plot the scatter
plt.scatter(range(NoiseNum),np.mean(SuccessOrNot_WS_001,axis=0)*100,s=MarkerSize,marker='o',color=color_WS,edgecolors=edge_color,linewidth=lw3)
plt.scatter(range(NoiseNum),SuccessOrNot_NSS_SINDy001,s=MarkerSize,marker='o',color=color_NSS_SINDy,edgecolors=edge_color,linewidth=lw3)

plt.xticks(range(NoiseNum),NoisePercentageToSwipe)

#plt.yscale('log')
plt.ylim([0,105])
plt.yticks([0,25,50,75,100])
plt.grid()
plt.savefig("Plots/SuccessRate001.pdf")



