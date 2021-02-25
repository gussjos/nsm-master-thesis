#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:35:40 2021

@author: gustaf
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as IO
A = IO.loadmat('/home/gustaf/nsm_data/mixed data benchmarks barbora/collection_D11.mat')
#%% Load collections
plt.close('all')
N = A['collection']['N'][0][0][0]

idcs = np.where(N>40)

Deff_mean = A['collection']['Deff_mean'][0][0][0][idcs]
iOC_mean = A['collection']['iOC_mean'][0][0][0][idcs]

#plt.hist(Deff_mean,100)
plt.hist(iOC_mean,100)
#%% Add N of each prediction
iOC_preds = np.zeros(0)
D_preds = np.zeros(0)
for j,iOC_val in enumerate(iOC_mean):
    
    nbr = int(N[j])
    iOC = iOC_mean[j]
    D = Deff_mean[j]
    
    iOC_vals = nbr*[iOC]
    D_vals = nbr*[D]
    iOC_preds = np.concatenate((iOC_preds,np.array(iOC_vals)),axis=0)
    D_preds = np.concatenate((D_preds,np.array(D_vals)),axis=0)

plt.hist(iOC_preds,bins=100)
#%% Gaussian fits to iOC
yolo_preds = np.copy(iOC_preds)*10000

#% Gaussian fits
from pylab import *
from scipy.optimize import curve_fit
    
yolo_preds = np.array(yolo_preds)

# For iOC 1 and 2
plt.figure(figsize=(6,3))
data=np.copy(yolo_preds[yolo_preds<3])
nbr_bins = 300
y,x,_=hist(data,nbr_bins,alpha=.3,label='data')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

#expected=(1,0.2,1, 2,.5,3)
expected=(1,0.2,len(iOC_preds)/10, 2,.5,len(iOC_preds)/10)
params,cov=curve_fit(bimodal,x,y,expected)

sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=2,label='model')
legend()

ioc1_gauss_params = params[:3]
ioc2_gauss_params = params[3:]

x_middle = np.linspace(ioc1_gauss_params[0],ioc2_gauss_params[0])
y_middle = bimodal(x_middle,*params)
plt.plot(x_middle,y_middle,color='blue')
idx = np.argmin(y_middle)
ioc12 = x_middle[idx]
plt.axvline(ioc12,color='green')
ioc12

#% For iOC 2 and iOC 5
plt.figure(figsize=(6,3))
data=np.copy(yolo_preds[yolo_preds>ioc12])
y,x2,_=hist(data,nbr_bins,alpha=.3,label='data')

x2=(x2[1:]+x2[:-1])/2 # for len(x)==len(y)

expected=(2,.5,len(iOC_preds)/10, 5,.5,len(iOC_preds)/10) # for mixed data
expected=(2,.5,len(iOC_preds)/10, 5,.5,len(iOC_preds)/10)
params,cov=curve_fit(bimodal,x2,y,expected)
sigma=sqrt(diag(cov))
plot(x2,bimodal(x2,*params),color='red',lw=2,label='model')
legend()
ioc5_gauss_params = params[3:]

x_middle = np.linspace(ioc2_gauss_params[0],ioc5_gauss_params[0])
y_middle = bimodal(x_middle,*params)
plt.plot(x_middle,y_middle,color='blue')
idx = np.argmin(y_middle)
ioc25 = x_middle[idx]
plt.axvline(ioc25,color='green')
ioc25

plt.figure(figsize=(12,3))
hist(yolo_preds,nbr_bins,alpha=.3,label='data')
g1 = gauss(x,*ioc1_gauss_params)
g2 = gauss(x,*ioc2_gauss_params)
g5 = gauss(x2,*ioc5_gauss_params)
plt.plot(x,g1,color='red')
plt.plot(x,g2,color='red')
plt.plot(x2,g5,color='red')

ioc1_gauss_sigma = ioc2_gauss_params[1]
ioc2_gauss_sigma = ioc2_gauss_params[1]
ioc5_gauss_sigma = ioc5_gauss_params[1]
ioc1_gauss_peak = ioc1_gauss_params[0]
ioc2_gauss_peak = ioc2_gauss_params[0]
ioc5_gauss_peak = ioc5_gauss_params[0]

nbr_sigma = 3
plt.axvline(ioc12,color='red',linestyle='--',label=r"cutoff between ioc1 and ioc2")
plt.axvline(ioc2_gauss_peak + nbr_sigma*ioc2_gauss_sigma,color='black',linestyle='--',label=r"cutoffs at {}$\sigma$".format(nbr_sigma))
plt.axvline(ioc5_gauss_peak + nbr_sigma*ioc5_gauss_sigma,color='black',linestyle='--')
plt.axvline(ioc5_gauss_peak - nbr_sigma*ioc5_gauss_sigma,color='black',linestyle='--')
print('ioc2 upper cutoff: {:.2f}'.format(ioc2_gauss_peak + nbr_sigma*ioc2_gauss_sigma))
print('ioc5 lower cutoff: {:.2f}'.format(ioc5_gauss_peak - nbr_sigma*ioc5_gauss_sigma))
print('ioc5 upper cutoff: {:.2f}'.format(ioc5_gauss_peak + nbr_sigma*ioc5_gauss_sigma))
print(ioc1_gauss_peak)
print(ioc2_gauss_peak)
print(ioc5_gauss_peak)
print(ioc1_gauss_sigma)
print(ioc2_gauss_sigma)
print(ioc5_gauss_sigma)
plt.legend()



# Save the parameters
gauss_params = {"iOC0.0001":{"mu":ioc1_gauss_peak,"sigma":ioc1_gauss_sigma,"ioc12_cutoff":ioc12},
               "iOC0.0002":{"mu":ioc2_gauss_peak,"sigma":ioc2_gauss_sigma},
               "iOC0.0005":{"mu":ioc5_gauss_peak,"sigma":ioc5_gauss_sigma}}

#%% Filter measurements w.r.t gauss peak and width

# init dict
iOCs = [1,2,5]
filtered_preds = {}
for iOC in iOCs:
    iOC_str = "iOC0.000" + str(iOC)
    filtered_preds[iOC_str] = {"D":[],"iOC":[]}
    
#% Filter iOC preds

nbr_sigma = 3
iOC_preds = 1e4*np.copy(iOC_preds)
for iOC in iOCs: 
    iOC_str = "iOC0.000" + str(iOC)
    mu = gauss_params[iOC_str]["mu"]
    sigma = gauss_params[iOC_str]["sigma"]
    
    lower = mu - nbr_sigma*sigma
    upper = mu + nbr_sigma*sigma
    idcs = [(iOC_preds > lower) & (iOC_preds < upper)]
    filtered_preds[iOC_str]["iOC"] = iOC_preds[idcs]
    filtered_preds[iOC_str]["D"] = D_preds[idcs]
    
#%% Plot histograms of filtered preds
Ds = [50,20,10]
iOCs = [1,2,5]
for j,iOC in enumerate(iOCs):
    iOC_str = "iOC0.000" + str(iOC)
    D = Ds[j]
    D_preds = filtered_preds[iOC_str]["D"]
    nbr_bins = int(len(D_preds)/2)
    y,x,_=hist(D_preds,nbr_bins,alpha=.3,label='data')
        
    mu = D
    sigma = D/10
    print(len(y))
    height = len(y)/5
    expected_params = (mu,sigma,height)
    
    params,cov=curve_fit(gauss,x[:-1],y,expected_params)
    plot(x,gauss(x,*params),color='red',lw=2,label='model')
    print(params)
    
    
#%% 


iOC_means = [4.97, 1.92, 1.10]
iOC_stds = np.array([1.84, 1.178, 1.174])/10
D_means = [43.7,20.15,10.07]
D_stds = np.array([1.25,3.125,19])

plt.plot(iOC_means,np.flip(D_means),'.')
plt.errorbar(iOC_means,np.flip(D_means),
             xerr=np.flip(iOC_stds)/2,
             yerr=D_stds/2,
             capsize=4,fmt='.')
plt.grid(True)
plt.xlim(1,6)
plt.ylim(5,55)