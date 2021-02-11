#%% Imports
import sys
sys.path.append("..") #Adds the module to path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as IO
import os
import tqdm as tqdm
import skimage.measure
from scipy.signal import convolve2d

#%%  Define preprocessing functions
# Doesn't scale vs std of background (used for intensity predictions) /7 jan
import skimage.measure
from scipy.signal import convolve2d

def intensity_preprocess(A):
    B=(A-np.mean(A,axis=0))/np.mean(A,axis=0) # Remove the
    ono=np.ones((200,1))
    ono=ono/np.sum(ono)

    B-=convolve2d(B,ono,mode="same")
    B-=convolve2d(B,np.transpose(ono),mode="same")
    # remove local mean \pm 100 pixels
    
    B-=np.expand_dims(np.mean(B,axis=0),axis=0)
    B=1000*skimage.measure.block_reduce(B,(1,4),np.mean)
    
    return B

# Scales image vs background (used for segmentation and diffusion predictions) /7 jan
def diffusion_preprocess(A,downscaling_factor):
    B2=(A-np.mean(A,axis=0))/np.mean(A,axis=0)

    ono=np.ones((200,1))
    ono[0:80]=1
    ono[120:]=1
    ono=ono/np.sum(ono)
    ono2=np.ones((1,50))
    ono2/=np.sum(ono2)

    B2-=convolve2d(B2,ono,mode="same")
    B2-=convolve2d(B2,np.transpose(ono),mode="same")
    B2-=np.expand_dims(np.mean(B2,axis=0),axis=0)
    a=np.std(B2,axis=0)
    B2=B2/a
    B2=skimage.measure.block_reduce(B2,(1,downscaling_factor),np.mean)
    return B2,a
#%% Preprocess and save

downscaling_factor = 4
save_imgs = 1

img_path = '/home/gustaf/nsm data/7.5e-5/'
save_path_intensity = '/home/gustaf/nsm data/7.5e-4 preprocessed for intensity/'
save_path_diffusion = '/home/gustaf/nsm data/7.5e-4 preprocessed for diffusion/'

try:
    os.mkdir(save_path_intensity)
except:
    print('Directory already exists.')
try:
    os.mkdir(save_path_diffusion)
except:
    print('Directory already exists.')

files = os.listdir(img_path)
for file in tqdm.tqdm(files):
    ExpData = IO.loadmat(img_path + file)
    data = ExpData["data"]["Im"][0][0]
    
    Bsm_D,a = diffusion_preprocess(data,downscaling_factor)
    Bsm_iOC = intensity_preprocess(data)
        
    # Plot diffusion img
    plt.figure(figsize=(16,2))
    i1 = 256
    plt.imshow(Bsm_D.T[:,i1:1024+i1],aspect='auto')
    plt.colorbar()
    plt.show()
        
    # Plot intensity img
    plt.figure(figsize=(16,2))
    i1 = 256
    plt.imshow(Bsm_iOC.T[:,i1:1024+i1],aspect='auto')
    plt.colorbar()
    plt.show()
    
    if save_imgs:
        np.save(save_path_diffusion + file, Bsm_D)
        np.save(save_path_intensity + file, Bsm_iOC)
        print('Files saved.')