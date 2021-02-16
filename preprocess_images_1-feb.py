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

main_load_path = '/home/gustaf/nsm_data/experimental_data/'
folder = 'ferritin/2020-10-13-ferritin/ferritin/channel_17_30x85/'
load_path = main_load_path + folder


img_path = '/home/gustaf/nsm_data/experimental data/'

main_save_path = '/home/gustaf/nsm_data/preprocessed_experimental_data/'
save_path = main_save_path + folder

try:
   os.makedirs(save_path + '/diffusion/', exist_ok=True)
except:
    print()
    print('Directory already exists.')
try:
   os.makedirs(save_path + '/intensity/', exist_ok=True)
except:
    print('Directory already exists.')

files = os.listdir(load_path)

for filename in tqdm.tqdm(files):
    print(filename)
    if os.path.isfile(save_path+ '/diffusion/' + filename):
        continue
    ExpData = IO.loadmat(load_path + filename)
    data = ExpData["data"]["Im"][0][0]
    
    Bsm_D,a = diffusion_preprocess(data,downscaling_factor)
    #Bsm_iOC = intensity_preprocess(data)
        
    # Plot diffusion img
    plt.figure(figsize=(16,2))
    i1 = 256
    plt.imshow(Bsm_D.T[:,i1:1024+i1],aspect='auto')
    plt.colorbar()
    plt.show()
        
    # Plot intensity img
    #plt.figure(figsize=(16,2))
    #i1 = 256
    #plt.imshow(Bsm_iOC.T[:,i1:1024+i1],aspect='auto')
    #plt.colorbar()
    #plt.show()
    
    if save_imgs:
        np.save(save_path + '/diffusion/' + filename, Bsm_D)
        #np.save(save_path + '/intensity/' + filename, Bsm_iOC)
        print(' Files saved.')