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
save_files = 1
save_pictures = 1

img_path = '/home/gustaf/nsm_data/experimental_data/'

folders = os.listdir(img_path)
for folder in folders:
    
    load_path = img_path + folder
    main_save_path = '/home/gustaf/nsm_data/preprocessed_experimental_data/'
    save_path = main_save_path + folder
    diffusion_save_path = save_path + '/diffusion/'
    intensity_save_path = save_path + '/intensity/'
    pictures_save_path = save_path + '/pictures/'
    
    try:
       os.makedirs(diffusion_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + diffusion_save_path + ' already exists, passing.')
        continue
    try:
       os.makedirs(intensity_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + intensity_save_path + ' already exists, passing.')
        continue
    try:
       os.makedirs(pictures_save_path, exist_ok=True)
    except:
        print()
        print('Directory:' + intensity_save_path + ' already exists, passing.')
        continue
    
    files = os.listdir(load_path)
    print('len(files):' + str(len(files)))
    
    for filename in tqdm.tqdm(files):
        ExpData = IO.loadmat(load_path + '/' + filename)
        data = ExpData["data"]["Im"][0][0]
        
        Bsm_D,a = diffusion_preprocess(data,downscaling_factor)
        Bsm_iOC = intensity_preprocess(data)
            
        # Plot diffusion img
        plt.figure(figsize=(16,2))
        i1 = 256
        plt.imshow(Bsm_D.T[:,i1:1024+i1],aspect='auto')
        plt.title(filename)
        plt.colorbar()
        
        if save_files:
            np.save(diffusion_save_path + filename, Bsm_D)
            np.save(intensity_save_path + filename, Bsm_iOC)
            print(' Files saved.')
            
        if save_pictures:
            plt.savefig(pictures_save_path+filename[:-4]+'.pdf')
            plt.close('all')