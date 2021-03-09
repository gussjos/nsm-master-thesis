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
def diffusion_preprocess(A):
    B2=(A-np.mean(A,axis=0))/np.mean(A,axis=0)

    ono=np.ones((200,1))
    ono=ono/np.sum(ono)

    B2-=convolve2d(B2,ono,mode="same")
    B2-=convolve2d(B2,np.transpose(ono),mode="same")
    B2-=np.expand_dims(np.mean(B2,axis=0),axis=0)
    a=np.std(B2,axis=0)
    B2=B2/a
    B2=skimage.measure.block_reduce(B2,(1,4),np.mean)
    return B2,a
#%% Preprocess and save
plt.close('all')

downscaling_factor = 4
save_files = 1
plot = 1
save_pictures = 1

img_path = '/home/gustaf/nsm_data/experimental_data/'

folders = os.listdir(img_path)
folders = [folder for folder in folders if 'conc7' in folder]

for folder in tqdm.tqdm(folders):
    
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
    frame_nbr = ''
    split_images = 0
    for filename in tqdm.tqdm(files):
        ExpData = IO.loadmat(load_path + '/' + filename)
        full_data = ExpData["data"]["Im"][0][0]
        
        idcs = []
        for j,timestep in enumerate(full_data[:,0]):
            if np.all(full_data[j,:] == np.zeros(full_data[j,:].shape)):
                idcs.append(j)
        if len(idcs) < 0:
            idcs = [len(full_data)]
                
        prev_idx = 0
        shift = 50
        for j,idx in enumerate(idcs):
            data = full_data[prev_idx+shift:idx-shift]
            prev_idx = idx
            
            Bsm_D,a = diffusion_preprocess(data)
            Bsm_iOC = intensity_preprocess(data)
            
            #plt.imshow(data.T,aspect='auto')
            #plt.colorbar()
            #plt.plot(np.std(Bsm_D,axis=0))
                
            # Plot diffusion img
            if plot:
                plt.figure(figsize=(16,2))
                i1 = 256
                plt.imshow(Bsm_D[11:-11].T,aspect='auto')#,vmin=-1,vmax=1)
                plt.title(filename)
                plt.colorbar()
            
            if idx != len(files):
                frame_nbr = str(idx)
            else:
                frame_nbr = ''
            
            if save_files:
                np.save(diffusion_save_path + filename + frame_nbr, Bsm_D)
                np.save(intensity_save_path + filename + frame_nbr, Bsm_iOC)
                print(' Files saved.')
                
            if save_pictures:
                plt.savefig(pictures_save_path+filename[:-4]+ frame_nbr +'.png')
                plt.close('all')
            

#%% Function fit
A = np.array([4.8,5,5.2])
A = np.sqrt(A)
#A = np.array([1.8,2,2.2])/2
#A = np.array([0.95,1,1.05])/1
#A += np.min(A)
B = np.array([10,20,50])
plt.plot(B,A)


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b):
    return a*np.log(b*x)

popt, pcov = curve_fit(func, B, A)

x = np.linspace(5,100)
plt.plot(x,func(x,*popt))

D = 75
print(func(D,*popt))

#%%        ### Delete blank camera shots ###
idcs = []
for j,timestep in enumerate(data[:,0]):
    if np.all(data[j,:] == np.zeros(data[j,:].shape)):
        idcs.append(j)
J = 5 # nbr of frames to remove on each side of blank shot
shifts = np.arange(-J,J)
shifted_idcs = []

for shift in shifts:
    idcs_to_add = np.array(idcs)+shift
    idcs_to_add = list(idcs_to_add)
    shifted_idcs+= idcs_to_add
shifted_idcs += idcs
#data = np.delete(data,shifted_idcs,0)
###