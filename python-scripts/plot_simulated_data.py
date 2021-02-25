import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as IO

#%% Load datafiles
path = '/home/gustaf/Desktop/velocity0_distance20_timesteps10000/'
filename = filenames[0]
file = path + filename

filenames = os.listdir(path)

SimData = IO.loadmat(file)

#%%
Im = SimData["data"]["Im"][0][0]
Yum = SimData["data"]["Yum"][0][0]
time = SimData["data"]["time"][0][0]
response = SimData["data"]["responce"][0][0]
 